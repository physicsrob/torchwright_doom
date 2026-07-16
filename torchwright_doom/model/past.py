"""Graph-construction facade for the original's ``Past``-style reads.

``GraphPast`` stores graph ``Node`` references in lightweight handles and
lowers each read to an existing torchwright attention primitive. It does
not allocate runtime KV state and is not a compiler primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torchwright.graph import Node, RopeConfig
from torchwright.ops.linear import negate
from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_argmin_above_in_bucket,
    attend_argmin_above_integer,
    attend_mean_where,
    attend_most_recent_globally,
    attend_to_offset,
)
from torchwright.ops.swiglu.global_recency import global_position_from_bos

from .extract import input_type_code, is_type
from .render_constants import RECENCY_GAIN
from .vocab import BOS

# Build-time strategy flag for global_position(): True = the op's smoothed
# path (exact causal mean of the raw recovery, ×2).  Exists only so probes
# and A/B replays can rebuild the raw-tiebreak graph; production is True.
_SMOOTHED_GLOBAL_POSITION = True


@dataclass(frozen=True)
class PastHandle:
    """Capability for reading one graph-published channel."""

    name: str
    node: Node
    width: int
    source: Literal["user", "input"]
    _owner: object = field(repr=False, compare=False)


class GraphPast:
    """Thin graph-building facade over torchwright attention ops."""

    def __init__(self, *, input_vec: Node, rope: RopeConfig):
        self._check_node(input_vec, "input_vec")
        if not isinstance(rope, RopeConfig):
            raise TypeError(
                "GraphPast: rope must be a torchwright.graph.RopeConfig, "
                f"got {type(rope).__name__}"
            )
        self._input_vec = input_vec
        self._rope = rope
        self._global_position_node: Node | None = None
        self._owner = object()
        self._published_names: set[str] = set()
        self._input_type_handle: PastHandle | None = None

    def global_position(self) -> Node:
        """Each token's approximate absolute position (RoPE-derived, via BOS).

        Built once and cached.  This is the graph-computed monotone position
        scalar that replaces the old host counter column: ``global_position_from_bos``
        recovers position ``0..max_positions`` from the softmax weight every token
        places on the inert BOS token (``is_type(input_vec, BOS)``, 1.0 only at
        position 0).  It is used both as the absolute-position scalar for bounded
        pixel-index math (``pos - marker``) and as the recency tiebreak in
        :meth:`pick_most_recent`.  Provenance-clean: derived from rotary attention,
        never host-seeded.

        **Smoothed.**  The raw PWL recovery's compiled fp32 evaluation wanders
        with position (measured ±0.5 at 3.7k → ±10.4 at 54k; adjacent steps
        down to 0.53, which at ``RECENCY_GAIN=8`` left ~1.5% softmax leak on
        recency reads — the n_heads=32 regression class).  Production
        therefore takes the op's ``smoothed=True`` path: an exact uniform
        causal mean of the recovery, ×2, which restores adjacent steps to
        ≥0.965 at 54k and cuts the absolute envelope to ~±1.7 (receipts in
        ``smooth_recency_rank_derisked.md``).  ``_SMOOTHED_GLOBAL_POSITION``
        is a build-time A/B flag for probes; production keeps it True.
        """
        if self._global_position_node is None:
            bos_indicator = is_type(self._input_vec, BOS)
            self._global_position_node = global_position_from_bos(
                self._rope, bos_indicator, smoothed=_SMOOTHED_GLOBAL_POSITION
            )
        return self._global_position_node

    def publish(self, name: str, value: Node) -> PastHandle:
        """Name ``value`` for subsequent graph-past attention reads."""
        if not isinstance(name, str):
            raise TypeError(
                f"GraphPast.publish: name must be a str, got {type(name).__name__}"
            )
        if name.startswith("input."):
            raise ValueError(
                f"GraphPast.publish: cannot publish reserved input channel {name!r}"
            )
        if name in self._published_names:
            raise RuntimeError(
                f"GraphPast.publish: channel {name!r} was already published"
            )
        self._check_node(value, f"publish({name!r})")
        self._published_names.add(name)
        return PastHandle(
            name=name,
            node=value,
            width=len(value),
            source="user",
            _owner=self._owner,
        )

    def input_type(self) -> PastHandle:
        """Handle for the compact 8-wide E8 input type code."""
        if self._input_type_handle is None:
            node = input_type_code(self._input_vec)
            self._input_type_handle = PastHandle(
                name="input.type",
                node=node,
                width=len(node),
                source="input",
                _owner=self._owner,
            )
        return self._input_type_handle

    def input_slot(self, name: str) -> PastHandle:
        """Deferred: use typed token extraction instead of flat slots."""
        raise NotImplementedError(
            "GraphPast.input_slot is intentionally deferred. Use typed "
            "TOKEN.extract(input_vec, slot) / extract_derived(...) at the "
            "call site; full input_slot parity also needs an explicit "
            "validity handle to preserve the original published-bit semantics."
        )

    def pick_argmax(
        self,
        query: Node,
        key: PastHandle,
        value: PastHandle,
        *,
        match_gain: float = 200.0,
        exclude_self: bool = False,
    ) -> Node:
        """Pick the causal row with maximum ``query · key``."""
        if exclude_self:
            raise NotImplementedError(
                "GraphPast.pick_argmax(exclude_self=True) is not implemented; "
                "use attend_to_offset for explicit previous-position reads"
            )
        self._check_node(query, "pick_argmax query")
        key_node = self._check_handle(key, "pick_argmax key")
        value_node = self._check_handle(value, "pick_argmax value")
        self._check_query_width(query, key, "pick_argmax")
        return attend_argmax_dot(
            self._rope,
            query,
            key_node,
            value_node,
            match_gain=match_gain,
        )

    def pick_argmin(
        self,
        query: Node,
        key: PastHandle,
        value: PastHandle,
        *,
        match_gain: float = 200.0,
        exclude_self: bool = False,
    ) -> Node:
        """Dense-key argmin of ``query · key``."""
        if exclude_self:
            raise NotImplementedError(
                "GraphPast.pick_argmin(exclude_self=True) is not implemented; "
                "use attend_to_offset for explicit previous-position reads"
            )
        self._check_node(query, "pick_argmin query")
        key_node = self._check_handle(key, "pick_argmin key")
        value_node = self._check_handle(value, "pick_argmin value")
        self._check_query_width(query, key, "pick_argmin")
        return attend_argmax_dot(
            self._rope,
            negate(query),
            key_node,
            value_node,
            match_gain=match_gain,
        )

    def pick_most_recent(
        self,
        query: Node,
        key: PastHandle,
        value: PastHandle,
        *,
        match_gain: float = 200.0,
        recency_scale: float = RECENCY_GAIN,
    ) -> Node:
        """Pick the most recent causal row whose key matches ``query`` — **global**.

        Recency is resolved by the graph-derived absolute position
        (:meth:`global_position`), not the bounded rotary lobe: the tiebreak among
        content-matching keys is monotone over the whole ``[0, max_positions]``
        rollout, so a recency read can reach arbitrarily far back with no window
        cliff (the unbounded clip read DOOM needs).  The KV cache is unbounded, so
        every committed row stays readable for the whole run.

        This is the same mechanism the pre-RoPE renderer used — a global absolute
        position scaled by a per-position gain — so it inherits that scheme's
        sharpness.  ``recency_scale`` is that gain (``RECENCY_GAIN``, the old
        ``SCORE_GAIN = 8``): adjacent positions differ by ``recency_scale`` in the
        logit, so the most recent matching key wins the softmax sharply
        (``exp(8)`` ⇒ cond ≈ 0.9993), which a ±1 boolean marker read needs.  The
        torchwright op default ``recency_scale=1`` is far too soft (``exp(1)`` ⇒
        ~0.73 blend); DOOM threads ``8`` here.

        The per-adjacent-position logit gap is ``recency_scale`` times the
        position signal's real adjacent step.  The smoothed
        :meth:`global_position` keeps that step ≥ 0.965 over a full-cap
        rollout (measured; the raw recovery dipped to 0.53 at scale, leaking
        ~1.5% of the softmax to the runner-up — the n_heads=32 regression
        class; see ``smooth_recency_rank_derisked.md``).

        Content-dominance invariant (the caller must satisfy): a content-matched
        older key must beat an unmatched newer key, i.e.
        ``match_gain · min_match_dot_gap > recency_scale · max_positions``.  At
        ``recency_scale=8`` and the ``max_positions=65536`` cap that needs
        ``match_gain · min_match_dot_gap > 524_288``; DOOM callers pass
        ``MATCH_GAIN_LONG`` / ``MATCH_GAIN_CLIP`` (600,000 — 12.6% headroom).
        The facade default ``match_gain=200.0`` is the underlying op default
        and is not relied on by any caller.
        """
        self._check_node(query, "pick_most_recent query")
        key_node = self._check_handle(key, "pick_most_recent key")
        value_node = self._check_handle(value, "pick_most_recent value")
        self._check_query_width(query, key, "pick_most_recent")
        return attend_most_recent_globally(
            self._rope,
            query,
            key_node,
            self.global_position(),
            value_node,
            match_gain=match_gain,
            recency_scale=recency_scale,
        )

    def attend_to_offset(self, value: PastHandle, delta_pos: int = -1) -> Node:
        """Read ``value`` at a fixed causal offset.

        The KV cache is unbounded, so any causal offset stays readable for
        the whole run.
        """
        if not isinstance(delta_pos, int):
            raise TypeError(
                "GraphPast.attend_to_offset: delta_pos must be an int, "
                f"got {type(delta_pos).__name__}"
            )
        if delta_pos > 0:
            raise NotImplementedError(
                "GraphPast.attend_to_offset is causal-only; "
                f"got delta_pos={delta_pos}"
            )
        value_node = self._check_handle(value, "attend_to_offset value")
        return attend_to_offset(self._rope, value_node, delta_pos=delta_pos)

    def mean_where(self, validity: PastHandle, value: PastHandle) -> Node:
        """Uniform mean of ``value`` over rows whose validity is +1."""
        validity_node = self._check_handle(validity, "mean_where validity")
        value_node = self._check_handle(value, "mean_where value")
        return attend_mean_where(self._rope, validity_node, value_node)

    def pick_argmin_above(
        self,
        score: PastHandle,
        indicators_above: PastHandle,
        threshold_onehot: Node,
        value: PastHandle,
    ) -> Node:
        """Argmin of ``score`` among rows above a runtime threshold."""
        score_node = self._check_handle(score, "pick_argmin_above score")
        indicators_node = self._check_handle(
            indicators_above, "pick_argmin_above indicators_above"
        )
        self._check_node(threshold_onehot, "pick_argmin_above threshold_onehot")
        value_node = self._check_handle(value, "pick_argmin_above value")
        if len(indicators_above.node) != len(threshold_onehot):
            raise ValueError(
                "GraphPast.pick_argmin_above: indicators_above width "
                f"{len(indicators_above.node)} must match threshold_onehot "
                f"width {len(threshold_onehot)}"
            )
        return attend_argmin_above_integer(
            self._rope,
            score_node,
            indicators_node,
            threshold_onehot,
            value_node,
        )

    def pick_argmin_above_in_bucket(
        self,
        score: PastHandle,
        validity: PastHandle,
        key_bucket_onehot: PastHandle,
        score_above_each_threshold: PastHandle,
        query_bucket_onehot: Node,
        threshold_onehot: Node,
        value: PastHandle,
        *,
        assert_hardness_gt: float | None = None,
    ) -> Node:
        """Bucket-filtered argmin-above read over graph-published rows."""
        score_node = self._check_handle(score, "pick_argmin_above_in_bucket score")
        validity_node = self._check_handle(
            validity, "pick_argmin_above_in_bucket validity"
        )
        key_bucket_node = self._check_handle(
            key_bucket_onehot, "pick_argmin_above_in_bucket key_bucket_onehot"
        )
        above_node = self._check_handle(
            score_above_each_threshold,
            "pick_argmin_above_in_bucket score_above_each_threshold",
        )
        self._check_node(
            query_bucket_onehot, "pick_argmin_above_in_bucket query_bucket_onehot"
        )
        self._check_node(
            threshold_onehot, "pick_argmin_above_in_bucket threshold_onehot"
        )
        value_node = self._check_handle(value, "pick_argmin_above_in_bucket value")

        if len(key_bucket_node) != len(query_bucket_onehot):
            raise ValueError(
                "GraphPast.pick_argmin_above_in_bucket: key_bucket_onehot "
                f"width {len(key_bucket_node)} must match query_bucket_onehot "
                f"width {len(query_bucket_onehot)}"
            )
        if len(above_node) != len(threshold_onehot):
            raise ValueError(
                "GraphPast.pick_argmin_above_in_bucket: "
                "score_above_each_threshold width "
                f"{len(above_node)} must match threshold_onehot width "
                f"{len(threshold_onehot)}"
            )

        return attend_argmin_above_in_bucket(
            self._rope,
            score_node,
            validity_node,
            key_bucket_node,
            above_node,
            query_bucket_onehot,
            threshold_onehot,
            value_node,
            assert_hardness_gt=assert_hardness_gt,
        )

    @staticmethod
    def _check_node(node: Node, where: str) -> None:
        if not isinstance(node, Node):
            raise TypeError(
                f"GraphPast.{where}: expected a torchwright graph Node, "
                f"got {type(node).__name__}"
            )

    def _check_handle(self, handle: PastHandle, where: str) -> Node:
        if not isinstance(handle, PastHandle):
            raise TypeError(
                f"GraphPast.{where}: expected a PastHandle, "
                f"got {type(handle).__name__}"
            )
        if handle._owner is not self._owner:
            raise RuntimeError(
                f"GraphPast.{where}: handle {handle.name!r} belongs to a "
                "different GraphPast instance"
            )
        return handle.node

    @staticmethod
    def _check_query_width(query: Node, key: PastHandle, where: str) -> None:
        if len(query) != key.width:
            raise ValueError(
                f"GraphPast.{where}: query width {len(query)} does not match "
                f"key {key.name!r} width {key.width}"
            )


class PastHandleScope:
    """Per-build bundle of handles returned by ``GraphPast.publish``."""

    def __init__(self, past: GraphPast):
        self._past = past
        self._handles: dict[str, PastHandle] = {}

    def publish(self, name: str, value: Node) -> PastHandle:
        handle = self._past.publish(name, value)
        self._handles[name] = handle
        return handle

    # input_type / input_slot / pick_argmin_above / pick_argmin_above_in_bucket
    # are intentionally NOT redefined here: they are pure pass-throughs to the
    # wrapped GraphPast, which __getattr__ (below) already forwards. __getitem__'s
    # self.input_type() / self.input_slot(...) calls resolve through it too.

    def __getitem__(self, name: str) -> PastHandle:
        if name.startswith("input."):
            if name == "input.type":
                return self.input_type()
            return self.input_slot(name.removeprefix("input."))
        try:
            return self._handles[name]
        except KeyError as exc:
            raise RuntimeError(
                f"handle scope has no published channel {name!r}; use the "
                "PastHandle returned by publish(), or publish the channel "
                "before reading it"
            ) from exc

    def __getattr__(self, name: str):
        return getattr(self._past, name)
