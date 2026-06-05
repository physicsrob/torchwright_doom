"""Graph-construction facade for sandbox-style ``Past`` reads.

``GraphPast`` stores graph ``Node`` references in lightweight handles and
lowers each read to an existing torchwright attention primitive. It does
not allocate runtime KV state and is not a compiler primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torchwright.graph import Node, PosEncoding
from torchwright.ops import negate
from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_argmin_above_in_bucket,
    attend_argmin_above_integer,
    attend_mean_where,
    attend_most_recent_matching,
)

from .extract import input_type_code


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

    def __init__(self, *, input_vec: Node, pos_encoding: PosEncoding):
        self._check_node(input_vec, "input_vec")
        if not isinstance(pos_encoding, PosEncoding):
            raise TypeError(
                "GraphPast: pos_encoding must be a torchwright.graph.PosEncoding, "
                f"got {type(pos_encoding).__name__}"
            )
        self._input_vec = input_vec
        self._pos = pos_encoding
        self._owner = object()
        self._published_names: set[str] = set()
        self._input_type_handle: PastHandle | None = None

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
            "validity handle to preserve sandbox published-bit semantics."
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
    ) -> Node:
        """Pick the most recent causal row whose key matches ``query``.

        Plan C resolved the precision concern for long-span callers:
        torchwright's current attention paths run this matmul in fp32,
        so callers can pass ``MATCH_GAIN_LONG = 300_000.0``. At unit
        content gap that gain is safe up to roughly 37,500 positions;
        the facade default remains the underlying op default, ``200.0``.
        """
        self._check_node(query, "pick_most_recent query")
        key_node = self._check_handle(key, "pick_most_recent key")
        value_node = self._check_handle(value, "pick_most_recent value")
        self._check_query_width(query, key, "pick_most_recent")
        return attend_most_recent_matching(
            self._pos,
            query,
            key_node,
            value_node,
            match_gain=match_gain,
        )

    def attend_to_offset(self, value: PastHandle, delta_pos: int = -1) -> Node:
        """Read ``value`` at a fixed causal offset."""
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
        return self._pos.attend_to_offset(value_node, delta_pos=delta_pos)

    def mean_where(self, validity: PastHandle, value: PastHandle) -> Node:
        """Uniform mean of ``value`` over rows whose validity is +1."""
        validity_node = self._check_handle(validity, "mean_where validity")
        value_node = self._check_handle(value, "mean_where value")
        return attend_mean_where(self._pos, validity_node, value_node)

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
            self._pos,
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
            self._pos,
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

    def input_type(self) -> PastHandle:
        return self._past.input_type()

    def input_slot(self, name: str) -> PastHandle:
        return self._past.input_slot(name)

    def pick_argmin_above(
        self,
        score: PastHandle,
        indicators_above: PastHandle,
        threshold_onehot: Node,
        value: PastHandle,
    ) -> Node:
        return self._past.pick_argmin_above(
            score,
            indicators_above,
            threshold_onehot,
            value,
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
        return self._past.pick_argmin_above_in_bucket(
            score,
            validity,
            key_bucket_onehot,
            score_above_each_threshold,
            query_bucket_onehot,
            threshold_onehot,
            value,
            assert_hardness_gt=assert_hardness_gt,
        )

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
