"""Past-published solid wall fragments used as horizontal occlusion state.

The seg scan fills this channel at ``R_STORE_WALL_RANGE`` and queries it at
``FIND_RUN`` to decide which screen columns a later seg may still draw —
DOOM's ``solidsegs`` horizontal occlusion, expressed as a queryable union of
prior one-sided drawseg fragments.

The query is a *radix successor* search: screen columns in ``[0, SCREEN_WIDTH]``
are split into a high *bucket* digit and a low digit (``_RADIX_BASE =
ceil(sqrt(width))``), so a "next occupied column after X" key fits ~``2*sqrt``
columns instead of ``width``. The answer is the next occupied column in the same
bucket, or — if that bucket is exhausted — the lowest in the next non-empty
higher bucket (a *carry*). See ``GLOSSARY.md``.

Changes from the original: ``Vec`` -> ``Node``; the original ``api``
imports map to the real ``std`` / ``past`` / ``render_ops`` shims.

Sentinel/constant nodes are built inside the publish methods, not at module
scope — see GLOSSARY.md 'the import-time-node rule'.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node
from torchwright.graph import annotated
from torchwright.ops.arithmetic_ops import compare, mod_const, thermometer_floor_div

from .attention_handles import RecentMarkerHandle
from .constants import SCREEN_WIDTH
from .render_constants import PRESENT_THRESHOLD
from .past import PastHandle, PastHandleScope
from .protocol_tokens import ProtocolTokenView
from .render_ops import (
    COL_RADIX_BASE as _RADIX_BASE,
    MUL_SCREEN,
    N_COL_BUCKETS as _N_BUCKETS,
    add_const,
    and_,
    one_minus,
    snap_bool,
)
from .scene_index import SceneIndex
from .std import (
    bool_and,
    clamp,
    concat,
    constant,
    linear,
    one_hot,
    pick_by_one_hot,
    select,
    split,
)

# Radix base/buckets are the shared screen-column scheme in render_ops
# (imported above as the historical local names).
_INVALID_HI = _N_BUCKETS

# Plain weight data (no graph nodes), so module scope is safe. The runtime graph
# forms the row index with ``one_hot`` and uses these tables to publish the small
# strict-above indicator bases that H1/H2 consume.
_LO_ABOVE_TABLE = [
    [1.0 if lo > threshold else 0.0 for threshold in range(_RADIX_BASE)]
    for lo in range(_RADIX_BASE)
]
_HI_FOR_H2_ABOVE_TABLE = [
    [1.0 if hi > threshold else 0.0 for threshold in range(_N_BUCKETS)]
    for hi in range(_N_BUCKETS + 1)
]


@dataclass(frozen=True)
class SolidIntervals:
    """Queryable union of prior one-sided drawseg fragments."""

    past: PastHandleScope
    key: PastHandle
    interval_state: PastHandle
    solid_emit: PastHandle
    start_s: PastHandle
    start_hi: PastHandle
    start_lo: PastHandle
    start_bucket_onehot: PastHandle
    start_above_lo: PastHandle
    start_hi_for_h2: PastHandle
    start_hi_above_for_h2: PastHandle
    start_above_all: PastHandle
    same_payload: PastHandle
    carry_payload: PastHandle

    @classmethod
    @annotated("stor")
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        scene: SceneIndex,
    ) -> "SolidIntervals":
        """Publish the current input fragment if it is a solid drawseg."""
        # R_STORE_WALL_RANGE rows publish the completed fragment. The store row
        # carries the seg id directly; x1 is the most recent FIND_RUN column
        # and x2 is recovered from the most recent semantic EMIT_X2 row.
        find_run_row = RecentMarkerHandle.publish(
            past,
            "solid_find_run",
            inp.is_find_run,
        )
        emit_x2_row = RecentMarkerHandle.publish(
            past,
            "solid_emit_x2",
            inp.is_emit_x2,
        )
        input_x_or_zero = past.publish("solid_input_x_or_zero", inp.seg_x_or_zero)
        emit_x2_x = past.publish("emit_x2_x_or_zero", inp.emit_x2_x)
        x1 = find_run_row.pick(past, input_x_or_zero)
        x2 = emit_x2_row.pick(past, emit_x2_x)
        is_open_portal = scene.segs.is_portal(inp.store_i)
        solid_emit = and_(inp.is_store_wall_range, one_minus(is_open_portal))

        sentinel_key = constant([0.0, 0.0, 1.0])
        sentinel_start = constant(float(SCREEN_WIDTH))

        interval_key = _interval_key(x1, x2)
        key = past.publish(
            "solid_interval_key",
            select(solid_emit, interval_key, sentinel_key),
        )
        interval_state = past.publish(
            "solid_interval_state",
            concat(solid_emit, select(solid_emit, x2, sentinel_start)),
        )
        start_s = select(solid_emit, x1, sentinel_start)
        successor = _publish_successor_fields(past, start_s, solid_emit)
        return cls(
            past=past,
            key=key,
            interval_state=interval_state,
            solid_emit=successor.solid_emit,
            start_s=successor.start_s,
            start_hi=successor.start_hi,
            start_lo=successor.start_lo,
            start_bucket_onehot=successor.start_bucket_onehot,
            start_above_lo=successor.start_above_lo,
            start_hi_for_h2=successor.start_hi_for_h2,
            start_hi_above_for_h2=successor.start_hi_above_for_h2,
            start_above_all=successor.start_above_all,
            same_payload=successor.same_payload,
            carry_payload=successor.carry_payload,
        )

    # DOOM: solidsegs scan predicate (r_bsp.c:R_ClipSolidWallSegment/R_ClipPassWallSegment)
    @annotated("stor")
    def covered_and_end(
        self,
        column: Node,
        column_square: Node,
    ) -> tuple[Node, Node]:
        """Return coverage and covering interval end using one shared query.

        The query ``[col², col, 1]`` dotted with a published key
        ``[-2, 2(a+b), -2ab]`` scores ``-2(col-a)(col-b)`` — positive when
        ``col`` lies inside the padded interval ``[a, b] = [x1-1, x2+1]``. The
        sentinel key ``[0, 0, 1]`` scores a flat 1, so a column outside every
        interval (or only at a padded edge) picks the sentinel and reads "not
        covered".
        """
        query = concat(column_square, column, constant(1.0))
        covered, end = split(
            self.past.pick_argmax(
                query,
                self.key,
                self.interval_state,
            ),
            [1, 1],
        )
        # ``covered`` is the matched key's ±1 ``solid_emit``; the fp32 softmax
        # recovers it as ≈±1 with ~1e-5 noise. Both consumers use it only as a
        # ``select`` cond, so snap it to a clean ±1 — the noise would otherwise
        # leak the discarded branch's head and flip a thin integer-slot argmax
        # (see ``render_ops.snap_bool``). ``end`` stays raw (a screen column).
        return snap_bool(covered), end

    @annotated("stor")
    def next_start_after(self, column: Node) -> Node:
        """Return the nearest solid interval start strictly after `column`."""
        query_hi = thermometer_floor_div(column, _RADIX_BASE, SCREEN_WIDTH)
        query_lo = mod_const(column, _RADIX_BASE, SCREEN_WIDTH)
        query_bucket_onehot = one_hot(query_hi, _N_BUCKETS)
        query_lo_threshold = one_hot(query_lo, _RADIX_BASE)

        same = self.past.pick_argmin_above_in_bucket(
            self.start_lo,
            self.solid_emit,
            self.start_bucket_onehot,
            self.start_above_lo,
            query_bucket_onehot,
            query_lo_threshold,
            self.same_payload,
        )
        same_s, same_solid, same_bucket_oh, same_above_lo = split(
            same,
            [1, 1, _N_BUCKETS, _RADIX_BASE],
        )
        same_bucket_score = pick_by_one_hot(query_bucket_onehot, same_bucket_oh)
        same_above_score = pick_by_one_hot(query_lo_threshold, same_above_lo)
        same_present = bool_and(
            snap_bool(same_solid),
            compare(same_bucket_score, PRESENT_THRESHOLD),
            compare(same_above_score, PRESENT_THRESHOLD),
        )

        higher_hi = self.past.pick_argmin_above(
            self.start_hi_for_h2,
            self.start_hi_above_for_h2,
            query_bucket_onehot,
            self.start_hi_for_h2,
        )
        higher_bucket_query = one_hot(
            clamp(higher_hi, 0.0, float(_N_BUCKETS - 1)), _N_BUCKETS
        )

        carry = self.past.pick_argmin_above_in_bucket(
            self.start_lo,
            self.solid_emit,
            self.start_bucket_onehot,
            self.start_above_all,
            higher_bucket_query,
            constant([1.0]),
            self.carry_payload,
        )
        carry_s, carry_solid, carry_bucket_oh = split(carry, [1, 1, _N_BUCKETS])
        higher_is_real_bucket = one_minus(compare(higher_hi, float(_N_BUCKETS) - 0.5))
        carry_bucket_score = pick_by_one_hot(higher_bucket_query, carry_bucket_oh)
        carry_present = bool_and(
            higher_is_real_bucket,
            snap_bool(carry_solid),
            compare(carry_bucket_score, PRESENT_THRESHOLD),
        )

        return select(
            same_present,
            same_s,
            select(carry_present, carry_s, constant(float(SCREEN_WIDTH))),
        )


@dataclass(frozen=True)
class _SuccessorHandles:
    solid_emit: PastHandle
    start_s: PastHandle
    start_hi: PastHandle
    start_lo: PastHandle
    start_bucket_onehot: PastHandle
    start_above_lo: PastHandle
    start_hi_for_h2: PastHandle
    start_hi_above_for_h2: PastHandle
    start_above_all: PastHandle
    same_payload: PastHandle
    carry_payload: PastHandle


def _publish_successor_fields(
    past: PastHandleScope,
    start_s: Node,
    solid_emit: Node,
) -> _SuccessorHandles:
    solid_emit_h = past.publish("solid_interval_solid_emit", solid_emit)
    start_s_h = past.publish("solid_interval_start", start_s)

    start_hi = thermometer_floor_div(start_s, _RADIX_BASE, SCREEN_WIDTH)
    start_lo = mod_const(start_s, _RADIX_BASE, SCREEN_WIDTH)
    start_bucket_onehot = one_hot(start_hi, _N_BUCKETS)
    start_above_lo = linear(one_hot(start_lo, _RADIX_BASE), _LO_ABOVE_TABLE)

    start_hi_for_h2 = select(solid_emit, start_hi, constant(float(_INVALID_HI)))
    start_hi_above_for_h2 = linear(
        one_hot(start_hi_for_h2, _N_BUCKETS + 1),
        _HI_FOR_H2_ABOVE_TABLE,
    )
    start_above_all = constant([1.0])

    start_hi_h = past.publish("solid_interval_start_hi", start_hi)
    start_lo_h = past.publish("solid_interval_start_lo", start_lo)
    start_bucket_h = past.publish(
        "solid_interval_start_bucket_onehot",
        start_bucket_onehot,
    )
    start_above_lo_h = past.publish("solid_interval_start_above_lo", start_above_lo)
    start_hi_for_h2_h = past.publish(
        "solid_interval_start_hi_for_h2",
        start_hi_for_h2,
    )
    start_hi_above_for_h2_h = past.publish(
        "solid_interval_start_hi_above_for_h2",
        start_hi_above_for_h2,
    )
    start_above_all_h = past.publish("solid_interval_start_above_all", start_above_all)

    same_payload = concat(start_s, solid_emit, start_bucket_onehot, start_above_lo)
    carry_payload = concat(start_s, solid_emit, start_bucket_onehot)
    same_payload_h = past.publish("solid_interval_same_payload", same_payload)
    carry_payload_h = past.publish("solid_interval_carry_payload", carry_payload)

    return _SuccessorHandles(
        solid_emit=solid_emit_h,
        start_s=start_s_h,
        start_hi=start_hi_h,
        start_lo=start_lo_h,
        start_bucket_onehot=start_bucket_h,
        start_above_lo=start_above_lo_h,
        start_hi_for_h2=start_hi_for_h2_h,
        start_hi_above_for_h2=start_hi_above_for_h2_h,
        start_above_all=start_above_all_h,
        same_payload=same_payload_h,
        carry_payload=carry_payload_h,
    )


# Encode solid interval (x1, x2) as query key for argmax matching (DOOM: cliprange_t, r_bsp.c:88-92)
def _interval_key(x1: Node, x2: Node) -> Node:
    a = add_const(x1, -1.0)
    b = add_const(x2, 1.0)
    ab = MUL_SCREEN(a, b)
    minus_two_ab = linear(ab, [[-2.0]])
    sum_ab = linear(concat(a, b), [[2.0], [2.0]])
    return concat(constant(-2.0), sum_ab, minus_two_ab)
