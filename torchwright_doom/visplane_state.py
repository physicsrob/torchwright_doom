"""Runtime-visplane state for the SegProjection ``planes`` subcontext (Phase H).

DOOM: R_FindPlane / R_CheckPlane (r_plane.c) over visplane_t — tracks which
ceiling/floor regions share the same height, texture, and light level, and
manages per-visplane column coverage (top[]/bottom[]) to detect conflicts when
new column ranges must be merged.

Ported from ``doom_sandbox/implementation/forward/visplane_state.py``. The
sandbox keeps several module-level ``constant(...)`` / ``compare_const(...)``
nodes; on the real side a ``constant`` is a graph ``Node`` with a global id, so
every node-building literal is relocated inside the function that uses it (the
no-import-time-nodes rule). The plain-list ``linear`` matrices (raw arrays) stay
at module level. ``compare_const(c, sharpness)`` becomes the real ``compare(node,
c, sharpness)`` (the sandbox ``input_range`` has no real counterpart).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.ops.arithmetic_ops import (
    compare,
    mod_const,
    piecewise_linear,
    thermometer_floor_div,
)

from .past import PastHandle, PastHandleScope
from .render_ops import SCREEN_X_CLAMP, add_const, neg, or_, snap_bool
from .std import (
    bool_and,
    clamp,
    concat,
    constant,
    gate,
    linear,
    one_hot,
    pick_by_one_hot,
    select,
    split,
)
from .std import sum as vec_sum
from .vocab import (
    N_PLANE_SENTINEL,
    N_PLANES_MAX,
    N_VISPLANE_MAX,
    N_VP_PER_PLANE_MAX,
    N_VP_SENTINEL,
)
from torchwright.graph import Node

from .constants import SCREEN_HEIGHT, SCREEN_WIDTH

if TYPE_CHECKING:
    from .protocol_tokens import ProtocolTokenView
    from .wall_column_state import WallColumnState


def _scale_matrix(width: int, scale: float) -> list[list[float]]:
    return [
        [scale if row == col else 0.0 for col in range(width)]
        for row in range(width)
    ]


_INSTANCE_IDX_LINEAR = [[float(N_VP_PER_PLANE_MAX)], [1.0]]
_OCC_PLANE_SCALE = _scale_matrix(N_PLANES_MAX, 128.0)
_OCC_VP_SCALE = _scale_matrix(N_VP_PER_PLANE_MAX, 128.0)
_USED_PLANE_ABOVE = [
    [1.0 if p > (threshold_idx - 1) else 0.0
     for threshold_idx in range(N_PLANES_MAX + 1)]
    for p in range(N_PLANES_MAX + 1)
]
_USED_VP_ABOVE = [
    [1.0 if vp > (threshold_idx - 1) else 0.0
     for threshold_idx in range(N_VP_PER_PLANE_MAX + 1)]
    for vp in range(N_VP_PER_PLANE_MAX + 1)
]
_USED_PLANE_KEY_SCALE = _scale_matrix(N_PLANES_MAX, 64.0)
_USED_VP_THRESHOLD_SCALE = _scale_matrix(N_VP_PER_PLANE_MAX + 1, 16.0)

# --- R_CheckPlane overlap test: instance-filtered radix successor -------------
#
# ``check_conflict`` asks: does visplane instance ``(plane, vp)`` already occupy a
# screen column in ``[x1, x2]``? The old form dotted a width-``SCREEN_WIDTH``
# per-column one-hot (``occupied_key`` 63 cols fixture, 163 real) — the binding
# ``d_head`` driver. This radixes the column the way the solid successor does
# (``solid_intervals.next_start_after``), so the screen column collapses to a
# (bucket, local-digit) pair and the key shrinks to ~21 cols (``d_head`` 64 -> 32).
#
# Equivalent boolean (gate-identical): ``c* = smallest instance-occupied column
# with c >= x1``; conflict iff ``c*`` exists and ``c* <= x2``. ``c*`` is the
# successor's same/higher-bucket/carry search with an INSTANCE filter folded in
# and an inclusive (>= x1) lower bound; the single ``c* <= x2`` compare then
# subsumes the same-bucket two-sided range case with no extra staging.

_RADIX_BASE = math.ceil(math.sqrt(SCREEN_WIDTH + 1))  # 8 at SW=60, 13 at SW=160
_N_BUCKETS = SCREEN_WIDTH // _RADIX_BASE + 1
_INSTANCE_BREAKPOINTS = list(range(N_VISPLANE_MAX))

# Plain weight data (no graph nodes), so module scope is safe. The runtime graph
# forms the row index with ``one_hot`` and reads these to publish the small
# indicator bases the bucketed-argmin op consumes.
#   _LO_GE_TABLE[lo][k]  = I(lo >= k)  — INCLUSIVE lower bound for H1 (c >= x1).
#   _HI_ABOVE_TABLE[hi][t]= I(hi > t)  — STRICT next-bucket test for H2.
_LO_GE_TABLE = [
    [1.0 if lo >= k else 0.0 for k in range(_RADIX_BASE)]
    for lo in range(_RADIX_BASE)
]
_HI_ABOVE_TABLE = [
    [1.0 if hi > t else 0.0 for t in range(_N_BUCKETS)]
    for hi in range(_N_BUCKETS)
]


def _lifted_instance_key(instance_idx: Node) -> Node:
    """Producer key for instance equality: ``[id, -id^2, 1]`` (width 3).

    Dotted with ``_lifted_instance_query(q)`` this scores ``1 - (id - q)^2`` —
    exactly 1 on an instance match, dropping by >= 1 per unit of id mismatch. The
    square is exact at integer ids (breakpoints on every integer). The key is
    used UNSCALED (gain 1): the bucketed-argmin op multiplies the bucket dot by
    ``_BUCKET_BONUS`` (256), so the matched-row common-mode logit is
    ``256 * (q^2 + 1)`` <= ``256 * 65026`` ~ 1.66e7, which stays under fp32's
    exact-integer ceiling (2^24) so the gained local-digit score (8 per digit)
    still resolves. Any extra gain on this key (e.g. the ``G_inst`` an AND-by-sum
    reading would suggest) pushes that common mode past ~1.3e8, where the fp32
    ulp (16) swamps the score and the within-bucket argmin silently blends.
    Dominance over the co-located ``col_hi`` one-hot needs no extra gain: each
    contributes an independent unit to the bucket sum, so failing either costs a
    full ``_BUCKET_BONUS`` (256) >> the 56-wide score swing.
    """
    square = piecewise_linear(
        instance_idx,
        _INSTANCE_BREAKPOINTS,
        lambda v: v * v,
        name="visplane_instance_square",
    )
    return concat(instance_idx, neg(square), constant(1.0))


def _lifted_instance_query(instance_idx: Node) -> Node:
    """Query ``[2q, 1, 1 - q^2]`` for the lifted instance equality (width 3).

    Against the key ``[id, -id^2, 1]`` this scores ``2q*id - id^2 + 1 - q^2 =
    1 - (id - q)^2``. The ``-q^2`` folded into the constant column CANCELS the
    ``q^2`` that the bare ``[2q, 1, 1]`` query would leave in every matched row's
    dot — keeping the post-``_BUCKET_BONUS`` matched logit at ~1024 instead of
    ~1.66e7, well clear of the fp32 exact-integer ceiling. (Both forms resolve
    the score at the project's id range, but this leaves ~14 bits of headroom
    rather than riding just under 2^24.)
    """
    square = piecewise_linear(
        instance_idx,
        _INSTANCE_BREAKPOINTS,
        lambda v: v * v,
        name="visplane_instance_query_square",
    )
    return concat(
        linear(instance_idx, [[2.0]]),
        constant(1.0),
        add_const(neg(square), 1.0),
    )


@dataclass(frozen=True)
class VisplaneColumnValues:
    top: Node
    bottom: Node


@dataclass(frozen=True)
class OccupancyRadix:
    """Per-occupied-column radix state read by ``check_conflict``.

    One row per occupied screen column of a visplane instance. The screen column
    is split into ``hi = col // B`` (bucket) and ``lo = col % B`` (local digit);
    the bucketed-argmin op filters by ``(bucket, instance)`` and picks the
    smallest local digit above a threshold. ``validity`` (the ±1 occupancy
    marker) is the op's static validity — inactive rows are rejected by it, so
    the bucket keys are published raw (no gate, hence no ``cond_gate`` bound).
    """

    validity: PastHandle  # ±1 occupancy marker (op static validity)
    lo: PastHandle  # local digit col % B (score for H1/H3)
    hi: PastHandle  # bucket col // B (score for H2)
    composite_bucket: PastHandle  # concat(col_bucket_onehot[N_BUCKETS], lift[3])
    instance_bucket: PastHandle  # lift[3] only (H2 filters by instance, not bucket)
    lo_ge: PastHandle  # I(lo >= k) thermometer, width B (H1 above-table)
    hi_above: PastHandle  # I(hi > t) thermometer, width N_BUCKETS (H2 above-table)
    above_all: PastHandle  # constant [1.0] (H3 "above everything" threshold)
    h13_value: PastHandle  # concat(col, validity, col_bucket_oh, lo_ge, inst_oh)
    h2_value: PastHandle  # concat(hi, validity, inst_oh, hi_above)


@dataclass(frozen=True)
class RuntimeVisplaneState:
    """Runtime visplane occupancy and used-instance scans."""

    occupancy: OccupancyRadix
    occupied_x: PastHandle
    bounds_min_key: PastHandle
    bounds_max_key: PastHandle
    col_key: PastHandle
    col_range: PastHandle
    used_plane_score: PastHandle
    used_plane_above: PastHandle
    used_plane_value: PastHandle
    used_vp_key: PastHandle
    used_vp_value: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        wall_column: WallColumnState,
        plane_mark_p_or_zero: PastHandle,
        plane_mark_vp_or_zero: PastHandle,
    ) -> "RuntimeVisplaneState":
        visplane_col_sentinel_bias = constant(-2.5)
        screen_height_value = constant(float(SCREEN_HEIGHT))
        neg_one_value = constant(-1.0)
        vp_sentinel = constant(float(N_VP_SENTINEL))
        plane_sentinel = constant(float(N_PLANE_SENTINEL))
        zero_used_plane_above = constant([0.0] * (N_PLANES_MAX + 1))

        occupied_active = inp.screen_range_after_plane_mark
        occupied_p_value = past.attend_to_offset(plane_mark_p_or_zero, delta_pos=-1)
        occupied_vp_value = past.attend_to_offset(plane_mark_vp_or_zero, delta_pos=-1)
        occupied_x_value = wall_column.pick(past, wall_column.x)
        occupied_x_oh_value = wall_column.pick(past, wall_column.x_key)

        occupied_instance_idx = linear(
            concat(occupied_p_value, occupied_vp_value),
            _INSTANCE_IDX_LINEAR,
        )
        occupied_instance_oh_value = one_hot(occupied_instance_idx, N_VISPLANE_MAX)
        occupancy = _publish_occupancy_radix(
            past,
            occupied_active,
            occupied_x_value,
            occupied_instance_idx,
            occupied_instance_oh_value,
        )
        occupied_x = past.publish("visplane_occupied_x", occupied_x_value)
        bounds_min_key = past.publish(
            "visplane_bounds_min_key",
            gate(
                occupied_active,
                concat(
                    linear(one_hot(occupied_p_value, N_PLANES_MAX), _OCC_PLANE_SCALE),
                    linear(
                        one_hot(occupied_vp_value, N_VP_PER_PLANE_MAX),
                        _OCC_VP_SCALE,
                    ),
                    neg(occupied_x_value),
                ),
            ),
        )
        bounds_max_key = past.publish(
            "visplane_bounds_max_key",
            gate(
                occupied_active,
                concat(
                    linear(one_hot(occupied_p_value, N_PLANES_MAX), _OCC_PLANE_SCALE),
                    linear(
                        one_hot(occupied_vp_value, N_VP_PER_PLANE_MAX),
                        _OCC_VP_SCALE,
                    ),
                    occupied_x_value,
                ),
            ),
        )
        col_key_value = concat(
            one_hot(occupied_p_value, N_PLANES_MAX),
            one_hot(occupied_vp_value, N_VP_PER_PLANE_MAX),
            occupied_x_oh_value,
            visplane_col_sentinel_bias,
        )
        col_key = past.publish(
            "visplane_col_key",
            gate(occupied_active, col_key_value),
        )
        col_range = past.publish(
            "visplane_col_range",
            concat(
                select(occupied_active, inp.screen_range_y1, screen_height_value),
                select(occupied_active, inp.screen_range_y2, neg_one_value),
            ),
        )

        used_plane_active = or_(occupied_active, inp.is_draw_planes_begin)
        used_plane_value_raw = select(
            inp.is_draw_planes_begin,
            plane_sentinel,
            occupied_p_value,
        )
        used_plane_oh = one_hot(used_plane_value_raw, N_PLANES_MAX + 1)
        used_plane_score = past.publish(
            "used_plane_score",
            select(used_plane_active, used_plane_value_raw, plane_sentinel),
        )
        used_plane_above = past.publish(
            "used_plane_above",
            select(
                used_plane_active,
                linear(used_plane_oh, _USED_PLANE_ABOVE),
                zero_used_plane_above,
            ),
        )
        used_plane_value = past.publish(
            "used_plane_value",
            select(used_plane_active, used_plane_value_raw, plane_sentinel),
        )

        used_vp_sentinel_active = inp.is_plane_def
        used_vp_active = or_(occupied_active, used_vp_sentinel_active)
        used_vp_p = select(used_vp_sentinel_active, inp.plane_def_p, occupied_p_value)
        used_vp_value_raw = select(
            used_vp_sentinel_active,
            vp_sentinel,
            occupied_vp_value,
        )
        used_vp_above = linear(
            one_hot(used_vp_value_raw, N_VP_PER_PLANE_MAX + 1),
            _USED_VP_ABOVE,
        )
        used_vp_key_value = concat(
            linear(one_hot(used_vp_p, N_PLANES_MAX), _USED_PLANE_KEY_SCALE),
            linear(used_vp_above, _USED_VP_THRESHOLD_SCALE),
            neg(used_vp_value_raw),
        )
        used_vp_key = past.publish(
            "used_vp_key",
            gate(used_vp_active, used_vp_key_value),
        )
        used_vp_value = past.publish(
            "used_vp_value",
            select(used_vp_active, used_vp_value_raw, vp_sentinel),
        )

        return cls(
            occupancy=occupancy,
            occupied_x=occupied_x,
            bounds_min_key=bounds_min_key,
            bounds_max_key=bounds_max_key,
            col_key=col_key,
            col_range=col_range,
            used_plane_score=used_plane_score,
            used_plane_above=used_plane_above,
            used_plane_value=used_plane_value,
            used_vp_key=used_vp_key,
            used_vp_value=used_vp_value,
        )

    # DOOM: R_CheckPlane (r_plane.c) — does a new column range overlap an existing visplane's coverage?
    def check_conflict(
        self,
        past: PastHandleScope,
        *,
        plane_id: Node,
        candidate_vp: Node,
        x1: Node,
        x2: Node,
    ) -> Node:
        """±1: does instance ``(plane_id, candidate_vp)`` occupy a column in
        ``[x1, x2]``? Computed as ``c* = smallest instance column >= x1`` (the
        instance-filtered radix successor over occupied columns), then
        ``conflict = c* exists and c* <= x2``."""
        occ = self.occupancy
        query_instance_idx = linear(
            concat(plane_id, candidate_vp),
            _INSTANCE_IDX_LINEAR,
        )
        query_instance_oh = one_hot(query_instance_idx, N_VISPLANE_MAX)
        query_instance_lift = _lifted_instance_query(query_instance_idx)

        hi1 = thermometer_floor_div(x1, _RADIX_BASE, SCREEN_WIDTH)
        lo1 = mod_const(x1, _RADIX_BASE, SCREEN_WIDTH)
        hi1_bucket_oh = one_hot(hi1, _N_BUCKETS)
        lo1_threshold = one_hot(lo1, _RADIX_BASE)  # reads I(lo >= lo1) from lo_ge
        hi1_threshold = one_hot(hi1, _N_BUCKETS)  # reads I(hi > hi1) from hi_above

        # H1 — same bucket: smallest local digit in bucket hi1, this instance,
        # with lo >= lo1. The composite bucket ANDs (col bucket == hi1) with the
        # lifted instance equality; both are unit terms in the op's bucket dot.
        same = past.pick_argmin_above_in_bucket(
            occ.lo,
            occ.validity,
            occ.composite_bucket,
            occ.lo_ge,
            concat(hi1_bucket_oh, query_instance_lift),
            lo1_threshold,
            occ.h13_value,
        )
        same_col, same_valid, same_bucket_oh, same_lo_ge, same_inst_oh = split(
            same, [1, 1, _N_BUCKETS, _RADIX_BASE, N_VISPLANE_MAX]
        )
        # Recompute presence from the selected row's carried one-hots (a blend of
        # non-matching rows dotted against the query one-hots stays below 1).
        same_present = bool_and(
            snap_bool(same_valid),
            compare(pick_by_one_hot(hi1_bucket_oh, same_bucket_oh), 0.9),
            compare(pick_by_one_hot(lo1_threshold, same_lo_ge), 0.9),
            compare(pick_by_one_hot(query_instance_oh, same_inst_oh), 0.9),
        )

        # H2 — next bucket: smallest bucket strictly above hi1 that this instance
        # occupies (instance is the op's bucket here; the score is the hi digit).
        higher = past.pick_argmin_above_in_bucket(
            occ.hi,
            occ.validity,
            occ.instance_bucket,
            occ.hi_above,
            query_instance_lift,
            hi1_threshold,
            occ.h2_value,
        )
        higher_hi, higher_valid, higher_inst_oh, higher_hi_above = split(
            higher, [1, 1, N_VISPLANE_MAX, _N_BUCKETS]
        )
        higher_present = bool_and(
            snap_bool(higher_valid),
            compare(pick_by_one_hot(query_instance_oh, higher_inst_oh), 0.9),
            compare(pick_by_one_hot(hi1_threshold, higher_hi_above), 0.9),
        )
        higher_bucket_oh = one_hot(
            clamp(higher_hi, 0.0, float(_N_BUCKETS - 1)), _N_BUCKETS
        )

        # H3 — carry: smallest local digit in the carried bucket, this instance
        # (every digit qualifies, so the threshold is "above all").
        carry = past.pick_argmin_above_in_bucket(
            occ.lo,
            occ.validity,
            occ.composite_bucket,
            occ.above_all,
            concat(higher_bucket_oh, query_instance_lift),
            constant([1.0]),
            occ.h13_value,
        )
        carry_col, carry_valid, carry_bucket_oh, _carry_lo_ge, carry_inst_oh = split(
            carry, [1, 1, _N_BUCKETS, _RADIX_BASE, N_VISPLANE_MAX]
        )
        carry_present = bool_and(
            higher_present,
            snap_bool(carry_valid),
            compare(pick_by_one_hot(higher_bucket_oh, carry_bucket_oh), 0.9),
            compare(pick_by_one_hot(query_instance_oh, carry_inst_oh), 0.9),
        )

        # c* = smallest instance column >= x1: the same-bucket hit if present,
        # else the carry. conflict = c* exists and c* <= x2 (x2 - c* > -0.5).
        cstar = select(same_present, same_col, carry_col)
        cstar_present = or_(same_present, carry_present)
        le_x2 = compare(vec_sum(x2, neg(cstar)), -0.5, sharpness=1000.0)
        return bool_and(cstar_present, le_x2)

    # DOOM: R_DrawPlanes (r_plane.c) — iterate active visplanes (for pl = visplanes; pl < lastvisplane; pl++)
    def next_plane_after(self, past: PastHandleScope, threshold: Node) -> Node:
        threshold_idx = add_const(threshold, 1.0)
        return past.pick_argmin_above(
            self.used_plane_score,
            self.used_plane_above,
            one_hot(threshold_idx, N_PLANES_MAX + 1),
            self.used_plane_value,
        )

    # DOOM: R_DrawPlanes (r_plane.c) — nested iteration over a plane's visplane instances (merge slots)
    def next_vp_after(self, past: PastHandleScope, plane_id: Node, threshold: Node) -> Node:
        one = constant(1.0)
        threshold_idx = add_const(threshold, 1.0)
        query = concat(
            linear(one_hot(plane_id, N_PLANES_MAX), _USED_PLANE_KEY_SCALE),
            linear(
                one_hot(threshold_idx, N_VP_PER_PLANE_MAX + 1),
                _USED_VP_THRESHOLD_SCALE,
            ),
            one,
        )
        return past.pick_argmax(query, self.used_vp_key, self.used_vp_value)

    # DOOM: visplane_t.minx (r_plane.c) — lowest column the visplane spans
    def min_x(self, past: PastHandleScope, plane_id: Node, vp: Node) -> Node:
        one = constant(1.0)
        query = concat(
            linear(one_hot(plane_id, N_PLANES_MAX), _OCC_PLANE_SCALE),
            linear(one_hot(vp, N_VP_PER_PLANE_MAX), _OCC_VP_SCALE),
            one,
        )
        return past.pick_argmax(query, self.bounds_min_key, self.occupied_x)

    # DOOM: visplane_t.maxx (r_plane.c) — highest column the visplane spans
    def max_x(self, past: PastHandleScope, plane_id: Node, vp: Node) -> Node:
        one = constant(1.0)
        query = concat(
            linear(one_hot(plane_id, N_PLANES_MAX), _OCC_PLANE_SCALE),
            linear(one_hot(vp, N_VP_PER_PLANE_MAX), _OCC_VP_SCALE),
            one,
        )
        return past.pick_argmax(query, self.bounds_max_key, self.occupied_x)

    # DOOM: visplane_t.top[]/bottom[] (r_plane.c) — per-column coverage lookup for one visplane
    def column_range(
        self,
        past: PastHandleScope,
        plane_id: Node,
        vp: Node,
        x: Node,
    ) -> VisplaneColumnValues:
        query = _visplane_col_query(plane_id, vp, x)
        return VisplaneColumnValues(
            *split(past.pick_argmax(query, self.col_key, self.col_range), [1, 1])
        )


def _publish_occupancy_radix(
    past: PastHandleScope,
    occupied_active: Node,
    occupied_x_value: Node,
    occupied_instance_idx: Node,
    occupied_instance_oh: Node,
) -> OccupancyRadix:
    """Publish the per-occupied-column radix rows ``check_conflict`` reads.

    The bucket keys are published raw (not gated): the bucketed-argmin op rejects
    inactive rows via ``validity`` (= ``occupied_active``), so no ``gate`` /
    ``cond_gate`` is involved. ``thermometer_floor_div`` / ``mod_const`` are exact
    on integer columns (the radix-successor derisk showed ``floor_int(v/B +
    0.5/B)`` is wrong at B=13; these place transitions at ``k*B - 0.5``)."""
    col_hi = thermometer_floor_div(occupied_x_value, _RADIX_BASE, SCREEN_WIDTH)
    col_lo = mod_const(occupied_x_value, _RADIX_BASE, SCREEN_WIDTH)
    col_bucket_onehot = one_hot(col_hi, _N_BUCKETS)
    lo_ge = linear(one_hot(col_lo, _RADIX_BASE), _LO_GE_TABLE)
    hi_above = linear(one_hot(col_hi, _N_BUCKETS), _HI_ABOVE_TABLE)
    lift = _lifted_instance_key(occupied_instance_idx)

    # Value payloads use the op's identity V/O (any width, no d_qk cost). They
    # carry exactly the one-hots the presence recompute re-tests, plus the column
    # scalar (H1/H3) / bucket scalar (H2) the combine step reads.
    h13_value = concat(
        occupied_x_value,
        occupied_active,
        col_bucket_onehot,
        lo_ge,
        occupied_instance_oh,
    )
    h2_value = concat(
        col_hi,
        occupied_active,
        occupied_instance_oh,
        hi_above,
    )
    return OccupancyRadix(
        validity=past.publish("visplane_occ_validity", occupied_active),
        lo=past.publish("visplane_occ_lo", col_lo),
        hi=past.publish("visplane_occ_hi", col_hi),
        composite_bucket=past.publish(
            "visplane_occ_composite_bucket", concat(col_bucket_onehot, lift)
        ),
        instance_bucket=past.publish("visplane_occ_instance_bucket", lift),
        lo_ge=past.publish("visplane_occ_lo_ge", lo_ge),
        hi_above=past.publish("visplane_occ_hi_above", hi_above),
        above_all=past.publish("visplane_occ_above_all", constant([1.0])),
        h13_value=past.publish("visplane_occ_h13_value", h13_value),
        h2_value=past.publish("visplane_occ_h2_value", h2_value),
    )


def _visplane_col_query(plane_id: Node, vp: Node, x: Node) -> Node:
    one = constant(1.0)
    return concat(
        one_hot(plane_id, N_PLANES_MAX),
        one_hot(vp, N_VP_PER_PLANE_MAX),
        one_hot(SCREEN_X_CLAMP(x), SCREEN_WIDTH),
        one,
    )
