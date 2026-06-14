"""Runtime-visplane state for the SegProjection ``planes`` subcontext (Phase H).

DOOM: R_FindPlane / R_CheckPlane (r_plane.c) over visplane_t — tracks which
ceiling/floor regions share the same height, texture, and light level, and
manages per-visplane column coverage (top[]/bottom[]) to detect conflicts when
new column ranges must be merged.

The original plain-Python implementation keeps several module-level
``constant(...)`` / ``compare_const(...)``
nodes; on the real side a ``constant`` is a graph ``Node`` with a global id, so
every node-building literal is relocated inside the function that uses it (the
no-import-time-nodes rule). The plain-list ``linear`` matrices (raw arrays) stay
at module level. ``compare_const(c, sharpness)`` becomes the real ``compare(node,
c, sharpness)``.

Layout (top to bottom):

  - Plain weight tables + radix-key helpers (``_radix_plane_key``,
    ``_lifted_instance_key`` / ``_lifted_instance_query``): the column- and
    plane-id encodings the attention reads dot against. The lift turns an id
    equality into a small dot that scores 1 on a match and drops by >= 1 per
    unit of mismatch, so it costs far fewer residual columns than a wide one-hot.
  - State dataclasses (``VisplaneColumnValues``, ``OccupancyRadix``,
    ``UsedPlaneSuccessor``, ``RuntimeVisplaneState``): the bundles of published
    rows and the values read back off them.
  - ``RuntimeVisplaneState.publish``: builds every per-position channel — the
    occupancy radix rows, the per-instance min/max-x and per-column-range keys,
    and the used-plane / used-vp successor rows.
  - ``check_conflict`` (R_CheckPlane): the three-head successor scan asking
    whether instance (plane, vp) already occupies a column in [x1, x2]
    (same-bucket / next-bucket / carry).
  - Small accessors (``next_plane_after``, ``next_vp_after``, ``min_x`` /
    ``max_x``, ``column_range``): read the channels ``publish`` built — the
    plane/vp iteration successors and the per-instance column-coverage lookups.
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

from torchwright.graph import annotated

from .past import PastHandle, PastHandleScope
from .render_ops import (
    COL_RADIX_BASE as _RADIX_BASE,
    N_COL_BUCKETS as _N_BUCKETS,
    SCREEN_X_CLAMP,
    add_const,
    neg,
    one_minus,
    or_,
    radix_col_key as _radix_col_key,
    snap_bool,
)
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
from .render_constants import PRESENT_THRESHOLD

if TYPE_CHECKING:
    from .protocol_tokens import ProtocolTokenView
    from .wall_column_state import WallColumnState


def _scale_matrix(width: int, scale: float) -> list[list[float]]:
    return [
        [scale if row == col else 0.0 for col in range(width)] for row in range(width)
    ]


_INSTANCE_IDX_LINEAR = [[float(N_VP_PER_PLANE_MAX)], [1.0]]
_OCC_VP_SCALE = _scale_matrix(N_VP_PER_PLANE_MAX, 128.0)
_USED_VP_ABOVE = [
    [
        1.0 if vp > (threshold_idx - 1) else 0.0
        for threshold_idx in range(N_VP_PER_PLANE_MAX + 1)
    ]
    for vp in range(N_VP_PER_PLANE_MAX + 1)
]
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

# The screen-column radix key is the shared render_ops scheme
# (radix_col_key, imported above as the historical local name).
_INSTANCE_BREAKPOINTS = list(range(N_VISPLANE_MAX))

# Plain weight data (no graph nodes), so module scope is safe. The runtime graph
# forms the row index with ``one_hot`` and reads these to publish the small
# indicator bases the bucketed-argmin op consumes.
#   _LO_GE_TABLE[lo][k]  = I(lo >= k)  — INCLUSIVE lower bound for H1 (c >= x1).
#   _HI_ABOVE_TABLE[hi][t]= I(hi > t)  — STRICT next-bucket test for H2.
_LO_GE_TABLE = [
    [1.0 if lo >= k else 0.0 for k in range(_RADIX_BASE)] for lo in range(_RADIX_BASE)
]
_HI_ABOVE_TABLE = [
    [1.0 if hi > t else 0.0 for t in range(_N_BUCKETS)] for hi in range(_N_BUCKETS)
]

# --- Plane-id radix (min_x / max_x / next_vp_after / next_plane_after) --------
#
# The plane-keyed visplane reads used a full ``one_hot(plane, N_PLANES_MAX)``
# query component (d_qk 41-42 for the argmax reads; d_head 35 for the
# ``next_plane_after`` argmin-above basis), all above the d_head=32 floor. Radix
# the plane id exactly the way ClipMemory / OccupancyRadix radix a screen column:
# ``plane -> (bucket = p // B, digit = p % B)`` one-hots, so an EXACT plane
# equality needs only ``NB + B`` cols (12 vs 32) and is a sum of one-hot products
# — NO large-magnitude cancellation (the lifted-square form would, and these
# heads resolve a small post-equality gap: argmin ``occupied_x`` / ``vp``, where
# the cancellation noise blends the tie — see the d_head reduction notes). The
# successor scan (``next_plane_after``) buckets the used-plane set and runs the
# same/higher/carry search ``solid_intervals.next_start_after`` uses over columns.
#
# One config covers both ranges: real planes 0..N_PLANES_MAX-1 (min_x/max_x/
# next_vp_after) and the scored set 0..N_PLANES_MAX (next_plane_after publishes
# the sentinel plane as a used row), since 32 // 6 = 5 still fits bucket width 6.
_PLANE_RADIX_BASE = math.ceil(math.sqrt(N_PLANES_MAX + 1))  # 6 at N_PLANES_MAX=32
_PLANE_N_BUCKETS = N_PLANES_MAX // _PLANE_RADIX_BASE + 1  # 6
_PLANE_RADIX_WIDTH = _PLANE_N_BUCKETS + _PLANE_RADIX_BASE  # 12
_PLANE_RADIX_SCALE = _scale_matrix(_PLANE_RADIX_WIDTH, 128.0)
_PLANE_INVALID_HI = _PLANE_N_BUCKETS

# Strict-above tables for the plane successor (mirror ``solid_intervals``):
#   _PLANE_LO_ABOVE_TABLE[lo][t]      = I(lo > t)  — same-bucket digit search.
#   _PLANE_HI_FOR_H2_ABOVE_TABLE[hi][t] = I(hi > t) — next-non-empty-bucket search
#                                        (row hi == _PLANE_INVALID_HI is all-zero).
# width B+1: column j encodes I(lo > j - 1), so slot 0 is "above -1" (every
# digit), used by next_plane_after's threshold == -1 find-first query.
_PLANE_LO_ABOVE_TABLE = [
    [1.0 if lo > (j - 1) else 0.0 for j in range(_PLANE_RADIX_BASE + 1)]
    for lo in range(_PLANE_RADIX_BASE)
]
_PLANE_HI_FOR_H2_ABOVE_TABLE = [
    [1.0 if hi > t else 0.0 for t in range(_PLANE_N_BUCKETS)]
    for hi in range(_PLANE_N_BUCKETS + 1)
]


def _radix_plane_key(plane_scalar: Node) -> Node:
    """Exact plane-equality key: ``concat(one_hot(p // B), one_hot(p % B))``
    (width ``NB + B = 12``). Two such keys dot to ``bucket_match + digit_match``
    — 2 on an exact plane match, <= 1 otherwise — a sum of one-hot products with
    no large-magnitude cancellation."""
    hi = thermometer_floor_div(plane_scalar, _PLANE_RADIX_BASE, N_PLANES_MAX)
    lo = mod_const(plane_scalar, _PLANE_RADIX_BASE, N_PLANES_MAX)
    return concat(one_hot(hi, _PLANE_N_BUCKETS), one_hot(lo, _PLANE_RADIX_BASE))


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
class UsedPlaneSuccessor:
    """Radix-bucketed used-plane set read by ``next_plane_after``.

    One row per used plane (every occupied column's plane, plus the sentinel
    plane published at ``DRAW_PLANES_BEGIN``). The plane id is split into ``hi =
    p // B`` (bucket) and ``lo = p % B`` (digit); ``next_plane_after`` runs the
    same same/next-bucket/carry successor scan ``solid_intervals.next_start_after``
    uses over screen columns. This replaces a single width-(N_PLANES_MAX+1)
    ``pick_argmin_above`` (d_head 35) with three heads of d_qk <= 14.
    """

    validity: PastHandle  # +/-1 used marker (bucketed-argmin static validity)
    lo: PastHandle  # plane digit p % B (score for the same/carry searches)
    hi_for_h2: PastHandle  # bucket p // B, or _PLANE_INVALID_HI on invalid rows
    bucket_onehot: PastHandle  # one_hot(p // B, NB)
    above_lo: PastHandle  # I(lo > t-1), width B+1 (the t=-1 slot finds the first plane)
    hi_above_for_h2: PastHandle  # I(hi > t), width NB (next-non-empty-bucket search)
    above_all: PastHandle  # constant [1.0] (carry "above everything" threshold)
    same_payload: PastHandle  # concat(plane, validity, bucket_oh, above_lo)
    carry_payload: PastHandle  # concat(plane, validity, bucket_oh)


@dataclass(frozen=True)
class RuntimeVisplaneState:
    """Runtime visplane occupancy and used-instance scans."""

    occupancy: OccupancyRadix
    occupied_x: PastHandle
    bounds_min_key: PastHandle
    bounds_max_key: PastHandle
    col_key: PastHandle
    col_range: PastHandle
    used_plane: UsedPlaneSuccessor
    used_vp_key: PastHandle
    used_vp_value: PastHandle

    @classmethod
    @annotated("pmrk")
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

        # --- Occupancy rows: one per screen column a marked seg covers. Read by
        # check_conflict to test instance overlap. The plane-mark seg's (p, vp)
        # comes from the PLANE_MARK row one position back; the column from the
        # wall-column cursor.
        occupied_active = inp.screen_range_after_plane_mark
        occupied_p_value = past.attend_to_offset(plane_mark_p_or_zero, delta_pos=-1)
        occupied_vp_value = past.attend_to_offset(plane_mark_vp_or_zero, delta_pos=-1)
        occupied_x_value = wall_column.pick(past, wall_column.x)

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
        # --- Per-instance min/max-x keys: read by min_x / max_x. A (plane, vp)
        # query argmaxes these over every occupied column; the +/-occupied_x tail
        # term picks the smallest (min) or largest (max) matching column.
        # Radix the plane equality (was one_hot(plane, 32), the d_qk driver) into
        # a (bucket, digit) pair; the vp one-hot (width 8) stays. Both query and
        # key scale by 128, so a matched digit/vp column contributes 128*128 to
        # the dot, dominating the +/-occupied_x term (<= SCREEN_WIDTH) so the
        # (plane, vp) equality wins; among matches, +/-occupied_x picks min/max x.
        bounds_min_key = past.publish(
            "visplane_bounds_min_key",
            gate(
                occupied_active,
                concat(
                    linear(_radix_plane_key(occupied_p_value), _PLANE_RADIX_SCALE),
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
                    linear(_radix_plane_key(occupied_p_value), _PLANE_RADIX_SCALE),
                    linear(
                        one_hot(occupied_vp_value, N_VP_PER_PLANE_MAX),
                        _OCC_VP_SCALE,
                    ),
                    occupied_x_value,
                ),
            ),
        )
        # --- Per-column coverage rows: read by column_range. A (plane, vp, x)
        # query argmaxes col_key to find the row for exactly that column of that
        # instance, returning its [top, bottom] from col_range (an empty range if
        # the column is unoccupied).
        # column_range key: lift the (plane, vp) instance to a width-3 scalar-id
        # equality (the same form OccupancyRadix uses) and radix the screen column
        # (was a width-(SCREEN_WIDTH+1) one_hot), so a (plane, vp, x) lookup needs
        # 3 + 16 + 1 cols (was 101). The dot of a full match is
        #   (1) instance + (2) col(hi+lo) - 2.5 (sentinel bias) = 0.5 > 0;
        # any partial match is <= -0.5 and an inactive (gated-zero) row is 0, so a
        # column with no occupancy picks an inactive row -> empty range. Identical
        # separation to the old raw-one-hot form (instance lift contributes 1 on a
        # match in place of the old plane(1)+vp(1)=2, and the col radix contributes
        # 2 in place of the old x(1)=1 -> same total 3).
        col_key_value = concat(
            _lifted_instance_key(occupied_instance_idx),
            _radix_col_key(occupied_x_value),
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

        # --- Used-plane successor rows: read by next_plane_after to iterate
        # planes in R_DrawPlanes order. Every occupied column's plane is a used
        # row, plus the sentinel plane published once at DRAW_PLANES_BEGIN so a
        # successor is always present.
        used_plane_active = or_(occupied_active, inp.is_draw_planes_begin)
        used_plane_value_raw = select(
            inp.is_draw_planes_begin,
            plane_sentinel,
            occupied_p_value,
        )
        used_plane = _publish_used_plane_successor(
            past, used_plane_active, used_plane_value_raw
        )

        # --- Used-vp successor rows: read by next_vp_after to iterate a plane's
        # visplane instances. Each occupied column contributes its vp; the
        # per-plane sentinel vp (published at the PLANE_DEF row) bounds the scan.
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
            linear(_radix_plane_key(used_vp_p), _PLANE_RADIX_SCALE),
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
            used_plane=used_plane,
            used_vp_key=used_vp_key,
            used_vp_value=used_vp_value,
        )

    # DOOM: R_CheckPlane (r_plane.c) — does a new column range overlap an existing visplane's coverage?
    @annotated("pmrk/R_CheckPlane")
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
            compare(pick_by_one_hot(hi1_bucket_oh, same_bucket_oh), PRESENT_THRESHOLD),
            compare(pick_by_one_hot(lo1_threshold, same_lo_ge), PRESENT_THRESHOLD),
            compare(
                pick_by_one_hot(query_instance_oh, same_inst_oh), PRESENT_THRESHOLD
            ),
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
            compare(
                pick_by_one_hot(query_instance_oh, higher_inst_oh), PRESENT_THRESHOLD
            ),
            compare(pick_by_one_hot(hi1_threshold, higher_hi_above), PRESENT_THRESHOLD),
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
            compare(
                pick_by_one_hot(higher_bucket_oh, carry_bucket_oh), PRESENT_THRESHOLD
            ),
            compare(
                pick_by_one_hot(query_instance_oh, carry_inst_oh), PRESENT_THRESHOLD
            ),
        )

        # c* = smallest instance column >= x1: the same-bucket hit if present,
        # else the carry. conflict = c* exists and c* <= x2 (x2 - c* > -0.5).
        cstar = select(same_present, same_col, carry_col)
        cstar_present = or_(same_present, carry_present)
        le_x2 = compare(vec_sum(x2, neg(cstar)), -0.5, sharpness=1000.0)
        return bool_and(cstar_present, le_x2)

    # DOOM: R_DrawPlanes (r_plane.c) — iterate active visplanes (for pl = visplanes; pl < lastvisplane; pl++)
    @annotated("pmrk")
    def next_plane_after(self, past: PastHandleScope, threshold: Node) -> Node:
        """Smallest used plane STRICTLY greater than ``threshold``.

        The plane-id successor over the bucketed used-plane set, mirroring
        ``solid_intervals.next_start_after``: H1 finds the smallest digit above
        ``threshold % B`` in ``threshold``'s bucket; H2 finds the next non-empty
        bucket; H3 carries to that bucket's smallest digit. ``threshold == -1``
        (R_DrawPlanes start, "find the first plane") works because ``threshold``'s
        digit floor-divides/mods to ``(0, -1)`` and the ``above_lo`` table's
        leading ``t == -1`` slot makes every digit in bucket 0 qualify. The
        sentinel plane published at DRAW_PLANES_BEGIN keeps a successor always
        present, so the ``plane_sentinel`` fallback is unreachable in practice."""
        plane_sentinel = constant(float(N_PLANE_SENTINEL))
        up = self.used_plane
        query_hi = thermometer_floor_div(threshold, _PLANE_RADIX_BASE, N_PLANES_MAX)
        query_lo = mod_const(threshold, _PLANE_RADIX_BASE, N_PLANES_MAX)
        query_bucket_oh = one_hot(query_hi, _PLANE_N_BUCKETS)
        # Digit threshold shifted +1: slot 0 = "above -1" (find-first), slots
        # 1..B = "above 0..B-1". So threshold == -1 -> query_lo == -1 -> slot 0.
        query_lo_threshold = one_hot(add_const(query_lo, 1.0), _PLANE_RADIX_BASE + 1)

        same = past.pick_argmin_above_in_bucket(
            up.lo,
            up.validity,
            up.bucket_onehot,
            up.above_lo,
            query_bucket_oh,
            query_lo_threshold,
            up.same_payload,
        )
        same_plane, same_valid, same_bucket_oh, same_above_lo = split(
            same, [1, 1, _PLANE_N_BUCKETS, _PLANE_RADIX_BASE + 1]
        )
        same_present = bool_and(
            snap_bool(same_valid),
            compare(
                pick_by_one_hot(query_bucket_oh, same_bucket_oh), PRESENT_THRESHOLD
            ),
            compare(
                pick_by_one_hot(query_lo_threshold, same_above_lo), PRESENT_THRESHOLD
            ),
        )

        higher_hi = past.pick_argmin_above(
            up.hi_for_h2,
            up.hi_above_for_h2,
            query_bucket_oh,
            up.hi_for_h2,
        )
        higher_bucket_query = one_hot(
            clamp(higher_hi, 0.0, float(_PLANE_N_BUCKETS - 1)), _PLANE_N_BUCKETS
        )

        carry = past.pick_argmin_above_in_bucket(
            up.lo,
            up.validity,
            up.bucket_onehot,
            up.above_all,
            higher_bucket_query,
            constant([1.0]),
            up.carry_payload,
        )
        carry_plane, carry_valid, carry_bucket_oh = split(
            carry, [1, 1, _PLANE_N_BUCKETS]
        )
        higher_is_real = one_minus(compare(higher_hi, float(_PLANE_N_BUCKETS) - 0.5))
        carry_present = bool_and(
            higher_is_real,
            snap_bool(carry_valid),
            compare(
                pick_by_one_hot(higher_bucket_query, carry_bucket_oh), PRESENT_THRESHOLD
            ),
        )

        return select(
            same_present,
            same_plane,
            select(carry_present, carry_plane, plane_sentinel),
        )

    # DOOM: R_DrawPlanes (r_plane.c) — nested iteration over a plane's visplane instances (merge slots)
    @annotated("pmrk")
    def next_vp_after(
        self, past: PastHandleScope, plane_id: Node, threshold: Node
    ) -> Node:
        one = constant(1.0)
        threshold_idx = add_const(threshold, 1.0)
        query = concat(
            linear(_radix_plane_key(plane_id), _PLANE_RADIX_SCALE),
            linear(
                one_hot(threshold_idx, N_VP_PER_PLANE_MAX + 1),
                _USED_VP_THRESHOLD_SCALE,
            ),
            one,
        )
        return past.pick_argmax(query, self.used_vp_key, self.used_vp_value)

    # DOOM: visplane_t.minx (r_plane.c) — lowest column the visplane spans
    @annotated("pmrk")
    def min_x(self, past: PastHandleScope, plane_id: Node, vp: Node) -> Node:
        one = constant(1.0)
        query = concat(
            linear(_radix_plane_key(plane_id), _PLANE_RADIX_SCALE),
            linear(one_hot(vp, N_VP_PER_PLANE_MAX), _OCC_VP_SCALE),
            one,
        )
        return past.pick_argmax(query, self.bounds_min_key, self.occupied_x)

    # DOOM: visplane_t.maxx (r_plane.c) — highest column the visplane spans
    @annotated("pmrk")
    def max_x(self, past: PastHandleScope, plane_id: Node, vp: Node) -> Node:
        one = constant(1.0)
        query = concat(
            linear(_radix_plane_key(plane_id), _PLANE_RADIX_SCALE),
            linear(one_hot(vp, N_VP_PER_PLANE_MAX), _OCC_VP_SCALE),
            one,
        )
        return past.pick_argmax(query, self.bounds_max_key, self.occupied_x)

    # DOOM: visplane_t.top[]/bottom[] (r_plane.c) — per-column coverage lookup for one visplane
    @annotated("pmrk")
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


def _publish_used_plane_successor(
    past: PastHandleScope,
    used_plane_active: Node,
    used_plane_value_raw: Node,
) -> UsedPlaneSuccessor:
    """Publish the per-used-plane radix rows ``next_plane_after`` scans.

    Mirrors ``solid_intervals._publish_successor_fields`` over plane ids: the
    plane splits into ``hi = p // B`` / ``lo = p % B``; the same/carry searches
    use ``validity`` (= ``used_plane_active``) as the bucketed-argmin static
    validity, so the bucket keys are published raw (no gate). The ``above_lo``
    table is width ``B + 1`` (a leading ``t == -1`` slot) so a ``threshold == -1``
    query — R_DrawPlanes' find-first — admits every digit in bucket 0. Invalid
    rows get ``hi = _PLANE_INVALID_HI`` so the H2 next-bucket argmin returns that
    sentinel when no higher bucket exists (``next_plane_after`` reads it as absent).
    """
    plane_hi = thermometer_floor_div(
        used_plane_value_raw, _PLANE_RADIX_BASE, N_PLANES_MAX
    )
    plane_lo = mod_const(used_plane_value_raw, _PLANE_RADIX_BASE, N_PLANES_MAX)
    bucket_onehot = one_hot(plane_hi, _PLANE_N_BUCKETS)
    above_lo = linear(one_hot(plane_lo, _PLANE_RADIX_BASE), _PLANE_LO_ABOVE_TABLE)
    hi_for_h2 = select(used_plane_active, plane_hi, constant(float(_PLANE_INVALID_HI)))
    hi_above_for_h2 = linear(
        one_hot(hi_for_h2, _PLANE_N_BUCKETS + 1),
        _PLANE_HI_FOR_H2_ABOVE_TABLE,
    )
    same_payload = concat(
        used_plane_value_raw, used_plane_active, bucket_onehot, above_lo
    )
    carry_payload = concat(used_plane_value_raw, used_plane_active, bucket_onehot)
    return UsedPlaneSuccessor(
        validity=past.publish("used_plane_validity", used_plane_active),
        lo=past.publish("used_plane_lo", plane_lo),
        hi_for_h2=past.publish("used_plane_hi_for_h2", hi_for_h2),
        bucket_onehot=past.publish("used_plane_bucket_onehot", bucket_onehot),
        above_lo=past.publish("used_plane_above_lo", above_lo),
        hi_above_for_h2=past.publish("used_plane_hi_above_for_h2", hi_above_for_h2),
        above_all=past.publish("used_plane_above_all", constant([1.0])),
        same_payload=past.publish("used_plane_same_payload", same_payload),
        carry_payload=past.publish("used_plane_carry_payload", carry_payload),
    )


def _visplane_col_query(plane_id: Node, vp: Node, x: Node) -> Node:
    """Query matching the radixed ``col_key`` (see ``publish``): the lifted
    ``(plane, vp)`` instance equality, the radixed screen column, and a ``1`` for
    the ``-2.5`` sentinel bias. Dot of a full match is ``1 + 2 - 2.5 = 0.5``."""
    one = constant(1.0)
    query_instance = linear(concat(plane_id, vp), _INSTANCE_IDX_LINEAR)
    return concat(
        _lifted_instance_query(query_instance),
        _radix_col_key(SCREEN_X_CLAMP(x)),
        one,
    )
