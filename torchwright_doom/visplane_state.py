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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.ops.arithmetic_ops import compare

from .past import PastHandle, PastHandleScope
from .render_ops import SCREEN_X_CLAMP, add_const, and_, neg, or_
from .std import (
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


_RANGE_START_TO_GE = [
    [1.0 if x >= start else 0.0 for x in range(SCREEN_WIDTH)]
    for start in range(SCREEN_WIDTH)
]
_RANGE_END_TO_LE = [
    [1.0 if x <= end else 0.0 for x in range(SCREEN_WIDTH)]
    for end in range(SCREEN_WIDTH)
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


def _range_overlap(score: Node) -> Node:
    """±1: a picked column's coverage count > 1.5 (overlap). The thermometer
    score is exactly 1 (covered, non-overlap) or 2 (overlap); sharpness=1000
    gives a 1e-3 deadband, so the integer scores sit 0.5 either side. Sandbox
    ``_RANGE_OVERLAP = compare_const(1.5, sharpness=1000)``."""
    return compare(score, 1.5, sharpness=1000.0)


def _instance_match(score: Node) -> Node:
    """±1: the picked row's flattened (plane, vp) instance equals the query's
    (a one-hot dot, exactly 1 on match else 0). Sandbox ``_INSTANCE_MATCH =
    compare_const(0.5, sharpness=1000)``."""
    return compare(score, 0.5, sharpness=1000.0)


@dataclass(frozen=True)
class VisplaneConflictValues:
    present: Node
    x_oh: Node
    instance_oh: Node


@dataclass(frozen=True)
class VisplaneColumnValues:
    top: Node
    bottom: Node


@dataclass(frozen=True)
class RuntimeVisplaneState:
    """Runtime visplane occupancy and used-instance scans."""

    occupied_key: PastHandle
    occupied_x: PastHandle
    occupied_state: PastHandle
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

        occupied_key_value = concat(
            linear(one_hot(occupied_p_value, N_PLANES_MAX), _OCC_PLANE_SCALE),
            linear(one_hot(occupied_vp_value, N_VP_PER_PLANE_MAX), _OCC_VP_SCALE),
            occupied_x_oh_value,
        )
        occupied_instance_idx = linear(
            concat(occupied_p_value, occupied_vp_value),
            _INSTANCE_IDX_LINEAR,
        )
        occupied_instance_oh_value = one_hot(
            occupied_instance_idx,
            N_VISPLANE_MAX,
        )
        occupied_key = past.publish(
            "visplane_occupied_key",
            gate(occupied_active, occupied_key_value),
        )
        occupied_x = past.publish("visplane_occupied_x", occupied_x_value)
        occupied_state = past.publish(
            "visplane_occupied_state",
            concat(occupied_active, occupied_x_oh_value, occupied_instance_oh_value),
        )
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
            occupied_key=occupied_key,
            occupied_x=occupied_x,
            occupied_state=occupied_state,
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
        range_score = _range_score_bits(x1, x2)
        query = concat(
            linear(one_hot(plane_id, N_PLANES_MAX), _OCC_PLANE_SCALE),
            linear(one_hot(candidate_vp, N_VP_PER_PLANE_MAX), _OCC_VP_SCALE),
            range_score,
        )
        query_instance_idx = linear(
            concat(plane_id, candidate_vp),
            _INSTANCE_IDX_LINEAR,
        )
        query_instance_oh = one_hot(query_instance_idx, N_VISPLANE_MAX)
        picked = VisplaneConflictValues(
            *split(
                past.pick_argmax(query, self.occupied_key, self.occupied_state),
                [1, SCREEN_WIDTH, N_VISPLANE_MAX],
            )
        )
        picked_x_score = pick_by_one_hot(picked.x_oh, range_score)
        exact_instance = _instance_match(
            pick_by_one_hot(query_instance_oh, picked.instance_oh)
        )
        return and_(
            picked.present,
            and_(exact_instance, _range_overlap(picked_x_score)),
        )

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


def _range_score_bits(x1: Node, x2: Node) -> Node:
    start_bits = linear(one_hot(x1, SCREEN_WIDTH), _RANGE_START_TO_GE)
    end_bits = linear(one_hot(x2, SCREEN_WIDTH), _RANGE_END_TO_LE)
    return vec_sum(start_bits, end_bits)


def _visplane_col_query(plane_id: Node, vp: Node, x: Node) -> Node:
    one = constant(1.0)
    return concat(
        one_hot(plane_id, N_PLANES_MAX),
        one_hot(vp, N_VP_PER_PLANE_MAX),
        one_hot(SCREEN_X_CLAMP(x), SCREEN_WIDTH),
        one,
    )
