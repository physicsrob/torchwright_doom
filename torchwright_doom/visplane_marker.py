"""Read-only branch owner for visplane marker/check transitions (Phase H).

DOOM: R_CheckPlane (r_plane.c) — owns ceiling/floor visplane instance
assignment and the handoff to wall-column rendering.

Ported from ``doom_sandbox/implementation/forward/visplane_marker.py``:
``make_token`` -> ``make_token_head``, ``Vec`` -> ``Node``; module-level
``constant`` nodes relocated inside methods. The sandbox ``_VP_AT_CAP`` overflow
``assert_`` (a never-fires safety check on e1m1 — no two coplanar runs exhaust
``N_VP_PER_PLANE_MAX``) has no real-side predicate-assert counterpart and is
dropped; the gate verifies the actual conflict behaviour instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .render_ops import add_const, and_, gt_height, one_minus, same_int
from .std import constant, make_token_head, select
from .vocab import (
    R_CHECK_PLANE,
    R_CHECK_PLANE_RESULT,
    SCREEN_RANGE,
    SET_CURSOR_X,
)

if TYPE_CHECKING:
    from .seg_projection import SegProjection


@dataclass(frozen=True)
class VisplaneMarker:
    """Owns R_CheckPlane / PLANE_MARK branch entry points."""

    projection: "SegProjection"

    def after_r_check_plane(self):
        projection = self.projection
        kind = projection.core.inp.r_check_plane_kind
        candidate_vp = projection.core.inp.r_check_plane_vp
        plane_id = self.plane_id_for_kind(kind)
        conflict = projection.planes.runtime_visplanes.check_conflict(
            projection.core.past,
            plane_id=plane_id,
            candidate_vp=candidate_vp,
            x1=projection.drawseg.store_x1,
            x2=projection.drawseg.stop_x,
        )
        return select(
            conflict,
            make_token_head(
                R_CHECK_PLANE,
                kind=kind,
                vp=add_const(candidate_vp, 1.0),
            ),
            make_token_head(R_CHECK_PLANE_RESULT, kind=kind, vp=candidate_vp),
        )

    def after_r_check_plane_result(self):
        projection = self.projection
        plane_kind_floor = constant(1.0)
        vp_zero = constant(0.0)
        was_ceiling = one_minus(
            same_int(projection.core.inp.r_check_plane_result_kind, plane_kind_floor)
        )
        return select(
            and_(was_ceiling, self.floor_check_needed()),
            make_token_head(R_CHECK_PLANE, kind=plane_kind_floor, vp=vp_zero),
            self.start_wall_columns(),
        )

    def after_plane_mark(self):
        projection = self.projection
        plane_kind_floor = constant(1.0)
        is_floor = same_int(projection.core.inp.plane_mark_kind, plane_kind_floor)
        ranges = projection.wall.wall_column.plane_range_values(projection.core.past)
        return make_token_head(
            SCREEN_RANGE,
            y1=select(is_floor, ranges.floor_y1, ranges.ceiling_y1),
            y2=select(is_floor, ranges.floor_y2, ranges.ceiling_y2),
        )

    def first_check_or_start_wall_columns(self):
        plane_kind_ceiling = constant(0.0)
        plane_kind_floor = constant(1.0)
        vp_zero = constant(0.0)
        return select(
            self.ceiling_check_needed(),
            make_token_head(R_CHECK_PLANE, kind=plane_kind_ceiling, vp=vp_zero),
            select(
                self.floor_check_needed(),
                make_token_head(R_CHECK_PLANE, kind=plane_kind_floor, vp=vp_zero),
                self.start_wall_columns(),
            ),
        )

    def start_wall_columns(self):
        return make_token_head(
            SET_CURSOR_X,
            x=self.projection.drawseg.store_x1,
        )

    def plane_id_for_kind(self, kind):
        projection = self.projection
        plane_kind_floor = constant(1.0)
        return select(
            same_int(kind, plane_kind_floor),
            projection.wall.wall_column.current_floor_plane_id,
            projection.wall.wall_column.current_ceiling_plane_id,
        )

    def ceiling_check_needed(self):
        projection = self.projection
        scene = projection.core.scene
        seg_i = projection.drawseg.store_i
        portal = scene.segs.is_portal(seg_i)
        same_ceil = same_int(
            scene.segs.back_ceiling(seg_i),
            scene.segs.front_ceiling(seg_i),
        )
        base = one_minus(and_(portal, same_ceil))
        return and_(
            base,
            gt_height(scene.segs.front_ceiling(seg_i), scene.view.z),
        )

    def floor_check_needed(self):
        projection = self.projection
        scene = projection.core.scene
        seg_i = projection.drawseg.store_i
        portal = scene.segs.is_portal(seg_i)
        same_floor = same_int(
            scene.segs.back_floor(seg_i),
            scene.segs.front_floor(seg_i),
        )
        base = one_minus(and_(portal, same_floor))
        return and_(
            base,
            gt_height(scene.view.z, scene.segs.front_floor(seg_i)),
        )
