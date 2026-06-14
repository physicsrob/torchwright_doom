"""Read-only branch owner for visplane marker/check transitions (Phase H).

DOOM: R_CheckPlane (r_plane.c) — owns ceiling/floor visplane instance
assignment and the handoff to wall-column rendering. "Marker" here is DOOM's
plane-mark sense (record a visplane region for the later flat pass to fill),
NOT the protocol "marker" token (the recency row a ``RecentMarkerHandle``
publishes elsewhere).

These branches run DURING the wall walk, BEFORE the flat pass, and decide which
visplane instance each wall seg's floor/ceiling region belongs to. The full
flat-pass control map (this is its left edge) lives in ``flat_pass_renderer.py``.
The branches, in the order one wall seg flows through them:

  - first_check_or_start_wall_columns: entry for a freshly-stored drawseg.
    Decides whether this seg needs a ceiling visplane and/or a floor visplane
    (a sky/portal with matching back surface needs neither). If ceiling is
    needed it emits R_CHECK_PLANE(ceiling, vp = 0); else if floor is needed
    R_CHECK_PLANE(floor, vp = 0); else nothing to mark -> start_wall_columns.

  - after_r_check_plane: tests whether candidate instance (plane, vp) already
    covers a column in this seg's screen range [x1, x2] (check_conflict). On a
    conflict it re-emits R_CHECK_PLANE with vp + 1 — looping through candidate
    visplane ids until one is free. On no conflict the candidate is free, so it
    emits R_CHECK_PLANE_RESULT(kind, vp) carrying the chosen instance.

  - after_r_check_plane_result: chains the ceiling check to the floor check.
    If the result just resolved a ceiling AND a floor is also needed, emit
    R_CHECK_PLANE(floor, vp = 0); otherwise both planes are resolved, so
    start_wall_columns (SET_CURSOR_X) begins the wall-column pass for this seg.

  - after_plane_mark: emits SCREEN_RANGE carrying the seg's per-column clip
    bounds (floor_y1/y2 or ceiling_y1/y2). That following SCREEN_RANGE row is
    what ``visplane_state.publish`` reads back (its ``screen_range_after_plane_mark``
    predicate) to record this seg's screen range into the chosen visplane
    instance, so the flat pass can read the region's coverage later.

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

from torchwright.graph import annotated

from .render_constants import PLANE_KIND_CEILING, PLANE_KIND_FLOOR
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

    @annotated("pmrk/R_CheckPlane")
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

    @annotated("pmrk/R_CheckPlane")
    def after_r_check_plane_result(self):
        projection = self.projection
        plane_kind_floor = constant(PLANE_KIND_FLOOR)
        vp_zero = constant(0.0)
        was_ceiling = one_minus(
            same_int(projection.core.inp.r_check_plane_result_kind, plane_kind_floor)
        )
        return select(
            and_(was_ceiling, self.floor_check_needed()),
            make_token_head(R_CHECK_PLANE, kind=plane_kind_floor, vp=vp_zero),
            self.start_wall_columns(),
        )

    @annotated("pmrk/R_CheckPlane")
    def after_plane_mark(self):
        projection = self.projection
        plane_kind_floor = constant(PLANE_KIND_FLOOR)
        is_floor = same_int(projection.core.inp.plane_mark_kind, plane_kind_floor)
        ranges = projection.wall.wall_column.plane_range_values(projection.core.past)
        return make_token_head(
            SCREEN_RANGE,
            y1=select(is_floor, ranges.floor_y1, ranges.ceiling_y1),
            y2=select(is_floor, ranges.floor_y2, ranges.ceiling_y2),
        )

    @annotated("pmrk/R_CheckPlane")
    def first_check_or_start_wall_columns(self):
        plane_kind_ceiling = constant(PLANE_KIND_CEILING)
        plane_kind_floor = constant(PLANE_KIND_FLOOR)
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
        plane_kind_floor = constant(PLANE_KIND_FLOOR)
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
