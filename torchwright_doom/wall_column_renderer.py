"""Read-only branch owner for wall-column render transitions (Phase H).

Ported from ``doom_sandbox/implementation/forward/wall_column_renderer.py``:
``make_token`` -> ``make_token_head``, ``make_value`` -> ``value_scalar``,
``Vec`` -> ``Node``; every module-level ``constant`` node relocated inside the
method that uses it (no-import-time-nodes rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import CENTER_Y, SCREEN_HEIGHT
from .render_constants import PART_NONE, PLANE_KIND_CEILING, PLANE_KIND_FLOOR
from .render_ops import (
    CEIL_Y,
    CLIP_Y_CLAMP,
    FLOOR_Y_WIDE,
    SCALE_CLAMP,
    add_const,
    and_,
    gt_height,
    gt_screen,
    le_span_y,
    mul_column_scalestep,
    mul_height_scale,
    one_minus,
    or_,
    same_int,
    sub,
)
from .seg_scanner import SegScanner
from .std import (
    constant,
    indicator_to_bool,
    linear,
    make_token_head,
    one_hot,
    select,
    value_scalar,
)
from .std import sum as vec_sum
from .value_ranges import ValueRange
from .vocab import (
    CLIP_UPDATE,
    FIND_RUN,
    PLANE_MARK,
    SCREEN_RANGE,
    SET_CURSOR_X,
    SET_CURSOR_Y,
    WALL_SPAN_META,
)

if TYPE_CHECKING:
    from .seg_projection import SegProjection


_PART_IS_MID_LINEAR = [[1.0], [0.0], [0.0]]
_PART_IS_UPPER_LINEAR = [[0.0], [1.0], [0.0]]


def _part_is_mid(part_oh):
    return indicator_to_bool(linear(part_oh, _PART_IS_MID_LINEAR))


def _part_is_upper(part_oh):
    return indicator_to_bool(linear(part_oh, _PART_IS_UPPER_LINEAR))


@dataclass(frozen=True)
class WallColumnRenderer:
    """Owns wall-column branch entry points before pixel dispatch."""

    projection: "SegProjection"

    def after_wall_col_u(self):
        projection = self.projection
        # delta_pos=-1 assumes WALL_COL_U sits exactly one row after SET_CURSOR_X
        # in protocol order, so the SET_CURSOR_X screen-x is one position back.
        wall_col_x = projection.core.past.attend_to_offset(
            projection.inputs.x_or_zero, delta_pos=-1
        )
        return value_scalar(ValueRange.R5, self.wall_column_scale(wall_col_x))

    def after_wall_span_meta(self):
        return make_token_head(
            SET_CURSOR_Y, y=self.projection.core.inp.wall_span_meta_y
        )

    def after_clip_update(self):
        projection = self.projection
        y1, y2 = projection.wall.wall_column.clip_range_values(projection.core.past)
        return make_token_head(SCREEN_RANGE, y1=y1, y2=y2)

    def after_screen_y_value(self, fallback_out):
        projection = self.projection
        return select(
            projection.core.inp.screen_y_after_wall_column_scale,
            self.first_plane_mark_or_span(),
            fallback_out,
        )

    def after_completed_plane_mark(self):
        projection = self.projection
        plane_kind_floor = constant(PLANE_KIND_FLOOR)
        prev_kind = projection.core.past.attend_to_offset(
            projection.planes.plane_mark_kind_or_zero, delta_pos=-1,
        )
        was_ceiling = one_minus(same_int(prev_kind, plane_kind_floor))
        floor_emit = projection.wall.wall_column.pick(
            projection.core.past, projection.wall.wall_column.floor_emit,
        )
        return select(
            and_(was_ceiling, floor_emit),
            self.make_floor_plane_mark(),
            self.first_span_or_clip_update(),
        )

    def first_span_or_clip_update(self):
        projection = self.projection
        span_ordinal_0 = constant(0.0)
        span_ordinal_1 = constant(1.0)
        span_ordinal_2 = constant(2.0)
        seg_i = projection.drawseg.store_i
        wall_span = projection.wall.wall_column.span_values(projection.core.past)
        has_mid = projection.wall.seg_facts.has_mid(seg_i)
        has_upper = projection.wall.seg_facts.has_upper(seg_i)
        has_lower = projection.wall.seg_facts.has_lower(seg_i)
        mid_visible = and_(has_mid, wall_span.middle_ok)
        upper_visible = and_(has_upper, wall_span.upper_ok)
        lower_visible = and_(has_lower, wall_span.lower_ok)
        k_part_0 = projection.wall.seg_facts.K_part_0(seg_i)
        k_part_1 = projection.wall.seg_facts.K_part_1(seg_i)
        k_part_2 = projection.wall.seg_facts.K_part_2(seg_i)
        k0_visible = self.part_idx_visible(
            k_part_0, mid_visible, upper_visible, lower_visible,
        )
        k1_visible = self.part_idx_visible(
            k_part_1, mid_visible, upper_visible, lower_visible,
        )
        k2_visible = self.part_idx_visible(
            k_part_2, mid_visible, upper_visible, lower_visible,
        )
        k0_y1, k1_y1, k2_y1 = projection.wall.wall_span_runtime.wallcol_k_y1_values(
            projection.core.past,
            projection.wall.wall_column,
        )
        first_y1 = select(k0_visible, k0_y1, select(k1_visible, k1_y1, k2_y1))
        first_ordinal = select(
            k0_visible,
            span_ordinal_0,
            select(k1_visible, span_ordinal_1, span_ordinal_2),
        )
        first_span = self.make_span_meta_at_y(first_y1, first_ordinal)
        has_span = or_(k0_visible, or_(k1_visible, k2_visible))
        return select(has_span, first_span, self.make_clip_update_or_advance())

    def first_plane_mark_or_span(self):
        projection = self.projection
        ceiling_emit = projection.wall.wall_column.current_ceiling_emit
        floor_emit = projection.wall.wall_column.current_floor_emit
        return select(
            ceiling_emit,
            self.make_ceiling_plane_mark_current(),
            select(
                floor_emit,
                self.make_floor_plane_mark_current(),
                self.first_span_or_clip_update(),
            ),
        )

    def make_ceiling_plane_mark_current(self):
        projection = self.projection
        plane_kind_ceiling = constant(PLANE_KIND_CEILING)
        return make_token_head(
            PLANE_MARK,
            p=projection.wall.wall_column.current_ceiling_plane_id,
            kind=plane_kind_ceiling,
            vp=projection.planes.assigned_vp_for_kind(
                projection.core.past, plane_kind_ceiling
            ),
        )

    def make_floor_plane_mark_current(self):
        projection = self.projection
        plane_kind_floor = constant(PLANE_KIND_FLOOR)
        return make_token_head(
            PLANE_MARK,
            p=projection.wall.wall_column.current_floor_plane_id,
            kind=plane_kind_floor,
            vp=projection.planes.assigned_vp_for_kind(
                projection.core.past, plane_kind_floor
            ),
        )

    def make_floor_plane_mark(self):
        projection = self.projection
        plane_kind_floor = constant(PLANE_KIND_FLOOR)
        return make_token_head(
            PLANE_MARK,
            p=projection.wall.wall_column.pick(
                projection.core.past, projection.wall.wall_column.floor_plane_id,
            ),
            kind=plane_kind_floor,
            vp=projection.planes.assigned_vp_for_kind(
                projection.core.past, plane_kind_floor
            ),
        )

    def after_completed_span(self):
        projection = self.projection
        span = projection.wall.wall_span_runtime.span_start_values(projection.core.past)
        return select(
            span.has_next,
            self.make_span_meta_at_y(span.next_y, span.next_ordinal),
            self.make_clip_update_or_advance(),
        )

    def part_idx_visible(self, part_idx, mid_visible, upper_visible, lower_visible):
        part_sentinel = constant(PART_NONE)
        part_oh = one_hot(part_idx, 3)
        part_is_mid = _part_is_mid(part_oh)
        part_is_upper = _part_is_upper(part_oh)
        part_visible = select(
            part_is_mid,
            mid_visible,
            select(part_is_upper, upper_visible, lower_visible),
        )
        part_exists = one_minus(same_int(part_idx, part_sentinel))
        return and_(part_exists, part_visible)

    def after_completed_clip_update(self):
        return self.advance_after_column()

    def make_clip_update_or_advance(self):
        projection = self.projection
        changed = projection.wall.wall_column.pick(
            projection.core.past, projection.wall.wall_column.clip_changed,
        )
        return select(changed, self.make_clip_update(), self.advance_after_column())

    def advance_after_column(self):
        projection = self.projection
        next_x = add_const(
            projection.wall.wall_column.pick(
                projection.core.past, projection.wall.wall_column.x
            ),
            1.0,
        )
        return select(
            gt_screen(next_x, projection.drawseg.stop_x),
            self.continue_after_range(),
            make_token_head(SET_CURSOR_X, x=next_x),
        )

    def make_span_meta_at_y(self, y, ordinal):
        return make_token_head(WALL_SPAN_META, y=y, ordinal=ordinal)

    def make_clip_update(self):
        return make_token_head(CLIP_UPDATE)

    def continue_after_range(self):
        projection = self.projection
        next_x = add_const(projection.drawseg.stop_x, 1.0)
        seg_i = projection.drawseg.store_i
        advance = SegScanner(projection).advance_after_seg(
            seg_i,
            projection.seg.cycle.subsector_id,
            projection.seg.cycle.tree_depth,
        )
        return select(
            gt_screen(next_x, projection.seg.columns.last),
            advance,
            make_token_head(FIND_RUN, x=next_x),
        )

    def wall_column_scale(self, wall_col_x):
        """Per-column scale by linear interpolation from scale1 and scalestep.

        DOOM: R_RenderSegLoop (r_segs.c:360) — rw_scale += rw_scalestep per column;
        scale is 1/distance and feeds dc_iscale for texture stepping.
        """
        projection = self.projection
        scale1 = projection.drawseg.scale1
        scalestep = projection.drawseg.pick_scalestep()
        return SCALE_CLAMP(
            vec_sum(
                scale1,
                mul_column_scalestep(
                    sub(wall_col_x, projection.drawseg.store_x1),
                    scalestep,
                ),
            )
        )

    def wall_column_new_ceiling_from_value(self):
        """The new ceiling clip bound for a wall column (yl from worldtop, clamped
        by prior ceiling occlusion, upper tier checked, ceiling plane marked).

        DOOM: R_RenderSegLoop (r_segs.c:221-226, 294-313).
        """
        projection = self.projection
        center_y = constant(float(CENTER_Y))
        screen_height = constant(float(SCREEN_HEIGHT))
        seg_i = projection.drawseg.store_i
        scale = SCALE_CLAMP(projection.inputs.drawseg_scale_vec)

        scene = projection.core.scene
        worldtop = sub(scene.segs.front_ceiling(seg_i), scene.view.z)
        worldhigh = sub(scene.segs.back_ceiling(seg_i), scene.view.z)

        top_y_raw = sub(center_y, mul_height_scale(worldtop, scale))
        yl_unclipped = CEIL_Y(top_y_raw)
        ceiling_min = add_const(projection.wall.clip.ceiling, 1.0)
        floor_max = add_const(projection.wall.clip.floor, -1.0)
        yl = select(
            gt_screen(yl_unclipped, ceiling_min),
            yl_unclipped,
            ceiling_min,
        )

        portal = scene.segs.is_portal(seg_i)
        solid_or_closed = one_minus(portal)
        front_ceiling = scene.segs.front_ceiling(seg_i)
        back_ceiling = scene.segs.back_ceiling(seg_i)
        markceiling = and_(
            scene.segs.two_sided(seg_i),
            one_minus(same_int(back_ceiling, front_ceiling)),
        )
        markceiling = and_(markceiling, gt_height(front_ceiling, scene.view.z))

        upper_geom = gt_height(worldtop, worldhigh)
        high_y_raw = sub(center_y, mul_height_scale(worldhigh, scale))
        upper_mid_unclipped = FLOOR_Y_WIDE(high_y_raw)
        upper_mid = select(
            gt_screen(upper_mid_unclipped, floor_max),
            floor_max,
            upper_mid_unclipped,
        )
        upper_textured = and_(upper_geom, scene.segs.upper_texture(seg_i))
        upper_visible = and_(upper_textured, le_span_y(yl, upper_mid))
        ceiling_if_upper_textured = select(
            upper_visible,
            upper_mid,
            add_const(yl, -1.0),
        )
        ceiling_if_upper_geom = select(
            scene.segs.upper_texture(seg_i),
            ceiling_if_upper_textured,
            select(markceiling, add_const(yl, -1.0), projection.wall.clip.ceiling),
        )
        portal_ceiling = select(
            upper_geom,
            ceiling_if_upper_geom,
            select(markceiling, add_const(yl, -1.0), projection.wall.clip.ceiling),
        )
        return select(solid_or_closed, screen_height, CLIP_Y_CLAMP(portal_ceiling))
