"""Read-only branch owner for R_StoreWallRange / drawseg setup.

Ported from ``doom_sandbox/implementation/forward/wall_range_builder.py``. Owns
the drawseg/range transitions: from ``R_STORE_WALL_RANGE`` it emits the
``SEG_KPART`` → ``SEG_DC_TMID_*`` → ``DRAWSEG_META`` → scale chain
(``SCALE1_DEN`` … ``SCALESTEP``) → silhouettes → ``DRAWSEG_U_PHASE`` token
sequence, computing per-segment perspective scale, texture origins, and
sprite-clipping silhouette heights.

At the end of that sequence it hands off to the downstream owners:
``after_drawseg_u_angle_value`` starts the visplane check (``VisplaneMarker``);
the ``is_value_after_wall_column`` arm of ``after_drawseg_value`` emits a wall
column's ``SCREEN_Y_VALUE`` (``WallColumnRenderer``); and the
``is_value_after_set_cursor_y`` arm emits a ``PIXEL`` (``PixelDispatcher``).

Changes from the sandbox source: ``Vec`` -> ``Node``; ``make_token`` ->
``make_token_head`` (the dispatch folds over emit heads); module-level
``constant`` sentinels move inside the methods (import-time node rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import Node

from .render_ops import (
    DIST_GT_ONE,
    FAR_DEN_CLAMP,
    MAX_SCALE_VALUE,
    MUL_FAR_DEN,
    MUL_NEAR_DEN,
    MUL_NEAR_FLOOR_SCALE,
    MUL_UNIT,
    NEAR_DEN_CLAMP,
    NEAR_DEN_SCALE_UP,
    NEAR_FLOOR_NUMERATOR,
    PROJECT_SCALE,
    SCALE_CLAMP,
    SINEB_ABOVE_FLOOR,
    SINEB_CLAMP,
    and_,
    gt_height,
    mul_far_scale,
    mul_scalestep,
    one_minus,
    or_,
    sub,
    wrap_signed_angle,
)
from .std import (
    ScalarEmit,
    angle_scalar,
    bool_to_01,
    concat,
    constant,
    linear,
    make_token_head,
    pick_by_one_hot,
    select,
    value_scalar,
)
from .std import sum as vec_sum
from .value_ranges import ValueRange
from .wall_range_state import rw_distance_for
from .vocab import (
    DRAWSEG_BSILHEIGHT,
    DRAWSEG_META,
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE1_DEN,
    DRAWSEG_SCALE2,
    DRAWSEG_SCALE2_DEN,
    DRAWSEG_SCALESTEP,
    DRAWSEG_SCALESTEP_DEN,
    DRAWSEG_TSILHEIGHT,
    DRAWSEG_U_PHASE,
    SEG_DC_TMID_LOWER,
    SEG_DC_TMID_MID,
    SEG_DC_TMID_UPPER,
    SEG_KPART,
)

if TYPE_CHECKING:
    from .seg_projection import SegProjection

# Silhouette flags (reference.py: SIL_BOTTOM=1, SIL_TOP=2, SIL_BOTH=3).
_SIL_BOTH_VALUE = 3.0
_SIL_HEIGHT_MAX = 256.0
_SIL_HEIGHT_MIN = -256.0
_WALL_KIND_LINEAR = [[1.0], [2.0]]
_PAT_LINEAR = [[4.0], [2.0], [1.0]]


@dataclass(frozen=True)
class WallRangeBuilder:
    """Owns drawseg/range branch transitions and rendering setup math.

    DOOM: R_StoreWallRange (r_segs.c) — computes per-segment scale values,
    texture origins, and silhouette clipping heights for a drawseg.
    """

    projection: "SegProjection"

    def after_store_wall_range(self) -> Node:
        projection = self.projection
        seg_i = projection.drawseg.store_i
        segs = projection.core.scene.segs
        front_ceiling = segs.front_ceiling(seg_i)
        front_floor = segs.front_floor(seg_i)
        back_ceiling = segs.back_ceiling(seg_i)
        back_floor = segs.back_floor(seg_i)
        portal = segs.is_portal(seg_i)
        solid_or_closed = one_minus(portal)
        mid_present = segs.mid_texture(seg_i)
        upper_present = segs.upper_texture(seg_i)
        lower_present = segs.lower_texture(seg_i)
        upper_geom = gt_height(front_ceiling, back_ceiling)
        lower_geom = gt_height(back_floor, front_floor)
        has_mid = and_(solid_or_closed, mid_present)
        has_upper = and_(portal, and_(upper_present, upper_geom))
        has_lower = and_(portal, and_(lower_present, lower_geom))
        pat = linear(
            concat(bool_to_01(has_mid), bool_to_01(has_upper), bool_to_01(has_lower)),
            _PAT_LINEAR,
        )
        return make_token_head(SEG_KPART, pat=pat)

    def after_seg_kpart(self) -> Node:
        return make_token_head(SEG_DC_TMID_MID)

    def dc_tmid_compute(self, part: str) -> Node:
        # DOOM: rw_*texturemid setup (r_segs.c:R_StoreWallRange) — texture
        # vertical origin (mid/top/bottom) for a wall part.
        projection = self.projection
        seg_i = projection.drawseg.store_i
        scene = projection.core.scene
        viewz = scene.view.z
        front_ceiling = scene.segs.front_ceiling(seg_i)
        front_floor = scene.segs.front_floor(seg_i)
        back_ceiling = scene.segs.back_ceiling(seg_i)
        back_floor = scene.segs.back_floor(seg_i)
        rowoffset = scene.segs.rowoffset(seg_i)
        dontpegtop = scene.segs.dontpegtop(seg_i)
        dontpegbottom = scene.segs.dontpegbottom(seg_i)
        worldtop_v = sub(front_ceiling, viewz)
        if part == "mid":
            mid_tex_id = scene.segs.mid_tex_id(seg_i)
            tex_h_native = scene.assets.walls.height(mid_tex_id)
            mid_pegged_v = sub(vec_sum(front_floor, tex_h_native), viewz)
            return vec_sum(
                select(dontpegbottom, mid_pegged_v, worldtop_v),
                rowoffset,
            )
        if part == "upper":
            upper_tex_id = scene.segs.upper_tex_id(seg_i)
            tex_h_native = scene.assets.walls.height(upper_tex_id)
            upper_default_v = sub(vec_sum(back_ceiling, tex_h_native), viewz)
            return vec_sum(
                select(dontpegtop, worldtop_v, upper_default_v),
                rowoffset,
            )
        if part == "lower":
            lower_default_v = sub(back_floor, viewz)
            return vec_sum(
                select(dontpegbottom, worldtop_v, lower_default_v),
                rowoffset,
            )
        raise ValueError(f"unknown dc_tmid part: {part!r}")

    def after_seg_dc_tmid_mid(self) -> ScalarEmit:
        return value_scalar(ValueRange.R3, self.dc_tmid_compute("mid"))

    def after_seg_dc_tmid_upper(self) -> ScalarEmit:
        return value_scalar(ValueRange.R4, self.dc_tmid_compute("upper"))

    def after_seg_dc_tmid_lower(self) -> ScalarEmit:
        return value_scalar(ValueRange.R4, self.dc_tmid_compute("lower"))

    def after_drawseg_meta(self) -> Node:
        return make_token_head(DRAWSEG_SCALE1_DEN)

    def after_drawseg_scale1_den(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R6,
            self.scale_denominator_for_inverse(is_stop=False),
        )

    def after_drawseg_scale1(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R5,
            self.scale_from_recent_denominator(is_stop=False),
        )

    def after_drawseg_scale2_den(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R6,
            self.scale_denominator_for_inverse(is_stop=True),
        )

    def after_drawseg_scale2(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R5,
            self.scale_from_recent_denominator(is_stop=True),
        )

    def after_drawseg_scalestep_den(self) -> ScalarEmit:
        projection = self.projection
        return value_scalar(
            ValueRange.R7,
            sub(projection.drawseg.stop_x, projection.drawseg.store_x1),
        )

    def after_drawseg_scalestep(self) -> ScalarEmit:
        return value_scalar(ValueRange.R8, self.scale_step())

    def after_drawseg_bsilheight(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R9,
            self.bsilheight_value(self.projection.drawseg.store_i),
        )

    def after_drawseg_tsilheight(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R9,
            self.tsilheight_value(self.projection.drawseg.store_i),
        )

    def after_drawseg_u_phase(self) -> ScalarEmit:
        projection = self.projection
        scene = projection.core.scene
        normal_angle = scene.segs.normal_angle(projection.drawseg.store_i)
        u_phase_angle = wrap_signed_angle(sub(scene.view.angle, normal_angle))
        return angle_scalar(u_phase_angle)

    def after_drawseg_u_angle_value(self) -> Node:
        # The DRAWSEG_U_PHASE -> ANGLE_VALUE handoff starts the visplane
        # find/check chain (or the wall-column pass if no plane is marked).
        from .visplane_marker import VisplaneMarker

        return VisplaneMarker(self.projection).first_check_or_start_wall_columns()

    def after_drawseg_value(self, fallback_out: Node) -> Node:
        projection = self.projection
        seg_i = projection.drawseg.store_i
        inp = projection.core.inp
        drawseg_meta_out = make_token_head(
            DRAWSEG_META,
            i=seg_i,
            wall_kind=self.wall_kind(seg_i),
            silhouette=self.silhouette(seg_i),
        )
        # A VALUE after a wall column (SCREEN_Y_VALUE marker carrier)
        # emits the new ceiling clip bound for the next column.
        from .vocab import SCREEN_Y_VALUE
        from .wall_column_renderer import WallColumnRenderer

        wall_column_screen_y_out = make_token_head(
            SCREEN_Y_VALUE,
            y=WallColumnRenderer(projection).wall_column_new_ceiling_from_value(),
        )
        drawseg_value_out = select(
            inp.is_value_after_wall_column,
            wall_column_screen_y_out,
            select(
                inp.is_value_after_seg_dc_tmid_mid,
                make_token_head(SEG_DC_TMID_UPPER),
                select(
                    inp.is_value_after_seg_dc_tmid_upper,
                    make_token_head(SEG_DC_TMID_LOWER),
                    select(
                        inp.is_value_after_seg_dc_tmid_lower,
                        drawseg_meta_out,
                        select(
                            inp.is_value_after_drawseg_tsilheight,
                            make_token_head(DRAWSEG_U_PHASE),
                            select(
                                inp.is_value_after_drawseg_bsilheight,
                                make_token_head(DRAWSEG_TSILHEIGHT),
                                select(
                                    inp.is_value_after_drawseg_scalestep,
                                    make_token_head(DRAWSEG_BSILHEIGHT),
                                    select(
                                        inp.is_value_after_drawseg_scalestep_den,
                                        make_token_head(DRAWSEG_SCALESTEP),
                                        select(
                                            inp.is_value_after_drawseg_scale2,
                                            make_token_head(DRAWSEG_SCALESTEP_DEN),
                                            select(
                                                inp.is_value_after_drawseg_scale2_den,
                                                make_token_head(DRAWSEG_SCALE2),
                                                select(
                                                    inp.is_value_after_drawseg_scale1,
                                                    make_token_head(DRAWSEG_SCALE2_DEN),
                                                    select(
                                                        inp.is_value_after_drawseg_scale1_den,
                                                        make_token_head(DRAWSEG_SCALE1),
                                                        fallback_out,
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        # A VALUE after SET_CURSOR_Y (the per-span v0 carrier) emits the
        # span's first wall pixel (pixel_index 0).
        from .pixel_dispatcher import PixelDispatcher

        return select(
            inp.is_value_after_set_cursor_y,
            PixelDispatcher(projection).make_first_pixel_color(),
            drawseg_value_out,
        )

    def wall_kind(self, seg_i: Node) -> Node:
        segs = self.projection.core.scene.segs
        closed = segs.closed_door(seg_i)
        portal = segs.is_portal(seg_i)
        return linear(concat(bool_to_01(closed), bool_to_01(portal)), _WALL_KIND_LINEAR)

    def silhouette(self, seg_i: Node) -> Node:
        # DOOM: ds_p->silhouette setup (r_segs.c:R_StoreWallRange) —
        # sprite-clipping silhouette flags (SIL_TOP, SIL_BOTTOM, SIL_BOTH).
        portal = self.projection.core.scene.segs.is_portal(seg_i)
        solid_or_closed = one_minus(portal)
        bottom = self.portal_bottom_silhouette(seg_i)
        top = self.portal_top_silhouette(seg_i)
        portal_silhouette = linear(
            concat(bool_to_01(bottom), bool_to_01(top)),
            [[1.0], [2.0]],
        )
        return select(solid_or_closed, constant(_SIL_BOTH_VALUE), portal_silhouette)

    def portal_bottom_silhouette(self, seg_i: Node) -> Node:
        segs = self.projection.core.scene.segs
        front_floor = segs.front_floor(seg_i)
        back_floor = segs.back_floor(seg_i)
        return or_(
            gt_height(front_floor, back_floor),
            gt_height(back_floor, self.projection.core.scene.view.z),
        )

    def portal_top_silhouette(self, seg_i: Node) -> Node:
        segs = self.projection.core.scene.segs
        front_ceiling = segs.front_ceiling(seg_i)
        back_ceiling = segs.back_ceiling(seg_i)
        return or_(
            gt_height(back_ceiling, front_ceiling),
            gt_height(self.projection.core.scene.view.z, back_ceiling),
        )

    def scale_denominator_for_inverse(self, *, is_stop: bool) -> Node:
        # DOOM: rw_scale perspective denominator (r_segs.c:R_StoreWallRange).
        distance = self.rw_distance(self.projection.drawseg.store_i)
        xtova_cos = self.projection.rows.pick_range_xtova_cos(is_stop=is_stop)
        near = self.is_near(distance)
        far_den = FAR_DEN_CLAMP(MUL_FAR_DEN(distance, xtova_cos))
        near_den = NEAR_DEN_SCALE_UP(NEAR_DEN_CLAMP(MUL_NEAR_DEN(distance, xtova_cos)))
        return select(near, near_den, far_den)

    def scale_from_recent_denominator(self, *, is_stop: bool) -> Node:
        projection = self.projection
        distance = self.rw_distance(projection.drawseg.store_i)
        sineb = self.scale_sineb(is_stop=is_stop)
        near = self.is_near(distance)
        near_high = SINEB_ABOVE_FLOOR(sineb)
        den_inverse = projection.drawseg.pick_scale_den_inverse()
        far_scale = SCALE_CLAMP(mul_far_scale(PROJECT_SCALE(sineb), den_inverse))
        near_floor_scale = SCALE_CLAMP(
            MUL_NEAR_FLOOR_SCALE(NEAR_FLOOR_NUMERATOR(sineb), den_inverse)
        )
        near_scale = select(near_high, MAX_SCALE_VALUE(near_high), near_floor_scale)
        return select(near, near_scale, far_scale)

    def scale_step(self) -> Node:
        projection = self.projection
        scale1 = projection.drawseg.scale1
        scale2 = projection.drawseg.pick_scale2()
        return mul_scalestep(
            sub(scale2, scale1),
            projection.drawseg.pick_scalestep_den_inverse(),
        )

    def rw_distance(self, seg_i: Node) -> Node:
        # DOOM: rw_distance (r_segs.c:R_StoreWallRange) — the shared
        # definition in wall_range_state (SegLevelFacts publishes the same
        # chain on its fact row).
        return rw_distance_for(self.projection.core.scene, seg_i)

    def scale_sineb(self, *, is_stop: bool) -> Node:
        projection = self.projection
        scene = projection.core.scene
        screen_key = projection.rows.pick_range_screen_key(is_stop=is_stop)
        ray_x = pick_by_one_hot(screen_key, scene.view.ray_x_by_screen)
        ray_y = pick_by_one_hot(screen_key, scene.view.ray_y_by_screen)
        seg_i = projection.drawseg.store_i
        return SINEB_CLAMP(
            vec_sum(
                MUL_UNIT(ray_x, scene.segs.normal_cos(seg_i)),
                MUL_UNIT(ray_y, scene.segs.normal_sin(seg_i)),
            )
        )

    def is_near(self, distance: Node) -> Node:
        return one_minus(DIST_GT_ONE(distance))

    def bsilheight_value(self, seg_i: Node) -> Node:
        scene = self.projection.core.scene
        portal = scene.segs.is_portal(seg_i)
        solid_or_closed = one_minus(portal)
        front_floor = scene.segs.front_floor(seg_i)
        back_floor = scene.segs.back_floor(seg_i)
        portal_value = select(
            gt_height(front_floor, back_floor),
            front_floor,
            select(
                gt_height(back_floor, scene.view.z),
                constant(_SIL_HEIGHT_MAX),
                constant(0.0),
            ),
        )
        return select(solid_or_closed, constant(_SIL_HEIGHT_MAX), portal_value)

    def tsilheight_value(self, seg_i: Node) -> Node:
        scene = self.projection.core.scene
        portal = scene.segs.is_portal(seg_i)
        solid_or_closed = one_minus(portal)
        front_ceiling = scene.segs.front_ceiling(seg_i)
        back_ceiling = scene.segs.back_ceiling(seg_i)
        portal_value = select(
            gt_height(back_ceiling, front_ceiling),
            front_ceiling,
            select(
                gt_height(scene.view.z, back_ceiling),
                constant(_SIL_HEIGHT_MIN),
                constant(0.0),
            ),
        )
        return select(solid_or_closed, constant(_SIL_HEIGHT_MIN), portal_value)
