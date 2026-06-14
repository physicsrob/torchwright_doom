"""Read-only mixed branch owner for wall/flat pixel transitions.

The three shared pixel/cursor branches (``after_wall_column`` on SET_CURSOR_X,
``after_set_cursor_y`` on SET_CURSOR_Y, ``after_pixel_color`` on PIXEL) are
**shared** between the wall and flat passes; each forks on the runtime boolean
``flat_span_seen`` (``+1`` once a ``SPAN_ROW`` has fired this frame, ``-1``
before). ``SPAN_ROW`` is only emitted by ``FlatPassRenderer`` after the BSP walk,
so before the flat pass ``flat_span_seen`` is structurally false and every fork
degenerates to the wall arm. The full flat-pass token sequence (where these
shared branches sit in it) is mapped in ``flat_pass_renderer.py``.

Changes from the original: ``Vec`` -> ``Node``; ``make_token`` ->
``make_token_head``; ``make_value`` -> the eager R3 VALUE head (so the
``after_set_cursor_y`` ``select`` chooses between two head ``Node``\\ s, not a head
vs. a ``ScalarEmit``); ``ONE`` -> ``add_const(., -1.0)``; the module-level
multiply / floor / zero nodes relocated to ``render_ops`` shims /
inside-function ``constant`` (no import-time nodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import annotated

from .constants import COLUMN_COUNT
from .flat_pass_renderer import FlatPassRenderer
from .lighting import apply_colormap_row
from .pwl_banks import MOD64_PWL
from .render_ops import (
    FLOOR_NATIVE,
    add_const,
    column_from_screen_x,
    gt_screen,
    mul_k_step,
    mul_u_native,
    screen_x_from_column,
    sub,
)
from .std import constant, linear, make_token_head, pick_by_index, select
from .std import sum as vec_sum
from .uv_compute import compute_v_at_pixel, compute_v_native_at_screen_y
from .value_ranges import ValueRange
from .vocab import PIXEL, SET_CURSOR_X, WALL_COL_U, make_value
from .wall_column_renderer import WallColumnRenderer

if TYPE_CHECKING:
    from torchwright.graph.node import Node

    from .seg_projection import SegProjection


_NEG1_LINEAR = [[-1.0]]  # plain list weight matrix — not a node, fine at module level


@dataclass(frozen=True)
class PixelDispatcher:
    """Owns protocol branches that merge wall and flat pixel paths."""

    projection: "SegProjection"

    # --- The three shared pixel/cursor branches (forked on flat_span_seen) ----

    @annotated("pix")
    def after_wall_column(self) -> "Node":
        projection = self.projection
        flat_first_pixel = make_token_head(
            PIXEL,
            color=self.flat_pixel_atlas_color(constant(0.0)),
            w=constant(1.0),
        )
        return select(
            projection.flats.flat_pass.flat_span_seen,
            flat_first_pixel,
            self.wall_column_output(),
        )

    @annotated("pix")
    def after_set_cursor_y(self) -> "Node":
        projection = self.projection
        flat_span = projection.flats.flat_pass.flat_span_values(projection.core.past)
        return select(
            projection.flats.flat_pass.flat_span_seen,
            make_token_head(SET_CURSOR_X, x=screen_x_from_column(flat_span.x1)),
            make_value(ValueRange.R3, self.span_v0_at_top()),
        )

    @annotated("pix")
    def after_pixel_color(self) -> "Node":
        projection = self.projection
        return select(
            projection.flats.flat_pass.flat_span_seen,
            self.after_flat_pixel_color(),
            self.after_wall_pixel_color(),
        )

    # --- Wall texel pass -----------------------------------------------------

    @annotated("pix/R_DrawColumn")
    def wall_column_output(self) -> "Node":
        projection = self.projection
        seg_i_active = projection.drawseg.store_i
        u_tan_by_column = projection.rows.pick_u_tan_by_column()
        tan_rel = pick_by_index(
            column_from_screen_x(projection.core.inp.cursor_x),
            u_tan_by_column,
            COLUMN_COUNT,
        )
        rw_distance_active = projection.wall.seg_facts.rw_distance(seg_i_active)
        mul_dist_tan = mul_u_native(rw_distance_active, tan_rel)
        u_native = linear(mul_dist_tan, _NEG1_LINEAR)
        u_floor = FLOOR_NATIVE(u_native)
        return make_token_head(WALL_COL_U, u_idx=u_floor)

    @annotated("pix/R_DrawColumn")
    def after_wall_pixel_color(self) -> "Node":
        projection = self.projection
        span = projection.wall.wall_span_runtime.span_start_values(projection.core.past)
        span_v0 = projection.wall.wall_span_runtime.span_v0_values(projection.core.past)
        pixel_index = add_const(sub(projection.core.pos, span_v0.pos), -1.0)
        next_pixel_index = add_const(pixel_index, 1.0)
        next_in_span = gt_screen(span.height, next_pixel_index)
        return select(
            next_in_span,
            self.make_pixel_color(next_pixel_index),
            WallColumnRenderer(projection).after_completed_span(),
        )

    @annotated("pix/R_DrawColumn")
    def make_first_pixel_color(self) -> "Node":
        return self.make_pixel_color(constant(0.0))

    @annotated("pix/R_DrawColumn")
    def make_pixel_color(self, pixel_index_vec: "Node") -> "Node":
        return make_token_head(
            PIXEL,
            color=self.pixel_lit_palette_index(pixel_index_vec),
            w=constant(1.0),
        )

    @annotated("pix/R_DrawColumn")
    def span_v0_at_top(self) -> "Node":
        span = self.projection.wall.wall_span_runtime.span_start_values(
            self.projection.core.past
        )
        return compute_v_native_at_screen_y(
            screen_y=span.y_start,
            dc_iscale=span.dc_iscale,
            dc_texturemid=span.dc_texturemid,
        )

    @annotated("pix/R_DrawColumn")
    def pixel_lit_palette_index(self, pixel_index_vec: "Node") -> "Node":
        projection = self.projection
        span = projection.wall.wall_span_runtime.span_start_values(projection.core.past)
        span_v0 = projection.wall.wall_span_runtime.span_v0_values(projection.core.past)
        v_scaled_mod_H = compute_v_at_pixel(
            pixel_index_vec=pixel_index_vec,
            dc_iscale=span.dc_iscale,
            v_0_at_top=span_v0.v0_at_top,
            h_idx_oh=span.h_idx_oh,
            sawtooth_bank=projection.core.scene.assets.sawtooth_bank,
        )
        raw_palette_index = projection.core.scene.assets.walls.palette_index(
            span.tex_id,
            span.u_native,
            v_scaled_mod_H,
        )
        return apply_colormap_row(raw_palette_index, span.cmap_row)

    # --- Flat span pass ------------------------------------------------------

    @annotated("pix/R_DrawSpan")
    def after_flat_pixel_color(self) -> "Node":
        projection = self.projection
        cursor = projection.flats.flat_pass.flat_cursor_values(projection.core.past)
        flat_span = projection.flats.flat_pass.flat_span_values(projection.core.past)
        span_width = add_const(sub(flat_span.x2, flat_span.x1), 1.0)
        pixel_index = add_const(sub(projection.core.pos, cursor.pos), -1.0)
        next_pixel_index = add_const(pixel_index, 1.0)
        next_in_span = gt_screen(span_width, next_pixel_index)
        return select(
            next_in_span,
            make_token_head(
                PIXEL,
                color=self.flat_pixel_atlas_color(next_pixel_index),
                w=constant(1.0),
            ),
            FlatPassRenderer(projection).after_completed_flat_span_row(),
        )

    @annotated("pix/R_DrawSpan")
    def flat_pixel_atlas_color(self, pixel_index_vec: "Node") -> "Node":
        projection = self.projection
        cursor = projection.flats.flat_pass.flat_cursor_values(projection.core.past)
        k_xstep = mul_k_step(pixel_index_vec, cursor.xstep)
        k_ystep = mul_k_step(pixel_index_vec, cursor.ystep)
        xfrac_k = vec_sum(cursor.xfrac0, k_xstep)
        yfrac_k = vec_sum(cursor.yfrac0, k_ystep)
        u = MOD64_PWL(FLOOR_NATIVE(xfrac_k))
        v = MOD64_PWL(FLOOR_NATIVE(yfrac_k))
        raw_palette = projection.core.scene.assets.flats.palette_index(
            cursor.flat_id,
            u,
            v,
        )
        return apply_colormap_row(raw_palette, cursor.cmap_row)
