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
from torchwright.graph.asserts import assert_in_range

from .constants import COLUMN_COUNT, HUD_ENABLED, PIXEL_WIDTH
from .flat_pass_renderer import FlatPassRenderer
from .lighting import apply_colormap_row
from .psprite_renderer import PspriteRenderer
from .statusbar_renderer import StatusBarRenderer
from .pwl_banks import FLOOR_MOD64
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
from .std import (
    bool_and,
    bool_not,
    constant,
    linear,
    make_token_head,
    pick_by_index,
    select,
    type_switch,
)
from .std import sum as vec_sum
from .uv_compute import compute_v_mods_at_pixel, compute_v_native_at_screen_y
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

    # Each shared branch is a PRIORITY over the pass latch flags — hud (drawn
    # last) over weapon over flat over wall. The naive encoding is a select
    # ladder, but a ladder's depth is the sum of its rungs and this one sits on
    # the compiled critical path (layers 42-45 of the 49-layer spine). The flat
    # form below derives mutually-exclusive masks from the latch flags in one
    # parallel boolean layer and picks the winning arm with a single
    # type_switch. The masks derive from flags that exist by layer ~1, so the
    # mask network runs in parallel with the (much deeper) branch-arm chains
    # and adds no depth of its own; measured DAG floor 49 -> 47.
    #
    # Numerically this extends the pattern the main dispatch already certifies
    # (flat-folding emit heads through type_switch/cond_gate): at a clean +-1
    # condition the losing branch contributes exactly zero in fp32. Every mask
    # input here is a latch flag or a bool_* output, so all conds are snapped
    # +-1 booleans. Degenerate case preserved: before the flat pass,
    # flat_span_seen is structurally false, so m_wall is the one true mask and
    # the switch degenerates to the wall arm — same behavior as the ladder.

    def _pixel_priority_switch(
        self,
        weapon_arm: "Node",
        flat_arm: "Node",
        wall_arm: "Node",
        hud_arm: "Node | None",
    ) -> tuple[tuple["Node", "Node"], ...]:
        """Exclusive-mask (cond, arm) pairs for hud > weapon > flat > wall.

        Returns the pairs instead of switching here: the dispatch folds them
        into its own flat ``type_switch`` (each cond AND-ed with the branch's
        transition predicate), so the branch pays no gate+sum level of its
        own. Exactly one cond is +1 (same exclusive-mask argument as before).
        """
        flats = self.projection.flats
        weapon_seen = flats.weapon.weapon_seen
        flat_seen = flats.flat_pass.flat_span_seen
        not_weapon = bool_not(weapon_seen)
        not_flat = bool_not(flat_seen)
        # HUD off: no HUD arm built at all (bit-identical to pre-HUD).
        if hud_arm is None:
            return (
                (weapon_seen, weapon_arm),
                (bool_and(flat_seen, not_weapon), flat_arm),
                (bool_and(not_flat, not_weapon), wall_arm),
            )
        hud = flats.hud
        assert hud is not None  # built on the HUD_ENABLED path (guarded above)
        not_hud = bool_not(hud.hud_seen)
        return (
            (hud.hud_seen, hud_arm),
            (bool_and(weapon_seen, not_hud), weapon_arm),
            (bool_and(flat_seen, not_weapon, not_hud), flat_arm),
            (bool_and(not_flat, not_weapon, not_hud), wall_arm),
        )

    @annotated("pix")
    def after_wall_column(self) -> "Node":
        projection = self.projection
        flat_first_pixel = make_token_head(
            PIXEL,
            color=self.flat_pixel_atlas_color(constant(0.0)),
            # Flat span painted PIXEL_WIDTH cells wide at low-detail: the span
            # emits (x2-x1+1) column samples and the host blits each w cells in
            # +X so they tile the full screen extent [x1*PW, (x2+1)*PW). With
            # w=1 the span covered only its left half (screen-x < ~COLUMN_COUNT)
            # -- the missing center/right flats. Matches the pydoom drafter
            # (Token(PIXEL, w=PIXEL_WIDTH)); identity at high-detail.
            w=constant(float(PIXEL_WIDTH)),
        )
        return self._pixel_priority_switch(
            weapon_arm=PspriteRenderer(projection).after_set_cursor_x_weapon(),
            flat_arm=flat_first_pixel,
            wall_arm=self.wall_column_output(),
            hud_arm=(
                StatusBarRenderer(projection).after_set_cursor_x_hud()
                if HUD_ENABLED
                else None
            ),
        )

    @annotated("pix")
    def after_set_cursor_y(self) -> "Node":
        projection = self.projection
        flat_span = projection.flats.flat_pass.flat_span_values(projection.core.past)
        return self._pixel_priority_switch(
            weapon_arm=PspriteRenderer(projection).decision(),
            flat_arm=make_token_head(
                SET_CURSOR_X, x=screen_x_from_column(flat_span.x1)
            ),
            wall_arm=make_value(ValueRange.R3, self.span_v0_at_top()),
            hud_arm=(StatusBarRenderer(projection).decision() if HUD_ENABLED else None),
        )

    @annotated("pix")
    def after_pixel_color(self) -> "Node":
        projection = self.projection
        return self._pixel_priority_switch(
            weapon_arm=PspriteRenderer(projection).decision(),
            flat_arm=self.after_flat_pixel_color(),
            wall_arm=self.after_wall_pixel_color(),
            hud_arm=(StatusBarRenderer(projection).decision() if HUD_ENABLED else None),
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
            # Wall column painted PIXEL_WIDTH screen cells wide (R_DrawColumnLow):
            # the host blits w cells horizontally per row (decode.py), so a 2-wide
            # column needs w=2 at low-detail. Matches the pydoom drafter's
            # Token(PIXEL, w=PIXEL_WIDTH); identity at high-detail.
            w=constant(float(PIXEL_WIDTH)),
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
        v_mods = compute_v_mods_at_pixel(
            pixel_index_vec=pixel_index_vec,
            dc_iscale=span.dc_iscale,
            v_0_at_top=span_v0.v0_at_top,
            v_mod_bank=projection.core.scene.assets.v_mod_bank,
        )
        raw_palette_index = projection.core.scene.assets.walls.palette_index(
            span.tex_id,
            span.u_native,
            v_mods,
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
                # PIXEL_WIDTH-wide flat-span cell (see after_wall_column).
                w=constant(float(PIXEL_WIDTH)),
            ),
            FlatPassRenderer(projection).after_completed_flat_span_row(),
        )

    @annotated("pix/R_DrawSpan")
    def flat_pixel_atlas_color(self, pixel_index_vec: "Node") -> "Node":
        projection = self.projection
        cursor = projection.flats.flat_pass.flat_cursor_values(projection.core.past)
        k_xstep = mul_k_step(pixel_index_vec, cursor.xstep)
        k_ystep = mul_k_step(pixel_index_vec, cursor.ystep)
        # Range claims: FLOOR_MOD64's input contract (pwl_banks: raw
        # native coordinate in [-1023, 1023]).  Interval arithmetic puts
        # these Linears at ±1.4e9 — outside the contract the floor
        # saturates and the pixel is discarded junk, but the slack
        # poisons any range-driven analysis of the flat texel chain.
        xfrac_k = assert_in_range(vec_sum(cursor.xfrac0, k_xstep), -1023.0, 1023.0)
        yfrac_k = assert_in_range(vec_sum(cursor.yfrac0, k_ystep), -1023.0, 1023.0)
        u = FLOOR_MOD64(xfrac_k)
        v = FLOOR_MOD64(yfrac_k)
        raw_palette = projection.core.scene.assets.flats.palette_index(
            cursor.flat_id,
            u,
            v,
        )
        return apply_colormap_row(raw_palette, cursor.cmap_row)
