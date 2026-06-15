"""Read-only branch owner for the player-weapon pass (R_DrawPlayerSprites).

DOOM draws the ready pistol last, on top of the 3D view (``R_DrawPlayerSprites``
-> ``R_DrawPSprite``), under the painter order (last-write-wins). The weapon is
baked once into a screen-space masked picture (``weapon_assets.py`` ->
``AssetBanks.weapon_table_2d``): a dense bounding-box table of *lit* palette
indices with ``WEAPON_TRANSPARENT`` (256) for transparent cells.

This pass walks that bounding box (``weapon_min_col..weapon_max_col`` x
``weapon_top..weapon_bottom``) one column at a time, cursor advancing in Y like a
wall column. Per (col, row) it reads the baked opacity at the cursor and emits a
``PIXEL`` (opaque) or a ``SET_CURSOR_Y`` skip (transparent); the host paints the
pixel and the transparent gaps show the 3D scene through (Option 3 — reuse the
existing tokens, no new pixel token, no host change).

Token transitions (the owner of each arrow is this module unless noted):

    [flat pass end] -> R_DrawPlayerSprites              [flat_pass_renderer, HUD-gated]
    R_DrawPlayerSprites -> SET_CURSOR_DIRECTION_Y       (posts advance in Y)
    SET_CURSOR_DIRECTION_Y -> SET_CURSOR_X(min_col)     (weapon arm; weapon_seen)
    SET_CURSOR_X(col) -> col>max_col ? DONE : SET_CURSOR_Y(top)
    SET_CURSOR_Y(r)   -> D(col, r)
    PIXEL             -> D(col, current_row)

    D(col, y):                                   # the shared per-pixel decision
      current_row > weapon_bottom ? SET_CURSOR_X(col+1)              # next column
       : transparent(col, current_row) ? SET_CURSOR_Y(current_row+1) # skip
         : PIXEL(color = baked lit index)                            # paint

The SET_CURSOR_X / SET_CURSOR_Y / PIXEL arrows are SHARED with the wall and flat
passes; ``pixel_dispatcher`` forks each on ``weapon_seen`` (an OUTER select, so
the wall/flat width keystone stays structurally intact). ``current_row`` is
recovered from the most-recent weapon ``SET_CURSOR_Y`` (see ``WeaponPassState``);
``col`` from the most-recent weapon ``SET_CURSOR_X`` (both are stale on a PIXEL
row — pixels advance Y, not X, and a column's whole opaque run shares one
``SET_CURSOR_X``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import annotated
from torchwright.ops.arithmetic_ops import compare

from .constants import PIXEL_WIDTH
from .render_ops import (
    add_const,
    column_from_screen_x,
    gt_screen,
    screen_x_from_column,
    sub,
)
from .std import constant, make_token_head, select
from .std import sum as vec_sum
from .vocab import DONE, PIXEL, SET_CURSOR_DIRECTION_Y, SET_CURSOR_X, SET_CURSOR_Y
from .weapon_assets import WEAPON_TRANSPARENT

if TYPE_CHECKING:
    from torchwright.graph.node import Node

    from .seg_projection import SegProjection


@dataclass(frozen=True)
class PspriteRenderer:
    """Owns the player-weapon pass transitions (forked on ``weapon_seen``)."""

    projection: "SegProjection"

    # --- The phase prefix (one transition each, never shared) ----------------

    @annotated("pspr/R_DrawPlayerSprites")
    def after_draw_psprites_begin(self) -> "Node":
        # Posts advance in Y, so the weapon walk sets the cursor Y direction first.
        return make_token_head(SET_CURSOR_DIRECTION_Y)

    @annotated("pspr/R_DrawPlayerSprites")
    def after_set_cursor_direction_y_weapon(self) -> "Node":
        # Arm the walk at the first (leftmost) bounding-box column.
        banks = self.projection.core.scene.assets.banks
        return make_token_head(
            SET_CURSOR_X,
            x=screen_x_from_column(constant(float(banks.weapon_min_col))),
        )

    @annotated("pspr/R_DrawPlayerSprites")
    def after_set_cursor_x_weapon(self) -> "Node":
        # On the SET_CURSOR_X row cursor_x is fresh (it is this row's value);
        # past the last column finish the weapon (DONE), else start the column at
        # the bbox top.
        projection = self.projection
        banks = projection.core.scene.assets.banks
        col = column_from_screen_x(projection.core.inp.cursor_x)
        return select(
            gt_screen(col, constant(float(banks.weapon_max_col))),
            make_token_head(DONE),
            make_token_head(SET_CURSOR_Y, y=constant(float(banks.weapon_top))),
        )

    # --- The shared per-pixel decision (the SET_CURSOR_Y and PIXEL weapon arms) -

    @annotated("pspr/R_DrawPlayerSprites")
    def current_row(self) -> "Node":
        """The painted row: ``sy_value + (pos - sy_pos)`` (see WeaponCursorValues).

        Fresh on a SET_CURSOR_Y row (``pos == sy_pos``); on a PIXEL row it adds
        the count of opaque pixels emitted since the run's SET_CURSOR_Y.
        """
        projection = self.projection
        cur = projection.flats.weapon.weapon_cursor_values(projection.core.past)
        return vec_sum(cur.sy_value, sub(projection.core.pos, cur.sy_pos))

    @annotated("pspr/R_DrawPlayerSprites")
    def decision(self) -> "Node":
        projection = self.projection
        banks = projection.core.scene.assets.banks
        # col is recovered (the column's whole Y-run shares one SET_CURSOR_X);
        # current_row is recovered (stale on a PIXEL row).
        col = projection.flats.weapon.weapon_column(projection.core.past)
        current_row = self.current_row()
        # The baked picture is in-range over the bbox; the value is a lit palette
        # index (0..255) or WEAPON_TRANSPARENT (256) for a transparent cell.
        color = projection.core.scene.assets.weapon.color_or_transparent(
            col, current_row
        )
        # WEAPON_TRANSPARENT (256) sits above every palette index (0..255), so a
        # threshold at 255.5 cleanly separates opaque from transparent.
        transparent = compare(color, WEAPON_TRANSPARENT - 0.5)
        return select(
            gt_screen(current_row, constant(float(banks.weapon_bottom))),
            # Column finished: advance to the next column (its SET_CURSOR_X arm
            # then decides DONE vs. SET_CURSOR_Y(top)).
            make_token_head(
                SET_CURSOR_X,
                x=screen_x_from_column(add_const(col, 1.0)),
            ),
            select(
                transparent,
                # Skip the transparent row: the 3D scene shows through here.
                make_token_head(SET_CURSOR_Y, y=add_const(current_row, 1.0)),
                # Paint the baked lit index, PIXEL_WIDTH cells wide (matching the
                # wall/flat passes; one baked column == one rendered column).
                make_token_head(PIXEL, color=color, w=constant(float(PIXEL_WIDTH))),
            ),
        )
