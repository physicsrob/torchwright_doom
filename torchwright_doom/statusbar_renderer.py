"""Read-only branch owner for the status-bar pass (ST_Drawer).

DOOM draws the status bar last, on top of everything, as a SEQUENCE of
``V_DrawPatch`` calls (st_stuff.c / st_lib.c): the plate, then the ammo/health/
armor numbers, the ARMS panel + weapon numbers, the face. Each ``V_DrawPatch`` is
a raw masked blit at a fixed screen position. We bake that sequence into a
draw-list (``hud_assets.py`` -> ``AssetBanks.hud_item_*``) and the patches into one
banked table (``AssetBanks.hud_table_2d``).

This pass walks the draw-list one patch at a time. It is the player-weapon pass
(``psprite_renderer``) generalized from a single fixed picture to a LIST of
pictures: a ``HUD_ITEM`` marks the next draw-list entry (its patch id + screen
origin + size), then the patch is rasterized column-major exactly like the
weapon's bounding box. Per (col, row) it reads the baked opacity at the cursor
and emits a ``PIXEL`` (opaque) or a ``SET_CURSOR_Y`` skip (transparent); the host
paints under last-write-wins, so later patches overwrite the plate beneath them —
DOOM's painter order.

Token transitions (this module owns each arrow unless noted):

    [weapon pass end] -> ST_Drawer                     [psprite_renderer, HUD-gated]
    ST_Drawer -> HUD_ITEM(0)                            (first draw-list entry)
    HUD_ITEM(i) -> SET_CURSOR_X(origin_x[i])            (arm the patch's first col)
    SET_CURSOR_X -> local_col >= width[i]
                       ? (i last ? DONE : HUD_ITEM(i+1))   # patch done
                       : SET_CURSOR_Y(origin_y[i])         # open the column
    SET_CURSOR_Y(r) -> D(i, r)
    PIXEL           -> D(i, current_row)

    D(i, y):                                   # the shared per-pixel decision
      current_row > origin_y[i]+height[i]-1 ? SET_CURSOR_X(cursor_x+1)  # next col
       : transparent(i, local_col, v) ? SET_CURSOR_Y(current_row+1)     # skip
         : PIXEL(color, w=1)                                            # paint

The cursor direction is Y, inherited from the weapon pass that always precedes
the bar (both gated on the HUD). The bar paints at native screen resolution
(``w = 1``, one host pixel per screen column — crisp UI, unlike the doubled 3D
view), so the screen column is the raw ``cursor_x``: local patch column =
``cursor_x - origin_x[i]``. The ``SET_CURSOR_X`` / ``SET_CURSOR_Y`` / ``PIXEL``
arrows are SHARED with the wall/flat/weapon passes; ``pixel_dispatcher`` and
``render_main`` fork each on ``hud_seen`` (an OUTER select, so the wall/flat width
keystone stays intact). ``i`` is the most-recent ``HUD_ITEM``; ``cursor_x`` the
most-recent HUD ``SET_CURSOR_X`` (both via ``HudPassState``); ``current_row`` the
most-recent HUD ``SET_CURSOR_Y`` plus the opaque pixels since.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import annotated
from torchwright.ops.relu.arithmetic_ops import compare

from .hud_assets import HUD_TRANSPARENT
from .render_ops import add_const, gt_screen, sub
from .std import constant, make_token_head, pick_by_index, select
from .std import sum as vec_sum
from .vocab import DONE, HUD_ITEM, PIXEL, SET_CURSOR_X, SET_CURSOR_Y

if TYPE_CHECKING:
    from torchwright.graph.node import Node

    from .flat_state import HudPassState
    from .seg_projection import SegProjection


@dataclass(frozen=True)
class StatusBarRenderer:
    """Owns the status-bar pass transitions (forked on ``hud_seen``)."""

    projection: "SegProjection"

    # --- Per-item draw-list table reads (indexed by the draw-list item) --------

    @property
    def _hud(self) -> "HudPassState":
        # StatusBarRenderer is only built on the HUD_ENABLED path (the
        # `if not HUD_ENABLED: return below` guards in pixel_dispatcher), where
        # the flats projection always carries a HudPassState -- never None here.
        hud = self.projection.flats.hud
        assert hud is not None
        return hud

    def _item(self) -> "Node":
        return self._hud.hud_item(self.projection.core.past)

    def _table(self, values: list[float], index: "Node") -> "Node":
        banks = self.projection.core.scene.assets.banks
        return pick_by_index(index, constant(values), banks.n_hud_items)

    # --- The phase prefix (one transition each, never shared) ------------------

    @annotated("hud/ST_Drawer")
    def after_hud_begin(self) -> "Node":
        # Start the draw-list at item 0. The cursor direction is already Y
        # (inherited from the weapon pass that always precedes the bar).
        return make_token_head(HUD_ITEM, item=constant(0.0))

    @annotated("hud/ST_Drawer")
    def after_hud_item(self) -> "Node":
        # On a HUD_ITEM row the item index is fresh (this row's value); arm the
        # patch's first column at its screen origin.
        item = self.projection.core.inp.hud_item_value
        origin_x = self._table(
            self.projection.core.scene.assets.banks.hud_item_origin_x, item
        )
        return make_token_head(SET_CURSOR_X, x=origin_x)

    @annotated("hud/ST_Drawer")
    def after_set_cursor_x_hud(self) -> "Node":
        # Open the column at the patch top. The item-done transition lives in
        # the decision (it fires ON the last column, so the cursor never has to
        # advance to `width` — which for a full-width patch like the plate would
        # exceed the SET_CURSOR_X range [0, SCREEN_WIDTH) and clamp, sticking the
        # walk on the last column).
        item = self._item()
        return make_token_head(
            SET_CURSOR_Y,
            y=self._table(
                self.projection.core.scene.assets.banks.hud_item_origin_y, item
            ),
        )

    # --- The shared per-pixel decision (the SET_CURSOR_Y and PIXEL HUD arms) ----

    @annotated("hud/ST_Drawer")
    def current_row(self) -> "Node":
        """The painted row: ``sy_value + (pos - sy_pos)`` (cf. the weapon)."""
        projection = self.projection
        cur = self._hud.hud_cursor_values(projection.core.past)
        return vec_sum(cur.sy_value, sub(projection.core.pos, cur.sy_pos))

    @annotated("hud/ST_Drawer")
    def decision(self) -> "Node":
        projection = self.projection
        banks = projection.core.scene.assets.banks
        item = self._item()
        # cursor_x is stale on a PIXEL/SET_CURSOR_Y row, so recover the column
        # from the most-recent HUD SET_CURSOR_X.
        cursor_x = self._hud.hud_cursor_x(projection.core.past)
        origin_x = self._table(banks.hud_item_origin_x, item)
        origin_y = self._table(banks.hud_item_origin_y, item)
        width = self._table(banks.hud_item_width, item)
        height = self._table(banks.hud_item_height, item)
        patch_id = self._table(banks.hud_item_patch_id, item)
        current_row = self.current_row()
        local_col = sub(cursor_x, origin_x)
        local_row = sub(current_row, origin_y)  # v into the patch
        # Baked color: a raw (unlit) palette index or HUD_TRANSPARENT (256).
        color = projection.core.scene.assets.hud.color_or_transparent(
            patch_id, local_col, local_row
        )
        transparent = compare(color, HUD_TRANSPARENT - 0.5)
        # bottom row of this column = origin_y + height - 1.
        bottom = add_const(vec_sum(origin_y, height), -1.0)
        # On the LAST column (local_col == width-1) finishing, advance the item
        # (or DONE) directly — never emit SET_CURSOR_X(origin_x + width), which a
        # full-width patch can't encode. Otherwise advance to the next column.
        # gt_screen(a, b) is the INTEGER "a > b" (it bakes in the 0.5), so the
        # last-column test "local_col >= width-1" is gt_screen(local_col, width-2)
        # and the last-item test "item >= n_items-1" is gt_screen(item, n_items-2).
        is_last_col = gt_screen(local_col, sub(width, constant(2.0)))
        last_item = gt_screen(item, constant(float(banks.n_hud_items) - 2.0))
        column_done = select(
            is_last_col,
            select(
                last_item,
                make_token_head(DONE),
                make_token_head(HUD_ITEM, item=add_const(item, 1.0)),
            ),
            make_token_head(SET_CURSOR_X, x=add_const(cursor_x, 1.0)),
        )
        return select(
            gt_screen(current_row, bottom),
            column_done,
            select(
                transparent,
                # Transparent: skip the row, whatever is beneath shows through.
                make_token_head(SET_CURSOR_Y, y=add_const(current_row, 1.0)),
                # Paint the baked index, one host cell wide (native bar resolution).
                make_token_head(PIXEL, color=color, w=constant(1.0)),
            ),
        )
