"""Per-pixel v coordinate decode for texture sampling.

This module runs once at compile time: it builds part of the computation
graph that torchwright lowers into the transformer's weights. Nothing here
executes during inference — at render time, only the compiled transformer
runs. Coined terms: see GLOSSARY.md.

The chain mirrors DOOM ``R_DrawColumn`` texture stepping (r_draw.c:132-145)::

    v_native = dc_texturemid + (y - centery) * dc_iscale
    v_floor  = floor(v_native)                       (ONE shared floor)
    v_mod_h  = v_floor % texture_height              (one PWL per bank height)

``build_v_mod_bank`` gives one ``v_floor mod H`` PWL per ``WALL_HEIGHT_BANK``
height; this module floors once and evaluates ALL of them, returning the
tuple.  Each wall bank is single-height, so ``assets.WallAssets.palette_index``
feeds each bank its own height's entry directly — no ``h_idx_oh`` pick on the
row-address path (unselected banks see their own-height mod of a possibly-junk
``v_native``: bounded junk, discarded by the bank mask exactly like their
junk u/row candidates today).  The floor is shared, not folded per-height —
see ``pwl_banks`` for the width story.

All graph nodes are built inside the functions, never at module scope — see
GLOSSARY.md 'the import-time-node rule'.

``mul_pixel_dc_iscale`` is exact on its first axis because ``pixel_index`` /
``screen_y - CENTER_Y`` are always integers (``CENTER_Y = VIEW_HEIGHT / 2``,
an integer at every committed config), so they land on the grid's unit step
and the dc_iscale axis cell precision does not matter.
"""

from __future__ import annotations

from torchwright.graph import Node
from torchwright.graph import annotated
from torchwright.graph.asserts import assert_in_range

from ..constants import CENTER_Y
from ..assets.pwl_banks import V_MOD_BANK
from ..render_ops import FLOOR_NATIVE, add_const, mul_pixel_dc_iscale
from ..std import sum as vec_sum


@annotated("paint")
def compute_v_mods_at_pixel(
    *,
    pixel_index_vec: Node,
    dc_iscale: Node,
    v_0_at_top: Node,
    v_mod_bank: tuple | None = None,
) -> tuple[Node, ...]:
    """Return ``floor(v) mod H`` for one pixel, one entry per bank height."""
    v_mod_bank = V_MOD_BANK if v_mod_bank is None else v_mod_bank
    v_offset = mul_pixel_dc_iscale(pixel_index_vec, dc_iscale)
    # Range claim: FLOOR_NATIVE's own input contract (render_ops /
    # pwl_banks: raw native coordinate in [-1023, 1023]).  Interval
    # arithmetic alone puts this Linear at ±9.5e8 — slack that outside
    # the contract means nothing (the floor saturates and downstream is
    # discarded junk), but that poisons any range-driven analysis of
    # the texel chain.
    v_native = assert_in_range(vec_sum(v_offset, v_0_at_top), -1023.0, 1023.0)
    v_floor = FLOOR_NATIVE(v_native)
    return tuple(mod_h(v_floor) for mod_h in v_mod_bank)


@annotated("paint")
def compute_v_native_at_screen_y(
    *,
    screen_y: Node,
    dc_iscale: Node,
    dc_texturemid: Node,
) -> Node:
    """Return native texture v at an absolute screen y before floor + mod H."""
    y_offset = add_const(screen_y, -float(CENTER_Y))
    v_offset = mul_pixel_dc_iscale(y_offset, dc_iscale)
    return vec_sum(dc_texturemid, v_offset)
