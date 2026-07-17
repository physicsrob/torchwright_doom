"""Bake the player weapon (the ready pistol, sprite ``PISGA0``) into a static
screen-space picture the renderer paints last, on top of the 3D view.

This module runs once at compile time: it builds part of the computation graph that torchwright lowers into the transformer's weights. Nothing here executes during inference — at render time, only the compiled transformer runs. Coined terms: see GLOSSARY.md.

DOOM draws the player weapon in ``R_DrawPlayerSprites`` -> ``R_DrawPSprite``
(``r_things.c``) -> ``R_DrawVisSprite`` -> ``R_DrawMaskedColumn``. The player
weapon has no perspective: every frame it is the same size at the same screen
position, so for a fixed pose it is a static picture. This module ports that
placement math for the *ready* pistol (state ``A_WeaponReady`` with no bob:
``sx = FRACUNIT``, ``sy = WEAPONTOP``) and rasterizes it into our coordinate
system (``COLUMN_COUNT`` rendered columns x ``VIEW_HEIGHT`` rows).

The result is consumed two ways, from this one source of truth so they cannot
disagree:

- the in-tree ``pydoom`` reference emits the matching token stream
  (``R_DrawPlayerSprites`` phase), and
- the compiled graph bakes the same picture into a ``table_lookup_2d`` weight.

Transparency is preserved (``None``): where the sprite is transparent no pixel
is emitted, so the 3D scene shows through. Colors are baked *lit* — DOOM lights
the ready pistol by the view sector's brightest scale-light
(``spritelights[MAXLIGHTSCALE-1]``); we apply that colormap row at bake time, so
the emit path stays unlit (a raw palette index straight to the host).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .wad_assets import DOOM1_WAD_PATH, WADReader

FRACBITS = 16
FRACUNIT = 1 << FRACBITS

# DOOM psprite constants (p_pspr.c / r_things.c). Native 320x200 screen.
_SCREENWIDTH_DOOM = 320
# screenblocks-10 view height (the windowed view with the status bar) — the
# layout this whole status-bar work targets. centery = viewheight/2.
_VIEWHEIGHT_DOOM = 168
_WEAPONTOP = 32  # p_pspr.c:48  (psp->sy for the ready weapon, in screen units)
_BASEYCENTER = 100  # r_things.c:47
_READY_SX = 1  # A_WeaponReady sets psp->sx = FRACUNIT (bob = 0 when standing)

_PISTOL_READY_LUMP = "PISGA0"


@dataclass(frozen=True)
class WeaponPicture:
    """A baked weapon, in our rendered-column coordinate system.

    ``pixels[col][row]`` is a *lit* palette index, or ``None`` for transparent;
    ``column_count`` x ``view_height`` is the grid. The non-empty bounding box
    the emit phase needs is computed from it by ``bake_weapon_table`` (see
    ``WeaponBake``).
    """

    column_count: int
    view_height: int
    pixels: list[list[int | None]]  # pixels[col][row], None = transparent


def _fixed_mul(a: int, b: int) -> int:
    return (a * b) >> FRACBITS


def bake_pistol(
    scale: int,
    pixel_width: int,
    colormap_row: int = 0,
    *,
    wad_path=DOOM1_WAD_PATH,
) -> WeaponPicture:
    """Rasterize the ready pistol into a ``(COLUMN_COUNT x VIEW_HEIGHT)`` grid.

    ``scale`` and ``pixel_width`` are the renderer's screen knobs (1/2 and 2 for
    the production low-detail config). ``colormap_row`` selects the COLORMAP row
    applied for lighting (0 = brightest); bake the row matching the start
    sector's brightest scale-light. The math runs at DOOM-native 320x168 and is
    nearest-neighbour resampled to our grid (the same left-of-block sample the
    low-detail 3D view uses).
    """
    wad = WADReader(wad_path)
    buf = wad.lump(_PISTOL_READY_LUMP)
    width, height, leftoffset, topoffset = struct.unpack_from("<hhhh", buf, 0)
    patch = wad._patch_image(_PISTOL_READY_LUMP)  # pixels[col][row], None = transp
    colormap = wad.colormap()
    cmap = colormap[colormap_row]

    # --- Native 320x168 placement (pspritescale = FRACUNIT, viewwidth = 320). ---
    centerx = _SCREENWIDTH_DOOM // 2  # 160
    centery = _VIEWHEIGHT_DOOM // 2  # 84
    centerxfrac = centerx << FRACBITS
    centeryfrac = centery << FRACBITS
    pspritescale = FRACUNIT  # FRACUNIT * viewwidth / SCREENWIDTH, viewwidth = 320

    # R_DrawPSprite horizontal edges.
    tx = (_READY_SX << FRACBITS) - (centerx << FRACBITS) - (leftoffset << FRACBITS)
    x1 = (centerxfrac + _fixed_mul(tx, pspritescale)) >> FRACBITS
    tx2 = tx + (width << FRACBITS)
    x2 = ((centerxfrac + _fixed_mul(tx2, pspritescale)) >> FRACBITS) - 1

    # R_DrawVisSprite vertical anchor (sy = WEAPONTOP for the ready weapon).
    texturemid = (
        (_BASEYCENTER << FRACBITS)
        + (FRACUNIT // 2)
        - ((_WEAPONTOP << FRACBITS) - (topoffset << FRACBITS))
    )
    spryscale = pspritescale
    sprtopscreen = centeryfrac - _fixed_mul(texturemid, spryscale)

    # Native screen grid (320 x 168), column-major; None = transparent.
    native: list[list[int | None]] = [
        [None for _ in range(_VIEWHEIGHT_DOOM)] for _ in range(_SCREENWIDTH_DOOM)
    ]
    for sx in range(max(0, x1), min(_SCREENWIDTH_DOOM, x2 + 1)):
        texcol = sx - x1  # pspriteiscale = FRACUNIT (no flip), startfrac = 0
        if texcol < 0 or texcol >= width:
            continue
        column = patch.pixels[texcol]
        # R_DrawMaskedColumn: each texel ty lands at sprtopscreen + ty*spryscale.
        for ty in range(height):
            color = column[ty]
            if color is None:
                continue
            topscreen = sprtopscreen + spryscale * ty
            sy = (topscreen + FRACUNIT - 1) >> FRACBITS
            if 0 <= sy < _VIEWHEIGHT_DOOM:
                native[sx][sy] = int(color)

    # --- Resample native 320x168 -> our (COLUMN_COUNT x VIEW_HEIGHT) grid. ---
    # Our column c spans screen x [c*pixel_width, ...]; at native resolution that
    # is x = c * pixel_width * scale. Row r -> native y = r * scale. Sample the
    # top-left of each block (the low-detail "left column" convention).
    screen_width = _SCREENWIDTH_DOOM // scale
    view_height = _VIEWHEIGHT_DOOM // scale
    column_count = screen_width // pixel_width
    pixels: list[list[int | None]] = [
        [None for _ in range(view_height)] for _ in range(column_count)
    ]
    for col in range(column_count):
        nx = col * pixel_width * scale
        if nx >= _SCREENWIDTH_DOOM:
            continue
        for row in range(view_height):
            ny = row * scale
            if ny >= _VIEWHEIGHT_DOOM:
                continue
            raw = native[nx][ny]
            if raw is None:
                continue
            pixels[col][row] = int(cmap[raw])
    return WeaponPicture(
        column_count=column_count, view_height=view_height, pixels=pixels
    )


# Transparent sentinel for the baked weapon table: a value outside the 0..255
# palette range. The render loop reads the table at the cursor and, on the
# sentinel, emits a setCursorY skip instead of a pixel — reusing an existing
# token, no new pixel token, no host change — so the sentinel never reaches
# the host.
WEAPON_TRANSPARENT = 256.0


@dataclass(frozen=True)
class WeaponBake:
    """A baked weapon ready for the asset banks.

    ``table`` is ``(bbox_height, bbox_width)`` of lit palette indices, with
    ``WEAPON_TRANSPARENT`` for transparent cells; it is addressed by the cursor
    offset into the bounding box. ``min_col``/``max_col`` (rendered columns) and
    ``top``/``bottom`` (rows) are the sprite's bounding box in screen space — the
    only bounds the emit phase needs; the per-pixel transparency lives in
    ``table`` and is resolved in the render loop, not preprocessed.
    """

    table: np.ndarray  # (bbox_height, bbox_width); float32
    min_col: int
    max_col: int
    top: int
    bottom: int


def bake_weapon_table(
    scale: int,
    pixel_width: int,
    colormap_row: int = 0,
    *,
    wad_path=DOOM1_WAD_PATH,
) -> WeaponBake:
    """Bake the ready pistol into a dense bounding-box table for the graph.

    The table holds a lit palette index per cell, ``WEAPON_TRANSPARENT`` where
    the sprite is transparent. Empty rows/cols outside the opaque region are
    trimmed to the bounding box so the ``table_lookup_2d`` weight stays small.
    """
    pic = bake_pistol(scale, pixel_width, colormap_row, wad_path=wad_path)
    occupied = [
        (col, row)
        for col in range(pic.column_count)
        for row in range(pic.view_height)
        if pic.pixels[col][row] is not None
    ]
    if not occupied:
        raise ValueError("baked pistol has no opaque pixels")
    cols = [c for c, _ in occupied]
    rows = [r for _, r in occupied]
    min_col, max_col = min(cols), max(cols)
    top, bottom = min(rows), max(rows)
    h = bottom - top + 1
    w = max_col - min_col + 1
    table = np.full((h, w), WEAPON_TRANSPARENT, dtype=np.float32)
    for col in range(min_col, max_col + 1):
        for row in range(top, bottom + 1):
            idx = pic.pixels[col][row]
            if idx is not None:
                table[row - top, col - min_col] = float(idx)
    return WeaponBake(
        table=table, min_col=min_col, max_col=max_col, top=top, bottom=bottom
    )
