"""Bake DOOM's status bar as a list of masked patches the renderer composites
last, into the reserved bottom rows of the screen.

This module runs once at compile time: it builds part of the computation graph that torchwright lowers into the transformer's weights. Nothing here executes during inference — at render time, only the compiled transformer runs. Coined terms: see GLOSSARY.md.

DOOM draws the status bar as a **sequence of ``V_DrawPatch`` calls**, not as one
pre-composited image: the ``STBAR`` plate first (``ST_refreshBackground``,
st_stuff.c:504), then each widget — the ammo/health/armor numbers, the ``STARMS``
panel and its weapon numbers, the face — painted on top in painter order
(``ST_drawWidgets``, st_stuff.c:1064-1083). ``V_DrawPatch`` (v_video.c:243) is a
raw masked blit: walk the patch's columns, walk each column's posts, copy palette
bytes — **no scaling, no colormap, no view-clip**. This module mirrors that
exactly: each lump stays its own masked patch (never merged), and a static
draw-list of ``(patch, x, y)`` records the ``V_DrawPatch`` sequence for the
hardcoded E1M1 pistol-start state.

This is deliberately **not** the weapon's path. The player weapon is a scaled,
lit psprite (``R_DrawPSprite`` -> ``R_DrawVisSprite`` -> the masked-column
drawer); the bar is the raw ``V_DrawPatch`` blit. They are different DOOM
operations and stay separate phases.

The output is consumed two ways, from this one source of truth so they cannot
disagree:

- the in-tree ``pydoom`` reference composites the same draw-list with a faithful
  ``V_DrawPatch`` blit (the pixel oracle's ground truth), and
- the compiled graph embeds each patch as a ``table_lookup_2d`` weight and walks
  the draw-list, emitting pixels under last-write-wins.

**Hardcoded E1M1 pistol-start state** (st_stuff.c widget values): health 100,
armor 0, ammo 50, pistol only — so ARMS number "2" is lit (yellow ``STYSNUM2``)
and "3".."7" are gray (``STGNUM3..7``) — no keys, neutral forward-facing face
(``STFST01``). Fixing the state this way means **no in-graph digit arithmetic,
no face state machine, no prefill scalars**: the draw-list is a constant.

Resolution: the bar is emitted at native screen resolution, one host pixel per
screen column (``w = 1``, unlike the doubled 3D view). At ``scale`` 1 that is
DOOM's exact 320x200 bar; at ``scale`` 2 every patch is decimated and every
position halved for the 160x100 preview. Both the reference and the graph
consume the decimated bank, so they match by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wad_assets import DOOM1_WAD_PATH, WADReader

# --- Status-bar layout constants (st_stuff.c / st_stuff.h, native 320x200). ---
_FULL_SCREEN_WIDTH = 320
_FULL_SCREEN_HEIGHT = 200
_FULL_BAR_HEIGHT = 32  # ST_HEIGHT
ST_X = 0  # st_stuff.c:86
ST_Y = _FULL_SCREEN_HEIGHT - _FULL_BAR_HEIGHT  # ST_Y = 168

# Widget anchors (st_stuff.c). ``*X`` for a number is the RIGHT edge (drawNum
# advances leftward); ``*Y`` is the absolute screen row.
ST_AMMOX, ST_AMMOY, ST_AMMOWIDTH = 44, 171, 3
ST_HEALTHX, ST_HEALTHY = 90, 171
ST_ARMORX, ST_ARMORY = 221, 171
ST_ARMSBGX, ST_ARMSBGY = 104, 168
ST_ARMSX, ST_ARMSY = 111, 172
ST_ARMSXSPACE, ST_ARMSYSPACE = 12, 10
ST_FACESX, ST_FACESY = 143, 168

# Per-type ammo (BULL/SHEL/RCKT/CELL) and max-ammo counts on the right of the
# bar. Index is the ammo enum (am_clip, am_shell, am_cell, am_misl); the Y
# positions (st_stuff.c) deliberately reorder am_cell/am_misl so the numbers
# line up with the plate's BULL/SHEL/RCKT/CELL labels — we just use the exact
# coordinates. These draw with the small yellow digits (shortnum / STYSNUM).
ST_AMMO_X, ST_MAXAMMO_X = 288, 314
ST_AMMO_Y = [173, 179, 191, 185]
ST_MAXAMMO_Y = [173, 179, 191, 185]
ST_AMMO_WIDTH = 3

# Lump names (st_stuff.c ST_loadGraphics).
_PLATE = "STBAR"
_TALLNUM = [f"STTNUM{i}" for i in range(10)]  # tall red digits (ammo/health/armor)
_SHORTNUM = [f"STYSNUM{i}" for i in range(10)]  # small yellow digits (arms/ammo counts)
_TALLPERCENT = "STTPRCNT"
_ARMSBG = "STARMS"
_FACE_NEUTRAL = "STFST01"  # neutral forward face, full health, no pain

# Hardcoded E1M1 pistol-start values.
_E1M1_HEALTH = 100
_E1M1_ARMOR = 0
_E1M1_AMMO = 50  # ready-weapon (pistol) ammo = bullets
# Per-type ammo / max ammo (am_clip, am_shell, am_cell, am_misl). Pistol start
# carries 50 bullets; maxammo {200, 50, 300, 50} (p_inter.c:58, g_game.c:831).
_E1M1_AMMO_BY_TYPE = [50, 0, 0, 0]
_E1M1_MAXAMMO_BY_TYPE = [200, 50, 300, 50]
# weaponowned[1..6] -> ARMS numbers "2".."7". Pistol (weapon 1) only.
_E1M1_WEAPONS_OWNED = [True, False, False, False, False, False]

# The patches the E1M1 draw-list touches (loaded into the bank). The ARMS
# numbers are gray ``STGNUM{n}`` for unowned weapons, yellow ``STYSNUM{n}`` for
# owned; only the ones this state uses are loaded.
_HUD_PATCH_NAMES = (
    [_PLATE, _TALLPERCENT, _ARMSBG, _FACE_NEUTRAL]
    + _TALLNUM
    + _SHORTNUM
    + [f"STGNUM{n}" for n in range(3, 8)]
)

# Sentinel for a transparent cell in a baked patch table (outside 0..255), the
# same convention the weapon bake uses (``WEAPON_TRANSPARENT``). The render loop
# reads the table at the cursor and, on the sentinel, skips the pixel.
HUD_TRANSPARENT = 256.0


@dataclass(frozen=True)
class HudPatch:
    """One status-bar lump as a masked image, native size.

    ``pixels[col][row]`` is a palette index or ``None`` (transparent).
    ``leftoffset``/``topoffset`` are the DOOM picture offsets ``V_DrawPatch``
    subtracts from the draw position.
    """

    name: str
    width: int
    height: int
    leftoffset: int
    topoffset: int
    pixels: list[list[int | None]]


@dataclass(frozen=True)
class DrawCall:
    """One ``V_DrawPatch(x, y, patch)`` in the bar's painter-order sequence.

    ``x``/``y`` are the native-screen arguments DOOM passes (before the offset
    subtraction ``V_DrawPatch`` itself applies).
    """

    patch: str
    x: int
    y: int


def load_hud_patches(
    names=_HUD_PATCH_NAMES, *, wad_path=DOOM1_WAD_PATH
) -> dict[str, HudPatch]:
    """Load each HUD lump as its own masked image (the bank). No compositing."""
    wad = WADReader(wad_path)
    bank: dict[str, HudPatch] = {}
    for name in names:
        img = wad.patch(name)
        bank[name] = HudPatch(
            name=name,
            width=img.width,
            height=img.height,
            leftoffset=img.leftoffset,
            topoffset=img.topoffset,
            pixels=img.pixels,
        )
    return bank


def _draw_num(
    value: int, x: int, y: int, numdigits: int, advance_w: int, digit_lumps
) -> list[DrawCall]:
    """Port ``STlib_drawNum`` (st_lib.c): right-justified digits.

    ``x`` is the right edge; digits are emitted right-to-left, each advancing by
    ``advance_w`` (= the width of digit "0", as DOOM uses ``n->p[0]->width``).
    The special case ``value == 0`` draws a single "0".
    """
    calls: list[DrawCall] = []
    num = value
    cursor = x
    if num == 0:
        # st_lib.c: V_DrawPatch(x - w, y, n->p[0])
        calls.append(DrawCall(digit_lumps[0], cursor - advance_w, y))
        return calls
    remaining = numdigits
    while num and remaining:
        cursor -= advance_w
        calls.append(DrawCall(digit_lumps[num % 10], cursor, y))
        num //= 10
        remaining -= 1
    return calls


def _draw_percent(
    value: int, x: int, y: int, advance_w: int, digit_lumps
) -> list[DrawCall]:
    """Port ``STlib_updatePercent`` (st_lib.c): the percent sign at ``x`` then a
    3-digit right-justified number ending at ``x``."""
    calls = [DrawCall(_TALLPERCENT, x, y)]
    calls += _draw_num(value, x, y, 3, advance_w, digit_lumps)
    return calls


def e1m1_draw_list(bank: dict[str, HudPatch]) -> list[DrawCall]:
    """The ordered ``V_DrawPatch`` sequence for the E1M1 pistol-start bar.

    Painter order matches ``ST_refreshBackground`` (plate) + ``ST_drawWidgets``
    (st_stuff.c:1064-1083): plate, ammo, health, armor, ARMS panel, ARMS
    numbers, face. The tall-digit advance is the width of ``STTNUM0`` (DOOM's
    ``n->p[0]->width``).
    """
    tall_w = bank[_TALLNUM[0]].width
    calls: list[DrawCall] = []

    # 1. The plate (ST_refreshBackground -> V_CopyRect lands it at ST_X, ST_Y).
    calls.append(DrawCall(_PLATE, ST_X, ST_Y))

    # 2. Ready-weapon ammo (a plain number, no percent).
    calls += _draw_num(_E1M1_AMMO, ST_AMMOX, ST_AMMOY, ST_AMMOWIDTH, tall_w, _TALLNUM)

    # 2b. Per-type ammo + max-ammo counts (BULL/SHEL/RCKT/CELL), small yellow
    #     digits (shortnum). ST_drawWidgets draws these interleaved right after
    #     w_ready.
    short_w = bank[_SHORTNUM[0]].width
    for i in range(4):
        calls += _draw_num(
            _E1M1_AMMO_BY_TYPE[i],
            ST_AMMO_X,
            ST_AMMO_Y[i],
            ST_AMMO_WIDTH,
            short_w,
            _SHORTNUM,
        )
        calls += _draw_num(
            _E1M1_MAXAMMO_BY_TYPE[i],
            ST_MAXAMMO_X,
            ST_MAXAMMO_Y[i],
            ST_AMMO_WIDTH,
            short_w,
            _SHORTNUM,
        )

    # 3-4. Health and armor percentages.
    calls += _draw_percent(_E1M1_HEALTH, ST_HEALTHX, ST_HEALTHY, tall_w, _TALLNUM)
    calls += _draw_percent(_E1M1_ARMOR, ST_ARMORX, ST_ARMORY, tall_w, _TALLNUM)

    # 5. ARMS panel background (single player: st_notdeathmatch true).
    calls.append(DrawCall(_ARMSBG, ST_ARMSBGX, ST_ARMSBGY))

    # 6. Weapon-owned numbers 2..7 in a 3x2 grid. Owned -> yellow STYSNUM,
    #    unowned -> gray STGNUM (st_stuff.c arms[i] = {STGNUM(i+2), STYSNUM(i+2)}).
    for i in range(6):
        number = i + 2
        gx = ST_ARMSX + (i % 3) * ST_ARMSXSPACE
        gy = ST_ARMSY + (i // 3) * ST_ARMSYSPACE
        lump = f"STYSNUM{number}" if _E1M1_WEAPONS_OWNED[i] else f"STGNUM{number}"
        calls.append(DrawCall(lump, gx, gy))

    # 7. Face (neutral, full health). No keys on E1M1 start.
    calls.append(DrawCall(_FACE_NEUTRAL, ST_FACESX, ST_FACESY))

    return calls


def _decimate(patch: HudPatch, scale: int) -> HudPatch:
    """Nearest-neighbour decimation by ``scale`` (identity at ``scale == 1``).

    Samples local ``(u*scale, v*scale)`` — the top-left of each block, the same
    convention the low-detail 3D view and the weapon bake use. Offsets divide by
    ``scale`` to keep the patch registered to its (also halved) draw position.
    """
    if scale == 1:
        return patch
    w = patch.width // scale
    h = patch.height // scale
    pixels = [[patch.pixels[u * scale][v * scale] for v in range(h)] for u in range(w)]
    return HudPatch(
        name=patch.name,
        width=w,
        height=h,
        leftoffset=patch.leftoffset // scale,
        topoffset=patch.topoffset // scale,
        pixels=pixels,
    )


def bar_dimensions(scale: int) -> tuple[int, int, int]:
    """Return ``(screen_width, bar_height, hud_top)`` for ``scale``.

    Mirrors ``constants.py``: ``bar_height = 32 * (200//scale) // 200``,
    ``hud_top = (200//scale) - bar_height``.
    """
    screen_w = _FULL_SCREEN_WIDTH // scale
    screen_h = _FULL_SCREEN_HEIGHT // scale
    bar_h = (_FULL_BAR_HEIGHT * screen_h) // _FULL_SCREEN_HEIGHT
    hud_top = screen_h - bar_h
    return screen_w, bar_h, hud_top


def composite_bar(scale: int = 1, *, wad_path=DOOM1_WAD_PATH) -> np.ndarray:
    """Faithful ``V_DrawPatch`` composite of the E1M1 bar — the oracle ground truth.

    Returns a ``(bar_height, screen_width)`` array of palette indices for the bar
    region (screen rows ``[hud_top, screen_height)``). Patches are decimated to
    ``scale`` first, then composited at target resolution in painter order, so
    this matches what the graph emits cell-for-cell. The plate covers the whole
    bar, so the result is fully opaque; ``-1`` marks any cell no patch wrote.
    """
    screen_w, bar_h, hud_top = bar_dimensions(scale)
    bank = load_hud_patches(wad_path=wad_path)
    draw_list = e1m1_draw_list(bank)

    buf = np.full((bar_h, screen_w), -1, dtype=np.int16)
    for call in draw_list:
        patch = _decimate(bank[call.patch], scale)
        # V_DrawPatch: patch-local (0,0) lands at screen (x-left, y-top).
        origin_x = (call.x // scale) - patch.leftoffset
        origin_y = (call.y // scale) - patch.topoffset
        for u in range(patch.width):
            sx = origin_x + u
            if sx < 0 or sx >= screen_w:
                continue
            column = patch.pixels[u]
            for v in range(patch.height):
                color = column[v]
                if color is None:  # transparent post gap: leave what's beneath
                    continue
                row = origin_y + v - hud_top
                if 0 <= row < bar_h:
                    buf[row, sx] = int(color)
    return buf


@dataclass(frozen=True)
class HudBank:
    """The HUD patch bank for the graph: every patch stacked into one table.

    ``table`` is ``(total_rows, max_width)`` of palette indices, ``HUD_TRANSPARENT``
    for transparent or padding cells. Patch ``patch_id`` occupies rows
    ``[base_rows[patch_id], base_rows[patch_id] + heights[patch_id])``; its pixel
    ``(u, v)`` is ``table[base_rows[patch_id] + v, u]``. ``patch_ids`` maps lump
    name -> patch_id. Mirrors the weapon's single sentinel table, but banked by
    ``patch_id`` like the flats so the draw-list can select a patch at runtime.
    """

    table: np.ndarray  # (total_rows, max_width); float32
    base_rows: list[int]
    widths: list[int]
    heights: list[int]
    patch_ids: dict[str, int]


def bake_hud_bank(scale: int = 1, *, wad_path=DOOM1_WAD_PATH) -> HudBank:
    """Stack every E1M1 HUD patch (decimated to ``scale``) into one sentinel table.

    The patch_id order is :data:`_HUD_PATCH_NAMES`; the draw-list baked by
    :func:`bake_hud_draw_list` (consumed by the statusbar renderer) keys
    patches by the same ``patch_ids`` map so the graph and the bake agree.
    """
    bank = load_hud_patches(wad_path=wad_path)
    names = list(_HUD_PATCH_NAMES)
    decimated = [_decimate(bank[name], scale) for name in names]
    max_width = max(p.width for p in decimated)
    total_rows = sum(p.height for p in decimated)

    table = np.full((total_rows, max_width), HUD_TRANSPARENT, dtype=np.float32)
    base_rows: list[int] = []
    widths: list[int] = []
    heights: list[int] = []
    row = 0
    for patch in decimated:
        base_rows.append(row)
        widths.append(patch.width)
        heights.append(patch.height)
        for u in range(patch.width):
            column = patch.pixels[u]
            for v in range(patch.height):
                color = column[v]
                if color is not None:
                    table[row + v, u] = float(color)
        row += patch.height
    return HudBank(
        table=table,
        base_rows=base_rows,
        widths=widths,
        heights=heights,
        patch_ids={name: i for i, name in enumerate(names)},
    )


@dataclass(frozen=True)
class HudDrawList:
    """The draw-list resolved for the graph spine: one entry per ``V_DrawPatch``.

    Item ``i`` draws patch ``patch_id[i]`` with its local (0,0) at screen
    ``(origin_x[i], origin_y[i])`` (the DOOM draw position minus the patch
    offset), covering ``width[i] x height[i]`` cells. The spine walks items in
    order; ``patch_id``/``origin``/``width``/``height`` are indexed by the item
    counter, exactly as the weapon indexes its (single) bbox bounds. Painter
    order = list order, so last-write-wins composites widgets over the plate.
    """

    patch_id: list[int]
    origin_x: list[int]
    origin_y: list[int]
    width: list[int]
    height: list[int]
    n_items: int


def bake_hud_draw_list(scale: int = 1, *, wad_path=DOOM1_WAD_PATH) -> HudDrawList:
    """Resolve the E1M1 draw-list to per-item graph tables at ``scale``.

    The same origin math as :func:`composite_bar` (decimate the patch, then
    ``origin = draw_pos // scale - decimated_offset``) and the same ``patch_id``
    map as :func:`bake_hud_bank`, so the graph spine, the reference blit, and the
    patch bank all agree cell-for-cell.
    """
    bank = load_hud_patches(wad_path=wad_path)
    patch_ids = {name: i for i, name in enumerate(_HUD_PATCH_NAMES)}
    draw_list = e1m1_draw_list(bank)
    patch_id, origin_x, origin_y, width, height = [], [], [], [], []
    for call in draw_list:
        patch = _decimate(bank[call.patch], scale)
        patch_id.append(patch_ids[call.patch])
        origin_x.append((call.x // scale) - patch.leftoffset)
        origin_y.append((call.y // scale) - patch.topoffset)
        width.append(patch.width)
        height.append(patch.height)
    return HudDrawList(
        patch_id=patch_id,
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
        n_items=len(draw_list),
    )


def bar_to_rgb(indices: np.ndarray, *, wad_path=DOOM1_WAD_PATH) -> np.ndarray:
    """Map a palette-index bar (from :func:`composite_bar`) to an ``(H, W, 3)``
    uint8 RGB image via PLAYPAL, for PNG eyeballing. ``-1`` (unwritten) -> black.
    """
    palette = WADReader(wad_path).palette()
    h, w = indices.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            idx = int(indices[y, x])
            if idx >= 0:
                rgb[y, x] = palette[idx]
    return rgb
