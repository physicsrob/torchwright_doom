"""Reference renderer transcribed line-for-line from DOOM (1993)'s C source.

This module exists *next to* ``render.py`` (the ray-cast reference
renderer); it does not replace it.  The goal is faithfulness to the
original DOOM rendering pipeline — same control flow, same function
names, same module-level globals — so that this Python next to
``orig-doom-renderer/r_*.c`` reads as a transcription, not a
reinterpretation.

Scope: walls only.  No flats (visplanes), no sprites, no masked
midtextures, no diminishing lighting, no sky.

Deliberate deviations from C (commented inline at use sites):

  * Fixed-point arithmetic dropped.  The ``fixed_t`` math in C is a
    hardware concern, not algorithmic; preserving it in Python adds
    noise without adding clarity.  ``FixedMul/FixedDiv`` become plain
    multiply/divide; ``>>FRACBITS`` and ``HEIGHTBITS`` shifts are
    dropped wherever they were just packing/unpacking sub-pixel
    precision.

  * ``finesine`` / ``finecosine`` / ``finetangent`` precomputed from
    ``math.sin/cos/tan`` instead of bundled tables.  Same 8192-entry
    shape, same indexing semantics.

  * Texture lookup keyed by name (``textures: dict[str, np.ndarray]``)
    instead of ``texturetranslation[]``.

  * The framebuffer is pre-filled with ceiling/floor colours before
    walls render, since we omit visplanes.  C never clears.

Module-level globals are the actual transcription convention: this
file's ``viewx`` *is* DOOM's ``viewx``, not ``state.viewx``.  Each
frame's setup functions (``R_SetupFrame``, ``R_ClearClipSegs``,
``R_ClearDrawSegs``, ``R_ClearPlanes``) reset them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from torchwright_doom.doom.wad import (
    SUBSECTOR_FLAG,
    BspNode,
    Linedef,
    MapData,
    Seg,
    Sector,
    Sidedef,
    Subsector,
    Vertex,
)
from torchwright_doom.reference_renderer.types import RenderConfig, Segment


# =====================================================================
# === r_main.c =========================================================
# =====================================================================


# --- BAM (Binary Angular Measurement) angles --------------------------
# Angles are 32-bit unsigned integers covering [0, 2pi).  We store them
# as Python ints and mask to 32 bits at construction (``& ANGMASK``).
ANG90 = 0x40000000
ANG180 = 0x80000000
ANG270 = 0xC0000000
ANGMASK = 0xFFFFFFFF

# 8192-entry finesine / finetangent tables, indexed by `bam >> ANGLETOFINESHIFT`.
FINEANGLES = 8192
ANGLETOFINESHIFT = 32 - 13  # 19; FINEANGLES = 1 << 13

# r_bsp.c sizing
MAXSEGS = 32
MAXDRAWSEGS = 256

# Linedef flags from doomdef.h (only the two we read).
ML_DONTPEGTOP = 0x0008
ML_DONTPEGBOTTOM = 0x0010

# r_defs.h bbox indices.
BOXTOP = 0
BOXBOTTOM = 1
BOXLEFT = 2
BOXRIGHT = 3

# NF_SUBSECTOR — the high bit of a BSP node child index marks a leaf.
NF_SUBSECTOR = SUBSECTOR_FLAG

# DOOM's middle-of-screen "centery" gets `viewz - z` scaled by rw_scale
# to project a world-z onto a screen row.  Sentinels for "no scale yet".
_SCALE_MAX = 64.0       # 64 * FRACUNIT in C; pixels-per-world-unit cap.
_SCALE_MIN = 1.0 / 256  # 256 in fixed_t; keeps division well-conditioned.


# --- finesine / finecosine / finetangent ------------------------------
# C builds these in tables.c.  Same shape (5*FINEANGLES/4 for finesine,
# FINEANGLES/2 for finetangent) and same offset semantics.
_finesine_full: np.ndarray = np.array(
    [math.sin((i + 0.5) * 2.0 * math.pi / FINEANGLES) for i in range(5 * FINEANGLES // 4)],
    dtype=np.float64,
)
finesine: np.ndarray = _finesine_full
# DOOM: ``fixed_t* finecosine = &finesine[FINEANGLES/4];`` — same array,
# offset by 90 degrees so finecosine[i] == finesine[i + 2048].
finecosine: np.ndarray = _finesine_full[FINEANGLES // 4:]
finetangent: np.ndarray = np.array(
    [math.tan((i - FINEANGLES // 4 + 0.5) * 2.0 * math.pi / FINEANGLES) for i in range(FINEANGLES // 2)],
    dtype=np.float64,
)


# --- module-level state (reset every frame) ---------------------------

viewx: float = 0.0
viewy: float = 0.0
viewz: float = 0.0
viewangle: int = 0
viewsin: float = 0.0
viewcos: float = 1.0

viewwidth: int = 0
viewheight: int = 0
centerx: int = 0
centery: int = 0
projection: float = 0.0
clipangle: int = 0

# viewangletox[bam>>ANGLETOFINESHIFT] = column for that view-relative angle.
viewangletox: np.ndarray = np.zeros(FINEANGLES // 2, dtype=np.int64)
# xtoviewangle[col] = smallest BAM angle (relative to viewangle) mapping to that column.
xtoviewangle: np.ndarray = np.zeros(0, dtype=np.int64)

framecount: int = 0
validcount: int = 0

_mapdata: Optional[MapData] = None
_textures: Dict[str, np.ndarray] = {}
_config: Optional[RenderConfig] = None
_screen: Optional[np.ndarray] = None


def R_PointOnSide(x: float, y: float, node: BspNode) -> int:
    """Return 0 (front) or 1 (back) for a point against a partition line.

    Mirrors r_main.c.  The XOR sign-bit fast path in C is dropped —
    that's a fixed-point microoptimization with no float analogue.
    """
    if node.dx == 0:
        if x <= node.px:
            return 1 if node.dy > 0 else 0
        return 1 if node.dy < 0 else 0
    if node.dy == 0:
        if y <= node.py:
            return 1 if node.dx < 0 else 0
        return 1 if node.dx > 0 else 0
    dx = x - node.px
    dy = y - node.py
    left = node.dy * dx
    right = dy * node.dx
    if right < left:
        return 0  # front
    return 1  # back


def R_PointToAngle(x: float, y: float) -> int:
    """BAM angle from (viewx,viewy) to (x,y).

    C uses an 8-octant tantoangle[] lookup; floats let us use atan2
    directly.  Result is masked to 32-bit unsigned.
    """
    dx = x - viewx
    dy = y - viewy
    if dx == 0.0 and dy == 0.0:
        return 0
    angle_rad = math.atan2(dy, dx)
    if angle_rad < 0:
        angle_rad += 2.0 * math.pi
    return int(angle_rad * (2.0**32) / (2.0 * math.pi)) & ANGMASK


def R_PointToDist(x: float, y: float) -> float:
    """Euclidean distance from view to (x,y)."""
    return math.hypot(x - viewx, y - viewy)


# rw_distance / rw_normalangle live in r_segs.c module state; declared
# here so R_ScaleFromGlobalAngle can read them as DOOM does.  Their
# definitions follow in the r_segs.c section; the forward reference is
# resolved by Python's late binding inside the function body.

def R_ScaleFromGlobalAngle(visangle: int) -> float:
    """Pixels-per-world-unit scale for a column whose ray has BAM angle visangle.

    Formula (after dropping fixed-point):
        scale = projection * sin(angleb) / (rw_distance * sin(anglea))
    where
        anglea = pi/2 + (visangle - viewangle)   — fishbowl correction
        angleb = pi/2 + (visangle - rw_normalangle)

    sin(pi/2 + theta) == cos(theta); the C names the formulation by the
    sine table it indexes into, so we keep that.
    """
    anglea = (ANG90 + (visangle - viewangle)) & ANGMASK
    angleb = (ANG90 + (visangle - rw_normalangle)) & ANGMASK
    sinea = finesine[anglea >> ANGLETOFINESHIFT]
    sineb = finesine[angleb >> ANGLETOFINESHIFT]
    den = rw_distance * sinea
    num = projection * sineb
    if den <= num / _SCALE_MAX:
        return _SCALE_MAX
    scale = num / den
    if scale > _SCALE_MAX:
        scale = _SCALE_MAX
    elif scale < _SCALE_MIN:
        scale = _SCALE_MIN
    return scale


def R_InitTextureMapping() -> None:
    """Build viewangletox[] / xtoviewangle[] for the current view size & FOV.

    C: r_main.c.  ``FIELDOFVIEW`` in C is hardcoded at 2048
    finetangent units (== 90 deg HFOV when FINEANGLES == 8192).  We
    derive it from ``RenderConfig.fov_columns`` which uses 256-unit
    BAM (full circle = 256).  fov_columns * FINEANGLES / 256 gives
    the equivalent finetangent-unit width.
    """
    global viewangletox, xtoviewangle, clipangle

    assert _config is not None
    field_of_view = _config.fov_columns * FINEANGLES // 256

    # focallength = centerx / tan(FOV/2)
    focallength = centerx / finetangent[FINEANGLES // 4 + field_of_view // 2]

    viewangletox = np.zeros(FINEANGLES // 2, dtype=np.int64)
    for i in range(FINEANGLES // 2):
        ft = finetangent[i]
        if ft > 2.0:
            t = -1
        elif ft < -2.0:
            t = viewwidth + 1
        else:
            t = math.ceil(centerx - ft * focallength)
            if t < -1:
                t = -1
            elif t > viewwidth + 1:
                t = viewwidth + 1
        viewangletox[i] = t

    # Smallest view angle that maps to each column.
    xtoviewangle = np.zeros(viewwidth + 1, dtype=np.int64)
    for x in range(viewwidth + 1):
        i = 0
        while viewangletox[i] > x:
            i += 1
        xtoviewangle[x] = ((i << ANGLETOFINESHIFT) - ANG90) & ANGMASK

    # Take out the fencepost cases (-1 / viewwidth+1) so callers see
    # in-range values.  C does this *after* xtoviewangle is built so
    # that the search above can detect the off-screen sentinels.
    for i in range(FINEANGLES // 2):
        if viewangletox[i] == -1:
            viewangletox[i] = 0
        elif viewangletox[i] == viewwidth + 1:
            viewangletox[i] = viewwidth

    clipangle = int(xtoviewangle[0])


def R_SetupFrame(
    player_x: float,
    player_y: float,
    player_z: float,
    player_angle: int,
) -> None:
    """Cache per-frame view state.  C: r_main.c."""
    global viewx, viewy, viewz, viewangle, viewsin, viewcos
    global framecount, validcount
    viewx = player_x
    viewy = player_y
    viewz = player_z
    viewangle = player_angle & ANGMASK
    viewsin = float(finesine[viewangle >> ANGLETOFINESHIFT])
    viewcos = float(finecosine[viewangle >> ANGLETOFINESHIFT])
    framecount += 1
    validcount += 1


def R_RenderPlayerView(
    player_x: float,
    player_y: float,
    player_z: float,
    player_angle: int,
    mapdata: MapData,
    config: RenderConfig,
    textures: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Top-level entry — render one frame.

    Args:
        player_x, player_y, player_z: View origin in world units.
        player_angle: BAM (0..2^32).  Callers using DOOM 0..255 angles
            should shift left by 24 before calling.
        mapdata: Parsed map (vertices, segs, subsectors, BSP).
        config: Screen size, FOV, ceiling/floor fill colours.
        textures: name -> (W,H,3) float64 atlas; missing names paint
            the seg's fallback colour (sector hash).

    Returns:
        (H, W, 3) float64 framebuffer.
    """
    global _mapdata, _textures, _config, _screen

    _mapdata = mapdata
    _textures = textures or {}
    _config = config
    _init_view_size(config.screen_width, config.screen_height)

    R_SetupFrame(player_x, player_y, player_z, player_angle)
    R_ClearClipSegs()
    R_ClearDrawSegs()
    R_ClearPlanes()

    # Pre-fill ceiling/floor since we omit visplanes (deviation from C —
    # noted at file top).  R_RenderSegLoop only paints wall pixels; the
    # background must be present beforehand.
    centery_local = viewheight // 2
    _screen[:centery_local] = config.ceiling_color
    _screen[centery_local:] = config.floor_color

    # Root BSP node is the last entry; pass its index (or -1 if no nodes).
    R_RenderBSPNode(len(mapdata.nodes) - 1)

    # R_DrawPlanes / R_DrawMasked omitted (out of scope).
    # Return a copy: ``_screen`` is a module-level buffer that the next
    # call will overwrite.  Callers that want to keep multiple frames
    # would otherwise see them all alias to the latest render.
    return _screen.copy()


def _init_view_size(width: int, height: int) -> None:
    """One-time per-frame init: viewwidth/height, projection, tables.

    Equivalent to the bookkeeping in C's R_ExecuteSetViewSize that we
    care about (centerx, centery, projection, viewangletox).
    """
    global viewwidth, viewheight, centerx, centery, projection
    global _screen, floorclip, ceilingclip, drawsegs

    viewwidth = width
    viewheight = height
    centerx = width // 2
    centery = height // 2
    projection = float(centerx)

    if _screen is None or _screen.shape != (height, width, 3):
        _screen = np.empty((height, width, 3), dtype=np.float64)
    if floorclip.shape[0] != width:
        floorclip = np.empty(width, dtype=np.int64)
        ceilingclip = np.empty(width, dtype=np.int64)
    if not drawsegs:
        drawsegs = [Drawseg() for _ in range(MAXDRAWSEGS)]

    # colfunc = R_DrawColumn (default opaque-column variant).
    global colfunc
    colfunc = R_DrawColumn

    R_InitTextureMapping()


# =====================================================================
# === r_bsp.c ==========================================================
# =====================================================================


@dataclass
class Cliprange:
    """One run of solid (occluded) screen columns.  C: cliprange_t."""

    first: int = 0
    last: int = 0


@dataclass
class Drawseg:
    """Per-emitted-seg log entry.  C: drawseg_t.

    We only fill enough fields to make the masked-pass code shape
    correct.  Without sprites/masked we never read these back, but
    populating x1/x2/scale1/scale2 lets a curious caller inspect the
    front-to-back order the BSP produced.
    """

    curline: Optional[Seg] = None
    x1: int = 0
    x2: int = 0
    scale1: float = 0.0
    scale2: float = 0.0
    scalestep: float = 0.0
    silhouette: int = 0
    bsilheight: float = 0.0
    tsilheight: float = 0.0


# solidsegs[] sized MAXSEGS+2 so the leftright sentinels never collide
# with real entries even on a fully-saturated screen.
solidsegs: List[Cliprange] = [Cliprange() for _ in range(MAXSEGS + 2)]
newend: int = 0  # one past the last valid solidsegs entry

drawsegs: List[Drawseg] = []
ds_p: int = 0  # one past the last filled drawsegs entry


def R_ClearDrawSegs() -> None:
    """C: r_bsp.c."""
    global ds_p
    ds_p = 0


def R_ClearClipSegs() -> None:
    """Initialize solidsegs to the off-screen sentinels.  C: r_bsp.c."""
    global newend
    solidsegs[0].first = -0x7FFFFFFF
    solidsegs[0].last = -1
    solidsegs[1].first = viewwidth
    solidsegs[1].last = 0x7FFFFFFF
    newend = 2


def R_ClipSolidWallSegment(first: int, last: int) -> None:
    """Insert a solid-wall column range into solidsegs[], drawing fragments.

    Faithful transcription of r_bsp.c.  ``next``/``start`` in C are
    pointers into the solidsegs[] array; here they're indices.
    """
    global newend

    # Find the first range that touches [first, last] (adjacent pixels touch).
    start = 0
    while solidsegs[start].last < first - 1:
        start += 1

    if first < solidsegs[start].first:
        if last < solidsegs[start].first - 1:
            # Post is entirely visible (above start) — insert a new clippost.
            R_StoreWallRange(first, last)
            nxt = newend
            newend += 1
            while nxt != start:
                solidsegs[nxt].first = solidsegs[nxt - 1].first
                solidsegs[nxt].last = solidsegs[nxt - 1].last
                nxt -= 1
            solidsegs[nxt].first = first
            solidsegs[nxt].last = last
            return
        # Fragment above *start.
        R_StoreWallRange(first, solidsegs[start].first - 1)
        solidsegs[start].first = first

    if last <= solidsegs[start].last:
        return

    nxt = start
    crunch = False
    while last >= solidsegs[nxt + 1].first - 1:
        # Fragment between two posts.
        R_StoreWallRange(solidsegs[nxt].last + 1, solidsegs[nxt + 1].first - 1)
        nxt += 1
        if last <= solidsegs[nxt].last:
            solidsegs[start].last = solidsegs[nxt].last
            crunch = True
            break

    if not crunch:
        # Fragment after *next.
        R_StoreWallRange(solidsegs[nxt].last + 1, last)
        solidsegs[start].last = last

    # crunch: remove start+1..next from the clip list.
    if nxt == start:
        return
    # C does:  while (next++ != newend) *++start = *next;
    # Loop runs while the *pre-increment* value of next != newend.  Once
    # nxt advances past newend, stop.  We mirror that by checking before
    # advancing.
    while nxt != newend:
        nxt += 1
        start += 1
        solidsegs[start].first = solidsegs[nxt].first
        solidsegs[start].last = solidsegs[nxt].last
    newend = start + 1


def R_ClipPassWallSegment(first: int, last: int) -> None:
    """Draw a portal's visible fragments without updating solidsegs[]."""
    start = 0
    while solidsegs[start].last < first - 1:
        start += 1

    if first < solidsegs[start].first:
        if last < solidsegs[start].first - 1:
            R_StoreWallRange(first, last)
            return
        R_StoreWallRange(first, solidsegs[start].first - 1)

    if last <= solidsegs[start].last:
        return

    while last >= solidsegs[start + 1].first - 1:
        R_StoreWallRange(solidsegs[start].last + 1, solidsegs[start + 1].first - 1)
        start += 1
        if last <= solidsegs[start].last:
            return

    R_StoreWallRange(solidsegs[start].last + 1, last)


def R_AddLine(seg_index: int) -> None:
    """Project, clip, and dispatch one seg.  C: r_bsp.c.

    ``seg_index`` is the index into ``mapdata.segs``; C passes
    ``seg_t*``.
    """
    global curline, sidedef, linedef, frontsector, backsector
    global rw_angle1

    assert _mapdata is not None
    seg = _mapdata.segs[seg_index]
    curline = seg

    v1 = _mapdata.vertices[seg.v1]
    v2 = _mapdata.vertices[seg.v2]

    # Resolve sidedef / linedef / sectors for this seg (C: stored as
    # pointers on seg_t; we walk the WAD-shaped data).
    ld = _mapdata.linedefs[seg.linedef]
    linedef = ld
    front_sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
    back_sd_idx = ld.back_sidedef if seg.side == 0 else ld.front_sidedef
    if front_sd_idx < 0:
        return
    sd_front = _mapdata.sidedefs[front_sd_idx]
    sidedef = sd_front
    frontsector = _mapdata.sectors[sd_front.sector]
    if 0 <= back_sd_idx < len(_mapdata.sidedefs):
        sd_back = _mapdata.sidedefs[back_sd_idx]
        backsector = _mapdata.sectors[sd_back.sector]
    else:
        backsector = None

    angle1 = R_PointToAngle(float(v1.x), float(v1.y))
    angle2 = R_PointToAngle(float(v2.x), float(v2.y))

    span = (angle1 - angle2) & ANGMASK

    # Backface cull (span >= ANG180 means v2 is on the wrong side).
    if span >= ANG180:
        return

    rw_angle1 = angle1
    angle1 = (angle1 - viewangle) & ANGMASK
    angle2 = (angle2 - viewangle) & ANGMASK

    tspan = (angle1 + clipangle) & ANGMASK
    if tspan > 2 * clipangle:
        tspan = (tspan - 2 * clipangle) & ANGMASK
        if tspan >= span:
            return  # totally off the left edge
        angle1 = clipangle
    tspan = (clipangle - angle2) & ANGMASK
    if tspan > 2 * clipangle:
        tspan = (tspan - 2 * clipangle) & ANGMASK
        if tspan >= span:
            return  # totally off the right edge
        angle2 = (-clipangle) & ANGMASK

    angle1 = ((angle1 + ANG90) & ANGMASK) >> ANGLETOFINESHIFT
    angle2 = ((angle2 + ANG90) & ANGMASK) >> ANGLETOFINESHIFT
    x1 = int(viewangletox[angle1])
    x2 = int(viewangletox[angle2])

    if x1 == x2:
        return  # doesn't cross a pixel

    # Single-sided ⇒ solid wall.
    if backsector is None:
        R_ClipSolidWallSegment(x1, x2 - 1)
        return

    # Closed door (back fully blocks sightline).
    if (
        backsector.ceiling_h <= frontsector.floor_h
        or backsector.floor_h >= frontsector.ceiling_h
    ):
        R_ClipSolidWallSegment(x1, x2 - 1)
        return

    # Window: at least one of ceiling/floor differs.
    if (
        backsector.ceiling_h != frontsector.ceiling_h
        or backsector.floor_h != frontsector.floor_h
    ):
        R_ClipPassWallSegment(x1, x2 - 1)
        return

    # Identical floor/ceiling on both sides + no middle texture: a
    # trigger linedef.  Skip — it would draw nothing anyway.
    if sidedef.middle == "-" or sidedef.middle == "":
        return

    R_ClipPassWallSegment(x1, x2 - 1)


# C: int checkcoord[12][4].  Index by (boxy<<2)+boxx; entry [0..3]
# pick which of {top,bottom,left,right} go into x1,y1,x2,y2.
_CHECKCOORD = (
    (3, 0, 2, 1),
    (3, 0, 2, 0),
    (3, 1, 2, 0),
    (0, 0, 0, 0),  # unused
    (2, 0, 2, 1),
    (0, 0, 0, 0),  # boxpos == 5: viewer inside box, always visible
    (3, 1, 3, 0),
    (0, 0, 0, 0),  # unused
    (2, 0, 3, 1),
    (2, 1, 3, 1),
    (2, 1, 3, 0),
)


def R_CheckBBox(bbox: tuple) -> bool:
    """Return True if any part of bbox might be visible.

    bbox is (top, bottom, left, right) — DOOM's BOXTOP/BOXBOTTOM/
    BOXLEFT/BOXRIGHT order, matching the parser in wad.py.
    """
    box_top, box_bot, box_left, box_right = bbox

    if viewx <= box_left:
        boxx = 0
    elif viewx < box_right:
        boxx = 1
    else:
        boxx = 2

    if viewy >= box_top:
        boxy = 0
    elif viewy > box_bot:
        boxy = 1
    else:
        boxy = 2

    boxpos = (boxy << 2) + boxx
    if boxpos == 5:
        return True  # viewer is inside the bbox

    cc = _CHECKCOORD[boxpos]

    def _coord(idx: int) -> float:
        return (box_top, box_bot, box_left, box_right)[idx]

    x1 = _coord(cc[0])
    y1 = _coord(cc[1])
    x2 = _coord(cc[2])
    y2 = _coord(cc[3])

    angle1 = (R_PointToAngle(x1, y1) - viewangle) & ANGMASK
    angle2 = (R_PointToAngle(x2, y2) - viewangle) & ANGMASK
    span = (angle1 - angle2) & ANGMASK

    if span >= ANG180:
        return True  # sitting on a line

    tspan = (angle1 + clipangle) & ANGMASK
    if tspan > 2 * clipangle:
        tspan = (tspan - 2 * clipangle) & ANGMASK
        if tspan >= span:
            return False
        angle1 = clipangle
    tspan = (clipangle - angle2) & ANGMASK
    if tspan > 2 * clipangle:
        tspan = (tspan - 2 * clipangle) & ANGMASK
        if tspan >= span:
            return False
        angle2 = (-clipangle) & ANGMASK

    angle1 = ((angle1 + ANG90) & ANGMASK) >> ANGLETOFINESHIFT
    angle2 = ((angle2 + ANG90) & ANGMASK) >> ANGLETOFINESHIFT
    sx1 = int(viewangletox[angle1])
    sx2 = int(viewangletox[angle2])

    if sx1 == sx2:
        return False
    sx2 -= 1

    start = 0
    while solidsegs[start].last < sx2:
        start += 1

    if sx1 >= solidsegs[start].first and sx2 <= solidsegs[start].last:
        return False  # this clippost already contains the new span
    return True


def R_Subsector(num: int) -> None:
    """Visit one subsector — emit each of its segs.  C: r_bsp.c."""
    global frontsector

    assert _mapdata is not None
    sub = _mapdata.subsectors[num]
    # Find the front sector for this subsector.  The C version stores
    # subsector_t.sector directly; we look it up from the first seg's
    # sidedef (any seg in a subsector references the same sector).
    if sub.seg_count > 0:
        first_seg = _mapdata.segs[sub.first_seg]
        ld = _mapdata.linedefs[first_seg.linedef]
        sd_idx = ld.front_sidedef if first_seg.side == 0 else ld.back_sidedef
        if 0 <= sd_idx < len(_mapdata.sidedefs):
            frontsector = _mapdata.sectors[_mapdata.sidedefs[sd_idx].sector]
        else:
            frontsector = None
    else:
        frontsector = None

    # R_AddSprites omitted (out of scope).
    for i in range(sub.seg_count):
        R_AddLine(sub.first_seg + i)


def R_RenderBSPNode(bspnum: int) -> None:
    """Recursive front-to-back BSP traversal.  C: r_bsp.c."""
    assert _mapdata is not None

    if bspnum & NF_SUBSECTOR:
        if bspnum == -1:
            R_Subsector(0)
        else:
            R_Subsector(bspnum & ~NF_SUBSECTOR)
        return

    bsp = _mapdata.nodes[bspnum]
    side = R_PointOnSide(viewx, viewy, bsp)

    front_child = bsp.front_child if side == 0 else bsp.back_child
    back_child = bsp.back_child if side == 0 else bsp.front_child
    front_bbox = bsp.front_bbox if side == 0 else bsp.back_bbox
    back_bbox = bsp.back_bbox if side == 0 else bsp.front_bbox
    del front_bbox  # only the back's bbox is checked

    R_RenderBSPNode(front_child)
    if R_CheckBBox(back_bbox):
        R_RenderBSPNode(back_child)


# =====================================================================
# === r_plane.c (init only — no visplanes) =============================
# =====================================================================


floorclip: np.ndarray = np.zeros(0, dtype=np.int64)
ceilingclip: np.ndarray = np.zeros(0, dtype=np.int64)


def R_ClearPlanes() -> None:
    """Reset per-column clip arrays.  C: r_plane.c (visplane init dropped)."""
    floorclip[:] = viewheight
    ceilingclip[:] = -1


# =====================================================================
# === r_segs.c =========================================================
# =====================================================================


# Per-seg working state.  Re-set every R_StoreWallRange call.
curline: Optional[Seg] = None
sidedef: Optional[Sidedef] = None
linedef: Optional[Linedef] = None
frontsector: Optional[Sector] = None
backsector: Optional[Sector] = None

rw_x: int = 0
rw_stopx: int = 0
rw_distance: float = 0.0
rw_scale: float = 0.0
rw_scalestep: float = 0.0
rw_centerangle: int = 0
rw_offset: float = 0.0
rw_normalangle: int = 0
rw_angle1: int = 0
rw_midtexturemid: float = 0.0
rw_toptexturemid: float = 0.0
rw_bottomtexturemid: float = 0.0

# In C these are texture indices; we keep texture *names* (or "" for
# "no texture", which mirrors C's ``midtexture == 0`` test).
midtexture: str = ""
toptexture: str = ""
bottomtexture: str = ""

segtextured: bool = False
markfloor: bool = False
markceiling: bool = False

worldtop: float = 0.0
worldbottom: float = 0.0
worldhigh: float = 0.0
worldlow: float = 0.0

topfrac: float = 0.0
topstep: float = 0.0
bottomfrac: float = 0.0
bottomstep: float = 0.0
pixhigh: float = 0.0
pixhighstep: float = 0.0
pixlow: float = 0.0
pixlowstep: float = 0.0


def _seg_color() -> tuple:
    """Fallback colour when a wall has no usable texture (sector hash)."""
    from torchwright_doom.doom.wad import sector_color

    if frontsector is None or sidedef is None:
        return (0.5, 0.5, 0.5)
    return sector_color(sidedef.sector)


def _resolve_texture(name: str) -> Optional[np.ndarray]:
    """Look up a wall texture by name.  ``"-"``/empty ⇒ None."""
    if not name or name == "-":
        return None
    return _textures.get(name)


def R_StoreWallRange(start: int, stop: int) -> None:
    """Drive R_RenderSegLoop for columns [start, stop] of the current seg.

    C: r_segs.c.  All the math has been ported to floats; the structure
    matches the C 1:1 (texture-mid calc, scale endpoints, world{top,
    bottom,high,low} stepping setup).
    """
    global ds_p
    global rw_x, rw_stopx, rw_normalangle, rw_distance, rw_scale, rw_scalestep
    global rw_centerangle, rw_offset
    global rw_midtexturemid, rw_toptexturemid, rw_bottomtexturemid
    global midtexture, toptexture, bottomtexture
    global segtextured, markfloor, markceiling
    global worldtop, worldbottom, worldhigh, worldlow
    global topfrac, topstep, bottomfrac, bottomstep
    global pixhigh, pixhighstep, pixlow, pixlowstep

    if ds_p == MAXDRAWSEGS:
        return

    assert curline is not None and frontsector is not None and sidedef is not None and linedef is not None

    v1 = _mapdata.vertices[curline.v1]

    # rw_distance: perpendicular distance from view to the seg's line.
    rw_normalangle = (curline.angle + ANG90) & ANGMASK
    # offsetangle = abs(rw_normalangle - rw_angle1).  C does this with
    # signed-int abs of an unsigned-wrapped subtraction; the effect is
    # "shortest arc magnitude" in [0, ANG180].
    diff = (rw_normalangle - rw_angle1) & ANGMASK
    if diff > ANG180:
        diff = (-diff) & ANGMASK  # negate mod 2^32 ⇒ value in [0, ANG180]
    offsetangle = diff
    if offsetangle > ANG90:
        offsetangle = ANG90
    distangle = ANG90 - offsetangle
    hyp = R_PointToDist(float(v1.x), float(v1.y))
    sineval = float(finesine[distangle >> ANGLETOFINESHIFT])
    rw_distance = hyp * sineval

    ds = drawsegs[ds_p]
    ds.x1 = rw_x = start
    ds.x2 = stop
    ds.curline = curline
    rw_stopx = stop + 1

    # Scale at column endpoints.
    visangle_start = (viewangle + int(xtoviewangle[start])) & ANGMASK
    ds.scale1 = rw_scale = R_ScaleFromGlobalAngle(visangle_start)
    if stop > start:
        visangle_stop = (viewangle + int(xtoviewangle[stop])) & ANGMASK
        ds.scale2 = R_ScaleFromGlobalAngle(visangle_stop)
        ds.scalestep = rw_scalestep = (ds.scale2 - rw_scale) / (stop - start)
    else:
        ds.scale2 = ds.scale1
        ds.scalestep = rw_scalestep = 0.0

    # World-relative wall heights.
    worldtop = frontsector.ceiling_h - viewz
    worldbottom = frontsector.floor_h - viewz

    midtexture = ""
    toptexture = ""
    bottomtexture = ""

    if backsector is None:
        # Single-sided wall.
        midtexture = sidedef.middle
        markfloor = markceiling = True
        if linedef.flags & ML_DONTPEGBOTTOM:
            tex = _resolve_texture(midtexture)
            tex_h = tex.shape[1] if tex is not None else (frontsector.ceiling_h - frontsector.floor_h)
            vtop = frontsector.floor_h + tex_h
            rw_midtexturemid = vtop - viewz
        else:
            rw_midtexturemid = worldtop
        rw_midtexturemid += sidedef.y_offset

        ds.silhouette = 3  # SIL_BOTH
        ds.bsilheight = float("inf")
        ds.tsilheight = -float("inf")
    else:
        ds.silhouette = 0
        if frontsector.floor_h > backsector.floor_h:
            ds.silhouette = 1  # SIL_BOTTOM
            ds.bsilheight = float(frontsector.floor_h)
        elif backsector.floor_h > viewz:
            ds.silhouette = 1
            ds.bsilheight = float("inf")

        if frontsector.ceiling_h < backsector.ceiling_h:
            ds.silhouette |= 2  # SIL_TOP
            ds.tsilheight = float(frontsector.ceiling_h)
        elif backsector.ceiling_h < viewz:
            ds.silhouette |= 2
            ds.tsilheight = -float("inf")

        if backsector.ceiling_h <= frontsector.floor_h:
            ds.bsilheight = float("inf")
            ds.silhouette |= 1
        if backsector.floor_h >= frontsector.ceiling_h:
            ds.tsilheight = -float("inf")
            ds.silhouette |= 2

        worldhigh = backsector.ceiling_h - viewz
        worldlow = backsector.floor_h - viewz

        # markfloor/markceiling: in C these gate visplane writes.  We
        # have no visplanes, so they're effectively unused — but track
        # them for parity (and to keep the structure identical).
        markfloor = (
            worldlow != worldbottom
            or backsector.floor_tex != frontsector.floor_tex
            or backsector.light != frontsector.light
        )
        markceiling = (
            worldhigh != worldtop
            or backsector.ceiling_tex != frontsector.ceiling_tex
            or backsector.light != frontsector.light
        )
        if (
            backsector.ceiling_h <= frontsector.floor_h
            or backsector.floor_h >= frontsector.ceiling_h
        ):
            markceiling = markfloor = True

        if worldhigh < worldtop:
            toptexture = sidedef.upper
            if linedef.flags & ML_DONTPEGTOP:
                rw_toptexturemid = worldtop
            else:
                tex = _resolve_texture(toptexture)
                tex_h = tex.shape[1] if tex is not None else (frontsector.ceiling_h - backsector.ceiling_h)
                vtop = backsector.ceiling_h + tex_h
                rw_toptexturemid = vtop - viewz

        if worldlow > worldbottom:
            bottomtexture = sidedef.lower
            if linedef.flags & ML_DONTPEGBOTTOM:
                rw_bottomtexturemid = worldtop
            else:
                rw_bottomtexturemid = worldlow

        rw_toptexturemid += sidedef.y_offset
        rw_bottomtexturemid += sidedef.y_offset

    segtextured = bool(midtexture or toptexture or bottomtexture)

    if segtextured:
        raw_diff = (rw_normalangle - rw_angle1) & ANGMASK
        offsetangle = raw_diff
        if offsetangle > ANG180:
            offsetangle = (-offsetangle) & ANGMASK
        if offsetangle > ANG90:
            offsetangle = ANG90
        sineval = float(finesine[offsetangle >> ANGLETOFINESHIFT])
        rw_offset = hyp * sineval
        # C: ``if (rw_normalangle-rw_angle1 < ANG180) rw_offset = -rw_offset;``
        if raw_diff < ANG180:
            rw_offset = -rw_offset
        rw_offset += sidedef.x_offset + curline.offset
        rw_centerangle = (ANG90 + viewangle - rw_normalangle) & ANGMASK

    # Floors/ceilings on the wrong side of view plane = invisible.
    if frontsector.floor_h >= viewz:
        markfloor = False
    if frontsector.ceiling_h <= viewz:
        markceiling = False

    # Compute step values in screen-y (pixel) space.  No HEIGHTBITS
    # shifts — we use floats throughout.
    topstep = -rw_scalestep * worldtop
    topfrac = centery - worldtop * rw_scale
    bottomstep = -rw_scalestep * worldbottom
    bottomfrac = centery - worldbottom * rw_scale

    if backsector is not None:
        if worldhigh < worldtop:
            pixhigh = centery - worldhigh * rw_scale
            pixhighstep = -rw_scalestep * worldhigh
        if worldlow > worldbottom:
            pixlow = centery - worldlow * rw_scale
            pixlowstep = -rw_scalestep * worldlow

    R_RenderSegLoop()

    # ds_p++.  We don't populate maskedtexturecol / sprite-clip arrays
    # because the masked pass is out of scope.
    ds_p += 1


def R_RenderSegLoop() -> None:
    """Per-column inner loop — paint mid/top/bottom and update floorclip/ceilingclip.

    C: r_segs.c.  This is the heart of the wall pipeline.
    """
    global rw_x, rw_scale, topfrac, bottomfrac, pixhigh, pixlow
    global dc_x, dc_yl, dc_yh, dc_iscale, dc_texturemid, dc_source
    global _dc_fallback_color

    while rw_x < rw_stopx:
        # yl: first row this column may paint (clipped against the
        # ceilingclip[] pixel already painted above).
        yl = int(math.ceil(topfrac))
        if yl < ceilingclip[rw_x] + 1:
            yl = int(ceilingclip[rw_x] + 1)

        # markceiling/markfloor would emit visplane spans here; out of
        # scope, so the structure is preserved but the body is empty.

        yh = int(math.floor(bottomfrac))
        if yh >= floorclip[rw_x]:
            yh = int(floorclip[rw_x] - 1)

        # texturecolumn — horizontal offset along the wall texture.
        if segtextured:
            angle = (rw_centerangle + int(xtoviewangle[rw_x])) & ANGMASK
            ft = float(finetangent[angle >> ANGLETOFINESHIFT])
            texturecolumn = int(math.floor(rw_offset - ft * rw_distance))
            dc_x = rw_x
            dc_iscale = 1.0 / rw_scale if rw_scale != 0.0 else 0.0
            _dc_fallback_color = _seg_color()
        else:
            texturecolumn = 0

        if midtexture:
            # Single-sided wall — paint everything from yl to yh.
            dc_yl = yl
            dc_yh = yh
            dc_texturemid = rw_midtexturemid
            dc_source = _column_source(midtexture, texturecolumn)
            colfunc()
            ceilingclip[rw_x] = viewheight
            floorclip[rw_x] = -1
        else:
            # Two-sided seg — top tier, bottom tier.
            if toptexture:
                mid = int(math.floor(pixhigh))
                pixhigh += pixhighstep
                if mid >= floorclip[rw_x]:
                    mid = int(floorclip[rw_x] - 1)
                if mid >= yl:
                    dc_yl = yl
                    dc_yh = mid
                    dc_texturemid = rw_toptexturemid
                    dc_source = _column_source(toptexture, texturecolumn)
                    colfunc()
                    ceilingclip[rw_x] = mid
                else:
                    ceilingclip[rw_x] = yl - 1
            else:
                if markceiling:
                    ceilingclip[rw_x] = yl - 1

            if bottomtexture:
                mid = int(math.ceil(pixlow))
                pixlow += pixlowstep
                if mid <= ceilingclip[rw_x]:
                    mid = int(ceilingclip[rw_x] + 1)
                if mid <= yh:
                    dc_yl = mid
                    dc_yh = yh
                    dc_texturemid = rw_bottomtexturemid
                    dc_source = _column_source(bottomtexture, texturecolumn)
                    colfunc()
                    floorclip[rw_x] = mid
                else:
                    floorclip[rw_x] = yh + 1
            else:
                if markfloor:
                    floorclip[rw_x] = yh + 1

        rw_scale += rw_scalestep
        topfrac += topstep
        bottomfrac += bottomstep
        rw_x += 1


def _column_source(tex_name: str, texturecolumn: int) -> Optional[np.ndarray]:
    """Texture column source for R_DrawColumn.

    Returns a (height, 3) slice of the texture's column, or None if no
    texture is loaded for this name (caller falls back to seg colour).

    C: ``R_GetColumn(texnum, col)`` walks DOOM's column-major patch
    format.  Our textures arrive as (W,H,3) numpy arrays already, so a
    column is just ``tex[col]``.  texturecolumn is masked by texture
    width, mirroring DOOM's ``& 127`` for 128-pixel-wide textures.
    """
    tex = _resolve_texture(tex_name)
    if tex is None:
        return None
    tw = tex.shape[0]
    return tex[texturecolumn % tw]


# =====================================================================
# === r_draw.c =========================================================
# =====================================================================


# Column-paint state.  Set by R_RenderSegLoop, consumed by R_DrawColumn.
dc_x: int = 0
dc_yl: int = 0
dc_yh: int = 0
dc_iscale: float = 0.0
dc_texturemid: float = 0.0
dc_source: Optional[np.ndarray] = None  # (height, 3) texture column

# Internal: fallback colour when dc_source is None (texture missing).
_dc_fallback_color: tuple = (0.5, 0.5, 0.5)

# Active column-paint function.  Bound to R_DrawColumn in
# _init_view_size.  (Transcribed for shape parity with C; we have only
# the one variant.)
colfunc: Optional[Callable[[], None]] = None


def R_DrawColumn() -> None:
    """Paint one screen column from dc_yl..dc_yh inclusive.

    C: r_draw.c.  Per-row v-coordinate is
        v = dc_texturemid + (row - centery) * dc_iscale
    where dc_iscale = 1 / rw_scale (world units per pixel).  The texel
    is sampled at row ``v`` modulo the texture height.
    """
    if _screen is None:
        return
    if dc_yh < dc_yl:
        return
    yl = max(0, dc_yl)
    yh = min(viewheight - 1, dc_yh)
    if yh < yl:
        return

    if dc_source is None:
        _screen[yl : yh + 1, dc_x] = _dc_fallback_color
        return

    th = dc_source.shape[0]
    # frac at first painted row = dc_texturemid + (yl - centery) * dc_iscale
    frac = dc_texturemid + (yl - centery) * dc_iscale
    fracstep = dc_iscale
    for row in range(yl, yh + 1):
        tex_row = int(math.floor(frac)) % th
        _screen[row, dc_x] = dc_source[tex_row]
        frac += fracstep


# =====================================================================
# === Helpers (synthetic-scene MapData builder) ========================
# =====================================================================


@dataclass
class _WallSpec:
    """One wall for ``single_sector_map``.

    a, b are (x, y).  ``texture`` is a name to put on the linedef's
    middle slot; "-" / "" ⇒ untextured (renderer falls back to sector
    colour).
    """

    a: tuple
    b: tuple
    texture: str = "-"


def single_sector_map(
    walls: List[_WallSpec],
    floor_h: float,
    ceiling_h: float,
    light: int = 255,
) -> MapData:
    """Build a MapData with one sector and zero BSP nodes.

    Used for synthetic test scenes (e.g. box rooms): when ``numnodes
    == 0`` the C ``R_RenderBSPNode(numnodes-1)`` falls through with
    ``bspnum == -1`` and we render the single subsector directly.

    Walls must be wound clockwise (DOOM: front == right of a→b points
    into the sector).  Each wall becomes one linedef + one sidedef +
    one seg.
    """
    vertices: List[Vertex] = []
    vmap: Dict[tuple, int] = {}

    def _vid(x, y) -> int:
        key = (int(x), int(y))
        if key in vmap:
            return vmap[key]
        idx = len(vertices)
        vertices.append(Vertex(int(x), int(y)))
        vmap[key] = idx
        return idx

    sectors = [
        Sector(
            floor_h=int(floor_h),
            ceiling_h=int(ceiling_h),
            floor_tex="-",
            ceiling_tex="-",
            light=light,
            special=0,
            tag=0,
        )
    ]
    sidedefs: List[Sidedef] = []
    linedefs: List[Linedef] = []
    segs: List[Seg] = []

    for w in walls:
        v1 = _vid(*w.a)
        v2 = _vid(*w.b)
        sd_idx = len(sidedefs)
        sidedefs.append(
            Sidedef(
                x_offset=0,
                y_offset=0,
                upper="-",
                lower="-",
                middle=w.texture,
                sector=0,
            )
        )
        ld_idx = len(linedefs)
        linedefs.append(
            Linedef(
                v1=v1,
                v2=v2,
                flags=0,
                special=0,
                tag=0,
                front_sidedef=sd_idx,
                back_sidedef=-1,
            )
        )
        # Seg angle: BAM angle of the (v2 - v1) direction.
        ax, ay = w.a
        bx, by = w.b
        ang_rad = math.atan2(by - ay, bx - ax)
        if ang_rad < 0:
            ang_rad += 2.0 * math.pi
        # Stored in 16-bit BAM-shifted form, as DOOM SEGS are.  We use
        # full 32-bit BAM internally (R_StoreWallRange reads
        # curline.angle); a 16-bit field is what the WAD stores, so we
        # widen at use sites.  Simpler: store full BAM here.
        bam = int(ang_rad * (2.0**32) / (2.0 * math.pi)) & ANGMASK
        segs.append(
            Seg(
                v1=v1,
                v2=v2,
                angle=bam,
                linedef=ld_idx,
                side=0,
                offset=0,
            )
        )

    subsectors = [Subsector(seg_count=len(segs), first_seg=0)]
    nodes: List[BspNode] = []  # zero nodes ⇒ R_RenderBSPNode(-1) → R_Subsector(0)

    return MapData(
        name="single_sector",
        vertices=vertices,
        linedefs=linedefs,
        sidedefs=sidedefs,
        sectors=sectors,
        segs=segs,
        subsectors=subsectors,
        nodes=nodes,
    )


def mapdata_from_segments(
    segments: List[Segment],
    textures: Optional[List[np.ndarray]] = None,
) -> tuple:
    """Build a MapData + name-keyed texture dict from a legacy Segment list.

    Migration adapter for code that authored scenes as ``List[Segment]``
    (the format the compiler still consumes).  Returns ``(MapData,
    textures_by_name)`` ready to feed straight into
    :func:`R_RenderPlayerView`.

    Each unique ``texture_id`` from the segment list gets a synthetic
    name ``"TEX{id}"``.  ``upper_texture_id`` / ``lower_texture_id``
    are honoured for two-sided segs.

    Sectors:
      * If every segment shares the same front floor/ceiling and no
        ``back_floor`` is set, builds a single sector with that height.
      * If segments have different sector heights or ``back_floor`` /
        ``back_ceiling`` is set, builds one sector per unique
        ``(floor, ceiling)`` pair and routes each seg to the matching
        front + back sectors.

    All segs go in a single subsector with no BSP nodes, so the
    renderer falls through ``R_RenderBSPNode(-1)`` → ``R_Subsector(0)``.
    Subsector closure isn't enforced — hand-authored scenes typically
    pass a closed loop, but the renderer doesn't require it.
    """
    if not segments:
        # Empty scene — single empty sector.
        return single_sector_map([], floor_h=0, ceiling_h=128), {}

    # Collect unique sector heights.
    sector_keys: Dict[tuple, int] = {}

    def _sector_id(fh, ch) -> int:
        key = (int(fh), int(ch))
        if key not in sector_keys:
            sector_keys[key] = len(sector_keys)
        return sector_keys[key]

    # First pass: assign sector IDs.
    for s in segments:
        _sector_id(s.front_floor, s.front_ceiling)
        if s.back_floor is not None and s.back_ceiling is not None:
            _sector_id(s.back_floor, s.back_ceiling)

    sectors_list = [
        Sector(
            floor_h=fh, ceiling_h=ch,
            floor_tex="-", ceiling_tex="-",
            light=255, special=0, tag=0,
        )
        for (fh, ch) in sector_keys.keys()
    ]

    # Texture name table.  Maps a texture_id (int) to a synthetic name.
    def _tex_name(tex_id: int) -> str:
        return f"TEX{tex_id}" if tex_id >= 0 else "-"

    textures_by_name: Dict[str, np.ndarray] = {}
    if textures is not None:
        for i, t in enumerate(textures):
            textures_by_name[_tex_name(i)] = t

    # Build vertices, linedefs, sidedefs, segs.
    vertices: List[Vertex] = []
    vmap: Dict[tuple, int] = {}

    def _vid(x, y) -> int:
        key = (int(round(x)), int(round(y)))
        if key in vmap:
            return vmap[key]
        idx = len(vertices)
        vertices.append(Vertex(*key))
        vmap[key] = idx
        return idx

    sidedefs: List[Sidedef] = []
    linedefs: List[Linedef] = []
    segs: List[Seg] = []

    for s in segments:
        v1 = _vid(s.ax, s.ay)
        v2 = _vid(s.bx, s.by)
        front_sec = _sector_id(s.front_floor, s.front_ceiling)
        is_two_sided = s.back_floor is not None and s.back_ceiling is not None

        front_sd = len(sidedefs)
        sidedefs.append(Sidedef(
            x_offset=0, y_offset=0,
            upper=_tex_name(s.upper_texture_id),
            lower=_tex_name(s.lower_texture_id),
            middle=_tex_name(s.texture_id) if not is_two_sided else "-",
            sector=front_sec,
        ))
        back_sd = -1
        if is_two_sided:
            back_sec = _sector_id(s.back_floor, s.back_ceiling)
            back_sd = len(sidedefs)
            sidedefs.append(Sidedef(
                x_offset=0, y_offset=0,
                upper="-", lower="-", middle="-",
                sector=back_sec,
            ))

        ld_idx = len(linedefs)
        linedefs.append(Linedef(
            v1=v1, v2=v2, flags=0, special=0, tag=0,
            front_sidedef=front_sd, back_sidedef=back_sd,
        ))

        ang_rad = math.atan2(s.by - s.ay, s.bx - s.ax)
        if ang_rad < 0:
            ang_rad += 2.0 * math.pi
        bam = int(ang_rad * (2.0**32) / (2.0 * math.pi)) & ANGMASK
        segs.append(Seg(
            v1=v1, v2=v2, angle=bam,
            linedef=ld_idx, side=0, offset=0,
        ))

    subsectors = [Subsector(seg_count=len(segs), first_seg=0)]
    md = MapData(
        name="from_segments",
        vertices=vertices,
        linedefs=linedefs,
        sidedefs=sidedefs,
        sectors=sectors_list,
        segs=segs,
        subsectors=subsectors,
        nodes=[],
    )
    return md, textures_by_name


def save_png(frame: np.ndarray, path: str, scale: float = 255.0) -> None:
    """Save a (H, W, 3) float frame as a PNG.

    Args:
        frame: Output of :func:`R_RenderPlayerView`.
        path:  Destination file path.
        scale: Multiplier to convert float colours to 0-255 range.
               Default 255.0 assumes colours are in [0.0, 1.0].
    """
    from PIL import Image as _Image  # local import — keeps doom_render's eager imports tight

    pixels = np.clip(frame * scale, 0, 255).astype(np.uint8)
    _Image.fromarray(pixels, mode="RGB").save(path)


__all__ = [
    # Top-level entry
    "R_RenderPlayerView",
    # r_main.c
    "R_PointOnSide",
    "R_PointToAngle",
    "R_PointToDist",
    "R_ScaleFromGlobalAngle",
    "R_InitTextureMapping",
    "R_SetupFrame",
    # r_bsp.c
    "Cliprange",
    "Drawseg",
    "R_AddLine",
    "R_CheckBBox",
    "R_ClearClipSegs",
    "R_ClearDrawSegs",
    "R_ClipPassWallSegment",
    "R_ClipSolidWallSegment",
    "R_RenderBSPNode",
    "R_Subsector",
    # r_segs.c / r_plane.c / r_draw.c
    "R_ClearPlanes",
    "R_DrawColumn",
    "R_RenderSegLoop",
    "R_StoreWallRange",
    # Helpers
    "single_sector_map",
    "mapdata_from_segments",
    "save_png",
    # Constants worth re-exporting
    "ANG90",
    "ANG180",
    "ANG270",
    "ANGMASK",
    "ANGLETOFINESHIFT",
    "FINEANGLES",
    "MAXSEGS",
    "MAXDRAWSEGS",
    "ML_DONTPEGTOP",
    "ML_DONTPEGBOTTOM",
    "NF_SUBSECTOR",
]
