"""The renderer's token vocabulary — prompt side and autoregressive loop.

Token types emitted by :func:`torchwright_doom.prompt.build.build_prompt`
(``PROMPT_TYPES``) plus the AR-loop types produced by the renderer's
``forward()`` (``AR_TYPES``). The combined ``VOCAB_TYPES`` is what the
embedding table is built against.

Capacity constants (``N_NODES_MAX`` etc.) bound the ``IntSlot`` ranges;
bumping them is a vocab-scale change that propagates to the embedding
table.

``ANGLE_VALUE``'s derived columns (``sin``, ``cos``, ``vatx``,
``u_tan_by_column``, the per-screen-column ``ray_x_*`` / ``ray_y_*``
entries) and ``VALUE``'s decoded / inverse / wall-scale columns hand
precomputed projection values to consumers at depth 0, avoiding PWL
chains across re-embedding boundaries. Screen-column tokens carry
per-column one-hot derived columns (``x_oh_*``) for direct
attention-keyed addressing.

Source of truth: ``doom_sandbox/implementations/spec09__ref/setup.py``
(this file is the faithful real-side mirror; the umbrella
``scripts/vocab_diff.py`` pins the contract).
"""

from __future__ import annotations

import math

from .asset_config import N_FLATS, N_WALL_TEXTURES, PLAYPAL
from .constants import SCREEN_HEIGHT, SCREEN_WIDTH
from .doom_lighting import doom_flat_startmap
from .tokens import Derived, FloatSlot, IntSlot, Token, TokenType
from .value_ranges import (
    ValueRange,
    make_value as _make_value,
    prefill_value as _prefill_value,
    value_derived_columns,
)

# ---------------------------------------------------------------------------
# Capacity constants
# ---------------------------------------------------------------------------

ANGLE_BAM = 8192
BACK_HEIGHT_SENTINEL = -4096.0

N_NODES_MAX = 64
N_SUBSECTORS_MAX = 64
N_SEGS_MAX = 128
N_ENTITY_MAX = N_NODES_MAX + N_SUBSECTORS_MAX  # unified child_u space
N_DEPTH_MAX = 16
N_PLANES_MAX = 32
N_VP_PER_PLANE_MAX = 8
N_LIGHT_LEVELS = 256

FOV_HALF_BAM = ANGLE_BAM // 8  # 1024
_TAN_FOV_HALF = math.tan(FOV_HALF_BAM * 2 * math.pi / ANGLE_BAM)


def viewangletox(theta_bam: int) -> int:
    """Map a signed BAM view angle to an integer screen column."""
    if theta_bam >= FOV_HALF_BAM:
        return 0
    if theta_bam <= -FOV_HALF_BAM:
        return SCREEN_WIDTH - 1
    t = math.tan(theta_bam * 2 * math.pi / ANGLE_BAM)
    return round((SCREEN_WIDTH - 1) * (1 - t / _TAN_FOV_HALF) / 2)


# ---------------------------------------------------------------------------
# Derived-column helpers (ported from spec09__ref/setup.py)
# ---------------------------------------------------------------------------


def _xtoviewangle_rad(x: int) -> float:
    """Fixed Doom-style xtoviewangle table entry for a screen column."""
    if SCREEN_WIDTH <= 1:
        return 0.0
    tan_fov_half = math.tan((ANGLE_BAM // 8) * 2 * math.pi / ANGLE_BAM)
    tangent = tan_fov_half * (1.0 - 2.0 * x / (SCREEN_WIDTH - 1))
    return math.atan(tangent)


def _xtova_sin(x: int) -> float:
    return math.sin(_xtoviewangle_rad(x))


def _xtova_cos(x: int) -> float:
    return math.cos(_xtoviewangle_rad(x))


def _distscale(x: int) -> float:
    return 1.0 / max(0.1, _xtova_cos(x))


def _theta_le_neg_fov_half(a: int) -> float:
    return 1.0 if int(a) <= -FOV_HALF_BAM else 0.0


def _theta_ge_pos_fov_half(a: int) -> float:
    return 1.0 if int(a) >= FOV_HALF_BAM else 0.0


def _u_tan_by_column(a: int) -> list[float]:
    """tan((view_angle - rw_normalangle) + xtoviewangle[column]) per column."""
    base = int(a) * 2 * math.pi / ANGLE_BAM
    values: list[float] = []
    for column in range(SCREEN_WIDTH):
        value = math.tan(base + _xtoviewangle_rad(column))
        values.append(max(-10.5, min(10.5, value)))
    return values


def _id_lifted_key(value: int) -> list[float]:
    iv = int(value)
    return [float(iv), float(-(iv * iv)), 1.0]


def _yslope(y: int) -> float:
    # Doom's plane distance scales as projection / distance from the horizon.
    center_y = SCREEN_HEIGHT / 2.0
    projection = (SCREEN_WIDTH - 1) / (2.0 * _TAN_FOV_HALF)
    return projection / max(0.5, abs(float(y) - center_y))


def _pixel_rgb_channel(color_idx: int, channel: int) -> float:
    idx = max(0, min(255, int(color_idx)))
    return float(PLAYPAL[idx][channel])


def _tex_id_present(tid: int) -> float:
    return 1.0 if int(tid) != 0 else 0.0


SCREEN_X_ONE_HOT_DERIVED_NAMES = tuple(f"x_oh_{x:03d}" for x in range(SCREEN_WIDTH))
ANGLE_RAY_X_DERIVED_NAMES = tuple(f"ray_x_{x:03d}" for x in range(SCREEN_WIDTH))
ANGLE_RAY_Y_DERIVED_NAMES = tuple(f"ray_y_{x:03d}" for x in range(SCREEN_WIDTH))

_SCREEN_X_ONE_HOT_DERIVED = {
    name: (lambda value, column=column: 1.0 if int(value) == column else 0.0)
    for column, name in enumerate(SCREEN_X_ONE_HOT_DERIVED_NAMES)
}
_SCREEN_X_ANGLE_DERIVED = {
    "xtova_sin": _xtova_sin,
    "xtova_cos": _xtova_cos,
    "distscale": _distscale,
}
_SCREEN_X_DERIVED = {
    **_SCREEN_X_ANGLE_DERIVED,
    **_SCREEN_X_ONE_HOT_DERIVED,
}
# Scan x is the canonical visible-run x1; it adds x_square for the
# quadratic key used by run-finding.
_SCAN_X_DERIVED = {
    **_SCREEN_X_DERIVED,
    "x_square": (lambda x: float(x * x)),
}

# Public name of the lifted scalar-id key derived column (producer key
# ``[id, -id**2, 1]``), referenced by the renderer read-side (Plan D).
ID_LIFTED_KEY_DERIVED_NAME = "id_lifted_key"

_ID_LIFTED_KEY_DERIVED = {
    ID_LIFTED_KEY_DERIVED_NAME: Derived(_id_lifted_key, width=3),
}

_TEX_ID_DERIVED = {"present": _tex_id_present}

_VALUE_DERIVED = value_derived_columns()


# bbox CHECKCOORD table (Doom R_CheckBBox): per boxpos, which bbox corners
# form the test edges and whether the box trivially fails (always visible).
_CHECKCOORD = (
    (3, 0, 2, 1),
    (3, 0, 2, 0),
    (3, 1, 2, 0),
    (0, 0, 0, 0),
    (2, 0, 2, 1),
    (0, 0, 0, 0),
    (3, 1, 3, 0),
    (0, 0, 0, 0),
    (2, 0, 3, 1),
    (2, 1, 3, 1),
    (2, 1, 3, 0),
    (0, 0, 0, 0),
)


def _checkcoord_row(boxpos: int) -> tuple[int, int, int, int]:
    return _CHECKCOORD[int(boxpos)]


_BOXPOS_DERIVED = {
    "check_a_x_right": (lambda boxpos: 1.0 if _checkcoord_row(boxpos)[0] == 3 else 0.0),
    "check_a_y_bottom": (
        lambda boxpos: 1.0 if _checkcoord_row(boxpos)[1] == 1 else 0.0
    ),
    "check_b_x_right": (lambda boxpos: 1.0 if _checkcoord_row(boxpos)[2] == 3 else 0.0),
    "check_b_y_bottom": (
        lambda boxpos: 1.0 if _checkcoord_row(boxpos)[3] == 1 else 0.0
    ),
    "fails_open": (lambda boxpos: 1.0 if int(boxpos) in (3, 5, 7, 11) else 0.0),
}


def _boxpos_slot() -> IntSlot:
    return IntSlot(0, 12, derived=_BOXPOS_DERIVED)


# Existence pattern -> K_part lookup tables. Index is
# ``pat = 4*has_mid + 2*has_upper + has_lower``. ``K_part_k[pat]`` is the
# part index (0=mid, 1=upper, 2=lower, 3=sentinel) for the K-th existing
# part at this seg.
_K_PART_TABLES = (
    (3, 2, 1, 1, 0, 0, 0, 0),
    (3, 3, 3, 2, 3, 2, 1, 1),
    (3, 3, 3, 3, 3, 3, 3, 2),
)


# ---------------------------------------------------------------------------
# Shared payload types
# ---------------------------------------------------------------------------

VALUE = TokenType(
    "value",
    slots={"v": FloatSlot(-1.0, 1.0, levels=65536, derived=_VALUE_DERIVED)},
)


def prefill_value(range_id: ValueRange, value: float) -> Token:
    """2-arg ``VALUE``-bound wrapper over the ``value_ranges`` 3-arg core:
    build a ``VALUE`` token carrying the range-encoded float. Mirrors the
    sandbox ``setup.prefill_value``; the prompt builder uses it so every
    VALUE payload is encoded into its range's ``[-1, 1]`` space.
    """
    return _prefill_value(VALUE, range_id, value)


def make_value(range_id: ValueRange, value):
    """2-arg ``VALUE``-bound wrapper over the ``value_ranges`` 3-arg core:
    emit a ``VALUE`` carrier (an emit head) for a computed node, range-encoded.
    Mirrors the sandbox ``protocol_tokens_decl.make_value``; the forward
    renderer's projection / drawseg owners call it to emit ``VALUE`` payloads.
    """
    return _make_value(VALUE, range_id, value)


ANGLE_VALUE = TokenType(
    "angleValue",
    slots={
        "angle": IntSlot(
            -ANGLE_BAM // 2,
            ANGLE_BAM // 2,
            derived={
                "sin": lambda a: math.sin(a * 2 * math.pi / ANGLE_BAM),
                "cos": lambda a: math.cos(a * 2 * math.pi / ANGLE_BAM),
                "vatx": lambda a: float(viewangletox(a)),
                "theta_le_neg_fov_half": _theta_le_neg_fov_half,
                "theta_ge_pos_fov_half": _theta_ge_pos_fov_half,
                "u_tan_by_column": Derived(_u_tan_by_column, width=SCREEN_WIDTH),
                **{
                    name: (
                        lambda a, column=column: math.cos(
                            a * 2 * math.pi / ANGLE_BAM + _xtoviewangle_rad(column)
                        )
                    )
                    for column, name in enumerate(ANGLE_RAY_X_DERIVED_NAMES)
                },
                **{
                    name: (
                        lambda a, column=column: math.sin(
                            a * 2 * math.pi / ANGLE_BAM + _xtoviewangle_rad(column)
                        )
                    )
                    for column, name in enumerate(ANGLE_RAY_Y_DERIVED_NAMES)
                },
            },
        ),
    },
)

# ---------------------------------------------------------------------------
# Section 1: Player state
# ---------------------------------------------------------------------------

PLAYER_X_MARK = TokenType("viewx")
PLAYER_Y_MARK = TokenType("viewy")
PLAYER_Z_MARK = TokenType("viewz")
PLAYER_ANGLE_MARK = TokenType("viewangle")

# ---------------------------------------------------------------------------
# Section 2: Per-node block (header + plane + children + bboxes)
# ---------------------------------------------------------------------------

NODE = TokenType(
    "node", slots={"j": IntSlot(0, N_NODES_MAX, derived=_ID_LIFTED_KEY_DERIVED)}
)

NODE_PX = TokenType("node.x")
NODE_PY = TokenType("node.y")
NODE_DX = TokenType("node.dx")
NODE_DY = TokenType("node.dy")

NODE_FRONT_CHILD = TokenType("node.child1", slots={"child_u": IntSlot(0, N_ENTITY_MAX)})
NODE_BACK_CHILD = TokenType("node.child0", slots={"child_u": IntSlot(0, N_ENTITY_MAX)})

BBOX_TOP_FRONT = TokenType("node.bbox1.top")
BBOX_BOT_FRONT = TokenType("node.bbox1.bottom")
BBOX_LEFT_FRONT = TokenType("node.bbox1.left")
BBOX_RIGHT_FRONT = TokenType("node.bbox1.right")
BBOX_TOP_BACK = TokenType("node.bbox0.top")
BBOX_BOT_BACK = TokenType("node.bbox0.bottom")
BBOX_LEFT_BACK = TokenType("node.bbox0.left")
BBOX_RIGHT_BACK = TokenType("node.bbox0.right")

# ---------------------------------------------------------------------------
# Section 3: Per-subsector + per-seg blocks
# ---------------------------------------------------------------------------

SS = TokenType(
    "SSECTOR", slots={"s": IntSlot(0, N_SUBSECTORS_MAX, derived=_ID_LIFTED_KEY_DERIVED)}
)

SEG = TokenType(
    "seg",
    slots={
        "i": IntSlot(0, N_SEGS_MAX, derived=_ID_LIFTED_KEY_DERIVED),
        "is_first_of_ss": IntSlot(0, 2),
    },
)
SEG_AX = TokenType("seg.v1.x")
SEG_AY = TokenType("seg.v1.y")
SEG_BX = TokenType("seg.v2.x")
SEG_BY = TokenType("seg.v2.y")
SEG_TWO_SIDED = TokenType("seg.hasBacksector", slots={"flag": IntSlot(0, 2)})
SEG_NORMAL_ANGLE = TokenType("seg.normalAngle")
SEG_FRONT_FLOOR = TokenType("seg.front.floor")
SEG_FRONT_CEILING = TokenType("seg.front.ceiling")
SEG_BACK_FLOOR = TokenType("seg.back.floor")
SEG_BACK_CEILING = TokenType("seg.back.ceiling")
SEG_MID_TEXTURE = TokenType(
    "seg.texture.mid",
    slots={"tex_id": IntSlot(0, N_WALL_TEXTURES + 1, derived=_TEX_ID_DERIVED)},
)
SEG_UPPER_TEXTURE = TokenType(
    "seg.texture.upper",
    slots={"tex_id": IntSlot(0, N_WALL_TEXTURES + 1, derived=_TEX_ID_DERIVED)},
)
SEG_LOWER_TEXTURE = TokenType(
    "seg.texture.lower",
    slots={"tex_id": IntSlot(0, N_WALL_TEXTURES + 1, derived=_TEX_ID_DERIVED)},
)
SEG_LIGHT_STATIC = TokenType("seg.lightStatic", slots={"light": IntSlot(0, 64)})
SEG_EMPTY_LINE = TokenType("seg.emptyLine", slots={"flag": IntSlot(0, 2)})
SEG_CLOSED_DOOR = TokenType("seg.closedDoor", slots={"flag": IntSlot(0, 2)})
SEG_PEGGING = TokenType(
    "seg.pegging",
    slots={
        "dontpegtop": IntSlot(0, 2),
        "dontpegbottom": IntSlot(0, 2),
    },
)
SEG_ROWOFFSET = TokenType("seg.rowoffset")  # marker; VALUE follows

# ---------------------------------------------------------------------------
# Section 3b: Visplane prefill
# ---------------------------------------------------------------------------

PLANE_DEF = TokenType(
    "planeDef",
    slots={
        "p": IntSlot(0, N_PLANES_MAX),
        "flat_id": IntSlot(0, N_FLATS),
    },
)
PLANE_HEIGHT = TokenType("planeHeight")
PLANE_LIGHT = TokenType(
    "planeLight",
    slots={
        "light": IntSlot(
            0,
            N_LIGHT_LEVELS,
            derived={"light_static": lambda v: float(doom_flat_startmap(int(v)))},
        )
    },
)
SS_FLOOR_PLANE = TokenType(
    "ssFloorPlane",
    slots={
        "s": IntSlot(0, N_SUBSECTORS_MAX, derived=_ID_LIFTED_KEY_DERIVED),
        "p": IntSlot(0, N_PLANES_MAX),
    },
)
SS_CEILING_PLANE = TokenType(
    "ssCeilingPlane",
    slots={
        "s": IntSlot(0, N_SUBSECTORS_MAX, derived=_ID_LIFTED_KEY_DERIVED),
        "p": IntSlot(0, N_PLANES_MAX),
    },
)

# ---------------------------------------------------------------------------
# Section 4: BEGIN
# ---------------------------------------------------------------------------

BEGIN = TokenType("begin")


PROMPT_TYPES = [
    # Shared payload
    VALUE,
    ANGLE_VALUE,
    # Section 1
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    PLAYER_Z_MARK,
    PLAYER_ANGLE_MARK,
    # Section 2: per-node
    NODE,
    NODE_PX,
    NODE_PY,
    NODE_DX,
    NODE_DY,
    NODE_FRONT_CHILD,
    NODE_BACK_CHILD,
    BBOX_TOP_FRONT,
    BBOX_BOT_FRONT,
    BBOX_LEFT_FRONT,
    BBOX_RIGHT_FRONT,
    BBOX_TOP_BACK,
    BBOX_BOT_BACK,
    BBOX_LEFT_BACK,
    BBOX_RIGHT_BACK,
    # Section 3: per-ss + per-seg
    SS,
    SEG,
    SEG_AX,
    SEG_AY,
    SEG_BX,
    SEG_BY,
    SEG_TWO_SIDED,
    SEG_NORMAL_ANGLE,
    SEG_FRONT_FLOOR,
    SEG_FRONT_CEILING,
    SEG_BACK_FLOOR,
    SEG_BACK_CEILING,
    SEG_MID_TEXTURE,
    SEG_UPPER_TEXTURE,
    SEG_LOWER_TEXTURE,
    SEG_LIGHT_STATIC,
    SEG_EMPTY_LINE,
    SEG_CLOSED_DOOR,
    SEG_PEGGING,
    SEG_ROWOFFSET,
    # Section 3b: visplane prefill
    PLANE_DEF,
    PLANE_HEIGHT,
    PLANE_LIGHT,
    SS_FLOOR_PLANE,
    SS_CEILING_PLANE,
    # Section 4
    BEGIN,
]


# ---------------------------------------------------------------------------
# AR-loop types (ported from spec09__ref/setup.py)
# ---------------------------------------------------------------------------

NO_OP = TokenType("noOp")
THINK_SIDE = TokenType("R_PointOnSide", slots={"node": IntSlot(0, N_NODES_MAX)})
SIDE_RECORD = TokenType(
    "pointOnSideResult",
    slots={
        "node": IntSlot(0, N_NODES_MAX, derived=_ID_LIFTED_KEY_DERIVED),
        "side": IntSlot(0, 2),
    },
)
WORLD_ANGLE_MARK_A = TokenType("angle1")
THETA_MARK_A = TokenType("theta1")
WORLD_ANGLE_MARK_B = TokenType("angle2")
THETA_MARK_B = TokenType("theta2")
BBOX_WORLD_ANGLE_MARK_A = TokenType("bbox.angle1")
BBOX_THETA_MARK_A = TokenType("bbox.theta1")
BBOX_WORLD_ANGLE_MARK_B = TokenType("bbox.angle2")
BBOX_THETA_MARK_B = TokenType("bbox.theta2")
BBOX_BOXPOS = TokenType("boxpos", slots={"boxpos": _boxpos_slot()})
BBOX_CORNER_X_MARK_A = TokenType("bbox.x1", slots={"boxpos": _boxpos_slot()})
BBOX_CORNER_Y_MARK_A = TokenType("bbox.y1", slots={"boxpos": _boxpos_slot()})
BBOX_CORNER_X_MARK_B = TokenType("bbox.x2", slots={"boxpos": _boxpos_slot()})
BBOX_CORNER_Y_MARK_B = TokenType("bbox.y2", slots={"boxpos": _boxpos_slot()})
TRAVERSE_ENTER = TokenType(
    "bspFront",
    slots={"node": IntSlot(0, N_NODES_MAX), "depth": IntSlot(0, N_DEPTH_MAX)},
)
TRAVERSE_BETWEEN = TokenType(
    "bspCheckBack",
    slots={"node": IntSlot(0, N_NODES_MAX), "depth": IntSlot(0, N_DEPTH_MAX)},
)
TRAVERSE_RETURN = TokenType(
    "bspReturn",
    slots={"entity_u": IntSlot(0, N_ENTITY_MAX), "depth": IntSlot(0, N_DEPTH_MAX)},
)
VISIT_SUBSECTOR = TokenType(
    "R_Subsector",
    slots={"s": IntSlot(0, N_SUBSECTORS_MAX), "depth": IntSlot(0, N_DEPTH_MAX)},
)
PROCESS_SEG = TokenType("R_AddLine", slots={"i": IntSlot(0, N_SEGS_MAX)})
FIND_RUN = TokenType(
    "clipScan", slots={"x": IntSlot(0, SCREEN_WIDTH + 1, derived=_SCAN_X_DERIVED)}
)
BBOX_SCAN = TokenType(
    "bboxClipScan", slots={"x": IntSlot(0, SCREEN_WIDTH + 1, derived=_SCAN_X_DERIVED)}
)
ADVANCE_SEG = TokenType("nextSeg", slots={"i": IntSlot(0, N_SEGS_MAX)})
EMIT_X2 = TokenType(
    "drawseg.x2", slots={"x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_DERIVED)}
)
R_STORE_WALL_RANGE = TokenType(
    "R_StoreWallRange",
    slots={"i": IntSlot(0, N_SEGS_MAX, derived=_ID_LIFTED_KEY_DERIVED)},
)
SEG_KPART = TokenType(
    "segKpart",
    slots={
        "pat": IntSlot(
            0,
            8,
            derived={
                "K_part_0": lambda p: float(_K_PART_TABLES[0][int(p)]),
                "K_part_1": lambda p: float(_K_PART_TABLES[1][int(p)]),
                "K_part_2": lambda p: float(_K_PART_TABLES[2][int(p)]),
                "has_mid": lambda p: float((int(p) >> 2) & 1),
                "has_upper": lambda p: float((int(p) >> 1) & 1),
                "has_lower": lambda p: float(int(p) & 1),
            },
        )
    },
)
SEG_DC_TMID_MID = TokenType("segDcTmidMid")
SEG_DC_TMID_UPPER = TokenType("segDcTmidUpper")
SEG_DC_TMID_LOWER = TokenType("segDcTmidLower")
DRAWSEG_META = TokenType(
    "drawseg.meta",
    slots={
        "i": IntSlot(0, N_SEGS_MAX),
        "wall_kind": IntSlot(0, 3),
        "silhouette": IntSlot(0, 4),
    },
)
DRAWSEG_SCALE1_DEN = TokenType("drawseg.scale1.den")
DRAWSEG_SCALE1 = TokenType("drawseg.scale1")
DRAWSEG_SCALE2_DEN = TokenType("drawseg.scale2.den")
DRAWSEG_SCALE2 = TokenType("drawseg.scale2")
DRAWSEG_SCALESTEP_DEN = TokenType("drawseg.scalestep.den")
DRAWSEG_SCALESTEP = TokenType("drawseg.scalestep")
DRAWSEG_BSILHEIGHT = TokenType("drawseg.bsilheight")
DRAWSEG_TSILHEIGHT = TokenType("drawseg.tsilheight")
DRAWSEG_U_PHASE = TokenType("drawseg.uPhase")
R_CHECK_PLANE = TokenType(
    "R_CheckPlane",
    slots={"kind": IntSlot(0, 2), "vp": IntSlot(0, N_VP_PER_PLANE_MAX)},
)
R_CHECK_PLANE_RESULT = TokenType(
    "R_CheckPlane.result",
    slots={"kind": IntSlot(0, 2), "vp": IntSlot(0, N_VP_PER_PLANE_MAX)},
)
SET_CURSOR_X = TokenType(
    "setCursorX", slots={"x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_DERIVED)}
)
WALL_COL_U = TokenType("wallColU", slots={"u_idx": IntSlot(-1024, 1024)})
WALL_SPAN_META = TokenType(
    "wallSpanMeta",
    slots={"y": IntSlot(-1, SCREEN_HEIGHT + 1), "ordinal": IntSlot(0, 3)},
)
SET_CURSOR_Y = TokenType("setCursorY", slots={"y": IntSlot(-1, SCREEN_HEIGHT + 1)})
PIXEL = TokenType(
    "pixel",
    slots={
        "color": IntSlot(
            0,
            256,
            derived={
                "pixel_r": lambda color: _pixel_rgb_channel(color, 0),
                "pixel_g": lambda color: _pixel_rgb_channel(color, 1),
                "pixel_b": lambda color: _pixel_rgb_channel(color, 2),
            },
        )
    },
)
CLIP_UPDATE = TokenType("clipUpdate")
PLANE_MARK = TokenType(
    "planeMark",
    slots={
        "p": IntSlot(0, N_PLANES_MAX),
        "kind": IntSlot(0, 2),
        "vp": IntSlot(0, N_VP_PER_PLANE_MAX),
    },
)
DRAW_PLANES_BEGIN = TokenType("R_DrawPlanes")
FLAT_NEXT_PLANE = TokenType(
    "R_DrawPlanes.nextPlane", slots={"p": IntSlot(-1, N_PLANES_MAX)}
)
FLAT_NEXT_VP = TokenType(
    "R_DrawPlanes.nextVp",
    slots={"p": IntSlot(0, N_PLANES_MAX), "vp": IntSlot(-1, N_VP_PER_PLANE_MAX)},
)
FLAT_VISPLANE_BEGIN = TokenType(
    "visplaneBegin",
    slots={"p": IntSlot(0, N_PLANES_MAX), "vp": IntSlot(0, N_VP_PER_PLANE_MAX)},
)
MAKE_SPANS_COL = TokenType(
    "R_MakeSpans.col",
    slots={"x": IntSlot(0, SCREEN_WIDTH + 1, derived=_SCAN_X_DERIVED)},
)
SPAN_CLOSE_SLOT = TokenType("R_MakeSpans.closeSlot", slots={"slot": IntSlot(0, 2)})
SPAN_ROW = TokenType(
    "R_MapPlane.row",
    slots={"y": IntSlot(0, SCREEN_HEIGHT, derived={"yslope": _yslope})},
)
SET_CURSOR_DIRECTION_Y = TokenType("setCursorDirectionY")
SET_CURSOR_DIRECTION_X = TokenType("setCursorDirectionX")
SCREEN_Y_VALUE = TokenType("screenY", slots={"y": IntSlot(-1, SCREEN_HEIGHT + 1)})
SCREEN_RANGE = TokenType(
    "screenRange",
    slots={
        "y1": IntSlot(-1, SCREEN_HEIGHT + 1),
        "y2": IntSlot(-1, SCREEN_HEIGHT + 1),
    },
)
DONE = TokenType("done")


AR_TYPES = [
    NO_OP,
    THINK_SIDE,
    SIDE_RECORD,
    TRAVERSE_ENTER,
    TRAVERSE_BETWEEN,
    TRAVERSE_RETURN,
    VISIT_SUBSECTOR,
    PROCESS_SEG,
    FIND_RUN,
    WORLD_ANGLE_MARK_A,
    THETA_MARK_A,
    WORLD_ANGLE_MARK_B,
    THETA_MARK_B,
    BBOX_BOXPOS,
    BBOX_CORNER_X_MARK_A,
    BBOX_CORNER_Y_MARK_A,
    BBOX_CORNER_X_MARK_B,
    BBOX_CORNER_Y_MARK_B,
    BBOX_WORLD_ANGLE_MARK_A,
    BBOX_THETA_MARK_A,
    BBOX_WORLD_ANGLE_MARK_B,
    BBOX_THETA_MARK_B,
    BBOX_SCAN,
    ADVANCE_SEG,
    EMIT_X2,
    R_STORE_WALL_RANGE,
    SEG_KPART,
    SEG_DC_TMID_MID,
    SEG_DC_TMID_UPPER,
    SEG_DC_TMID_LOWER,
    DRAWSEG_META,
    DRAWSEG_SCALE1_DEN,
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE2_DEN,
    DRAWSEG_SCALE2,
    DRAWSEG_SCALESTEP_DEN,
    DRAWSEG_SCALESTEP,
    DRAWSEG_BSILHEIGHT,
    DRAWSEG_TSILHEIGHT,
    DRAWSEG_U_PHASE,
    R_CHECK_PLANE,
    R_CHECK_PLANE_RESULT,
    SET_CURSOR_X,
    WALL_COL_U,
    WALL_SPAN_META,
    SET_CURSOR_Y,
    PIXEL,
    CLIP_UPDATE,
    PLANE_MARK,
    DRAW_PLANES_BEGIN,
    FLAT_NEXT_PLANE,
    FLAT_NEXT_VP,
    FLAT_VISPLANE_BEGIN,
    MAKE_SPANS_COL,
    SPAN_CLOSE_SLOT,
    SPAN_ROW,
    SET_CURSOR_DIRECTION_Y,
    SET_CURSOR_DIRECTION_X,
    SCREEN_Y_VALUE,
    SCREEN_RANGE,
    DONE,
]


VOCAB_TYPES = PROMPT_TYPES + AR_TYPES
