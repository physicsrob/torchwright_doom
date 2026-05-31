"""The renderer's token vocabulary — prompt side and autoregressive loop.

Token types emitted by :func:`torchwright_doom.prompt.build.build_prompt`
(``PROMPT_TYPES``) plus the AR-loop types produced by the renderer's
``forward()`` (``AR_TYPES``). The combined ``VOCAB_TYPES`` is what the
embedding table is built against.

Capacity constants (``N_NODES_MAX`` etc.) bound the ``IntSlot`` ranges;
bumping them is a vocab-scale change that propagates to the embedding
table.

``ANGLE_VALUE``'s derived columns (``sin``, ``cos``, ``vatx``, plus the
per-screen-column ``ray_x_*`` / ``ray_y_*`` entries) and ``VALUE``'s
inverse / abs / square columns hand precomputed projection values to
consumers at depth 0, avoiding PWL chains across re-embedding
boundaries. ``EMIT_X1`` / ``EMIT_X2`` / ``WALL_COLUMN`` / etc. carry
per-screen-column one-hot derived columns for direct attention-keyed
addressing.

Source of truth for the derived helpers and AR-loop type declarations:
``doom_sandbox/implementations/spec09__ref/setup.py``.
"""

from __future__ import annotations

import math

from .constants import SCREEN_HEIGHT, SCREEN_WIDTH
from .tokens import FloatSlot, IntSlot, TokenType


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
N_FLATS_MAX = 32
N_LIGHT_LEVELS = 256

# SCREEN_WIDTH / SCREEN_HEIGHT now live in `.constants` (imported above) so
# `value_ranges.py` can depend on them without importing `vocab.py` — see
# constants.py for why. Porting scale is 60×50; retarget to 160×100 is a
# deferred one-line change in constants.py.

FOV_HALF_BAM = ANGLE_BAM // 8  # 1024
_TAN_FOV_HALF = math.tan(FOV_HALF_BAM * 2 * math.pi / ANGLE_BAM)


def viewangletox(theta_bam: int) -> int:
    """Map a signed BAM view angle to an integer screen column.

    Mirrors Doom's ``viewangletox`` table at our ``SCREEN_WIDTH`` and
    ``ANGLE_BAM``. Used by ``ANGLE_VALUE``'s ``vatx`` derived column so
    consumers can read the screen-column projection of an emitted angle
    at depth 0.
    """
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


SCREEN_X_ONE_HOT_DERIVED_NAMES = tuple(
    f"x_oh_{x:03d}" for x in range(SCREEN_WIDTH)
)
ANGLE_RAY_X_DERIVED_NAMES = tuple(
    f"ray_x_{x:03d}" for x in range(SCREEN_WIDTH)
)
ANGLE_RAY_Y_DERIVED_NAMES = tuple(
    f"ray_y_{x:03d}" for x in range(SCREEN_WIDTH)
)
_SCREEN_X_ONE_HOT_DERIVED = {
    name: (lambda value, column=column: 1.0 if int(value) == column else 0.0)
    for column, name in enumerate(SCREEN_X_ONE_HOT_DERIVED_NAMES)
}
_SCREEN_X_ANGLE_DERIVED = {
    "xtova_sin": _xtova_sin,
    "xtova_cos": _xtova_cos,
}
_SCREEN_X_DERIVED = {
    **_SCREEN_X_ANGLE_DERIVED,
    **_SCREEN_X_ONE_HOT_DERIVED,
}


_VALUE_MAX_FLOAT = 4096.0
_VALUE_INVERSE_EPS = 0.001


def _value_inverse(v: float) -> float:
    if v == 0.0:
        return _VALUE_MAX_FLOAT
    return 1.0 / v


def _value_clamped_inverse(v: float) -> float:
    if v >= 0.0:
        return 1.0 / max(v, _VALUE_INVERSE_EPS)
    return 1.0 / min(v, -_VALUE_INVERSE_EPS)


def _value_inverse_abs(v: float) -> float:
    av = abs(v)
    if av == 0.0:
        return _VALUE_MAX_FLOAT
    return 1.0 / av


def _value_clamped_inverse_abs(v: float) -> float:
    return 1.0 / max(abs(v), _VALUE_INVERSE_EPS)


_VALUE_DERIVED = {
    "inverse": _value_inverse,
    "clamped_inverse": _value_clamped_inverse,
    "abs": abs,
    "inverse_abs": _value_inverse_abs,
    "clamped_inverse_abs": _value_clamped_inverse_abs,
    "square": lambda v: v * v,
}


# ---------------------------------------------------------------------------
# Shared payload types
#
# One ``VALUE`` carrier absorbs every continuous float payload
# (player position, plane params, seg endpoints, bbox edges).
# One ``ANGLE_VALUE`` carrier handles every angle-bearing token,
# with precomputed derived columns so projection is depth-0.
# ---------------------------------------------------------------------------

VALUE = TokenType(
    "value",
    slots={"v": FloatSlot(-4096.0, 4096.0, levels=65536, derived=_VALUE_DERIVED)},
)

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
                **{
                    name: (
                        lambda a, column=column: math.cos(
                            a * 2 * math.pi / ANGLE_BAM
                            + _xtoviewangle_rad(column)
                        )
                    )
                    for column, name in enumerate(ANGLE_RAY_X_DERIVED_NAMES)
                },
                **{
                    name: (
                        lambda a, column=column: math.sin(
                            a * 2 * math.pi / ANGLE_BAM
                            + _xtoviewangle_rad(column)
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

NODE = TokenType("node", slots={"j": IntSlot(0, N_NODES_MAX)})

NODE_PX = TokenType("node.x")
NODE_PY = TokenType("node.y")
NODE_DX = TokenType("node.dx")
NODE_DY = TokenType("node.dy")

NODE_FRONT_CHILD = TokenType(
    "node.child1", slots={"child_u": IntSlot(0, N_ENTITY_MAX)}
)
NODE_BACK_CHILD = TokenType(
    "node.child0", slots={"child_u": IntSlot(0, N_ENTITY_MAX)}
)

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

SS = TokenType("SSECTOR", slots={"s": IntSlot(0, N_SUBSECTORS_MAX)})

SEG = TokenType(
    "seg",
    slots={
        "i": IntSlot(0, N_SEGS_MAX),
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
SEG_MID_TEXTURE = TokenType("seg.texture.mid", slots={"present": IntSlot(0, 2)})
SEG_UPPER_TEXTURE = TokenType("seg.texture.upper", slots={"present": IntSlot(0, 2)})
SEG_LOWER_TEXTURE = TokenType("seg.texture.lower", slots={"present": IntSlot(0, 2)})
SEG_EMPTY_LINE = TokenType("seg.emptyLine", slots={"flag": IntSlot(0, 2)})
SEG_CLOSED_DOOR = TokenType("seg.closedDoor", slots={"flag": IntSlot(0, 2)})

# ---------------------------------------------------------------------------
# Section 3b: Visplane prefill
# ---------------------------------------------------------------------------

FLAT_DEF = TokenType("flatDef", slots={"flat_id": IntSlot(0, N_FLATS_MAX)})
FLAT_IS_SKY = TokenType("flatIsSky", slots={"flag": IntSlot(0, 2)})
PLANE_DEF = TokenType(
    "planeDef",
    slots={
        "p": IntSlot(0, N_PLANES_MAX),
        "flat_id": IntSlot(0, N_FLATS_MAX),
        "is_sky": IntSlot(0, 2),
    },
)
PLANE_HEIGHT = TokenType("planeHeight")
PLANE_LIGHT = TokenType("planeLight", slots={"light": IntSlot(0, N_LIGHT_LEVELS)})
SS_FLOOR_PLANE = TokenType(
    "ssFloorPlane",
    slots={
        "s": IntSlot(0, N_SUBSECTORS_MAX),
        "p": IntSlot(0, N_PLANES_MAX),
    },
)
SS_CEILING_PLANE = TokenType(
    "ssCeilingPlane",
    slots={
        "s": IntSlot(0, N_SUBSECTORS_MAX),
        "p": IntSlot(0, N_PLANES_MAX),
    },
)

# ---------------------------------------------------------------------------
# Section 4: BEGIN
# ---------------------------------------------------------------------------

BEGIN = TokenType("begin")


PROMPT_TYPES = [
    # Shared payload
    VALUE, ANGLE_VALUE,
    # Section 1
    PLAYER_X_MARK, PLAYER_Y_MARK, PLAYER_Z_MARK, PLAYER_ANGLE_MARK,
    # Section 2: per-node
    NODE,
    NODE_PX, NODE_PY, NODE_DX, NODE_DY,
    NODE_FRONT_CHILD, NODE_BACK_CHILD,
    BBOX_TOP_FRONT, BBOX_BOT_FRONT, BBOX_LEFT_FRONT, BBOX_RIGHT_FRONT,
    BBOX_TOP_BACK, BBOX_BOT_BACK, BBOX_LEFT_BACK, BBOX_RIGHT_BACK,
    # Section 3: per-ss + per-seg
    SS, SEG,
    SEG_AX, SEG_AY, SEG_BX, SEG_BY,
    SEG_TWO_SIDED, SEG_NORMAL_ANGLE,
    SEG_FRONT_FLOOR, SEG_FRONT_CEILING,
    SEG_BACK_FLOOR, SEG_BACK_CEILING,
    SEG_MID_TEXTURE, SEG_UPPER_TEXTURE, SEG_LOWER_TEXTURE,
    SEG_EMPTY_LINE, SEG_CLOSED_DOOR,
    # Section 3b: visplane prefill
    FLAT_DEF, FLAT_IS_SKY,
    PLANE_DEF, PLANE_HEIGHT, PLANE_LIGHT,
    SS_FLOOR_PLANE, SS_CEILING_PLANE,
    # Section 4
    BEGIN,
]


# ---------------------------------------------------------------------------
# AR-loop types (ported from spec09__ref/setup.py)
# ---------------------------------------------------------------------------

NO_OP = TokenType("noOp")
THINK_SIDE = TokenType(
    "R_PointOnSide", slots={"node": IntSlot(0, N_NODES_MAX)}
)
# Side record: emitted after THINK_SIDE(N) to ferry side(N) across a
# re-embedding boundary.
SIDE_RECORD = TokenType(
    "pointOnSideResult",
    slots={
        "node": IntSlot(0, N_NODES_MAX),
        "side": IntSlot(0, 2),
    },
)
# Per-cycle markers between successive ANGLE_VALUEs in the projection
# pipeline.
WORLD_ANGLE_MARK_A = TokenType("angle1")
THETA_MARK_A = TokenType("theta1")
WORLD_ANGLE_MARK_B = TokenType("angle2")
THETA_MARK_B = TokenType("theta2")
BBOX_WORLD_ANGLE_MARK_A = TokenType("bbox.angle1")
BBOX_THETA_MARK_A = TokenType("bbox.theta1")
BBOX_WORLD_ANGLE_MARK_B = TokenType("bbox.angle2")
BBOX_THETA_MARK_B = TokenType("bbox.theta2")
BBOX_BOXPOS = TokenType("boxpos", slots={"boxpos": IntSlot(0, 12)})
BBOX_CORNER_X_MARK_A = TokenType(
    "bbox.x1", slots={"boxpos": IntSlot(0, 12)}
)
BBOX_CORNER_Y_MARK_A = TokenType(
    "bbox.y1", slots={"boxpos": IntSlot(0, 12)}
)
BBOX_CORNER_X_MARK_B = TokenType(
    "bbox.x2", slots={"boxpos": IntSlot(0, 12)}
)
BBOX_CORNER_Y_MARK_B = TokenType(
    "bbox.y2", slots={"boxpos": IntSlot(0, 12)}
)
TRAVERSE_ENTER = TokenType(
    "bspFront",
    slots={
        "node": IntSlot(0, N_NODES_MAX),
        "depth": IntSlot(0, N_DEPTH_MAX),
    },
)
TRAVERSE_BETWEEN = TokenType(
    "bspCheckBack",
    slots={
        "node": IntSlot(0, N_NODES_MAX),
        "depth": IntSlot(0, N_DEPTH_MAX),
    },
)
TRAVERSE_RETURN = TokenType(
    "bspReturn",
    slots={
        "entity_u": IntSlot(0, N_ENTITY_MAX),
        "depth": IntSlot(0, N_DEPTH_MAX),
    },
)
VISIT_SUBSECTOR = TokenType(
    "R_Subsector",
    slots={
        "s": IntSlot(0, N_SUBSECTORS_MAX),
        "depth": IntSlot(0, N_DEPTH_MAX),
    },
)
PROCESS_SEG = TokenType("R_AddLine", slots={"i": IntSlot(0, N_SEGS_MAX)})
FIND_RUN = TokenType(
    "clipScan", slots={"x": IntSlot(0, SCREEN_WIDTH + 1)}
)
BBOX_SCAN = TokenType(
    "bboxClipScan", slots={"x": IntSlot(0, SCREEN_WIDTH + 1)}
)
NEXT_SOLID_START = TokenType(
    "nextCliprangeFirst", slots={"x": IntSlot(0, SCREEN_WIDTH + 1)}
)
RUN_START = TokenType("drawsegStart")
ADVANCE_SEG = TokenType("nextSeg", slots={"i": IntSlot(0, N_SEGS_MAX)})
EMIT_X1 = TokenType(
    "drawseg.x1",
    slots={"x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_ONE_HOT_DERIVED)},
)
EMIT_X2 = TokenType(
    "drawseg.x2",
    slots={"x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_ONE_HOT_DERIVED)},
)
R_STORE_WALL_RANGE = TokenType(
    "R_StoreWallRange",
    slots={
        "i": IntSlot(0, N_SEGS_MAX),
        "x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_DERIVED),
    },
)
STORE_STOP_X = TokenType(
    "storeWall.stopX",
    slots={"x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_DERIVED)},
)
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
WALL_COLUMN = TokenType(
    "wallColumn",
    slots={"x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_DERIVED)},
)
WALL_SPAN = TokenType(
    "wallSpan",
    slots={
        "x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_ONE_HOT_DERIVED),
        "part": IntSlot(0, 3),
    },
)
CLIP_UPDATE = TokenType(
    "clipUpdate",
    slots={"x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_ONE_HOT_DERIVED)},
)
CLIP_FLOOR_UPDATE = TokenType(
    "clipFloorUpdate",
    slots={"x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_ONE_HOT_DERIVED)},
)
PLANE_MARK = TokenType(
    "planeMark",
    slots={
        "p": IntSlot(0, N_PLANES_MAX),
        "x": IntSlot(0, SCREEN_WIDTH, derived=_SCREEN_X_ONE_HOT_DERIVED),
        "kind": IntSlot(0, 2),
    },
)
SCREEN_Y_VALUE = TokenType(
    "screenY", slots={"y": IntSlot(-1, SCREEN_HEIGHT + 1)}
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
    NEXT_SOLID_START,
    RUN_START,
    ADVANCE_SEG,
    EMIT_X1,
    EMIT_X2,
    R_STORE_WALL_RANGE,
    STORE_STOP_X,
    DRAWSEG_META,
    DRAWSEG_SCALE1_DEN,
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE2_DEN,
    DRAWSEG_SCALE2,
    DRAWSEG_SCALESTEP_DEN,
    DRAWSEG_SCALESTEP,
    DRAWSEG_BSILHEIGHT,
    DRAWSEG_TSILHEIGHT,
    WALL_COLUMN,
    WALL_SPAN,
    CLIP_UPDATE,
    CLIP_FLOOR_UPDATE,
    PLANE_MARK,
    SCREEN_Y_VALUE,
    DONE,
]


VOCAB_TYPES = PROMPT_TYPES + AR_TYPES
