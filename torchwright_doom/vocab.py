"""The renderer's token vocabulary — prompt side.

Token types emitted by :func:`torchwright_doom.prompt.build.build_prompt`.
The autoregressive stage (forward / extract) will introduce additional
types for traversal, drawseg fragments, plane marks, etc. — those are
added here as they land.

Capacity constants (``N_NODES_MAX`` etc.) bound the ``IntSlot`` ranges;
bumping them is a vocab-scale change that propagates to the embedding
table.
"""

from __future__ import annotations

from .tokens import FloatSlot, IntSlot, TokenType


ANGLE_BAM = 8192
BACK_HEIGHT_SENTINEL = -4096.0

N_NODES_MAX = 64
N_SUBSECTORS_MAX = 64
N_SEGS_MAX = 128
N_ENTITY_MAX = N_NODES_MAX + N_SUBSECTORS_MAX
N_PLANES_MAX = 32
N_FLATS_MAX = 32
N_LIGHT_LEVELS = 256


VALUE = TokenType("value", slots={"v": FloatSlot(-4096.0, 4096.0)})
ANGLE_VALUE = TokenType(
    "angleValue",
    slots={"angle": IntSlot(-ANGLE_BAM // 2, ANGLE_BAM // 2)},
)

PLAYER_X_MARK = TokenType("viewx")
PLAYER_Y_MARK = TokenType("viewy")
PLAYER_Z_MARK = TokenType("viewz")
PLAYER_ANGLE_MARK = TokenType("viewangle")

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

BEGIN = TokenType("begin")


PROMPT_TYPES = [
    VALUE, ANGLE_VALUE,
    PLAYER_X_MARK, PLAYER_Y_MARK, PLAYER_Z_MARK, PLAYER_ANGLE_MARK,
    NODE,
    NODE_PX, NODE_PY, NODE_DX, NODE_DY,
    NODE_FRONT_CHILD, NODE_BACK_CHILD,
    BBOX_TOP_FRONT, BBOX_BOT_FRONT, BBOX_LEFT_FRONT, BBOX_RIGHT_FRONT,
    BBOX_TOP_BACK, BBOX_BOT_BACK, BBOX_LEFT_BACK, BBOX_RIGHT_BACK,
    SS, SEG,
    SEG_AX, SEG_AY, SEG_BX, SEG_BY,
    SEG_TWO_SIDED, SEG_NORMAL_ANGLE,
    SEG_FRONT_FLOOR, SEG_FRONT_CEILING,
    SEG_BACK_FLOOR, SEG_BACK_CEILING,
    SEG_MID_TEXTURE, SEG_UPPER_TEXTURE, SEG_LOWER_TEXTURE,
    SEG_EMPTY_LINE, SEG_CLOSED_DOOR,
    FLAT_DEF, FLAT_IS_SKY,
    PLANE_DEF, PLANE_HEIGHT, PLANE_LIGHT,
    SS_FLOOR_PLANE, SS_CEILING_PLANE,
    BEGIN,
]
