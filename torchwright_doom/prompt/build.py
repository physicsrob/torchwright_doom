"""Build a prompt token sequence from a :class:`MapData` + :class:`GameState`.

The prompt is what the transformer sees: a flat sequence of typed tokens
encoding the scene's geometry and the player's per-frame state. Section
order mirrors the sandbox ``get_prefill``: player state -> per-node block
-> per-subsector + per-seg block -> visplane defs + per-subsector plane
refs -> ``BEGIN`` marker.

Every continuous float payload is emitted through :func:`prefill_value`,
which range-encodes it into the ``VALUE`` carrier's ``[-1, 1]`` space
(the per-site ``ValueRange`` mirrors ``get_prefill``). Flats are carried
by ``PLANE_DEF.flat_id`` (their pixels are weight-side); there are no
standalone flat tokens.
"""

from __future__ import annotations

from ..asset_config import WALL_TEXTURE_ID_BY_NAME
from ..doom_lighting import (
    doom_wall_light_static,
    doom_wall_orientation_light_bias,
)
from ..tokens import Token
from ..vocab import (
    ANGLE_BAM,
    ANGLE_VALUE,
    BACK_HEIGHT_SENTINEL,
    BBOX_BOT_BACK,
    BBOX_BOT_FRONT,
    BBOX_LEFT_BACK,
    BBOX_LEFT_FRONT,
    BBOX_RIGHT_BACK,
    BBOX_RIGHT_FRONT,
    BBOX_TOP_BACK,
    BBOX_TOP_FRONT,
    BEGIN,
    N_NODES_MAX,
    NODE,
    NODE_BACK_CHILD,
    NODE_DX,
    NODE_DY,
    NODE_FRONT_CHILD,
    NODE_PX,
    NODE_PY,
    PLANE_DEF,
    PLANE_HEIGHT,
    PLANE_LIGHT,
    PLAYER_ANGLE_MARK,
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    PLAYER_Z_MARK,
    SEG,
    SEG_AX,
    SEG_AY,
    SEG_BACK_CEILING,
    SEG_BACK_FLOOR,
    SEG_BX,
    SEG_BY,
    SEG_CLOSED_DOOR,
    SEG_EMPTY_LINE,
    SEG_FRONT_CEILING,
    SEG_FRONT_FLOOR,
    SEG_LIGHT_STATIC,
    SEG_LOWER_TEXTURE,
    SEG_MID_TEXTURE,
    SEG_NORMAL_ANGLE,
    SEG_PEGGING,
    SEG_ROWOFFSET,
    SEG_TWO_SIDED,
    SEG_UPPER_TEXTURE,
    SS,
    SS_CEILING_PLANE,
    SS_FLOOR_PLANE,
    ValueRange,
    prefill_value,
)
from .geometry import Segment, bake_segments
from .plane_tables import build_plane_tables
from .types import GameState, MapData, SUBSECTOR_FLAG


_GAMESTATE_TO_BAM = ANGLE_BAM // 256  # 32 at ANGLE_BAM=8192
_BAM_HALF = ANGLE_BAM // 2

ML_DONTPEGTOP = 0x0008
ML_DONTPEGBOTTOM = 0x0010


def _player_angle_signed(angle_256: int) -> int:
    unsigned = angle_256 * _GAMESTATE_TO_BAM
    return unsigned if unsigned < _BAM_HALF else unsigned - ANGLE_BAM


def _seg_normal_angle(raw_seg_angle: int) -> int:
    """Convert a seg's 16-bit BAM angle to the signed N-bit BAM the vocab uses."""
    seg_angle_n = round(raw_seg_angle * ANGLE_BAM / 65536)
    wrapped = (seg_angle_n + ANGLE_BAM // 4) % ANGLE_BAM
    return wrapped if wrapped < _BAM_HALF else wrapped - ANGLE_BAM


def _child_unified(encoded: int) -> int:
    if encoded & SUBSECTOR_FLAG:
        return N_NODES_MAX + (encoded & ~SUBSECTOR_FLAG)
    return encoded


def _has_texture(name: str) -> bool:
    return bool(name and name != "-")


def _texture_id(name: str, name_to_id: dict[str, int]) -> int:
    if not _has_texture(name):
        return 0
    return name_to_id.get(name.upper(), 0)


def _back_floor(seg: Segment) -> float:
    return seg.back_floor if seg.back_floor is not None else BACK_HEIGHT_SENTINEL


def _back_ceiling(seg: Segment) -> float:
    return seg.back_ceiling if seg.back_ceiling is not None else BACK_HEIGHT_SENTINEL


def _is_closed(seg: Segment) -> bool:
    return (
        seg.is_two_sided
        and seg.back_floor is not None
        and seg.back_ceiling is not None
        and (
            seg.back_ceiling <= seg.front_floor
            or seg.back_floor >= seg.front_ceiling
        )
    )


def _is_empty_line(md: MapData, seg_idx: int) -> bool:
    raw_seg = md.segs[seg_idx]
    linedef = md.linedefs[raw_seg.linedef]
    front_sd_idx = linedef.front_sidedef if raw_seg.side == 0 else linedef.back_sidedef
    back_sd_idx = linedef.back_sidedef if raw_seg.side == 0 else linedef.front_sidedef
    if front_sd_idx < 0 or back_sd_idx < 0:
        return False

    front_sd = md.sidedefs[front_sd_idx]
    back_sd = md.sidedefs[back_sd_idx]
    front_sector = md.sectors[front_sd.sector]
    back_sector = md.sectors[back_sd.sector]
    return (
        front_sector.floor_h == back_sector.floor_h
        and front_sector.ceiling_h == back_sector.ceiling_h
        and front_sector.floor_tex == back_sector.floor_tex
        and front_sector.ceiling_tex == back_sector.ceiling_tex
        and front_sector.light == back_sector.light
        and not _has_texture(front_sd.middle)
    )


def _seg_front_sector(md: MapData, seg_idx: int):
    raw_seg = md.segs[seg_idx]
    linedef = md.linedefs[raw_seg.linedef]
    front_sd_idx = linedef.front_sidedef if raw_seg.side == 0 else linedef.back_sidedef
    return md.sectors[md.sidedefs[front_sd_idx].sector]


def _seg_light_static(md: MapData, seg_idx: int, seg: Segment) -> int:
    front_sector = _seg_front_sector(md, seg_idx)
    orientation_bias = doom_wall_orientation_light_bias(
        seg.ax, seg.ay, seg.bx, seg.by
    )
    return doom_wall_light_static(
        front_sector.light, orientation_bias=orientation_bias
    )


def _seg_pegging_and_offset(md: MapData, seg_idx: int) -> tuple[int, int, int]:
    raw_seg = md.segs[seg_idx]
    linedef = md.linedefs[raw_seg.linedef]
    sd_idx = linedef.front_sidedef if raw_seg.side == 0 else linedef.back_sidedef
    y_offset = md.sidedefs[sd_idx].y_offset if sd_idx >= 0 else 0
    return (linedef.flags, y_offset, raw_seg.side)


def build_prompt(md: MapData, state: GameState) -> list[Token]:
    segments = bake_segments(md)
    plane_tables = build_plane_tables(md)
    tokens: list[Token] = []
    name_to_id = {"-": 0, "": 0, **WALL_TEXTURE_ID_BY_NAME}

    tokens.append(Token(PLAYER_X_MARK))
    tokens.append(prefill_value(ValueRange.R1, state.x))
    tokens.append(Token(PLAYER_Y_MARK))
    tokens.append(prefill_value(ValueRange.R1, state.y))
    tokens.append(Token(PLAYER_Z_MARK))
    tokens.append(prefill_value(ValueRange.R3, state.viewz))
    tokens.append(Token(PLAYER_ANGLE_MARK))
    tokens.append(Token(ANGLE_VALUE, {"angle": _player_angle_signed(state.angle)}))

    for j, node in enumerate(md.nodes):
        tokens.append(Token(NODE, {"j": j}))
        tokens.append(Token(NODE_PX))
        tokens.append(prefill_value(ValueRange.R1, node.px))
        tokens.append(Token(NODE_PY))
        tokens.append(prefill_value(ValueRange.R1, node.py))
        tokens.append(Token(NODE_DX))
        tokens.append(prefill_value(ValueRange.R2, node.dx))
        tokens.append(Token(NODE_DY))
        tokens.append(prefill_value(ValueRange.R2, node.dy))
        tokens.append(
            Token(NODE_FRONT_CHILD, {"child_u": _child_unified(node.front_child)})
        )
        tokens.append(
            Token(NODE_BACK_CHILD, {"child_u": _child_unified(node.back_child)})
        )
        ft, fb, fl, fr = node.front_bbox
        bt, bb, bl, br = node.back_bbox
        tokens.append(Token(BBOX_TOP_FRONT))
        tokens.append(prefill_value(ValueRange.R0, ft))
        tokens.append(Token(BBOX_BOT_FRONT))
        tokens.append(prefill_value(ValueRange.R0, fb))
        tokens.append(Token(BBOX_LEFT_FRONT))
        tokens.append(prefill_value(ValueRange.R0, fl))
        tokens.append(Token(BBOX_RIGHT_FRONT))
        tokens.append(prefill_value(ValueRange.R0, fr))
        tokens.append(Token(BBOX_TOP_BACK))
        tokens.append(prefill_value(ValueRange.R0, bt))
        tokens.append(Token(BBOX_BOT_BACK))
        tokens.append(prefill_value(ValueRange.R0, bb))
        tokens.append(Token(BBOX_LEFT_BACK))
        tokens.append(prefill_value(ValueRange.R0, bl))
        tokens.append(Token(BBOX_RIGHT_BACK))
        tokens.append(prefill_value(ValueRange.R0, br))

    for s, sub in enumerate(md.subsectors):
        tokens.append(Token(SS, {"s": s}))
        for k in range(sub.seg_count):
            i = sub.first_seg + k
            seg = segments[i]
            tokens.append(
                Token(SEG, {"i": i, "is_first_of_ss": 1 if k == 0 else 0})
            )
            tokens.append(Token(SEG_AX))
            tokens.append(prefill_value(ValueRange.R1, seg.ax))
            tokens.append(Token(SEG_AY))
            tokens.append(prefill_value(ValueRange.R1, seg.ay))
            tokens.append(Token(SEG_BX))
            tokens.append(prefill_value(ValueRange.R1, seg.bx))
            tokens.append(Token(SEG_BY))
            tokens.append(prefill_value(ValueRange.R1, seg.by))
            tokens.append(
                Token(SEG_TWO_SIDED, {"flag": 1 if seg.is_two_sided else 0})
            )
            tokens.append(Token(SEG_NORMAL_ANGLE))
            tokens.append(
                Token(ANGLE_VALUE, {"angle": _seg_normal_angle(md.segs[i].angle)})
            )
            tokens.append(Token(SEG_FRONT_FLOOR))
            tokens.append(prefill_value(ValueRange.R3, seg.front_floor))
            tokens.append(Token(SEG_FRONT_CEILING))
            tokens.append(prefill_value(ValueRange.R3, seg.front_ceiling))
            tokens.append(Token(SEG_BACK_FLOOR))
            tokens.append(prefill_value(ValueRange.R4, _back_floor(seg)))
            tokens.append(Token(SEG_BACK_CEILING))
            tokens.append(prefill_value(ValueRange.R4, _back_ceiling(seg)))
            tokens.append(
                Token(SEG_MID_TEXTURE, {
                    "tex_id": _texture_id(seg.middle_texture_name, name_to_id),
                })
            )
            tokens.append(
                Token(SEG_UPPER_TEXTURE, {
                    "tex_id": _texture_id(seg.upper_texture_name, name_to_id),
                })
            )
            tokens.append(
                Token(SEG_LOWER_TEXTURE, {
                    "tex_id": _texture_id(seg.lower_texture_name, name_to_id),
                })
            )
            tokens.append(
                Token(SEG_LIGHT_STATIC, {"light": _seg_light_static(md, i, seg)})
            )
            tokens.append(
                Token(SEG_EMPTY_LINE, {"flag": 1 if _is_empty_line(md, i) else 0})
            )
            tokens.append(
                Token(SEG_CLOSED_DOOR, {"flag": 1 if _is_closed(seg) else 0})
            )
            flags, rowoffset, _side = _seg_pegging_and_offset(md, i)
            tokens.append(
                Token(SEG_PEGGING, {
                    "dontpegtop": 1 if flags & ML_DONTPEGTOP else 0,
                    "dontpegbottom": 1 if flags & ML_DONTPEGBOTTOM else 0,
                })
            )
            tokens.append(Token(SEG_ROWOFFSET))
            tokens.append(prefill_value(ValueRange.R3, float(rowoffset)))

    for plane in plane_tables.planes:
        tokens.append(
            Token(PLANE_DEF, {"p": plane.plane_id, "flat_id": plane.flat_id})
        )
        tokens.append(Token(PLANE_HEIGHT))
        tokens.append(prefill_value(ValueRange.R3, plane.height))
        tokens.append(Token(PLANE_LIGHT, {"light": plane.light}))

    for s, info in enumerate(plane_tables.subsectors):
        if info is None:
            continue
        if info.floor_plane_id is not None:
            tokens.append(
                Token(SS_FLOOR_PLANE, {"s": s, "p": info.floor_plane_id})
            )
        if info.ceiling_plane_id is not None:
            tokens.append(
                Token(SS_CEILING_PLANE, {"s": s, "p": info.ceiling_plane_id})
            )

    tokens.append(Token(BEGIN))
    return tokens
