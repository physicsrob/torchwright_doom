"""Build a prompt token sequence from a :class:`MapData` + :class:`GameState`.

The prompt is what the transformer sees: a flat sequence of typed tokens
encoding the scene's geometry and the player's per-frame state. Section
order mirrors the original ``get_prefill``: player state -> per-node block
-> per-subsector + per-seg block -> visplane defs + per-subsector plane
refs -> ``BEGIN`` marker.

Every continuous float payload is emitted through :func:`prefill_value`,
which range-encodes it into the ``VALUE`` carrier's ``[-1, 1]`` space
(the per-site ``ValueRange`` mirrors ``get_prefill``). Flats are carried
by ``PLANE_DEF.flat_id`` (their pixels are weight-side); there are no
standalone flat tokens.
"""

from __future__ import annotations

from ..model.asset_config import DEFAULT_ASSET_CONFIG, AssetConfig
from ..model.doom_lighting import (
    doom_wall_light_static,
    doom_wall_orientation_light_bias,
)
from ..model.marker_ranges import MARKER_RANGE
from ..model.tokens import Token, TokenType
from ..model.vocab import (
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
    BOS,
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
    prefill_value,
)
from .geometry import Segment, bake_segments
from .plane_tables import build_plane_tables
from .types import GameState, MapData, SUBSECTOR_FLAG

_GAMESTATE_TO_BAM = ANGLE_BAM // 256  # 32 at ANGLE_BAM=8192
_BAM_HALF = ANGLE_BAM // 2

ML_DONTPEGTOP = 0x0008
ML_DONTPEGBOTTOM = 0x0010


def _marked(tokens: list[Token], marker: TokenType, value: float) -> None:
    """Append a marker token then its ``VALUE`` carrier — the prefill's basic
    marker->value pair (see GLOSSARY.md 'marker' / 'carrier'). The marker's
    range is looked up in the shared ``marker_ranges.MARKER_RANGE`` table (the
    single source shared with the drafter and the tokenizer surface), so the
    range can't drift between the code that *writes* the stream and the code
    that *reads* it."""
    tokens.append(Token(marker))
    tokens.append(prefill_value(MARKER_RANGE[marker], value))


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
        and (seg.back_ceiling <= seg.front_floor or seg.back_floor >= seg.front_ceiling)
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
    orientation_bias = doom_wall_orientation_light_bias(seg.ax, seg.ay, seg.bx, seg.by)
    return doom_wall_light_static(front_sector.light, orientation_bias=orientation_bias)


def _seg_pegging_and_offset(md: MapData, seg_idx: int) -> tuple[int, int, int]:
    raw_seg = md.segs[seg_idx]
    linedef = md.linedefs[raw_seg.linedef]
    sd_idx = linedef.front_sidedef if raw_seg.side == 0 else linedef.back_sidedef
    y_offset = md.sidedefs[sd_idx].y_offset if sd_idx >= 0 else 0
    return (linedef.flags, y_offset, raw_seg.side)


def build_prompt(
    md: MapData, state: GameState, asset_config: AssetConfig | None = None
) -> list[Token]:
    asset_config = asset_config or DEFAULT_ASSET_CONFIG
    segments = bake_segments(md)
    plane_tables = build_plane_tables(md, flat_ids=asset_config.flat_id_by_name)
    tokens: list[Token] = []
    name_to_id = {"-": 0, "": 0, **asset_config.wall_id_by_name}

    # Position 0: a true beginning-of-sequence anchor. Inert (dispatches to
    # NO_OP); every downstream read is content-addressed or uses relative
    # offsets, so this +1 shift of the prompt is harmless. BEGIN still closes
    # the prompt below as the prompt->AR boundary.
    tokens.append(Token(BOS))

    _marked(tokens, PLAYER_X_MARK, state.x)
    _marked(tokens, PLAYER_Y_MARK, state.y)
    _marked(tokens, PLAYER_Z_MARK, state.viewz)
    # Angle uses the ANGLE_VALUE carrier (not prefill_value), so it stays explicit.
    tokens.append(Token(PLAYER_ANGLE_MARK))
    tokens.append(Token(ANGLE_VALUE, {"angle": _player_angle_signed(state.angle)}))

    for j, node in enumerate(md.nodes):
        tokens.append(Token(NODE, {"j": j}))
        _marked(tokens, NODE_PX, node.px)
        _marked(tokens, NODE_PY, node.py)
        _marked(tokens, NODE_DX, node.dx)
        _marked(tokens, NODE_DY, node.dy)
        tokens.append(
            Token(NODE_FRONT_CHILD, {"child_u": _child_unified(node.front_child)})
        )
        tokens.append(
            Token(NODE_BACK_CHILD, {"child_u": _child_unified(node.back_child)})
        )
        # Front then back bbox edges, each marker paired with its edge value
        # (all ValueRange.R0). Order is load-bearing — it mirrors get_prefill.
        ft, fb, fl, fr = node.front_bbox
        bt, bb, bl, br = node.back_bbox
        for bbox_marker, edge in (
            (BBOX_TOP_FRONT, ft),
            (BBOX_BOT_FRONT, fb),
            (BBOX_LEFT_FRONT, fl),
            (BBOX_RIGHT_FRONT, fr),
            (BBOX_TOP_BACK, bt),
            (BBOX_BOT_BACK, bb),
            (BBOX_LEFT_BACK, bl),
            (BBOX_RIGHT_BACK, br),
        ):
            _marked(tokens, bbox_marker, edge)

    for s, sub in enumerate(md.subsectors):
        tokens.append(Token(SS, {"s": s}))
        for k in range(sub.seg_count):
            i = sub.first_seg + k
            # seg is the BAKED Segment (geometry.py); md.segs[i] below is the raw
            # WAD Seg (types.py) — two different types reached in this loop body.
            seg = segments[i]
            tokens.append(Token(SEG, {"i": i, "is_first_of_ss": 1 if k == 0 else 0}))
            _marked(tokens, SEG_AX, seg.ax)
            _marked(tokens, SEG_AY, seg.ay)
            _marked(tokens, SEG_BX, seg.bx)
            _marked(tokens, SEG_BY, seg.by)
            tokens.append(Token(SEG_TWO_SIDED, {"flag": 1 if seg.is_two_sided else 0}))
            tokens.append(Token(SEG_NORMAL_ANGLE))
            tokens.append(
                Token(ANGLE_VALUE, {"angle": _seg_normal_angle(md.segs[i].angle)})
            )
            _marked(tokens, SEG_FRONT_FLOOR, seg.front_floor)
            _marked(tokens, SEG_FRONT_CEILING, seg.front_ceiling)
            _marked(tokens, SEG_BACK_FLOOR, _back_floor(seg))
            _marked(tokens, SEG_BACK_CEILING, _back_ceiling(seg))
            tokens.append(
                Token(
                    SEG_MID_TEXTURE,
                    {
                        "tex_id": _texture_id(seg.middle_texture_name, name_to_id),
                    },
                )
            )
            tokens.append(
                Token(
                    SEG_UPPER_TEXTURE,
                    {
                        "tex_id": _texture_id(seg.upper_texture_name, name_to_id),
                    },
                )
            )
            tokens.append(
                Token(
                    SEG_LOWER_TEXTURE,
                    {
                        "tex_id": _texture_id(seg.lower_texture_name, name_to_id),
                    },
                )
            )
            tokens.append(
                Token(SEG_LIGHT_STATIC, {"light": _seg_light_static(md, i, seg)})
            )
            tokens.append(
                Token(SEG_EMPTY_LINE, {"flag": 1 if _is_empty_line(md, i) else 0})
            )
            tokens.append(Token(SEG_CLOSED_DOOR, {"flag": 1 if _is_closed(seg) else 0}))
            flags, rowoffset, _side = _seg_pegging_and_offset(md, i)
            tokens.append(
                Token(
                    SEG_PEGGING,
                    {
                        "dontpegtop": 1 if flags & ML_DONTPEGTOP else 0,
                        "dontpegbottom": 1 if flags & ML_DONTPEGBOTTOM else 0,
                    },
                )
            )
            _marked(tokens, SEG_ROWOFFSET, float(rowoffset))

    for plane in plane_tables.planes:
        tokens.append(Token(PLANE_DEF, {"p": plane.plane_id, "flat_id": plane.flat_id}))
        _marked(tokens, PLANE_HEIGHT, plane.height)
        tokens.append(Token(PLANE_LIGHT, {"light": plane.light}))

    for s, info in enumerate(plane_tables.subsectors):
        if info is None:
            continue
        # Both plane ids are always set (int) once info exists — no per-field guard.
        tokens.append(Token(SS_FLOOR_PLANE, {"s": s, "p": info.floor_plane_id}))
        tokens.append(Token(SS_CEILING_PLANE, {"s": s, "p": info.ceiling_plane_id}))

    tokens.append(Token(BEGIN))
    return tokens
