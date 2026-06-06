"""Declarative ownership table for the renderer token protocol.

The graph still builds branch nodes in the owner modules. This registry owns the
cross-cutting metadata that otherwise drifts: vocab phase, inert/replay policy,
payload-marker groups, and current-token dispatch wiring.

Ported from ``doom_sandbox/implementation/protocol_registry.py``. The only
changes from the sandbox source are the import block (``TokenType`` from the
real ``tokens``; the token declarations from ``vocab``) and a whitespace-clean
``render_protocol_table`` (empty payload rows no longer end in a trailing
``| ``). The entries, builders, and exports are a line-for-line port.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .tokens import TokenType
from .vocab import (
    ADVANCE_SEG,
    ANGLE_VALUE,
    BBOX_BOT_BACK,
    BBOX_BOT_FRONT,
    BBOX_BOXPOS,
    BBOX_CORNER_X_MARK_A,
    BBOX_CORNER_X_MARK_B,
    BBOX_CORNER_Y_MARK_A,
    BBOX_CORNER_Y_MARK_B,
    BBOX_LEFT_BACK,
    BBOX_LEFT_FRONT,
    BBOX_RIGHT_BACK,
    BBOX_RIGHT_FRONT,
    BBOX_SCAN,
    BBOX_THETA_MARK_A,
    BBOX_THETA_MARK_B,
    BBOX_TOP_BACK,
    BBOX_TOP_FRONT,
    BBOX_WORLD_ANGLE_MARK_A,
    BBOX_WORLD_ANGLE_MARK_B,
    BEGIN,
    CLIP_UPDATE,
    DONE,
    DRAWSEG_BSILHEIGHT,
    DRAWSEG_META,
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE1_DEN,
    DRAWSEG_SCALE2,
    DRAWSEG_SCALE2_DEN,
    DRAWSEG_SCALESTEP,
    DRAWSEG_SCALESTEP_DEN,
    DRAWSEG_TSILHEIGHT,
    DRAWSEG_U_PHASE,
    DRAW_PLANES_BEGIN,
    EMIT_X2,
    FIND_RUN,
    FLAT_NEXT_PLANE,
    FLAT_NEXT_VP,
    FLAT_VISPLANE_BEGIN,
    MAKE_SPANS_COL,
    NODE,
    NODE_BACK_CHILD,
    NODE_DX,
    NODE_DY,
    NODE_FRONT_CHILD,
    NODE_PX,
    NODE_PY,
    NO_OP,
    PIXEL,
    PLANE_DEF,
    PLANE_HEIGHT,
    PLANE_LIGHT,
    PLANE_MARK,
    PLAYER_ANGLE_MARK,
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    PLAYER_Z_MARK,
    PROCESS_SEG,
    R_CHECK_PLANE,
    R_CHECK_PLANE_RESULT,
    R_STORE_WALL_RANGE,
    SCREEN_RANGE,
    SCREEN_Y_VALUE,
    SEG,
    SEG_AX,
    SEG_AY,
    SEG_BACK_CEILING,
    SEG_BACK_FLOOR,
    SEG_BX,
    SEG_BY,
    SEG_CLOSED_DOOR,
    SEG_DC_TMID_LOWER,
    SEG_DC_TMID_MID,
    SEG_DC_TMID_UPPER,
    SEG_EMPTY_LINE,
    SEG_FRONT_CEILING,
    SEG_FRONT_FLOOR,
    SEG_KPART,
    SEG_LIGHT_STATIC,
    SEG_LOWER_TEXTURE,
    SEG_MID_TEXTURE,
    SEG_NORMAL_ANGLE,
    SEG_PEGGING,
    SEG_ROWOFFSET,
    SEG_TWO_SIDED,
    SEG_UPPER_TEXTURE,
    SET_CURSOR_DIRECTION_X,
    SET_CURSOR_DIRECTION_Y,
    SET_CURSOR_X,
    SET_CURSOR_Y,
    SIDE_RECORD,
    SPAN_CLOSE_SLOT,
    SPAN_ROW,
    SS,
    SS_CEILING_PLANE,
    SS_FLOOR_PLANE,
    THETA_MARK_A,
    THETA_MARK_B,
    THINK_SIDE,
    TRAVERSE_BETWEEN,
    TRAVERSE_ENTER,
    TRAVERSE_RETURN,
    VALUE,
    VISIT_SUBSECTOR,
    WALL_COL_U,
    WALL_SPAN_META,
    WORLD_ANGLE_MARK_A,
    WORLD_ANGLE_MARK_B,
)


@dataclass(frozen=True)
class ProtocolEntry:
    token: TokenType
    vocab_phase: str
    role: str
    owner: str
    predicate: str
    branch: str
    dispatch_order: int
    prefill_replay: str = "never"
    payload_group: str | None = None


@dataclass(frozen=True)
class DispatchTransition:
    predicate: str
    branch: str
    tokens: tuple[TokenType, ...]


@dataclass(frozen=True)
class TokenCheckPredicate:
    """A dispatch predicate that matches exactly one token type, so its
    implementation is precisely ``token.check(input_vec)`` (one E8 is_type
    test). Predicates shared by several token types are excluded —
    :func:`_build_token_check_predicates` keeps only the single-type groups."""

    predicate: str
    token: TokenType


def _inert(
    token: TokenType,
    *,
    phase: str = "prefill",
    owner: str = "scene",
    payload_group: str | None = None,
) -> ProtocolEntry:
    return ProtocolEntry(
        token=token,
        vocab_phase=phase,
        role="inert",
        owner=owner,
        predicate="is_inert_non_payload",
        branch="no_op",
        dispatch_order=0,
        prefill_replay="input" if phase == "prefill" else "never",
        payload_group=payload_group,
    )


def _carrier(
    token: TokenType,
    *,
    predicate: str,
    branch: str,
    order: int,
) -> ProtocolEntry:
    return ProtocolEntry(
        token=token,
        vocab_phase="prefill",
        role="carrier",
        owner="payload_router",
        predicate=predicate,
        branch=branch,
        dispatch_order=order,
        prefill_replay="scene_payload",
    )


def _branch(
    token: TokenType,
    *,
    predicate: str,
    branch: str,
    order: int,
    owner: str,
    phase: str = "ar",
    role: str = "branch",
    payload_group: str | None = None,
) -> ProtocolEntry:
    return ProtocolEntry(
        token=token,
        vocab_phase=phase,
        role=role,
        owner=owner,
        predicate=predicate,
        branch=branch,
        dispatch_order=order,
        payload_group=payload_group,
    )


PROTOCOL_ENTRIES: tuple[ProtocolEntry, ...] = (
    # Shared carrier token types. They are placed in the prefill vocab block,
    # but are routed by previous-token context in both prefill and AR positions.
    _carrier(VALUE, predicate="is_value", branch="value", order=24),
    _carrier(ANGLE_VALUE, predicate="is_angle_value", branch="angle", order=25),
    # Scene/player prefill rows.
    _inert(PLAYER_X_MARK, payload_group="scene_value"),
    _inert(PLAYER_Y_MARK, payload_group="scene_value"),
    _inert(PLAYER_Z_MARK, payload_group="scene_value"),
    _inert(PLAYER_ANGLE_MARK, payload_group="scene_angle"),
    # Node prefill rows.
    _inert(NODE),
    _inert(NODE_PX, payload_group="scene_value"),
    _inert(NODE_PY, payload_group="scene_value"),
    _inert(NODE_DX, payload_group="scene_value"),
    _inert(NODE_DY, payload_group="scene_value"),
    _inert(NODE_FRONT_CHILD),
    _inert(NODE_BACK_CHILD),
    _inert(BBOX_TOP_FRONT, payload_group="scene_value"),
    _inert(BBOX_BOT_FRONT, payload_group="scene_value"),
    _inert(BBOX_LEFT_FRONT, payload_group="scene_value"),
    _inert(BBOX_RIGHT_FRONT, payload_group="scene_value"),
    _inert(BBOX_TOP_BACK, payload_group="scene_value"),
    _inert(BBOX_BOT_BACK, payload_group="scene_value"),
    _inert(BBOX_LEFT_BACK, payload_group="scene_value"),
    _inert(BBOX_RIGHT_BACK, payload_group="scene_value"),
    # Subsector and seg prefill rows.
    _inert(SS),
    _inert(SEG),
    _inert(SEG_AX, payload_group="scene_value"),
    _inert(SEG_AY, payload_group="scene_value"),
    _inert(SEG_BX, payload_group="scene_value"),
    _inert(SEG_BY, payload_group="scene_value"),
    _inert(SEG_TWO_SIDED),
    _inert(SEG_NORMAL_ANGLE, payload_group="scene_angle"),
    _inert(SEG_FRONT_FLOOR, payload_group="scene_value"),
    _inert(SEG_FRONT_CEILING, payload_group="scene_value"),
    _inert(SEG_BACK_FLOOR, payload_group="scene_value"),
    _inert(SEG_BACK_CEILING, payload_group="scene_value"),
    _inert(SEG_MID_TEXTURE),
    _inert(SEG_UPPER_TEXTURE),
    _inert(SEG_LOWER_TEXTURE),
    _inert(SEG_LIGHT_STATIC),
    _inert(SEG_EMPTY_LINE),
    _inert(SEG_CLOSED_DOOR),
    _inert(SEG_PEGGING),
    _inert(SEG_ROWOFFSET, payload_group="scene_value"),
    # Visplane prefill rows.
    _inert(PLANE_DEF),
    _inert(PLANE_HEIGHT, payload_group="scene_value"),
    _inert(PLANE_LIGHT),
    _inert(SS_FLOOR_PLANE),
    _inert(SS_CEILING_PLANE),
    # Final prefill marker. Its prediction seeds the AR loop, so it is not replayed.
    _branch(
        BEGIN,
        phase="prefill",
        role="begin",
        owner="main",
        predicate="is_begin",
        branch="begin",
        order=1,
    ),
    # AR protocol rows.
    _inert(NO_OP, phase="ar", owner="main"),
    _branch(
        THINK_SIDE,
        owner="traversal",
        predicate="is_think_side",
        branch="think",
        order=2,
    ),
    _branch(
        SIDE_RECORD,
        owner="traversal",
        predicate="is_side_record",
        branch="side_record",
        order=3,
    ),
    _branch(
        TRAVERSE_ENTER, owner="traversal", predicate="is_enter", branch="enter", order=4
    ),
    _branch(
        TRAVERSE_BETWEEN,
        owner="traversal",
        predicate="is_between",
        branch="between",
        order=5,
    ),
    _branch(
        TRAVERSE_RETURN,
        owner="traversal",
        predicate="is_return",
        branch="return_",
        order=6,
    ),
    _branch(
        VISIT_SUBSECTOR,
        owner="projection",
        predicate="is_visit_subsector",
        branch="visit",
        order=7,
    ),
    _branch(
        PROCESS_SEG,
        owner="projection",
        predicate="is_process_seg",
        branch="process",
        order=8,
    ),
    _branch(
        FIND_RUN,
        owner="projection",
        predicate="is_find_run",
        branch="find_run",
        order=9,
    ),
    _branch(
        WORLD_ANGLE_MARK_A,
        owner="projection",
        predicate="is_world_a_mark",
        branch="world_a",
        order=10,
        payload_group="projection_angle",
    ),
    _branch(
        THETA_MARK_A,
        owner="projection",
        predicate="is_theta_a_mark",
        branch="theta_a",
        order=11,
        payload_group="projection_angle",
    ),
    _branch(
        WORLD_ANGLE_MARK_B,
        owner="projection",
        predicate="is_world_b_mark",
        branch="world_b",
        order=12,
        payload_group="projection_angle",
    ),
    _branch(
        THETA_MARK_B,
        owner="projection",
        predicate="is_theta_b_mark",
        branch="theta_b",
        order=13,
        payload_group="projection_angle",
    ),
    _branch(
        BBOX_BOXPOS,
        owner="traversal",
        predicate="is_bbox_boxpos",
        branch="bbox_boxpos",
        order=14,
    ),
    _branch(
        BBOX_CORNER_X_MARK_A,
        owner="traversal",
        predicate="is_bbox_corner_x_a_mark",
        branch="bbox_corner_x_a",
        order=15,
        payload_group="bbox_value",
    ),
    _branch(
        BBOX_CORNER_Y_MARK_A,
        owner="traversal",
        predicate="is_bbox_corner_y_a_mark",
        branch="bbox_corner_y_a",
        order=16,
        payload_group="bbox_value",
    ),
    _branch(
        BBOX_CORNER_X_MARK_B,
        owner="traversal",
        predicate="is_bbox_corner_x_b_mark",
        branch="bbox_corner_x_b",
        order=17,
        payload_group="bbox_value",
    ),
    _branch(
        BBOX_CORNER_Y_MARK_B,
        owner="traversal",
        predicate="is_bbox_corner_y_b_mark",
        branch="bbox_corner_y_b",
        order=18,
        payload_group="bbox_value",
    ),
    _branch(
        BBOX_WORLD_ANGLE_MARK_A,
        owner="traversal",
        predicate="is_bbox_world_a_mark",
        branch="bbox_world_a",
        order=19,
        payload_group="bbox_angle",
    ),
    _branch(
        BBOX_THETA_MARK_A,
        owner="traversal",
        predicate="is_bbox_theta_a_mark",
        branch="bbox_theta_a",
        order=20,
        payload_group="bbox_angle",
    ),
    _branch(
        BBOX_WORLD_ANGLE_MARK_B,
        owner="traversal",
        predicate="is_bbox_world_b_mark",
        branch="bbox_world_b",
        order=21,
        payload_group="bbox_angle",
    ),
    _branch(
        BBOX_THETA_MARK_B,
        owner="traversal",
        predicate="is_bbox_theta_b_mark",
        branch="bbox_theta_b",
        order=22,
        payload_group="bbox_angle",
    ),
    _branch(
        BBOX_SCAN,
        owner="traversal",
        predicate="is_bbox_scan",
        branch="bbox_scan",
        order=23,
    ),
    _branch(
        ADVANCE_SEG,
        owner="projection",
        predicate="is_advance_seg",
        branch="advance_seg",
        order=26,
    ),
    _branch(
        EMIT_X2, owner="projection", predicate="is_emit_x2", branch="emit_x2", order=27
    ),
    _branch(
        R_STORE_WALL_RANGE,
        owner="projection",
        predicate="is_store_wall_range",
        branch="store_wall_range",
        order=28,
    ),
    _branch(
        SEG_KPART,
        owner="projection",
        predicate="is_seg_kpart",
        branch="seg_kpart",
        order=29,
    ),
    _branch(
        SEG_DC_TMID_MID,
        owner="projection",
        predicate="is_seg_dc_tmid_mid",
        branch="seg_dc_tmid_mid",
        order=30,
        payload_group="drawseg_value",
    ),
    _branch(
        SEG_DC_TMID_UPPER,
        owner="projection",
        predicate="is_seg_dc_tmid_upper",
        branch="seg_dc_tmid_upper",
        order=31,
        payload_group="drawseg_value",
    ),
    _branch(
        SEG_DC_TMID_LOWER,
        owner="projection",
        predicate="is_seg_dc_tmid_lower",
        branch="seg_dc_tmid_lower",
        order=32,
        payload_group="drawseg_value",
    ),
    _branch(
        DRAWSEG_META,
        owner="projection",
        predicate="is_drawseg_meta",
        branch="drawseg_meta",
        order=33,
    ),
    _branch(
        DRAWSEG_SCALE1_DEN,
        owner="projection",
        predicate="is_drawseg_scale1_den",
        branch="drawseg_scale1_den",
        order=34,
        payload_group="drawseg_value",
    ),
    _branch(
        DRAWSEG_SCALE1,
        owner="projection",
        predicate="is_drawseg_scale1",
        branch="drawseg_scale1",
        order=35,
        payload_group="drawseg_value",
    ),
    _branch(
        DRAWSEG_SCALE2_DEN,
        owner="projection",
        predicate="is_drawseg_scale2_den",
        branch="drawseg_scale2_den",
        order=36,
        payload_group="drawseg_value",
    ),
    _branch(
        DRAWSEG_SCALE2,
        owner="projection",
        predicate="is_drawseg_scale2",
        branch="drawseg_scale2",
        order=37,
        payload_group="drawseg_value",
    ),
    _branch(
        DRAWSEG_SCALESTEP_DEN,
        owner="projection",
        predicate="is_drawseg_scalestep_den",
        branch="drawseg_scalestep_den",
        order=38,
        payload_group="drawseg_value",
    ),
    _branch(
        DRAWSEG_SCALESTEP,
        owner="projection",
        predicate="is_drawseg_scalestep",
        branch="drawseg_scalestep",
        order=39,
        payload_group="drawseg_value",
    ),
    _branch(
        DRAWSEG_BSILHEIGHT,
        owner="projection",
        predicate="is_drawseg_bsilheight",
        branch="drawseg_bsilheight",
        order=40,
        payload_group="drawseg_value",
    ),
    _branch(
        DRAWSEG_TSILHEIGHT,
        owner="projection",
        predicate="is_drawseg_tsilheight",
        branch="drawseg_tsilheight",
        order=41,
        payload_group="drawseg_value",
    ),
    _branch(
        DRAWSEG_U_PHASE,
        owner="projection",
        predicate="is_drawseg_u_phase",
        branch="drawseg_u_phase",
        order=42,
        payload_group="drawseg_u_angle",
    ),
    _branch(
        R_CHECK_PLANE,
        owner="projection",
        predicate="is_r_check_plane",
        branch="r_check_plane",
        order=43,
    ),
    _branch(
        R_CHECK_PLANE_RESULT,
        owner="projection",
        predicate="is_r_check_plane_result",
        branch="r_check_plane_result",
        order=44,
    ),
    _branch(
        SET_CURSOR_X,
        owner="projection",
        predicate="is_wall_column",
        branch="wall_column",
        order=45,
    ),
    _branch(
        WALL_COL_U,
        owner="projection",
        predicate="is_wall_col_u",
        branch="wall_col_u",
        order=46,
    ),
    _branch(
        WALL_SPAN_META,
        owner="projection",
        predicate="is_wall_span_meta",
        branch="wall_span_meta",
        order=47,
    ),
    _branch(
        SET_CURSOR_Y,
        owner="projection",
        predicate="is_set_cursor_y",
        branch="set_cursor_y",
        order=48,
    ),
    _branch(
        PIXEL,
        owner="projection",
        predicate="is_pixel_color",
        branch="pixel_color",
        order=49,
    ),
    _branch(
        CLIP_UPDATE,
        owner="projection",
        predicate="is_clip_update",
        branch="clip_update",
        order=50,
    ),
    _branch(
        PLANE_MARK,
        owner="projection",
        predicate="is_plane_mark",
        branch="plane_mark",
        order=51,
    ),
    _branch(
        DRAW_PLANES_BEGIN,
        owner="projection",
        predicate="is_draw_planes_begin",
        branch="draw_planes_begin",
        order=52,
    ),
    _branch(
        FLAT_NEXT_PLANE,
        owner="projection",
        predicate="is_flat_next_plane",
        branch="flat_next_plane",
        order=53,
    ),
    _branch(
        FLAT_NEXT_VP,
        owner="projection",
        predicate="is_flat_next_vp",
        branch="flat_next_vp",
        order=54,
    ),
    _branch(
        FLAT_VISPLANE_BEGIN,
        owner="projection",
        predicate="is_flat_visplane_begin",
        branch="flat_visplane_begin",
        order=55,
    ),
    _branch(
        MAKE_SPANS_COL,
        owner="projection",
        predicate="is_make_spans_col",
        branch="make_spans_col",
        order=56,
    ),
    _branch(
        SPAN_CLOSE_SLOT,
        owner="projection",
        predicate="is_span_close_slot",
        branch="span_close_slot",
        order=57,
    ),
    _branch(
        SPAN_ROW,
        owner="projection",
        predicate="is_span_row",
        branch="span_row",
        order=58,
    ),
    _branch(
        SET_CURSOR_DIRECTION_Y,
        owner="traversal",
        predicate="is_set_cursor_direction_y",
        branch="set_cursor_direction_y",
        order=59,
    ),
    _branch(
        SET_CURSOR_DIRECTION_X,
        owner="projection",
        predicate="is_set_cursor_direction_x",
        branch="set_cursor_direction_x",
        order=60,
    ),
    _branch(
        SCREEN_Y_VALUE,
        owner="projection",
        predicate="is_screen_y_value",
        branch="screen_y",
        order=61,
    ),
    _branch(
        SCREEN_RANGE,
        owner="projection",
        predicate="is_screen_range",
        branch="screen_range",
        order=62,
    ),
    _branch(
        DONE,
        owner="main",
        predicate="is_done",
        branch="done",
        order=63,
        role="terminal",
    ),
)


def _tokens_for_payload_group(group: str) -> tuple[TokenType, ...]:
    return tuple(
        entry.token for entry in PROTOCOL_ENTRIES if entry.payload_group == group
    )


def _tokens_for_role(role: str) -> tuple[TokenType, ...]:
    return tuple(entry.token for entry in PROTOCOL_ENTRIES if entry.role == role)


def _build_dispatch_transitions() -> tuple[DispatchTransition, ...]:
    grouped: OrderedDict[tuple[str, str], list[ProtocolEntry]] = OrderedDict()
    for entry in sorted(PROTOCOL_ENTRIES, key=lambda item: item.dispatch_order):
        grouped.setdefault((entry.predicate, entry.branch), []).append(entry)
    return tuple(
        DispatchTransition(
            predicate=predicate,
            branch=branch,
            tokens=tuple(entry.token for entry in entries),
        )
        for (predicate, branch), entries in grouped.items()
    )


def _entry_by_token() -> dict[TokenType, ProtocolEntry]:
    out: dict[TokenType, ProtocolEntry] = {}
    for entry in PROTOCOL_ENTRIES:
        if entry.token in out:
            raise ValueError(f"duplicate protocol entry for {entry.token.name}")
        out[entry.token] = entry
    return out


def _build_token_check_predicates() -> tuple[TokenCheckPredicate, ...]:
    """Predicates whose implementation is exactly `token.check(input_vec)`."""
    grouped: OrderedDict[str, list[ProtocolEntry]] = OrderedDict()
    for entry in PROTOCOL_ENTRIES:
        grouped.setdefault(entry.predicate, []).append(entry)
    return tuple(
        TokenCheckPredicate(predicate=predicate, token=entries[0].token)
        for predicate, entries in grouped.items()
        if len(entries) == 1
    )


def _build_prefill_replay_predicates() -> tuple[str, ...]:
    """ProtocolTokenView predicate names that should replay prefill inputs."""
    input_predicates: OrderedDict[str, None] = OrderedDict()
    scene_payload_predicates: OrderedDict[str, None] = OrderedDict()
    for entry in PROTOCOL_ENTRIES:
        if entry.prefill_replay == "never":
            continue
        if entry.prefill_replay == "input":
            input_predicates.setdefault(entry.predicate, None)
            continue
        if entry.prefill_replay == "scene_payload":
            # Carriers (VALUE / ANGLE_VALUE) dispatch on a broad predicate
            # (is_value / is_angle_value), but only payloads read in a *scene*
            # context replay — so replay keys on the narrower is_scene_*_payload
            # predicate, not the carrier's own dispatch predicate.
            if entry.token == VALUE:
                scene_payload_predicates.setdefault("is_scene_value_payload", None)
            elif entry.token == ANGLE_VALUE:
                scene_payload_predicates.setdefault("is_scene_angle_payload", None)
            else:
                raise ValueError(
                    "scene_payload replay is only defined for VALUE/ANGLE_VALUE "
                    f"carriers, got {entry.token.name}"
                )
            continue
        raise ValueError(
            f"unknown prefill replay policy {entry.prefill_replay!r} "
            f"for {entry.token.name}"
        )
    return tuple(input_predicates) + tuple(scene_payload_predicates)


PROTOCOL_BY_TOKEN = _entry_by_token()
DISPATCH_TRANSITIONS = _build_dispatch_transitions()
TOKEN_CHECK_PREDICATES = _build_token_check_predicates()
PREFILL_REPLAY_PREDICATES = _build_prefill_replay_predicates()

INERT_NON_PAYLOAD_TYPES = _tokens_for_role("inert")
SCENE_VALUE_MARKERS = _tokens_for_payload_group("scene_value")
SCENE_ANGLE_MARKERS = _tokens_for_payload_group("scene_angle")
PROJECTION_ANGLE_MARKERS = _tokens_for_payload_group("projection_angle")
BBOX_ANGLE_MARKERS = _tokens_for_payload_group("bbox_angle")


def render_protocol_table() -> str:
    """Return a compact text table for docs and review (whitespace-clean)."""
    rows = [
        "token | phase | owner | role | predicate -> branch | payload",
        "----- | ----- | ----- | ---- | ------------------- | -------",
    ]
    for entry in PROTOCOL_ENTRIES:
        row = (
            f"{entry.token.name} | {entry.vocab_phase} | {entry.owner} | "
            f"{entry.role} | {entry.predicate} -> {entry.branch} | "
            f"{entry.payload_group or ''}"
        )
        rows.append(row.rstrip())
    return "\n".join(rows)
