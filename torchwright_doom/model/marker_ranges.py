"""Single source of truth: which marker token precedes a carrier, and (for a
``VALUE`` carrier) which :class:`ValueRange` decodes it.

A *marked* field in the token stream is a marker token immediately followed by
its **carrier**:

* a ``VALUE`` token — a float squashed into the marker's ``[-1, 1]`` range, or
* an ``ANGLE_VALUE`` token — a signed BAM integer (range-free).

The ``VALUE`` carrier does not store its own range; the *marker* chooses it
(see GLOSSARY.md 'carrier'). Historically that binding lived split across
``prompt/build.py`` (the prefill ``_marked`` call sites) and
``pydoom/drafter.py`` (the AR ``_PAYLOAD_RANGES`` / ``_TMID_RANGES`` tables).
This module lifts it into one declarative table that the prompt builder, the
drafter, and the tokenizer surface all import, so the binding can't drift
between the path that *produces* the stream and the path that *reads* it.

Scope: pure data over the vocab token types. Imports ``vocab`` (for the marker
``TokenType``\\ s) and ``value_ranges`` only — nothing imports this module at
vocab-build time, so the dependency runs one way (``vocab`` -> here).
"""

from __future__ import annotations

from .tokens import TokenType
from .value_ranges import ValueRange as R
from .vocab import (
    BBOX_BOT_BACK,
    BBOX_BOT_FRONT,
    BBOX_CORNER_X_MARK_A,
    BBOX_CORNER_X_MARK_B,
    BBOX_CORNER_Y_MARK_A,
    BBOX_CORNER_Y_MARK_B,
    BBOX_LEFT_BACK,
    BBOX_LEFT_FRONT,
    BBOX_RIGHT_BACK,
    BBOX_RIGHT_FRONT,
    BBOX_THETA_MARK_A,
    BBOX_THETA_MARK_B,
    BBOX_TOP_BACK,
    BBOX_TOP_FRONT,
    BBOX_WORLD_ANGLE_MARK_A,
    BBOX_WORLD_ANGLE_MARK_B,
    DRAWSEG_BSILHEIGHT,
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE1_DEN,
    DRAWSEG_SCALE2,
    DRAWSEG_SCALE2_DEN,
    DRAWSEG_SCALESTEP,
    DRAWSEG_SCALESTEP_DEN,
    DRAWSEG_TSILHEIGHT,
    DRAWSEG_U_PHASE,
    NODE_DX,
    NODE_DY,
    NODE_PX,
    NODE_PY,
    PLANE_HEIGHT,
    PLAYER_ANGLE_MARK,
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    PLAYER_Z_MARK,
    SEG_AX,
    SEG_AY,
    SEG_BACK_CEILING,
    SEG_BACK_FLOOR,
    SEG_BX,
    SEG_BY,
    SEG_DC_TMID_LOWER,
    SEG_DC_TMID_MID,
    SEG_DC_TMID_UPPER,
    SEG_FRONT_CEILING,
    SEG_FRONT_FLOOR,
    SEG_NORMAL_ANGLE,
    SEG_ROWOFFSET,
    SET_CURSOR_Y,
    THETA_MARK_A,
    THETA_MARK_B,
    WALL_COL_U,
    WORLD_ANGLE_MARK_A,
    WORLD_ANGLE_MARK_B,
)

# ---------------------------------------------------------------------------
# Prompt-path markers (produced by prompt/build.py).
# ---------------------------------------------------------------------------

_PROMPT_VALUE_MARKERS: dict[TokenType, R] = {
    PLAYER_X_MARK: R.R1,
    PLAYER_Y_MARK: R.R1,
    PLAYER_Z_MARK: R.R3,
    NODE_PX: R.R1,
    NODE_PY: R.R1,
    NODE_DX: R.R2,
    NODE_DY: R.R2,
    BBOX_TOP_FRONT: R.R0,
    BBOX_BOT_FRONT: R.R0,
    BBOX_LEFT_FRONT: R.R0,
    BBOX_RIGHT_FRONT: R.R0,
    BBOX_TOP_BACK: R.R0,
    BBOX_BOT_BACK: R.R0,
    BBOX_LEFT_BACK: R.R0,
    BBOX_RIGHT_BACK: R.R0,
    SEG_AX: R.R1,
    SEG_AY: R.R1,
    SEG_BX: R.R1,
    SEG_BY: R.R1,
    SEG_FRONT_FLOOR: R.R3,
    SEG_FRONT_CEILING: R.R3,
    SEG_BACK_FLOOR: R.R4,
    SEG_BACK_CEILING: R.R4,
    SEG_ROWOFFSET: R.R3,
    PLANE_HEIGHT: R.R3,
}

# ---------------------------------------------------------------------------
# AR-loop markers (produced by pydoom/drafter.py and the renderer forward()).
# ---------------------------------------------------------------------------

# Per-part dc_texturemid markers (drafter._TMID_RANGES).
_TMID_VALUE_MARKERS: dict[TokenType, R] = {
    SEG_DC_TMID_MID: R.R3,
    SEG_DC_TMID_UPPER: R.R4,
    SEG_DC_TMID_LOWER: R.R4,
}

# Drawseg scale / silhouette payloads (drafter._PAYLOAD_RANGES).
_DRAWSEG_VALUE_MARKERS: dict[TokenType, R] = {
    DRAWSEG_SCALE1_DEN: R.R6,
    DRAWSEG_SCALE1: R.R5,
    DRAWSEG_SCALE2_DEN: R.R6,
    DRAWSEG_SCALE2: R.R5,
    DRAWSEG_SCALESTEP_DEN: R.R7,
    DRAWSEG_SCALESTEP: R.R8,
    DRAWSEG_BSILHEIGHT: R.R9,
    DRAWSEG_TSILHEIGHT: R.R9,
}

# Markers emitted with their carrier inline at a drafter emit site (not via a
# named range table): the bbox visibility corners (R0), the per-column wall
# scale carried after wallColU (R5), and the texel-top value after setCursorY
# (R3). Kept here so the surface has the complete binding; the drafter remains
# the producer of record (the byte-exact rollout round-trip guards the match).
_AR_INLINE_VALUE_MARKERS: dict[TokenType, R] = {
    BBOX_CORNER_X_MARK_A: R.R0,
    BBOX_CORNER_Y_MARK_A: R.R0,
    BBOX_CORNER_X_MARK_B: R.R0,
    BBOX_CORNER_Y_MARK_B: R.R0,
    WALL_COL_U: R.R5,
    SET_CURSOR_Y: R.R3,
}

# ---------------------------------------------------------------------------
# The combined public tables.
# ---------------------------------------------------------------------------

#: Marker ``TokenType`` -> the :class:`ValueRange` that decodes its trailing
#: ``VALUE`` carrier. The single source for prefill, AR emit, and the surface.
MARKER_RANGE: dict[TokenType, R] = {
    **_PROMPT_VALUE_MARKERS,
    **_TMID_VALUE_MARKERS,
    **_DRAWSEG_VALUE_MARKERS,
    **_AR_INLINE_VALUE_MARKERS,
}

#: Markers whose trailing carrier is an ``ANGLE_VALUE`` (signed BAM), not a
#: ranged ``VALUE``. BAM is range-free, so these carry no :class:`ValueRange`.
ANGLE_MARKERS: frozenset[TokenType] = frozenset(
    {
        PLAYER_ANGLE_MARK,
        SEG_NORMAL_ANGLE,
        WORLD_ANGLE_MARK_A,
        WORLD_ANGLE_MARK_B,
        THETA_MARK_A,
        THETA_MARK_B,
        BBOX_WORLD_ANGLE_MARK_A,
        BBOX_WORLD_ANGLE_MARK_B,
        BBOX_THETA_MARK_A,
        BBOX_THETA_MARK_B,
        DRAWSEG_U_PHASE,
    }
)


def carrier_range(marker: TokenType) -> R | None:
    """The range decoding ``marker``'s ``VALUE`` carrier, or ``None`` if the
    marker carries an ``ANGLE_VALUE`` (BAM) instead — or isn't a marker."""
    return MARKER_RANGE.get(marker)
