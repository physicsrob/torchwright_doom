"""BBOX derived columns + the ``_CHECKCOORD`` table — anchored to DOOM's source.

Pinned against the in-tree
protocol tokens. ``torchwright_doom.model.vocab`` carries its own hand-transcribed
``_CHECKCOORD`` table (DOOM's ``R_CheckBBox`` corner-selection table); after the
renderer/drafter were vendored, nothing else pins it. This anchors that table —
and the per-corner derived columns built from it — against DOOM's ground truth.
"""

from __future__ import annotations

from torchwright_doom.model.constants import SCREEN_WIDTH
from torchwright_doom.model.protocol.protocol_tokens import (
    BBOX_BOXPOS,
    BBOX_CORNER_X_MARK_A,
    BBOX_CORNER_X_MARK_B,
    BBOX_CORNER_Y_MARK_A,
    BBOX_CORNER_Y_MARK_B,
    BBOX_SCAN,
    FIND_RUN,
)
from torchwright_doom.model.vocab import _CHECKCOORD as NATIVE_CHECKCOORD

# DOOM's R_CheckBBox corner table (r_bsp.c): per boxpos, the (x_a, y_a, x_b, y_b)
# BOXTOP/BOXBOTTOM/BOXLEFT/BOXRIGHT corner indices. Ground truth, hand-copied.
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


def test_native_checkcoord_table_matches_doom_ground_truth():
    assert tuple(tuple(row) for row in NATIVE_CHECKCOORD) == _CHECKCOORD


def test_boxpos_derived_columns_match_checkcoord_table():
    derived = BBOX_BOXPOS.slots["boxpos"].derived

    for boxpos, row in enumerate(_CHECKCOORD):
        assert derived["check_a_x_right"].fn(boxpos) == (1.0 if row[0] == 3 else 0.0)
        assert derived["check_a_y_bottom"].fn(boxpos) == (1.0 if row[1] == 1 else 0.0)
        assert derived["check_b_x_right"].fn(boxpos) == (1.0 if row[2] == 3 else 0.0)
        assert derived["check_b_y_bottom"].fn(boxpos) == (1.0 if row[3] == 1 else 0.0)
        assert derived["fails_open"].fn(boxpos) == (
            1.0 if boxpos in (3, 5, 7, 11) else 0.0
        )


def test_all_bbox_boxpos_slots_have_same_derived_columns():
    expected = set(BBOX_BOXPOS.slots["boxpos"].derived)

    for token_type in (
        BBOX_CORNER_X_MARK_A,
        BBOX_CORNER_Y_MARK_A,
        BBOX_CORNER_X_MARK_B,
        BBOX_CORNER_Y_MARK_B,
    ):
        assert set(token_type.slots["boxpos"].derived) == expected


def test_scan_x_square_derived_columns_match_integer_square():
    for token_type in (FIND_RUN, BBOX_SCAN):
        x_square = token_type.slots["x"].derived["x_square"].fn

        for x in (0, 1, 7, SCREEN_WIDTH - 1, SCREEN_WIDTH):
            assert x_square(x) == float(x * x)
