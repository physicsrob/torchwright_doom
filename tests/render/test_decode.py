"""Host pixel decode: cursor tracking, wall-Y / flat-X advance, first-write-wins."""

from __future__ import annotations

from torchwright_doom.asset_banks import PLAYPAL
from torchwright_doom.render.decode import (
    _PIXEL_START,
    decode_rows_to_pixels,
    pixel_color_index,
)
from torchwright_doom.render.tokens_bridge import row_index
from torchwright_doom.vocab import (
    PIXEL,
    SET_CURSOR_DIRECTION_X,
    SET_CURSOR_DIRECTION_Y,
    SET_CURSOR_X,
    SET_CURSOR_Y,
)


def _r(t, **v):
    return row_index(t, v)


def test_walls_advance_in_y():
    rows = [
        _r(SET_CURSOR_DIRECTION_Y),
        _r(SET_CURSOR_X, x=2),
        _r(SET_CURSOR_Y, y=3),
        _r(PIXEL, color=5),
        _r(PIXEL, color=6),
    ]
    assert decode_rows_to_pixels(rows) == {(2, 3): PLAYPAL[5], (2, 4): PLAYPAL[6]}


def test_flats_advance_in_x():
    rows = [
        _r(SET_CURSOR_DIRECTION_X),
        _r(SET_CURSOR_Y, y=1),
        _r(SET_CURSOR_X, x=4),
        _r(PIXEL, color=10),
        _r(PIXEL, color=11),
    ]
    assert decode_rows_to_pixels(rows) == {(4, 1): PLAYPAL[10], (5, 1): PLAYPAL[11]}


def test_first_write_wins():
    rows = [
        _r(SET_CURSOR_DIRECTION_Y),
        _r(SET_CURSOR_X, x=0),
        _r(SET_CURSOR_Y, y=0),
        _r(PIXEL, color=1),  # (0,0) -> color 1; cursor advances to (0,1)
        _r(SET_CURSOR_Y, y=0),  # reset back to (0,0)
        _r(PIXEL, color=2),  # (0,0) again -> ignored (first write wins)
    ]
    assert decode_rows_to_pixels(rows) == {(0, 0): PLAYPAL[1]}


def test_pixel_before_cursor_is_dropped():
    assert decode_rows_to_pixels([_r(PIXEL, color=1)]) == {}


def test_pixel_color_index_is_row_minus_start():
    assert pixel_color_index(_PIXEL_START + 7) == 7
