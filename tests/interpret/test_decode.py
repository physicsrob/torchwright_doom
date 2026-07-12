"""Host pixel decode: cursor tracking, wall-Y / flat-X advance, last-write-wins,
and the low-detail width slot (paint W cells, advance by counting)."""

from __future__ import annotations

from torchwright_doom.model.assets.asset_banks import PLAYPAL
from torchwright_doom.interpret.decode import (
    decode_rows_to_pixels,
    pixel_color_index,
)
from torchwright_doom.tokenizer.rows import row_index
from torchwright_doom.model.vocab import (
    PIXEL,
    SET_CURSOR_DIRECTION_X,
    SET_CURSOR_DIRECTION_Y,
    SET_CURSOR_X,
    SET_CURSOR_Y,
)


def _r(t, **v):
    return row_index(t, v)


def _px(color, w=1):
    return row_index(PIXEL, {"color": color, "w": w})


def test_walls_advance_in_y():
    rows = [
        _r(SET_CURSOR_DIRECTION_Y),
        _r(SET_CURSOR_X, x=2),
        _r(SET_CURSOR_Y, y=3),
        _px(5),
        _px(6),
    ]
    assert decode_rows_to_pixels(rows) == {(2, 3): PLAYPAL[5], (2, 4): PLAYPAL[6]}


def test_flats_advance_in_x():
    rows = [
        _r(SET_CURSOR_DIRECTION_X),
        _r(SET_CURSOR_Y, y=1),
        _r(SET_CURSOR_X, x=4),
        _px(10),
        _px(11),
    ]
    assert decode_rows_to_pixels(rows) == {(4, 1): PLAYPAL[10], (5, 1): PLAYPAL[11]}


def test_last_write_wins():
    # Painter's order (DOOM): the later write at a cell overwrites the earlier
    # one. This is what lets the weapon (emitted last) paint over the 3D scene.
    rows = [
        _r(SET_CURSOR_DIRECTION_Y),
        _r(SET_CURSOR_X, x=0),
        _r(SET_CURSOR_Y, y=0),
        _px(1),  # (0,0) -> color 1; cursor advances to (0,1)
        _r(SET_CURSOR_Y, y=0),  # reset back to (0,0)
        _px(2),  # (0,0) again -> overwrites (last write wins)
    ]
    assert decode_rows_to_pixels(rows) == {(0, 0): PLAYPAL[2]}


def test_pixel_before_cursor_is_dropped():
    assert decode_rows_to_pixels([_px(1)]) == {}


def test_width2_wall_paints_two_cells_and_steps_one_row():
    # A width-2 wall column paints two screen columns per row (the run is
    # horizontal, orthogonal to the Y advance) and steps one row down per pixel.
    rows = [
        _r(SET_CURSOR_DIRECTION_Y),
        _r(SET_CURSOR_X, x=2),
        _r(SET_CURSOR_Y, y=3),
        _px(5, w=2),
        _px(6, w=2),
    ]
    assert decode_rows_to_pixels(rows) == {
        (2, 3): PLAYPAL[5],
        (3, 3): PLAYPAL[5],
        (2, 4): PLAYPAL[6],
        (3, 4): PLAYPAL[6],
    }


def test_width2_flat_paints_two_cells_and_steps_two():
    # A width-2 flat span paints two screen columns per pixel and the cursor
    # resumes just past the run (steps by w), so consecutive pixels tile.
    rows = [
        _r(SET_CURSOR_DIRECTION_X),
        _r(SET_CURSOR_Y, y=1),
        _r(SET_CURSOR_X, x=4),
        _px(10, w=2),
        _px(11, w=2),
    ]
    assert decode_rows_to_pixels(rows) == {
        (4, 1): PLAYPAL[10],
        (5, 1): PLAYPAL[10],
        (6, 1): PLAYPAL[11],
        (7, 1): PLAYPAL[11],
    }


def test_width2_last_write_wins_per_cell():
    # Each of the w cells composites independently (last write wins per cell).
    rows = [
        _r(SET_CURSOR_DIRECTION_X),
        _r(SET_CURSOR_Y, y=0),
        _r(SET_CURSOR_X, x=0),
        _px(1, w=2),  # paints (0,0),(1,0); cursor -> x=2
        _r(SET_CURSOR_X, x=1),
        _px(2, w=2),  # paints (1,0) [overwritten], (2,0) [new]
    ]
    assert decode_rows_to_pixels(rows) == {
        (0, 0): PLAYPAL[1],
        (1, 0): PLAYPAL[2],
        (2, 0): PLAYPAL[2],
    }


def test_pixel_color_index_round_trips():
    # Color shares the row with the width slot; the decode recovers color
    # regardless of width.
    for color in (0, 7, 128, 255):
        for w in (1, 2):
            row = row_index(PIXEL, {"color": color, "w": w})
            assert pixel_color_index(row) == color
