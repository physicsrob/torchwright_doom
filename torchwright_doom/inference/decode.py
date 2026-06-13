"""Decode a generated row stream into a screen pixel buffer — the dumb host.

This is the host side of the autoregressive loop: it walks the model's emitted
tokens in order, tracks a cursor (position + direction), and blits each ``pixel``
token's palette color at the cursor. It does **no** geometry, arithmetic, or
visibility logic — only cursor bookkeeping, a palette table lookup, and a
conditional first-write-wins blit (front-to-back compositing). All rendering
decisions were made inside the transformer.

Generalizes ``tests/scene/test_flat_pixel_oracle.py::_decode_pixel_xy`` (+ its
``row - pixel_start`` color step) to consume ``W_EMBED`` row ids rather than
sandbox ``Token``s. Walls advance the cursor in Y (the default); flats advance
in X after a ``setCursorDirectionX``.
"""

from __future__ import annotations

from ..asset_banks import PLAYPAL
from ..embedding import TOKEN_VOCAB
from ..vocab import PIXEL

_PIXEL_START, _ = TOKEN_VOCAB.type_to_row_range[PIXEL]
_PIXEL_NAME = PIXEL.name

Rgb = tuple[int, int, int]


def _walk_pixels(rows):
    """The cursor state machine shared by both decoders: yield
    ``(stream_index, row, (x, y))`` for every ``pixel`` token with an
    established cursor, advancing per the direction marks.  Pixels emitted
    before a cursor is established are dropped (matches the sandbox host
    decode).  Cursor bookkeeping only — the dumb-host contract.

    Shared so the render decode and the teacher-forced diagnostic
    (:func:`decode_xy_by_position`) can never disagree about where a
    pixel landed.
    """
    cursor_dx, cursor_dy = 0, 1  # default: walls advance in Y
    cursor_x: int | None = None
    cursor_y: int | None = None
    for i, row in enumerate(rows):
        rtype, values = TOKEN_VOCAB.row_to_token[row]
        name = rtype.name
        if name == "setCursorDirectionX":
            cursor_dx, cursor_dy = 1, 0
        elif name == "setCursorDirectionY":
            cursor_dx, cursor_dy = 0, 1
        elif name == "setCursorX":
            cursor_x = int(values["x"])
        elif name == "setCursorY":
            cursor_y = int(values["y"])
        elif name == _PIXEL_NAME:
            if cursor_x is None or cursor_y is None:
                continue
            yield i, row, (cursor_x, cursor_y)
            cursor_x += cursor_dx
            cursor_y += cursor_dy


def decode_rows_to_pixels(
    rows,
    palette: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] = PLAYPAL,
) -> dict[tuple[int, int], Rgb]:
    """Walk a generated row stream -> ``{(x, y): rgb}``.

    First write at each ``(x, y)`` wins (front-to-back compositing: a conditional
    write, not computation).
    """
    buf: dict[tuple[int, int], Rgb] = {}
    for _, row, key in _walk_pixels(rows):
        if key not in buf:
            r, g, b = palette[row - _PIXEL_START]
            buf[key] = (int(r), int(g), int(b))
    return buf


def pixel_color_index(row: int) -> int:
    """Palette index carried by a ``pixel`` row (``row - pixel_start``)."""
    return row - _PIXEL_START


def decode_xy_by_position(rows) -> dict[int, tuple[int, int]]:
    """Map each ``pixel`` row's stream position -> its ``(x, y)`` screen cursor.

    Same cursor walk as :func:`decode_rows_to_pixels` (literally — both
    consume :func:`_walk_pixels`), keyed by stream position for the
    teacher-forced diagnostic, which checks the compiled pixel color
    against the option set at the reference cursor position.
    """
    return {i: xy for i, _, xy in _walk_pixels(rows)}
