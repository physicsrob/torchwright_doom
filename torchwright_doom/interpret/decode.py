"""Decode a generated row stream into a screen pixel buffer — the dumb host.

This is the host side of the autoregressive loop: it walks the model's emitted
tokens in order, tracks a cursor (position + direction), and blits each ``pixel``
token's palette color at the cursor. It does **no** geometry, arithmetic, or
visibility logic — only cursor bookkeeping, a palette table lookup, and an
overwriting blit (last-write-wins, DOOM's painter order). All rendering
decisions were made inside the transformer.

Generalizes ``tests/scene/test_flat_pixel_oracle.py::_decode_pixel_xy`` (+ its
``row - pixel_start`` color step) to consume ``W_EMBED`` row ids (a row id =
the tokenizer id, the embedding table's row index; GLOSSARY: row, W_EMBED)
rather than ``Token``s. Walls advance the cursor in Y (the default); flats advance
in X after a ``setCursorDirectionX``.
"""

from __future__ import annotations

from ..model.assets.asset_banks import PLAYPAL
from ..model.embedding import TOKEN_VOCAB
from ..model.vocab import PIXEL

_PIXEL_START, _ = TOKEN_VOCAB.type_to_row_range[PIXEL]
_PIXEL_NAME = PIXEL.name
# Width levels on the pixel `w` slot (IntSlot [lo, hi) -> hi - lo). Color and
# width share the pixel row: row = _PIXEL_START + color * _PIXEL_W_LEVELS + w_idx
# (color is the higher-order slot, declared first).
_PIXEL_W_LEVELS = PIXEL.slots["w"].hi - PIXEL.slots["w"].lo

Rgb = tuple[int, int, int]


def _walk_pixels(rows):
    """The cursor state machine of the render decode: yield
    ``(stream_index, row, (x, y), w)`` for every ``pixel`` token with an
    established cursor, advancing per the direction marks.  ``(x, y)`` is the
    base cursor; the host paints the ``w`` cells ``(x + k, y)`` for ``k`` in
    ``0..w-1``.  Pixels emitted before a cursor is established are dropped —
    a healthy stream always sets the cursor before its first pixel (see
    PROTOCOL.md).  Cursor bookkeeping only — the dumb-host contract: the
    model emits every cursor set, direction mark, and width; the host just
    applies them.
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
            w = int(values["w"])
            yield i, row, (cursor_x, cursor_y), w
            # Advance by counting, never multiplying: in a flat span (moving in
            # X) the cursor resumes just past the w-cell run it painted; in a
            # wall column (moving in Y) the run is horizontal, orthogonal to the
            # advance, so the cursor steps one row down (R_DrawColumnLow).
            if cursor_dx:
                cursor_x += w
            else:
                cursor_y += cursor_dy


def decode_rows_to_pixels(
    rows,
    palette: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] = PLAYPAL,
) -> dict[tuple[int, int], Rgb]:
    """Walk a generated row stream -> ``{(x, y): rgb}``.

    Each ``pixel`` token paints ``w`` adjacent cells in +X (low-detail blits two).
    Last write at each ``(x, y)`` wins — a plain overwrite, DOOM's painter order:
    the 3D scene is emitted first, then the weapon on top of it (and the bar into
    its own rows). The 3D pass itself writes each view pixel exactly once
    (column clipping inside the model), so the overwrite only matters where
    the weapon covers the scene.
    """
    buf: dict[tuple[int, int], Rgb] = {}
    for _, row, (x, y), w in _walk_pixels(rows):
        r, g, b = palette[pixel_color_index(row)]
        for k in range(w):
            buf[(x + k, y)] = (int(r), int(g), int(b))
    return buf


def pixel_color_index(row: int) -> int:
    """Palette index carried by a ``pixel`` row.

    Color shares the row with the width slot, enumerated as
    ``row = _PIXEL_START + color * _PIXEL_W_LEVELS + w_idx`` (color is the
    higher-order slot), so the color is the row offset floored by the number
    of width levels. This is pure detokenization — arithmetic-form-identical
    to reading ``values["color"]`` from ``TOKEN_VOCAB.row_to_token[row]``,
    the same table ``_walk_pixels`` already consults for ``w``.
    """
    return int((row - _PIXEL_START) // _PIXEL_W_LEVELS)
