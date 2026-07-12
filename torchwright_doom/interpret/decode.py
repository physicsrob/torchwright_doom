"""Decode a generated row stream into a screen pixel buffer — the dumb host.

This is the host side of the autoregressive loop: it walks the model's emitted
tokens in order, tracks a cursor (position + direction), and blits each ``pixel``
token's palette color at the cursor. It does **no** geometry, arithmetic, or
visibility logic — only cursor bookkeeping, a palette table lookup, and an
overwriting blit (last-write-wins, DOOM's painter order). All rendering
decisions were made inside the transformer.

Generalizes ``tests/scene/test_flat_pixel_oracle.py::_decode_pixel_xy`` (+ its
``row - pixel_start`` color step) to consume ``W_EMBED`` row ids rather than
``Token``s. Walls advance the cursor in Y (the default); flats advance
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
    """The cursor state machine shared by both decoders: yield
    ``(stream_index, row, (x, y), w)`` for every ``pixel`` token with an
    established cursor, advancing per the direction marks.  ``(x, y)`` is the
    base cursor; the host paints the ``w`` cells ``(x + k, y)`` for ``k`` in
    ``0..w-1``.  Pixels emitted before a cursor is established are dropped
    (matches the original host decode).  Cursor bookkeeping only — the
    dumb-host contract.

    Shared so the render decode and the per-position variant
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
    its own rows). The 3D pass itself writes each view pixel exactly once (column
    clipping), so this is identical to the old front-to-back rule there; the
    overwrite only matters where the weapon covers the scene.
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
    of width levels.
    """
    return int((row - _PIXEL_START) // _PIXEL_W_LEVELS)


def decode_xy_by_position(rows) -> dict[int, tuple[int, int]]:
    """Map each ``pixel`` row's stream position -> its ``(x, y)`` screen cursor.

    Same cursor walk as :func:`decode_rows_to_pixels` (literally — both
    consume :func:`_walk_pixels`), keyed by stream position for analyses
    that check an emitted pixel against the option set at the reference
    cursor position (the retired teacher-forced diagnostic consumed this).
    """
    return {i: xy for i, _, xy, _ in _walk_pixels(rows)}
