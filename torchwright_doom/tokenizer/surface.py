"""The readable surface: a **vanilla tokenizer** plus a **whitespace formatter**.

Two cleanly separable pieces, composed into the public ``render`` / ``parse``:

1. **The grammar** — ``decode(tokens) -> flat text`` / ``encode(text) -> tokens``,
   byte-exact at the token id level and **newline-free** (tokens separated by
   single spaces). It applies the per-id labels (:func:`.display.token_label`)
   plus the **one contextual rule**: a ``VALUE`` / ``ANGLE_VALUE`` carrier folds
   into the *preceding marker's* parens as its trailing positional arg,
   de-quantized through that marker's range (``marker_ranges.MARKER_RANGE``). The
   carrier stores only a ``[-1, 1]`` number; the marker chooses how to read it —
   so this is normal "vanilla" detokenizer behaviour (cf. a BPE byte-merge), one
   well-contained table-driven rule, not a third grammar. This is what the HF
   tokenizer uses; it is fully reversible.

2. **The formatter** — ``format(flat) -> laid-out text``. It owns ALL layout: a
   new line at each "header" token (indented by its nesting level under
   ``indent``) with the following tokens packed into aligned columns beneath it.
   Pure whitespace — :func:`encode` ignores it (``_scan`` drops whitespace and
   ``#`` comments) — so layout never affects the id stream.

Byte-exactness is anchored at the value level: a de-quantized carrier renders as
the *shortest decimal that re-quantizes to the same carrier level* (so
``encode(decode(v)) == v``). There is no lossy display knob; the figure shows the
reversible shortest-decimal (``1383.98``), never a rounded ``1384`` that could
re-encode to a neighbouring level.

The kept knobs are all reversible and audience-specific (none change the
re-encoded id stream): ``wall_names`` / ``flat_names`` (WAD asset names; ``None``
⇒ raw ids), ``angle_degrees`` (degrees vs BAM), ``origin`` (WAD vs scene-relative
coordinates), ``physical_values`` (physical vs raw ``[-1, 1]`` carrier). ``indent``
is a formatter option, not a grammar knob.

Imports only :mod:`.display` (the per-id labels), :mod:`.tokens`,
:mod:`.value_ranges`, the :mod:`.vocab` registry, and the shared
:mod:`.marker_ranges` table — no graph nodes at import or call time. Texture names
are passed in (not imported) so the module stays asset-agnostic.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from . import display
from ..model.marker_ranges import ANGLE_MARKERS, MARKER_RANGE
from ..model.tokens import Token, TokenType
from ..model.value_ranges import ValueRange, decode_float, encode_float
from ..model.vocab import ANGLE_BAM, ANGLE_VALUE, VALUE, VOCAB_TYPES

TokenLike = Token | tuple[TokenType, Mapping[str, int | float]]

# Name -> TokenType for the whole vocab (header lookup + marker-table sanity).
_TYPE_BY_NAME: dict[str, TokenType] = {t.name: t for t in VOCAB_TYPES}

# The 16-bit VALUE carrier: 65536 levels evenly over [-1, 1] (65535 steps).
_VALUE_STEPS = 65535

# Header token types -> their nesting LEVEL: a new display line starts at each
# header (the following non-header tokens join onto it). Layout only — keyed on
# canonical ``TokenType.name``, mapped to display aliases in HEADER_DISPLAY_LEVEL
# below. ``encode`` ignores whitespace and line breaks, so this table is freely
# tunable without affecting the id stream. Curated for entity-per-line density:
# one line per player block / node / subsector / seg / plane (prompt), and per
# traversal step / wall column / span action (AR output).
#
# The level drives the ``indent`` knob's per-block indentation (level *
# ``_INDENT_UNIT`` spaces; see _layout_fields). Levels are a SHALLOW STATIC phase
# hierarchy (0..2 here), not live recursion depth: BSP traversal nests to
# N_DEPTH_MAX=16, which the fixed ``_INDENT_MAXW`` budget cannot show as
# whitespace, so real depth stays in the ``depth=`` slot as a number. Keep the
# deepest level small enough that ``level * _INDENT_UNIT`` plus a typical field
# stays under ``_INDENT_MAXW``. ``setCursorX`` is held at level 2 in every
# context (wall column, flat span, weapon/HUD) so it never prints left of a
# parent header — it has no sub-headers of its own, so that can't invert.
_HEADER_LEVEL: dict[str, int] = {
    # --- prompt ---
    "viewx": 0,  # opens the player block (viewy/viewz/viewangle join it)
    "node": 0,
    "SSECTOR": 0,
    "seg": 1,  # segs nest under their subsector
    "planeDef": 0,
    "ssFloorPlane": 0,  # ssCeilingPlane joins it
    "begin": 0,
    # --- AR: traversal spine + visibility ---
    "noOp": 0,
    "R_PointOnSide": 0,  # pointOnSideResult is its child (the call's result)
    "bspFront": 0,
    "bspCheckBack": 0,
    "bspReturn": 0,
    "R_Subsector": 0,
    "boxpos": 1,  # bbox visibility check, under bspCheckBack
    "R_AddLine": 1,  # seg projection, under its subsector
    "nextSeg": 1,
    "drawseg.x2": 1,
    # --- AR: wall range + columns ---
    "R_StoreWallRange": 1,  # one visible run, under the seg
    "drawseg.meta": 2,  # wall-range setup detail
    "R_CheckPlane": 2,
    "setCursorX": 2,  # wall column, under the range (see note above)
    # --- AR: flat pass ---
    "R_DrawPlanes": 0,
    "R_DrawPlanes.nextPlane": 1,
    "R_DrawPlanes.nextVp": 1,
    "visplaneBegin": 1,
    "R_MakeSpans.col": 2,
    "R_MapPlane.row": 2,  # sibling of setCursorX so a span's setCursorX never inverts
    # --- AR: weapon + status bar + terminal ---
    "R_DrawPlayerSprites": 0,
    "ST_Drawer": 0,
    "ST_Drawer.item": 1,
    "done": 0,
}

#: Header DISPLAY name -> nesting level. The formatter scans flat text (which
#: carries display aliases) and looks the leading word up here. Built from
#: ``_HEADER_LEVEL`` (canonical) via the alias map; the alias is a bijection, so
#: no two headers collide on one display name.
HEADER_DISPLAY_LEVEL: dict[str, int] = {
    display.DISPLAY_NAME[_TYPE_BY_NAME[name]]: level
    for name, level in _HEADER_LEVEL.items()
}
assert len(HEADER_DISPLAY_LEVEL) == len(_HEADER_LEVEL), "header alias collision"

# Markers whose physical value is an absolute x / y map coordinate, shifted by
# the scene origin (centroid) under the WAD-coordinate display knob. Deltas,
# heights, scales, and screen-space values are never shifted.
_X_COORD_MARKERS: frozenset[str] = frozenset(
    {
        "viewx",
        "node.x",
        "seg.v1.x",
        "seg.v2.x",
        "node.bbox1.left",
        "node.bbox1.right",
        "node.bbox0.left",
        "node.bbox0.right",
        "bbox.x1",
        "bbox.x2",
    }
)
_Y_COORD_MARKERS: frozenset[str] = frozenset(
    {
        "viewy",
        "node.y",
        "seg.v1.y",
        "seg.v2.y",
        "node.bbox1.top",
        "node.bbox1.bottom",
        "node.bbox0.top",
        "node.bbox0.bottom",
        "bbox.y1",
        "bbox.y2",
    }
)

# Indented "semantic layout" (the ``indent`` formatter option). A header opens a
# block and its fields pack into aligned columns beneath it. The header itself is
# indented by its nesting level (``_HEADER_LEVEL[name] * _INDENT_UNIT`` spaces);
# its field rows take that same base indent plus a ``_FIELD_INDENT`` hang, so a
# block reads as the tree it is.
#
# Every block is laid on ONE uniform column grid: each column is as wide as its
# widest cell in the whole block plus a ``_COL_GUTTER`` gutter, so the gutters run
# as straight uninterrupted channels down the block and every row's columns line
# up. The column count is the largest that still fits the ``_INDENT_MAXW`` cap, so
# a block with wider values simply uses fewer, wider columns. ``_INDENT_MAXW`` is
# an ABSOLUTE cap: a row never crosses it (a lone field wider than its room still
# prints — the layout never splits a token). Pure whitespace, so encode ignores it
# and the id stream is unchanged; ``base == 0`` reproduces the un-nested layout.
# See :func:`_layout_fields`.
_INDENT_UNIT = 2  # spaces of indentation per nesting level
_FIELD_INDENT = 4  # fields hang this far beyond their header's base indent
_COL_GUTTER = 2  # spaces between adjacent columns (one uniform gutter, block-wide)
_INDENT_MAXW = 60  # absolute right margin (hard cap)
# When a block is too wide for one grid, a run of >= _UNIFORM_MIN consecutive plain
# cells whose widths span <= _UNIFORM_SLACK (a pixel flow) is split off as its own
# balanced grid, so a stray wide marker never widens the flow's columns; and a
# fall-back sub-grid never merges rows that would pad a cell beyond _MAX_PAD.
_UNIFORM_SLACK = 4
_UNIFORM_MIN = 2
_MAX_PAD = 12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _norm(tok: TokenLike) -> tuple[TokenType, dict[str, int | float]]:
    if isinstance(tok, Token):
        return tok.type, dict(tok.values)
    ttype, values = tok
    return ttype, dict(values)


def _origin_shift(marker_name: str, origin: tuple[float, float]) -> float:
    ox, oy = origin
    if marker_name in _X_COORD_MARKERS:
        return ox
    if marker_name in _Y_COORD_MARKERS:
        return oy
    return 0.0


def _level(v: float) -> int:
    """The VALUE carrier's quantization level for a ``[-1, 1]`` value — exactly
    the step ``row_index`` rounds it to (no clamp, matching the producer)."""
    return round((float(v) + 1.0) / 2.0 * _VALUE_STEPS)


def _fmt_decimal(value: float, places: int) -> str:
    if places <= 0:
        return str(int(round(value)))
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _shortest_value(range_id: ValueRange, carrier: float, shift: float) -> str:
    """Shortest decimal of the de-quantized physical that re-encodes to the
    same carrier level — guaranteeing ``encode(decode(v)) == v`` at the id."""
    target = _level(carrier)
    physical = decode_float(range_id, carrier) + shift
    for places in range(0, 10):
        candidate = round(physical, places)
        if _level(encode_float(range_id, candidate - shift)) == target:
            return _fmt_decimal(candidate, places)
    return repr(physical)  # full precision fallback (always round-trips)


def _shortest_angle(bam: int, degrees: bool) -> str:
    if not degrees:
        return str(int(bam))
    target = int(bam)
    physical = target * 360.0 / ANGLE_BAM
    for places in range(0, 10):
        candidate = round(physical, places)
        if round(candidate * ANGLE_BAM / 360.0) == target:
            return _fmt_decimal(candidate, places)
    return repr(physical)


# ---------------------------------------------------------------------------
# the grammar: decode (tokens -> flat text)
# ---------------------------------------------------------------------------


def decode(
    tokens: Iterable[TokenLike],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    wall_names: Sequence[str] | None = None,
    flat_names: Sequence[str] | None = None,
    physical_values: bool = True,
    angle_degrees: bool = False,
) -> str:
    """Detokenize a token stream to **flat** readable text (single spaces, no
    line breaks). Each token becomes its per-id label (:func:`.display.token_label`);
    a ``VALUE`` / ``ANGLE_VALUE`` carrier folds into the preceding marker's parens
    as its trailing positional arg, de-quantized through the marker's range.

    ``tokens`` are ``Token`` instances or ``(TokenType, values)`` tuples,
    *including* the carriers in stream order. Reversible: :func:`encode` recovers
    the identical id stream under matching knobs.
    """
    toks = [_norm(t) for t in tokens]
    units: list[str] = []
    i, n = 0, len(toks)
    while i < n:
        ttype, values = toks[i]
        if ttype is VALUE or ttype is ANGLE_VALUE:
            raise ValueError(f"carrier {ttype.name!r} at {i} has no preceding marker")

        # Marker-lookahead for the one contextual rule: if the next token is a
        # carrier, it is *this* token's (carriers only ever follow their marker).
        # A marker type like setCursorY only carries a value in some contexts —
        # keying on whether a carrier actually follows handles both.
        carrier: str | None = None
        if i + 1 < n and toks[i + 1][0] in (VALUE, ANGLE_VALUE):
            ctype, cvals = toks[i + 1]
            carrier = _render_carrier(
                ttype, ctype, cvals, origin, physical_values, angle_degrees
            )
            i += 1

        name, args = display.token_fields(
            ttype, values, wall_names=wall_names, flat_names=flat_names
        )
        if carrier is not None:
            args = [*args, carrier]
        units.append(name if not args else f"{name}({', '.join(args)})")
        i += 1

    return " ".join(units)


def _render_carrier(
    marker: TokenType,
    carrier_type: TokenType,
    carrier_values: Mapping[str, int | float],
    origin: tuple[float, float],
    physical_values: bool,
    angle_degrees: bool,
) -> str:
    if carrier_type is VALUE:
        range_id = MARKER_RANGE.get(marker)
        if range_id is None:
            raise ValueError(f"VALUE follows non-marker {marker.name!r}")
        carrier = float(carrier_values["v"])
        if physical_values:
            shift = _origin_shift(marker.name, origin)
            physical = decode_float(range_id, carrier) + shift
            word = display.decode_sentinel(marker.name, physical)
            if word is not None:
                return word
            return _shortest_value(range_id, carrier, shift)
        return repr(carrier)
    if marker not in ANGLE_MARKERS:
        raise ValueError(f"ANGLE_VALUE follows non-angle-marker {marker.name!r}")
    return _shortest_angle(int(carrier_values["angle"]), angle_degrees)


# ---------------------------------------------------------------------------
# the grammar: encode (text -> tokens)
# ---------------------------------------------------------------------------


def _scan(text: str) -> list[tuple[str, list[str] | None]]:
    """Split readable text into ``(type_name, args)`` units. ``args`` is the
    comma-separated list inside ``(...)`` (possibly empty), or ``None`` for a
    bare slotless type. Whitespace, newlines, and ``#`` comments are ignored;
    spaces inside the parens (``pixel(143, 2)``) are fine — this is what makes
    the whitespace formatter reversible (its layout is dropped here)."""
    body = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    out: list[tuple[str, list[str] | None]] = []
    i, n = 0, len(body)
    while i < n:
        if body[i].isspace():
            i += 1
            continue
        start = i
        while i < n and not body[i].isspace() and body[i] != "(":
            i += 1
        name = body[start:i]
        args: list[str] | None = None
        if i < n and body[i] == "(":
            close = body.index(")", i)
            inner = body[i + 1 : close].strip()
            args = [a.strip() for a in inner.split(",")] if inner else []
            i = close + 1
        if name:
            out.append((name, args))
    return out


def encode(
    text: str,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    wall_names: Sequence[str] | None = None,
    flat_names: Sequence[str] | None = None,
    physical_values: bool = True,
    angle_degrees: bool = False,
) -> list[tuple[TokenType, dict[str, int | float]]]:
    """Inverse of :func:`decode`: readable text -> ``(TokenType, values)`` list.

    Tolerant of whitespace, newlines, and ``#`` comments (so the formatter's
    layout is invisible here); named args are order-free, positional args map
    back in declaration order. The knobs mirror :func:`decode` and must match the
    decode settings for a byte-exact round trip."""
    out: list[tuple[TokenType, dict[str, int | float]]] = []
    for name, args in _scan(text):
        resolved = display.resolve_name(name)
        if resolved is None:
            raise ValueError(f"unknown token keyword: {name!r}")
        ttype, preset = resolved
        slot_values: dict[str, int | float] = dict(preset)
        bare: list[str] = []
        for arg in args or ():
            if "=" in arg:
                key, value = arg.split("=", 1)
                slot_values[key] = display.parse_slot(
                    ttype.name, key, value, wall_names, flat_names
                )
            else:
                bare.append(arg)

        # Positional bare args fill the still-unnamed slots in declaration order;
        # at most one leftover bare arg is the marker's carrier value.
        unfilled = [s for s in ttype.slots if s not in slot_values]
        carrier_word: str | None = None
        for index, value in enumerate(bare):
            if index < len(unfilled):
                slot_values[unfilled[index]] = display.parse_slot(
                    ttype.name, unfilled[index], value, wall_names, flat_names
                )
            elif carrier_word is None:
                carrier_word = value
            else:
                raise ValueError(f"token {name!r} has too many positional args: {bare}")
        out.append((ttype, slot_values))

        if carrier_word is None:
            continue
        range_id = MARKER_RANGE.get(ttype)
        if range_id is not None:
            sentinel = display.encode_sentinel(ttype.name, carrier_word)
            if sentinel is not None:
                carrier = encode_float(range_id, sentinel)
            elif physical_values:
                physical = float(carrier_word) - _origin_shift(ttype.name, origin)
                carrier = encode_float(range_id, physical)
            else:
                carrier = float(carrier_word)
            out.append((VALUE, {"v": carrier}))
        elif ttype in ANGLE_MARKERS:
            if angle_degrees:
                bam = round(float(carrier_word) * ANGLE_BAM / 360.0)
            else:
                bam = int(round(float(carrier_word)))
            out.append((ANGLE_VALUE, {"angle": bam}))
        else:
            raise ValueError(
                f"token {ttype.name!r} has a value {carrier_word!r} but is not a marker"
            )
    return out


# ---------------------------------------------------------------------------
# the whitespace formatter: format (flat text -> laid-out text)
# ---------------------------------------------------------------------------


def _scan_units(flat: str) -> list[tuple[str, str]]:
    """Re-segment flat grammar output into ``(leading_name, unit_text)`` pairs,
    each unit's exact substring preserved verbatim (so re-emitting it adds only
    whitespace). ``leading_name`` is the word before any ``(`` — what the header
    lookup keys on."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(flat)
    while i < n:
        if flat[i].isspace():
            i += 1
            continue
        start = i
        while i < n and not flat[i].isspace() and flat[i] != "(":
            i += 1
        name = flat[start:i]
        if i < n and flat[i] == "(":
            i = flat.index(")", i) + 1
        out.append((name, flat[start:i]))
    return out


# Scalar field families that pair by their trailing index: members of one family
# sharing an index sit on one row. ``angle1``+``theta1`` (one seg-projection
# endpoint per line); the bbox corner coords (``bx1``+``by1``) and corner angles
# (``bangle1``+``btheta1``) each pair too. Keyed by base name (trailing digits
# stripped) -> family tag. Other digit-suffixed scalars (``child1``/``child0``)
# are NOT here, so they keep packing together.
_PAIR_FAMILY: dict[str, str] = {
    "angle": "at",
    "theta": "at",  # seg-projection endpoint angles
    "bx": "bxy",
    "by": "bxy",  # bbox corner coords
    "bangle": "bat",
    "btheta": "bat",  # bbox corner angles
}


def _field_group(field: str) -> str | None:
    """Row-break key: the dotted prefix before a field's last segment
    (``bbox1.top`` -> ``bbox1``, ``v1.x`` -> ``v1``); for an indexed pair-family
    scalar, the family tag + the trailing index (so ``angle1``/``theta1`` share a
    row, but ``angle2`` starts a new one); else ``None`` for a plain scalar (no
    dot) so consecutive scalars pack together. Each distinct group starts a fresh
    row."""
    name = field.split("(", 1)[0]
    if "." in name:
        return name.rsplit(".", 1)[0]
    base = name.rstrip("0123456789")
    if base != name:
        family = _PAIR_FAMILY.get(base)
        if family is not None:
            return family + name[len(base) :]  # e.g. "at1", "bxy2", "bat1"
    return None


def _glues_to_prev(field: str, prev_field: str | None) -> bool:
    """Whether ``field`` should sit on the same row as the field before it (placed
    tight, one space after), because it is the *detail* of that field:

    * a ``screenRange`` details the plane-mark / clip-update it follows,
    * a ``bboxClipScan`` trails the bbox corners it tests (one merges onto the
      last corner-angle row; a run of them flows on, wrapping at the cap), and
    * a scale value follows its ``.den`` denominator (``ds.scale1.den ds.scale1``).

    Pure layout — the glue still honours the ``_INDENT_MAXW`` cap (the caller
    falls back to a normal wrap if it would overflow)."""
    if prev_field is None:
        return False
    name = field.split("(", 1)[0]
    if name in ("screenRange", "bboxClipScan"):
        return True  # keep-with-previous: a detail of the field before it
    prev_name = prev_field.split("(", 1)[0]
    return prev_name.endswith(".den")  # keep-with-next: the den's value


# A cell: a primary field plus the keep-with details glued to it (e.g. a marker
# and its ``screenRange``; a ``.den`` denominator and its value). A cell is laid
# out atomically — its fields stay together, in order, on one row. ``group`` is the
# primary's :func:`_field_group` row-break key.
_Cell = tuple  # (primary: str, tail: list[str], group: str | None)

# A field placement on a row: ``[text, col]`` — ``col`` is the character column the
# field starts at, filled in once the grid is known. A list so ``col`` is mutable.
_Placement = list


def _build_cells(fields: Sequence[str]) -> list[_Cell]:
    """Group the fields into cells: a primary field collects the run of keep-with
    details glued to it (:func:`_glues_to_prev`) — so a ``.den`` value or a
    ``screenRange`` rides its owner across a field-group boundary. Glue is
    cap-unaware here; an over-long detail run is wrapped in :func:`_greedy_rows`."""
    cells: list[_Cell] = []
    prev: str | None = None
    for field in fields:
        if cells and _glues_to_prev(field, prev):
            cells[-1][1].append(field)
        else:
            cells.append((field, [], _field_group(field)))
        prev = field
    return cells


def _runs(cells: Sequence[_Cell]) -> list[list[_Cell]]:
    """Split cells into maximal runs of the same field group. A group boundary
    always starts a new row, so the agreed pairings (seg ``angle``/``theta``, the
    bbox corners) and the dotted sub-blocks stay intact."""
    runs: list[list[_Cell]] = []
    for cell in cells:
        if runs and runs[-1][0][2] == cell[2]:
            runs[-1].append(cell)
        else:
            runs.append([cell])
    return runs


def _split_sizes(n: int, r: int) -> list[int]:
    """Split ``n`` cells into ``r`` balanced contiguous rows (sizes differ by at
    most one, the larger rows first) — so a run fills its rows evenly instead of
    dangling a lone leftover cell."""
    sizes, placed = [], 0
    for i in range(r):
        size = -(-(n - placed) // (r - i))  # ceil of the remaining even share
        sizes.append(size)
        placed += size
    return sizes


def _cell_row(cells: Sequence[_Cell]) -> list[_Placement]:
    """Flatten a chunk of cells into a row of placements — each cell's primary then
    its glued details, in stream order."""
    return [[f, 0] for cell in cells for f in (cell[0], *cell[1])]


def _balanced_rows(
    runs: Sequence[Sequence[_Cell]], cols_per_row: int
) -> list[list[_Placement]]:
    """Lay the block's runs into rows of at most ``cols_per_row`` cells, each run
    balanced across its rows (:func:`_split_sizes`); a group boundary still starts a
    fresh row. Cells stay atomic, so a row may hold more *fields* than
    ``cols_per_row`` when a cell carries glued details. Columns are assigned later."""
    rows: list[list[_Placement]] = []
    for run in runs:
        r = -(-len(run) // cols_per_row)  # fewest rows holding <= cols_per_row cells
        start = 0
        for size in _split_sizes(len(run), r):
            rows.append(_cell_row(run[start : start + size]))
            start += size
    return rows


def _greedy_rows(runs: Sequence[Sequence[_Cell]], left: int) -> list[list[_Placement]]:
    """Fall-back row builder for a block too wide for any single uniform grid: pack
    each run's cells onto a row until the next cell would cross the cap, then wrap.
    A cell stays whole — its glued details ride its primary on one row — so a marker
    keeps its ``screenRange`` beside it. (Only a lone cell wider than a full row
    spills its detail run onto continuation rows.) A group boundary starts a fresh
    row. Columns are assigned later."""

    def width(row: list[_Placement]) -> int:
        return left + sum(len(p[0]) for p in row) + _COL_GUTTER * (len(row) - 1)

    rows: list[list[_Placement]] = []
    for run in runs:
        row: list[_Placement] = []
        for primary, tail, _group in run:
            cell = len(primary) + sum(_COL_GUTTER + len(d) for d in tail)
            if row and width(row) + _COL_GUTTER + cell > _INDENT_MAXW:
                rows.append(row)  # the whole cell won't fit — wrap it as a unit
                row = []
            row.append([primary, 0])
            for detail in tail:
                if row and width(row) + _COL_GUTTER + len(detail) > _INDENT_MAXW:
                    rows.append(row)  # cell alone exceeds a row: spill the rest
                    row = [[detail, 0]]
                else:
                    row.append([detail, 0])
        if row:
            rows.append(row)
    return rows


def _grid(rows: Sequence[list[_Placement]], left: int) -> tuple[list[int], int]:
    """Column starts for one uniform grid over ``rows`` (each column as wide as its
    widest cell, ``_COL_GUTTER`` between), and the cap overshoot of the widest row
    (``<= 0`` if it fits)."""
    ncol = max((len(r) for r in rows), default=0)
    cols: list[int] = []
    colw: list[int] = []
    x = left
    for k in range(ncol):
        widest = max(len(r[k][0]) for r in rows if len(r) > k)
        if k:
            x = cols[k - 1] + colw[k - 1] + _COL_GUTTER
        cols.append(x)
        colw.append(widest)
    over = max((cols[len(r) - 1] + len(r[-1][0]) for r in rows if r), default=0)
    return cols, over - _INDENT_MAXW


def _grid_pad(rows: Sequence[list[_Placement]]) -> int:
    """The widest gap between a cell and its uniform column — how far the most
    over-padded cell sits from filling its column."""
    ncol = max((len(r) for r in rows), default=0)
    pad = 0
    for k in range(ncol):
        widths = [len(r[k][0]) for r in rows if len(r) > k]
        pad = max(pad, max(widths) - min(widths))
    return pad


def _assign(rows: Iterable[list[_Placement]], cols: Sequence[int]) -> None:
    for row in rows:
        for k, placement in enumerate(row):
            placement[1] = cols[k]


def _split_grids(rows: list[list[_Placement]], left: int) -> None:
    """Cut ``rows`` into consecutive uniform sub-grids, assigning columns within
    each — so columns stay straight within a section even where the whole block is
    too wide for one grid. A row joins the current sub-grid only while it still fits
    the cap and no cell would be padded past ``_MAX_PAD`` (so a wide marker never
    swallows a narrow neighbour into a sparse column)."""
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and (
            _grid(rows[i : j + 1], left)[1] <= 0
            and _grid_pad(rows[i : j + 1]) <= _MAX_PAD
        ):
            j += 1
        cols, _ = _grid(rows[i:j], left)
        _assign(rows[i:j], cols)
        i = j


def _render_row(row: list[_Placement]) -> str:
    text = ""
    for field, col in row:
        pad = col - len(text)
        if pad < 1 and text:  # a too-wide field overflowed; never let two touch
            pad = 1
        text += " " * max(pad, 0) + field
    return text


def _fit_grid(
    runs: Sequence[Sequence[_Cell]], left: int
) -> list[list[_Placement]] | None:
    """Lay ``runs`` as ONE uniform grid using the largest cell-count-per-row that
    still fits the cap (fewer, wider columns as values grow), or ``None`` if no
    column count fits. Above ~room/narrowest nothing fits, so bound the search
    there rather than at the cell count."""
    ceiling = max(1, (_INDENT_MAXW - left + _COL_GUTTER) // (1 + _COL_GUTTER))
    upper = min(max((len(run) for run in runs), default=1), ceiling)
    for cols_per_row in range(upper, 0, -1):
        rows = _balanced_rows(runs, cols_per_row)
        cols, over = _grid(rows, left)
        if over <= 0:
            _assign(rows, cols)
            return rows
    return None


def _sections(run: Sequence[_Cell]) -> list[tuple[str, list[_Cell]]]:
    """Segment a run's cells into sections: a ``"grid"`` section is a run of at least
    ``_UNIFORM_MIN`` consecutive plain cells whose widths span at most
    ``_UNIFORM_SLACK`` (a pixel flow), laid as its own balanced grid; a ``"flow"``
    section is everything else (markers and glued details), packed greedily. This
    keeps a pixel flow's columns straight without a stray wide marker stretching
    them."""
    out: list[tuple[str, list[_Cell]]] = []
    i, n = 0, len(run)
    while i < n:
        if not run[i][1]:  # plain cell — try to open a similar-width grid run
            lo = hi = len(run[i][0])
            j = i
            while j < n and not run[j][1]:
                w = len(run[j][0])
                if max(hi, w) - min(lo, w) > _UNIFORM_SLACK:
                    break
                lo, hi, j = min(lo, w), max(hi, w), j + 1
            if j - i >= _UNIFORM_MIN:
                out.append(("grid", list(run[i:j])))
                i = j
                continue
        if out and out[-1][0] == "flow":
            out[-1][1].append(run[i])
        else:
            out.append(("flow", [run[i]]))
        i += 1
    return out


def _fallback_rows(
    runs: Sequence[Sequence[_Cell]], left: int
) -> list[list[_Placement]]:
    """Lay a block too wide for one grid as a sequence of sections (:func:`_sections`):
    each pixel-flow section as its own balanced grid, each marker/detail section
    greedily, cut into the fewest consecutive uniform sub-grids (:func:`_split_grids`)
    so a run of ``label + detail`` rows still lines up."""
    rows: list[list[_Placement]] = []
    for run in runs:
        for kind, cells in _sections(run):
            grid = _fit_grid([cells], left) if kind == "grid" else None
            if grid is None:
                grid = _greedy_rows([cells], left)
                _split_grids(grid, left)
            rows.extend(grid)
    return rows


def _layout_fields(fields: Sequence[str], base: int = 0) -> list[str]:
    """Lay a header's fields into ONE uniform column grid beneath it, indented
    ``base + _FIELD_INDENT`` spaces. Every column is as wide as its widest cell in
    the block, so the gutters run straight down the whole block and every row lines
    up; the column count is the largest that still fits the ``_INDENT_MAXW`` cap
    (fewer, wider columns as a block's values grow). Fields glue into cells
    (keep-with details ride their owner) and split into runs at each field-group
    boundary; a run balances its cells across rows. A block too wide for any single
    grid — a wall column's wide ``wallColU`` and ``wallSpanMeta`` lines beside its
    pixel flow — falls back to per-section grids (:func:`_fallback_rows`), each
    internally straight-guttered. Stream order is preserved — only whitespace is
    inserted. ``base == 0`` is the un-nested layout."""
    left = base + _FIELD_INDENT
    runs = _runs(_build_cells(fields))
    rows = _fit_grid(runs, left) or _fallback_rows(runs, left)
    return [_render_row(r) for r in rows]


def format(flat: str, *, indent: bool = False) -> str:
    """Lay out flat grammar text: a new line at each header token, the following
    tokens joined onto it. With ``indent``, each header opens a block on its own
    line indented by its nesting level (``_HEADER_LEVEL`` * ``_INDENT_UNIT``
    spaces) and its fields pack into aligned columns beneath it; without it, each
    header group is one space-joined line. Pure whitespace — :func:`encode`
    ignores it — so it never changes the id stream."""
    lines: list[str] = []
    group: list[str] = []
    base = 0  # leading indent of the current header group (the indent option)

    def flush() -> None:
        if not group:
            return
        pad = " " * base if indent else ""
        if indent and len(group) > 1:
            head, *fields = group
            lines.append(pad + head)
            lines.extend(_layout_fields(fields, base))
        else:
            lines.append(pad + " ".join(group))
        group.clear()

    for name, unit in _scan_units(flat):
        level = HEADER_DISPLAY_LEVEL.get(name)
        if level is not None:
            flush()
            base = _INDENT_UNIT * level
        group.append(unit)

    flush()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# composed public API: render / parse
# ---------------------------------------------------------------------------


def render(
    tokens: Iterable[TokenLike],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    wall_names: Sequence[str] | None = None,
    flat_names: Sequence[str] | None = None,
    physical_values: bool = True,
    angle_degrees: bool = False,
    indent: bool = False,
) -> str:
    """Render a token stream to canonical readable text — the grammar's flat
    detokenization (:func:`decode`) laid out by the whitespace :func:`format`ter.

    Knobs (none change the re-encoded id stream):
      * ``origin`` — subset centroid; with it, x/y coordinate markers render as
        raw WAD units instead of scene-relative.
      * ``wall_names`` / ``flat_names`` — asset name tables; texture/flat slots
        render WAD names instead of raw ids.
      * ``physical_values`` — ``False`` renders the raw ``[-1, 1]`` carrier.
      * ``angle_degrees`` — ``True`` renders BAM angles as degrees.
      * ``indent`` — semantic nesting layout (see :func:`format`).
    """
    flat = decode(
        tokens,
        origin=origin,
        wall_names=wall_names,
        flat_names=flat_names,
        physical_values=physical_values,
        angle_degrees=angle_degrees,
    )
    return format(flat, indent=indent)


def parse(
    text: str,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    wall_names: Sequence[str] | None = None,
    flat_names: Sequence[str] | None = None,
    physical_values: bool = True,
    angle_degrees: bool = False,
    indent: bool = False,
) -> list[tuple[TokenType, dict[str, int | float]]]:
    """Inverse of :func:`render`: readable text -> ``(TokenType, values)`` list.
    Whitespace / layout is ignored (so the formatter is invisible), so this is
    just :func:`encode`; the knobs mirror :func:`render` (pass the same dict to
    both) and must match the render settings for a byte-exact round trip.
    ``indent`` is accepted for symmetry and ignored — layout never reaches the
    id stream."""
    del indent  # layout is dropped by `encode`'s whitespace-agnostic scan
    return encode(
        text,
        origin=origin,
        wall_names=wall_names,
        flat_names=flat_names,
        physical_values=physical_values,
        angle_degrees=angle_degrees,
    )
