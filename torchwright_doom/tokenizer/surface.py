"""The readable surface grammar: ``render(tokens) -> text`` and
``parse(text) -> tokens``, byte-exact at the token level.

Three tiers of readability, in increasing contextual reach:

1. **Baked context-free labels** (the bulk; pure lookup). Each token renders in
   functional style — ``TYPE(arg, ...)`` (bare ``TYPE`` when it has no args).
   Int slots are ``name=value`` (``seg(i=41, is_first_of_ss=0)``), except for a
   small set of self-evident types whose args are bare positional
   (``pixel(143, 2)``). Texture / flat slots render their WAD names. No
   neighbour is consulted.

2. **One contextual rule — continuous values.** A ``VALUE`` token's id encodes
   only a ``[-1, 1]`` carrier; its physical meaning needs the *preceding
   marker's* range (``marker_ranges.MARKER_RANGE``). So a marked field renders
   as ``marker(<physical>)`` — the de-quantized value as the marker's trailing
   positional arg. The ``physical_values=False`` knob falls back to the raw
   carrier; both re-encode to the same id, so the knob never changes the stream.

3. **Line layout by header-break.** A new line starts at each "header" token
   type (``_HEADER_TYPE_NAMES``); the following tokens join onto it with
   spaces. Pure whitespace layout — reversible because parse ignores it.

Byte-exactness is anchored at the value level by emitting the *shortest decimal
that re-quantizes to the same carrier level* (so ``encode(decode(v)) == v``).
Parse is tolerant (whitespace / newlines / ``#`` comments ignored, slot fields
order-free by name); render is canonical (fixed order, header breaks).

Imports only :mod:`.tokens`, :mod:`.value_ranges`, the :mod:`.vocab` registry,
and the shared :mod:`.marker_ranges` table — no graph nodes at import or call
time (mirrors ``tokens_bridge``'s discipline). Texture names are passed in (not
imported) so the module stays asset-agnostic; they are a reversible display
knob, not part of the WordLevel id<->label core.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from . import display
from ..marker_ranges import ANGLE_MARKERS, MARKER_RANGE
from ..tokens import Token, TokenType
from ..value_ranges import ValueRange, decode_float, encode_float
from ..vocab import ANGLE_BAM, ANGLE_VALUE, VALUE, VOCAB_TYPES

TokenLike = Token | tuple[TokenType, Mapping[str, int | float]]

# Name -> TokenType for the whole vocab (the keyword set the parser recognises).
_TYPE_BY_NAME: dict[str, TokenType] = {t.name: t for t in VOCAB_TYPES}

# (type_name, slot_name) -> asset kind for slots that render as WAD names.
_TEXTURE_SLOTS: dict[tuple[str, str], str] = {
    ("seg.texture.mid", "tex_id"): "wall",
    ("seg.texture.upper", "tex_id"): "wall",
    ("seg.texture.lower", "tex_id"): "wall",
    ("planeDef", "flat_id"): "flat",
}
_NO_TEXTURE = "none"  # tex_id == 0: the "-" / empty side

# Types whose int slots render as bare positional args (``pixel(143, 2)``)
# rather than ``name=value`` — the type name already says what each value is, so
# the names would be noise. Everything else stays named (``seg(i=41,
# is_first_of_ss=0)``). Layout only — parse maps positional args back by
# declaration order — so this set is freely tunable. Carriers are always
# positional (one value, named by the marker type).
_POSITIONAL_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "pixel",
        "seg.texture.mid",
        "seg.texture.upper",
        "seg.texture.lower",
    }
)

# The 16-bit VALUE carrier: 65536 levels evenly over [-1, 1] (65535 steps).
_VALUE_STEPS = 65535

# Header token types: a new display line starts at each (the following tokens
# join onto it). Layout only — parse ignores line breaks, so this set is freely
# tunable without affecting the id stream. Curated for entity-per-line density:
# one line per player block / node / subsector / seg / plane (prompt), and per
# traversal step / wall column / span action (AR output).
_HEADER_TYPE_NAMES: frozenset[str] = frozenset(
    {
        # --- prompt ---
        "viewx",  # opens the player block (viewy/viewz/viewangle join it)
        "node",
        "SSECTOR",
        "seg",
        "planeDef",
        "ssFloorPlane",  # ssCeilingPlane joins it
        "begin",
        # --- AR: traversal + visibility ---
        "noOp",
        "R_PointOnSide",
        "pointOnSideResult",
        "boxpos",
        "bspFront",
        "bspCheckBack",
        "bspReturn",
        "R_Subsector",
        "R_AddLine",
        "nextSeg",
        "drawseg.x2",
        # --- AR: wall range + columns ---
        "R_StoreWallRange",
        "drawseg.meta",
        "R_CheckPlane",
        "setCursorX",
        # --- AR: flat pass ---
        "R_DrawPlanes",
        "R_DrawPlanes.nextPlane",
        "R_DrawPlanes.nextVp",
        "visplaneBegin",
        "R_MakeSpans.col",
        "R_MapPlane.row",
        # --- AR: weapon + status bar + terminal ---
        "R_DrawPlayerSprites",
        "ST_Drawer",
        "ST_Drawer.item",
        "done",
    }
)

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
    same carrier level — guaranteeing ``parse(render(v)) == v`` at the id."""
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


def _slot_value_str(
    type_name: str,
    slot_name: str,
    value: int | float,
    wall_names: Sequence[str] | None,
    flat_names: Sequence[str] | None,
    decode_values: bool,
) -> str:
    """The displayed value for one int slot: a decoded word (``portal``, ``ss5``,
    ``yes``) when ``decode_values`` and the slot has a known meaning, else a WAD
    name for texture/flat slots, else the raw integer."""
    ivalue = int(value)
    if decode_values:
        decoded = display.decode_slot(type_name, slot_name, ivalue)
        if decoded is not None:
            return decoded
    kind = _TEXTURE_SLOTS.get((type_name, slot_name))
    if kind == "wall" and wall_names is not None:
        return _NO_TEXTURE if ivalue == 0 else wall_names[ivalue - 1]
    if kind == "flat" and flat_names is not None:
        return flat_names[ivalue]
    return str(ivalue)


def _parse_slot(
    type_name: str,
    slot_name: str,
    text: str,
    wall_names: Sequence[str] | None,
    flat_names: Sequence[str] | None,
    decode_values: bool,
) -> int:
    if decode_values:
        encoded = display.encode_slot(type_name, slot_name, text)
        if encoded is not None:
            return encoded
    kind = _TEXTURE_SLOTS.get((type_name, slot_name))
    if kind == "wall" and wall_names is not None and not _looks_int(text):
        if text == _NO_TEXTURE:
            return 0
        return wall_names.index(text) + 1
    if kind == "flat" and flat_names is not None and not _looks_int(text):
        return flat_names.index(text)
    return int(text)


def _looks_int(text: str) -> bool:
    body = text[1:] if text[:1] in "+-" else text
    return body.isdigit()


def _format_token(
    ttype: TokenType,
    values: Mapping[str, int | float],
    carrier: str | None,
    wall_names: Sequence[str] | None,
    flat_names: Sequence[str] | None,
    strip_prefixes: bool,
    decode_values: bool,
) -> str:
    """One token in functional style: ``TYPE(arg, ...)`` (bare ``TYPE`` if it has
    no args). The type name is the short alias under ``strip_prefixes``. Int
    slots are positional for ``_POSITIONAL_TYPE_NAMES`` (where the type name
    already says what the value is), for ``display.DECODE_POSITIONAL`` types when
    decoding (where the decoded word is self-evident), and ``name=value``
    otherwise; a marker's de-quantized ``carrier`` is the trailing positional
    arg."""
    name = display.DISPLAY_NAME[ttype] if strip_prefixes else ttype.name
    positional = ttype.name in _POSITIONAL_TYPE_NAMES or (
        decode_values and ttype.name in display.DECODE_POSITIONAL
    )
    args: list[str] = []
    for slot_name in ttype.slots:
        sval = _slot_value_str(
            ttype.name,
            slot_name,
            values[slot_name],
            wall_names,
            flat_names,
            decode_values,
        )
        args.append(sval if positional else f"{slot_name}={sval}")
    if carrier is not None:
        args.append(carrier)
    if not args:
        return name
    return f"{name}({', '.join(args)})"


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def render(
    tokens: Iterable[TokenLike],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    wall_names: Sequence[str] | None = None,
    flat_names: Sequence[str] | None = None,
    physical_values: bool = True,
    angle_degrees: bool = False,
    strip_prefixes: bool = False,
    decode_values: bool = False,
) -> str:
    """Render a token stream to canonical readable text.

    ``tokens`` are ``Token`` instances or ``(TokenType, values)`` tuples,
    *including* the ``VALUE`` / ``ANGLE_VALUE`` carriers in stream order — each
    carrier folds into its marker's parens as the trailing positional arg.

    Knobs (none change the re-encoded id stream):
      * ``origin`` — subset centroid; with it, x/y coordinate markers render as
        raw WAD units instead of scene-relative.
      * ``wall_names`` / ``flat_names`` — asset name tables; texture/flat slots
        render WAD names instead of raw ids.
      * ``physical_values`` — ``False`` renders the raw ``[-1, 1]`` carrier.
      * ``angle_degrees`` — ``True`` renders BAM angles as degrees.
      * ``strip_prefixes`` — drop the ``node.`` / ``seg.`` entity prefix from
        type names (see :mod:`.display`).
      * ``decode_values`` — render opaque integer slots / sentinels as their
        source-shaped words (see :mod:`.display`).
    """
    toks = [_norm(t) for t in tokens]
    lines: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            lines.append(" ".join(current))
            current.clear()

    i = 0
    n = len(toks)
    while i < n:
        ttype, values = toks[i]
        if ttype is VALUE or ttype is ANGLE_VALUE:
            raise ValueError(f"carrier {ttype.name!r} at {i} has no preceding marker")

        # Marker-lookahead for the single contextual rule: if the next token is a
        # carrier, it is *this* token's (carriers only ever follow their marker),
        # and we de-quantize it through this marker's binding. A marker type like
        # setCursorY only carries a value in some contexts — keying on whether a
        # carrier actually follows handles both.
        carrier: str | None = None
        if i + 1 < n and toks[i + 1][0] in (VALUE, ANGLE_VALUE):
            ctype, cvals = toks[i + 1]
            carrier = _render_carrier(
                ttype,
                ctype,
                cvals,
                origin,
                physical_values,
                angle_degrees,
                decode_values,
            )
            i += 1

        if ttype.name in _HEADER_TYPE_NAMES:
            flush()
        current.append(
            _format_token(
                ttype,
                values,
                carrier,
                wall_names,
                flat_names,
                strip_prefixes,
                decode_values,
            )
        )
        i += 1

    flush()
    return "\n".join(lines)


def _render_carrier(
    marker: TokenType,
    carrier_type: TokenType,
    carrier_values: Mapping[str, int | float],
    origin: tuple[float, float],
    physical_values: bool,
    angle_degrees: bool,
    decode_values: bool,
) -> str:
    if carrier_type is VALUE:
        range_id = MARKER_RANGE.get(marker)
        if range_id is None:
            raise ValueError(f"VALUE follows non-marker {marker.name!r}")
        carrier = float(carrier_values["v"])
        if physical_values:
            shift = _origin_shift(marker.name, origin)
            if decode_values:
                word = display.decode_sentinel(
                    marker.name, decode_float(range_id, carrier) + shift
                )
                if word is not None:
                    return word
            return _shortest_value(range_id, carrier, shift)
        return repr(carrier)
    if marker not in ANGLE_MARKERS:
        raise ValueError(f"ANGLE_VALUE follows non-angle-marker {marker.name!r}")
    return _shortest_angle(int(carrier_values["angle"]), angle_degrees)


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def _scan(text: str) -> list[tuple[str, list[str] | None]]:
    """Split readable text into ``(type_name, args)`` units. ``args`` is the
    comma-separated list inside ``(...)`` (possibly empty), or ``None`` for a
    bare slotless type. Whitespace, newlines, and ``#`` comments are ignored;
    spaces inside the parens (``pixel(143, 2)``) are fine."""
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


def parse(
    text: str,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    wall_names: Sequence[str] | None = None,
    flat_names: Sequence[str] | None = None,
    physical_values: bool = True,
    angle_degrees: bool = False,
    strip_prefixes: bool = False,
    decode_values: bool = False,
) -> list[tuple[TokenType, dict[str, int | float]]]:
    """Inverse of :func:`render`: readable text -> ``(TokenType, values)`` list.

    Tolerant of whitespace, newlines, and ``#`` comments; named args are
    order-free, positional args map back in declaration order. The knobs mirror
    :func:`render` and must match the render settings for a byte-exact round
    trip.
    """
    type_lookup = display.TYPE_BY_DISPLAY if strip_prefixes else _TYPE_BY_NAME
    out: list[tuple[TokenType, dict[str, int | float]]] = []
    for name, args in _scan(text):
        ttype = type_lookup.get(name)
        if ttype is None:
            raise ValueError(f"unknown token keyword: {name!r}")
        slot_values: dict[str, int | float] = {}
        bare: list[str] = []
        for arg in args or ():
            if "=" in arg:
                key, value = arg.split("=", 1)
                slot_values[key] = _parse_slot(
                    ttype.name, key, value, wall_names, flat_names, decode_values
                )
            else:
                bare.append(arg)

        # Positional bare args fill the still-unnamed slots in declaration order;
        # at most one leftover bare arg is the marker's carrier value.
        unfilled = [s for s in ttype.slots if s not in slot_values]
        carrier_word: str | None = None
        for index, value in enumerate(bare):
            if index < len(unfilled):
                slot_values[unfilled[index]] = _parse_slot(
                    ttype.name,
                    unfilled[index],
                    value,
                    wall_names,
                    flat_names,
                    decode_values,
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
            sentinel = (
                display.encode_sentinel(ttype.name, carrier_word)
                if decode_values
                else None
            )
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
