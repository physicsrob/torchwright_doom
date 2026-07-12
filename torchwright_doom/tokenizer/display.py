"""The vanilla per-id label: one context-free, reversible string per token id.

This module is the **single source of per-token text**. :func:`token_label`
turns one ``(TokenType, slot_values)`` row into its readable label —
``TYPE(arg, ...)`` (bare ``TYPE`` when it has no args) — and nothing else: it
reads only that one row, never a neighbour (the one cross-token rule, folding a
``VALUE`` carrier into its marker, lives in :mod:`.surface`, the grammar). Two
callers share it:

* :func:`embedding._row_label` — the HuggingFace ``WordLevel`` vocab labels,
  called with ``wall_names=flat_names=None`` so the vocab stays **config-free**
  (raw ids, asset-independent fingerprint), and
* the grammar renderer (:mod:`.surface`) — called *with* the asset name tables so
  the blog figure shows ``STARTAN3`` instead of a raw id.

Every transform here is a **bijection** (a 1:1 relabel of one row), so the
re-encoded token-id stream is byte-identical whether the label is the raw
``TYPE(slot=value)`` form or the pretty one. "Readability lives in the labels":
the pretty label *is* the canonical label — there is no on/off knob. The
transforms, all reversible:

* **type aliases** — drop the redundant entity prefix (``node.`` / ``seg.``) so
  ``node.x`` reads ``x`` and ``seg.front.floor`` reads ``floor``; the line's
  header already names the entity. ``bbox.`` / ``drawseg.`` / ``R_*`` families
  keep their prefix (it is meaningful and avoids name collisions, e.g.
  ``bbox.angle1`` vs the seg-projection ``angle1``). :data:`TYPE_BY_DISPLAY`
  inverts the alias map; the module raises at import if two types ever collide on
  one display name.
* **positional args** — for types whose name already says what each value is
  (``pixel(143, 2)``, ``setCursorX(68)``, a decoded ``child1(ss5)``), drop the
  redundant ``slot=`` and render the value bare. Parse maps positional args back
  by declaration order.
* **value decode** — turn an opaque integer slot into the source-shaped word DOOM
  means by it: enum codes (``wall_kind`` 2 → ``portal``), booleans (``flag`` 1 →
  ``yes``), the unified BSP child id (``child_u`` 69 → ``ss5``), the wall-part
  existence mask (``pat`` 5 → ``mid+lower``), and the bbox region as a 2-letter
  grid code ``XY`` — column X ∈ {L,C,R} (left/center/right), row Y ∈ {A,C,B}
  (above/center/below), so ``center-below`` reads ``CB``. Each has an exact
  inverse for parse; any value outside a table's domain falls back to the raw
  integer (still reversible).
* **value-in-name tags** — fold a dominant enum/bool slot into the word itself,
  for **sandbox / invented protocol tokens only**: ``hasBacksector(flag=1)`` →
  ``twoSided``, ``planeMark(kind=floor)`` → ``floorMark`` (keeping ``p`` / ``vp``),
  ``pointOnSideResult(side=front)`` → ``frontSideResult`` (keeping ``node``).

Honesty note (token-naming doctrine): this layer only *prettifies and decodes*.
It never folds a value into — or renames — a real DOOM call (``R_*`` / ``ST_*``):
``R_CheckPlane``'s ``kind`` goes positional, not into the name. See
``GLOSSARY.md`` for the provenance-based naming convention.

Imports only :mod:`.tokens` and the :mod:`.vocab` registry (plus two scalar
constants) — torch-free, never imports :mod:`.embedding`, so it can sit low in
the import graph and be shared by both callers without a cycle.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..model.tokens import FloatSlot, TokenType
from ..model.vocab import BACK_HEIGHT_SENTINEL, N_NODES_MAX, VOCAB_TYPES

# ---------------------------------------------------------------------------
# type aliases (drop the redundant entity prefix; short common words)
# ---------------------------------------------------------------------------

# The leading entity segments we drop entirely: the line's header already names
# the entity. The ``R_*`` family keeps its prefix (it is a real DOOM call name).
# The ``bbox.`` / ``drawseg.`` families can't drop the prefix (``bbox.angle1``
# would collide with the seg-projection ``angle1``), but the long form bloats
# every field, so they are *shortened* to a unique stub instead (see below).
_STRIP_PREFIXES = ("node.", "seg.")

# Long, repetitive family prefixes shortened to a unique stub (collision-free):
# ``bbox.`` fuses to ``b`` (``bbox.x1`` -> ``bx1``); ``drawseg.`` -> ``ds.``
# (``drawseg.scale1.den`` -> ``ds.scale1.den``).
_SHORTEN_PREFIXES = (("bbox.", "b"), ("drawseg.", "ds."))


def _alias_for(name: str) -> str:
    """The display alias for a canonical type name. A composition of reversible
    relabels: drop the entity prefix (``node.`` / ``seg.``); drop the default
    ``front.`` qualifier so a front height reads ``floor`` / ``ceil`` while the
    neighbour stays qualified (``back.floor`` / ``back.ceil``); shorten the two
    common words ``ceiling`` → ``ceil`` and ``SSECTOR`` → ``ss``; and shorten the
    long ``bbox.`` / ``drawseg.`` family prefixes (``bbox.x1`` → ``bx1``,
    ``drawseg.scale1`` → ``ds.scale1``)."""
    for pre in _STRIP_PREFIXES:
        if name.startswith(pre):
            name = name[len(pre) :]
            break
    if name.startswith("front."):
        name = name[len("front.") :]
    name = name.replace("ceiling", "ceil")
    if name == "SSECTOR":
        name = "ss"
    for pre, stub in _SHORTEN_PREFIXES:
        if name.startswith(pre):
            name = stub + name[len(pre) :]
            break
    return name


#: Canonical ``TokenType.name`` -> the type, for parse-side lookups + tag wiring.
_TYPE_BY_CANON: dict[str, TokenType] = {t.name: t for t in VOCAB_TYPES}

#: Token type -> its short display name (== the canonical name for unstripped
#: families). The blog figure and the HF vocab both render through these.
DISPLAY_NAME: dict[TokenType, str] = {t: _alias_for(t.name) for t in VOCAB_TYPES}

#: Inverse of :data:`DISPLAY_NAME`. Built eagerly so a future vocab rename that
#: collides two types onto one alias fails loud at import, not silently at parse.
TYPE_BY_DISPLAY: dict[str, TokenType] = {}
for _type, _alias in DISPLAY_NAME.items():
    if _alias in TYPE_BY_DISPLAY:
        raise ValueError(
            f"display-name collision: {_alias!r} from both {_type.name!r} and "
            f"{TYPE_BY_DISPLAY[_alias].name!r}. Keep one prefix (drop it from "
            "_STRIP_PREFIXES or special-case it) so the alias stays a bijection."
        )
    TYPE_BY_DISPLAY[_alias] = _type


# ---------------------------------------------------------------------------
# texture / flat slots (rendered as WAD asset names when the tables are passed)
# ---------------------------------------------------------------------------

# (type_name, slot_name) -> asset kind for slots that render as WAD names.
_TEXTURE_SLOTS: dict[tuple[str, str], str] = {
    ("seg.texture.mid", "tex_id"): "wall",
    ("seg.texture.upper", "tex_id"): "wall",
    ("seg.texture.lower", "tex_id"): "wall",
    ("planeDef", "flat_id"): "flat",
}
_NO_TEXTURE = "none"  # tex_id == 0: the "-" / empty side


# ---------------------------------------------------------------------------
# positional args (render bare, no ``slot=``)
# ---------------------------------------------------------------------------

# Types whose int slots render as bare positional args (``pixel(143, 2)``) rather
# than ``slot=value`` — the type name (or the decoded word) already says what each
# value is, so the names would be noise. Everything else stays named. Layout only
# — parse maps positional args back by declaration order — so this set is freely
# tunable. Carriers are always positional (one value, named by the marker type).
_POSITIONAL_TYPES: frozenset[str] = frozenset(
    {
        "pixel",
        "seg.texture.mid",
        "seg.texture.upper",
        "seg.texture.lower",
        # single self-evident index/coordinate slots
        "setCursorX",
        "setCursorY",
        "screenY",
        "clipScan",
        "bboxClipScan",
        "R_AddLine",
        "nextSeg",
        "R_StoreWallRange",
        "drawseg.x2",
        "R_MakeSpans.col",
        "R_MapPlane.row",
        # decoded-word slots whose meaning is self-evident positionally
        "node.child1",
        "node.child0",
        "boxpos",
        "bbox.x1",
        "bbox.y1",
        "bbox.x2",
        "bbox.y2",
        "seg.emptyLine",
        "seg.closedDoor",
        "seg.pegging",
        "segKpart",
        "R_CheckPlane",  # kind positional (NOT folded — honesty: real DOOM call)
        "R_CheckPlane.result",
    }
)


# ---------------------------------------------------------------------------
# value decoding
# ---------------------------------------------------------------------------

# (type_name, slot_name) -> ordered names; the slot int indexes the list.
_ENUMS: dict[tuple[str, str], list[str]] = {
    ("drawseg.meta", "wall_kind"): ["solid", "closed", "portal"],
    ("drawseg.meta", "silhouette"): ["none", "bottom", "top", "both"],
    ("R_CheckPlane", "kind"): ["ceiling", "floor"],
    ("R_CheckPlane.result", "kind"): ["ceiling", "floor"],
    ("planeMark", "kind"): ["ceiling", "floor"],
    ("pointOnSideResult", "side"): ["back", "front"],
}

# (type_name, slot_name) for 0/1 flags rendered as no/yes.
_BOOLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("seg", "is_first_of_ss"),
        ("seg.hasBacksector", "flag"),
        ("seg.emptyLine", "flag"),
        ("seg.closedDoor", "flag"),
        ("seg.pegging", "dontpegtop"),
        ("seg.pegging", "dontpegbottom"),
    }
)
_BOOL_WORDS = ("no", "yes")

# Unified BSP child id slots: < N_NODES_MAX is a node, else a subsector
# (index child_u - N_NODES_MAX). See render_ops.IS_SUBSECTOR.
_CHILD_SLOTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("node.child1", "child_u"),
        ("node.child0", "child_u"),
        ("bspReturn", "entity_u"),
    }
)

# R_CheckBBox region code slots: boxpos = 4*boxy + boxx (boxx 0..2, boxy 0..2),
# decoded to a 2-letter grid code XY — X column in {L,C,R} (left/center/right),
# Y row in {A,C,B} (above/center/below). So ``center-below`` reads ``CB``,
# ``right-center`` reads ``RC``. First char is always the column, second the row.
_BOXPOS_SLOTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("boxpos", "boxpos"),
        ("bbox.x1", "boxpos"),
        ("bbox.y1", "boxpos"),
        ("bbox.x2", "boxpos"),
        ("bbox.y2", "boxpos"),
    }
)
_BOXX = ("L", "C", "R")  # left / center / right (the column, first char)
_BOXY = ("A", "C", "B")  # above / center / below (the row, second char)

# Wall-part existence mask: pat = 4*has_mid + 2*has_upper + has_lower.
_KPART_SLOTS: frozenset[tuple[str, str]] = frozenset({("segKpart", "pat")})
_KPART_BITS = (("mid", 4), ("upper", 2), ("lower", 1))

# Markers whose VALUE carrier is the "no back sector" height sentinel.
_SENTINEL_MARKERS: frozenset[str] = frozenset({"seg.back.floor", "seg.back.ceiling"})


def decode_slot(type_name: str, slot_name: str, ivalue: int) -> str | None:
    """The source-shaped word for an int slot, or ``None`` to keep the integer."""
    key = (type_name, slot_name)
    names = _ENUMS.get(key)
    if names is not None and 0 <= ivalue < len(names):
        return names[ivalue]
    if key in _BOOLS and ivalue in (0, 1):
        return _BOOL_WORDS[ivalue]
    if key in _CHILD_SLOTS:
        if ivalue >= N_NODES_MAX:
            return f"ss{ivalue - N_NODES_MAX}"
        return f"node{ivalue}"
    if key in _BOXPOS_SLOTS:
        boxx, boxy = ivalue % 4, ivalue // 4
        if boxx < len(_BOXX) and boxy < len(_BOXY):
            return f"{_BOXX[boxx]}{_BOXY[boxy]}"
    if key in _KPART_SLOTS and 0 <= ivalue < 8:
        parts = [name for name, bit in _KPART_BITS if ivalue & bit]
        return "+".join(parts) if parts else "none"
    return None


def encode_slot(type_name: str, slot_name: str, text: str) -> int | None:
    """Inverse of :func:`decode_slot`, or ``None`` if ``text`` isn't a decoded
    word for this slot (then the caller parses it as a plain int / asset name)."""
    key = (type_name, slot_name)
    names = _ENUMS.get(key)
    if names is not None and text in names:
        return names.index(text)
    if key in _BOOLS and text in _BOOL_WORDS:
        return _BOOL_WORDS.index(text)
    if key in _CHILD_SLOTS:
        if text.startswith("ss") and text[2:].isdigit():
            return int(text[2:]) + N_NODES_MAX
        if text.startswith("node") and text[4:].isdigit():
            return int(text[4:])
        return None
    if key in _BOXPOS_SLOTS:
        if len(text) == 2 and text[0] in _BOXX and text[1] in _BOXY:
            return 4 * _BOXY.index(text[1]) + _BOXX.index(text[0])
        return None
    if key in _KPART_SLOTS:
        if text == "none":
            return 0
        bits = dict(_KPART_BITS)
        parts = text.split("+")
        if all(p in bits for p in parts):
            return sum(bits[p] for p in parts)
        return None
    return None


def decode_sentinel(marker_name: str, physical: float) -> str | None:
    """``none`` when a back-sector height marker carries the absence sentinel."""
    if marker_name in _SENTINEL_MARKERS and abs(physical - BACK_HEIGHT_SENTINEL) < 0.5:
        return "none"
    return None


def encode_sentinel(marker_name: str, word: str) -> float | None:
    """Inverse: the sentinel physical for ``none`` on a back-sector marker."""
    if marker_name in _SENTINEL_MARKERS and word == "none":
        return BACK_HEIGHT_SENTINEL
    return None


# ---------------------------------------------------------------------------
# value-in-name tags (fold a dominant enum/bool slot into the word)
# ---------------------------------------------------------------------------

# canonical type name -> (slot folded into the word, {value: tag word}, slots kept
# as ``name=value`` args). Only **sandbox / invented protocol** tokens (lowerCamel
# names) — never a real DOOM call. Honesty guard: ``R_CheckPlane``'s ``kind`` goes
# *positional* instead (see _POSITIONAL_TYPES), it is not folded into the name.
_TAG: dict[str, tuple[str, dict[int, str], tuple[str, ...]]] = {
    "seg.hasBacksector": ("flag", {0: "oneSided", 1: "twoSided"}, ()),
    "planeMark": ("kind", {0: "ceilMark", 1: "floorMark"}, ("p", "vp")),
    "pointOnSideResult": (
        "side",
        {0: "backSideResult", 1: "frontSideResult"},
        ("node",),
    ),
}

#: Parse-side inverse of :data:`_TAG`: tag word -> (type, the folded slot value).
TAG_INVERSE: dict[str, tuple[TokenType, dict[str, int]]] = {}
for _canon, (_fold, _vmap, _keep) in _TAG.items():
    _ttype = _TYPE_BY_CANON[_canon]
    for _val, _word in _vmap.items():
        if _word in TYPE_BY_DISPLAY or _word in TAG_INVERSE:
            raise ValueError(f"value-in-name tag collision: {_word!r}")
        TAG_INVERSE[_word] = (_ttype, {_fold: _val})


# ---------------------------------------------------------------------------
# per-slot value rendering / parsing (shared by token_fields and the grammar)
# ---------------------------------------------------------------------------


def slot_value_str(
    type_name: str,
    slot_name: str,
    value: int | float,
    wall_names: Sequence[str] | None,
    flat_names: Sequence[str] | None,
) -> str:
    """The displayed value for one int slot: a decoded word (``portal``, ``ss5``,
    ``yes``) when the slot has a known meaning, else a WAD name for texture/flat
    slots (when the name table is given), else the raw integer."""
    ivalue = int(value)
    decoded = decode_slot(type_name, slot_name, ivalue)
    if decoded is not None:
        return decoded
    kind = _TEXTURE_SLOTS.get((type_name, slot_name))
    if kind == "wall" and wall_names is not None:
        return _NO_TEXTURE if ivalue == 0 else wall_names[ivalue - 1]
    if kind == "flat" and flat_names is not None:
        return flat_names[ivalue]
    return str(ivalue)


def parse_slot(
    type_name: str,
    slot_name: str,
    text: str,
    wall_names: Sequence[str] | None,
    flat_names: Sequence[str] | None,
) -> int:
    """Inverse of :func:`slot_value_str`: a decoded word / WAD name / int back to
    the slot's integer value."""
    encoded = encode_slot(type_name, slot_name, text)
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


# ---------------------------------------------------------------------------
# the per-id label
# ---------------------------------------------------------------------------


def token_fields(
    ttype: TokenType,
    values: Mapping[str, int | float],
    *,
    wall_names: Sequence[str] | None = None,
    flat_names: Sequence[str] | None = None,
) -> tuple[str, list[str]]:
    """The context-free ``(name, args)`` for one token row — the per-id label
    split so the grammar can fold a marker's carrier onto the end of ``args``.

    ``name`` is the type's display alias (or a value-in-name tag word, when a
    dominant slot is folded into the word); ``args`` are its remaining slot values
    rendered positionally (for :data:`_POSITIONAL_TYPES`, where the name says what
    each value is) or ``slot=value`` otherwise. Reads only this one row — never a
    neighbour — so a marker's carrier is *not* here (that is the grammar's one
    contextual rule)."""
    tag = _TAG.get(ttype.name)
    if tag is not None and int(values[tag[0]]) in tag[1]:
        fold, vmap, keep = tag
        name = vmap[int(values[fold])]  # the folded slot becomes the word
        slot_names: tuple[str, ...] = keep  # kept slots stay named
        positional = False
    else:
        name = DISPLAY_NAME[ttype]
        slot_names = tuple(ttype.slots)
        positional = ttype.name in _POSITIONAL_TYPES
    args: list[str] = []
    for slot_name in slot_names:
        slot = ttype.slots[slot_name]
        if isinstance(slot, FloatSlot):
            sval = f"{float(values[slot_name]):g}"  # the raw carrier (no marker)
        else:
            sval = slot_value_str(
                ttype.name, slot_name, values[slot_name], wall_names, flat_names
            )
        args.append(sval if positional else f"{slot_name}={sval}")
    return name, args


def token_label(
    ttype: TokenType,
    values: Mapping[str, int | float],
    *,
    wall_names: Sequence[str] | None = None,
    flat_names: Sequence[str] | None = None,
) -> str:
    """The single context-free label for one token id: ``TYPE(arg, ...)`` (bare
    ``TYPE`` when it has no args). The shared source for the HF vocab
    (``embedding._row_label``, called config-free) and the grammar figure."""
    name, args = token_fields(
        ttype, values, wall_names=wall_names, flat_names=flat_names
    )
    return name if not args else f"{name}({', '.join(args)})"


def token_word(
    ttype: TokenType,
    values: Mapping[str, int | float],
    *,
    wall_names: Sequence[str] | None = None,
    flat_names: Sequence[str] | None = None,
) -> str:
    """The canonical stock-WordLevel word for one token row.

    This deliberately derives from :func:`token_fields`, exactly like
    :func:`token_label`.  The only difference is lexical: commas carry no
    following whitespace, so the complete row label is one unit under a
    ``WhitespaceSplit`` pre-tokenizer.  Keeping both spellings on the same
    field representation prevents the published tokenizer and the readable
    formatter from developing separate naming systems.
    """
    name, args = token_fields(
        ttype, values, wall_names=wall_names, flat_names=flat_names
    )
    return name if not args else f"{name}({','.join(args)})"


def resolve_name(name: str) -> tuple[TokenType, dict[str, int]] | None:
    """Parse-side inverse of the display name: the ``TokenType`` a leading label
    word denotes, plus any slot values folded into that word (the value-in-name
    tags, e.g. ``twoSided`` -> ``hasBacksector`` with ``flag=1``). ``None`` for an
    unknown word."""
    ttype = TYPE_BY_DISPLAY.get(name)
    if ttype is not None:
        return ttype, {}
    tagged = TAG_INVERSE.get(name)
    if tagged is not None:
        ttype, preset = tagged
        return ttype, dict(preset)
    return None
