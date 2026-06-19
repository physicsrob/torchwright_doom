"""Display layer for the readable surface: short type aliases and value
decoding. Pure, context-free, and reversible — every transform is a bijection,
so the re-encoded token-id stream is byte-identical with these knobs on or off.

Two independent display knobs consume this module (see :mod:`.surface`):

* **strip_prefixes** — drop the redundant entity prefix (``node.`` / ``seg.``)
  from a token's display name. The line's header already names the entity, so
  ``seg.front.floor`` reads as ``front.floor`` and ``node.x`` as ``x``. Only
  ``node.`` / ``seg.`` are stripped; the ``bbox.`` / ``drawseg.`` / ``R_*``
  families keep their prefix — it is meaningful (and keeping it is what avoids
  the only name collisions, e.g. ``bbox.angle1`` vs the seg-projection
  ``angle1``). :data:`TYPE_BY_DISPLAY` inverts the alias map and the module
  raises at import if two types ever collide on one display name.

* **decode_values** — turn an opaque integer slot (or sentinel carrier) into the
  source-shaped word DOOM means by it: enum codes (``wall_kind`` 2 -> ``portal``),
  booleans (``flag`` 1 -> ``yes``), the unified BSP child id (``child_u`` 69 ->
  ``ss5``), the R_CheckBBox region code (``boxpos`` 6 -> ``right-center``), the
  wall-part existence mask (``pat`` 5 -> ``mid+lower``), and the "no back sector"
  height sentinel (``-4096`` -> ``none``). Each has an exact inverse for parse;
  any value outside a table's domain falls back to the raw integer (still
  reversible).

Honesty note (token-naming doctrine): this layer only *prettifies and decodes*.
It never renames a serialized-data or sandbox-protocol token to look like a real
DOOM call — e.g. it does not promote the ``boxpos`` / ``bbox.*`` cluster to an
``R_CheckBBox`` prefix, and it leaves genuinely-internal carriers
(``segDcTmidMid``, the scale denominators) raw.

Imports only :mod:`.tokens` and the :mod:`.vocab` registry (plus two scalar
constants) — torch-free, like the rest of the surface.
"""

from __future__ import annotations

from ..tokens import TokenType
from ..vocab import BACK_HEIGHT_SENTINEL, N_NODES_MAX, VOCAB_TYPES

# ---------------------------------------------------------------------------
# type aliases (the strip_prefixes knob)
# ---------------------------------------------------------------------------

# The leading entity segments we drop. Everything after the first dot is kept;
# the bbox./drawseg./R_* families are intentionally NOT stripped.
_STRIP_PREFIXES = ("node.", "seg.")


def _alias_for(name: str) -> str:
    for pre in _STRIP_PREFIXES:
        if name.startswith(pre):
            return name[len(pre) :]
    return name


#: Token type -> its short display name (== the canonical name for unstripped
#: families). The blog figure renders with these; the HF tokenizer does not.
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
# value decoding (the decode_values knob)
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

# R_CheckBBox region code slots: boxpos = 4*boxy + boxx (boxx 0..2, boxy 0..2).
_BOXPOS_SLOTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("boxpos", "boxpos"),
        ("bbox.x1", "boxpos"),
        ("bbox.y1", "boxpos"),
        ("bbox.x2", "boxpos"),
        ("bbox.y2", "boxpos"),
    }
)
_BOXX = ("left", "center", "right")
_BOXY = ("above", "center", "below")

# Wall-part existence mask: pat = 4*has_mid + 2*has_upper + has_lower.
_KPART_SLOTS: frozenset[tuple[str, str]] = frozenset({("segKpart", "pat")})
_KPART_BITS = (("mid", 4), ("upper", 2), ("lower", 1))

#: Types whose single, self-explanatory decoded slot reads better positionally
#: (``child1(ss5)`` not ``child1(child_u=ss5)``). Only honored with decode on.
DECODE_POSITIONAL: frozenset[str] = frozenset(
    {
        "node.child1",
        "node.child0",
        "boxpos",
        "seg.hasBacksector",
        "seg.emptyLine",
        "seg.closedDoor",
        "seg.pegging",
        "segKpart",
    }
)

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
            return f"{_BOXX[boxx]}-{_BOXY[boxy]}"
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
    if key in _BOXPOS_SLOTS and "-" in text:
        bx, by = text.split("-", 1)
        if bx in _BOXX and by in _BOXY:
            return 4 * _BOXY.index(by) + _BOXX.index(bx)
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
