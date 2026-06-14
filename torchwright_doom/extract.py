"""In-graph extract helpers: reading a token's fields back from a
position's ``input_vec``.

The mirror image of ``emit.py``. ``emit`` writes a residual row that
matches one row of ``W_EMBED``; the embedding lookup produces an
``input_vec`` at every position, and these helpers decode that
``input_vec`` back into per-(type, slot) values, type indicators, and
derived-column reads. They are the primitives a ``forward()`` body
uses to dispatch on the input token's type and read its payload.

Public surface:

* :func:`is_type` — 1-wide 0/1 indicator that the active token is of a
  given type.
* :func:`extract_type_slot` / :func:`extract_type_slot_raw` — read one
  specific ``(token_type, slot_name)`` raw column, optionally
  auto-masked to zero when the active type is different.
* :func:`extract_int_slot` / :func:`extract_int_slot_raw`,
  :func:`extract_float_slot` / :func:`extract_float_slot_raw` —
  flat-namespace reads: sum the raw cols across every type that
  declares ``name``, returning the active type's value (or 0 / ``lo``
  when no declaring type is active).
* :func:`extract_derived` — read a derived column by name (depth 0
  data, the embedding builder pre-evaluated ``fn`` here).
* :func:`type_code` — the 8-wide E8 code for a token type, useful as
  the type-half of a composite attention key.

The compiler's heuristic scheduler does not fuse chained ``Linear``
nodes — each is its own layer. The helpers below follow the same
discipline as ``emit.py``: collapse the column-selector / scale / bias
into a single ``Linear`` so a single read is one layer that folds
into the consumer.

Threshold math for :func:`is_type`: every type's 8-wide E8 code is a
10x-scaled spherical-code unit vector, so the self-dot is 1600. In the
current vocab the maximum cross-dot between any two distinct types is
1200 — leaving a 400-unit gap on either side of the halfway threshold
1400. Far larger than the 1/sharpness = 0.1 ramp half-width of
``compare`` at the module default ``step_sharpness = 10``.
"""

from __future__ import annotations

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.arithmetic_ops import compare
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import cond_gate

from .embedding import D_CATEGORY, TOKEN_VOCAB
from .tokens import FloatSlot, IntSlot, TokenType

__all__ = [
    "is_type",
    "is_type_pm1",
    "type_code",
    "input_type_code",
    "type_matches",
    "type_matches_any",
    "indicator_to_bool",
    "extract_type_slot",
    "extract_type_slot_raw",
    "extract_int_slot",
    "extract_int_slot_raw",
    "extract_float_slot",
    "extract_float_slot_raw",
    "extract_derived",
]


# Minimum gap between self-dot and worst-case cross-dot before
# :func:`compare` saturation stops being trustworthy at the default
# sharpness. Self-dot is fixed at 1600 by the E8 code scaling; the gap
# is the slack ``compare`` has on either side of the halfway threshold.
_IS_TYPE_GAP_MARGIN: float = 200.0
_E8_SELF_DOT: float = 1600.0


# ---------------------------------------------------------------------------
# Internal: cached per-type threshold and dot weights
# ---------------------------------------------------------------------------


def _e8_code(token_type: TokenType) -> torch.Tensor:
    """Return the 8-wide E8 code for ``token_type``, as float32."""
    layout = TOKEN_VOCAB.layout
    if token_type.name not in layout.e8_indices:
        raise ValueError(f"token type {token_type.name!r} is not in the active vocab")
    return index_to_vector(layout.e8_indices[token_type.name]).to(torch.float32)


def _is_type_threshold(token_type: TokenType) -> float:
    """Compute the dot-product threshold for ``token_type``'s is_type
    compare: halfway between the self-dot (1600) and the worst-case
    cross-dot with any other type in the vocab.

    Validates that the gap on either side of the threshold is at least
    ``_IS_TYPE_GAP_MARGIN`` units so ``compare`` saturates cleanly at
    the default sharpness. Raises if not — that's a structural problem
    with the vocab's E8 assignment, not a runtime condition.
    """
    layout = TOKEN_VOCAB.layout
    target = _e8_code(token_type)
    max_cross = float("-inf")
    for other in layout.types:
        if other.name == token_type.name:
            continue
        dot = float(torch.dot(target, _e8_code(other)))
        if dot > max_cross:
            max_cross = dot
    if max_cross == float("-inf"):
        raise ValueError(
            f"vocab has only one type {token_type.name!r}; is_type cannot "
            f"establish a cross-dot threshold"
        )
    gap = _E8_SELF_DOT - max_cross
    if gap <= _IS_TYPE_GAP_MARGIN:
        raise ValueError(
            f"is_type({token_type.name!r}): self-dot {_E8_SELF_DOT} − "
            f"max-cross-dot {max_cross} = {gap}, below required margin "
            f"{_IS_TYPE_GAP_MARGIN}. compare cannot saturate cleanly."
        )
    return 0.5 * (_E8_SELF_DOT + max_cross)


def _dot_with_e8_code(input_vec: Node, token_type: TokenType, *, name: str) -> Node:
    """One ``Linear`` mapping ``input_vec`` to ``input_vec[0:8] · E8(T)``.

    Other columns of ``input_vec`` are zero-weighted; the dot collapses
    to the 8-wide E8 block. Width-1 output.
    """
    layout = TOKEN_VOCAB.layout
    if len(input_vec) != layout.d_embed:
        raise ValueError(
            f"is_type/extract: expected input_vec width "
            f"{layout.d_embed}, got {len(input_vec)}"
        )
    weights = torch.zeros((layout.d_embed, 1), dtype=torch.float32)
    weights[0:D_CATEGORY, 0] = _e8_code(token_type)
    return Linear(input_vec, weights, name=name)


def _is_type_pm1(input_vec: Node, token_type: TokenType) -> Node:
    """±1 indicator that the active token is ``token_type``.

    The compare's output saturates at +1 / −1 because the dot lands at
    ``_E8_SELF_DOT`` (active type) or at most ``max_cross_T`` (any other
    type) — a gap of ≥ ``_IS_TYPE_GAP_MARGIN`` on either side of the
    halfway threshold, far larger than the ``1/step_sharpness = 0.1``
    ramp half-width.
    """
    thresh = _is_type_threshold(token_type)
    dot = _dot_with_e8_code(
        input_vec, token_type, name=f"is_type_dot_{token_type.name}"
    )
    return compare(dot, thresh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_type(input_vec: Node, token_type: TokenType) -> Node:
    """1-wide 0/1 indicator that the active token has type ``token_type``.

    1.0 when the active token's type matches ``token_type``, 0.0
    otherwise. Depth: 1 MLP sublayer
    (the ``compare`` op); the surrounding affines (the E8 dot ``Linear``
    on the way in, the ±1 → 0/1 fold on the way out) fuse into their
    consumers / producers.
    """
    pm1 = _is_type_pm1(input_vec, token_type)
    # ±1 → 0/1 as a single fused Linear: ``0.5 * pm1 + 0.5``.
    return Linear(
        pm1,
        torch.tensor([[0.5]], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
        name=f"is_type_01_{token_type.name}",
    )


def type_code(token_type: TokenType) -> Node:
    """Constant 8-wide :class:`LiteralValue` carrying the E8 code for
    ``token_type``.

    Used as the type-half of a composite attention key, e.g.
    ``concat(type_code(T), slot_value)`` queries "the position whose
    type is T and whose payload matches". Depth 0 (constant).
    """
    return create_literal_value(
        _e8_code(token_type), name=f"type_code_{token_type.name}"
    )


def extract_type_slot_raw(
    input_vec: Node, token_type: TokenType, slot_name: str
) -> Node:
    """Single fused ``Linear`` extracting the ``(token_type, slot_name)``
    slot value from ``input_vec``, with no auto-mask.

    The W_EMBED column at ``(token_type, slot_name)`` holds the raw
    normalized value (``q / cardinality`` for ``IntSlot``,
    ``(2k + 1) / (2·levels)`` for ``FloatSlot``); this Linear inverts
    that mapping to recover the declared slot value.

    When the active token is a different type, the raw column is 0 and
    the inverse returns ``slot.lo`` (``IntSlot``) or ≈ ``slot.lo``
    (``FloatSlot`` — the half-LSB grid offset gives a residual of
    ``-span / (2·(levels - 1))``). Use :func:`extract_type_slot` for
    the zero-masked form most callers want.
    """
    slot = _require_slot(token_type, slot_name)
    layout = TOKEN_VOCAB.layout
    if len(input_vec) != layout.d_embed:
        raise ValueError(
            f"extract_type_slot_raw: expected input_vec width "
            f"{layout.d_embed}, got {len(input_vec)}"
        )
    col = layout.slot_columns[(token_type.name, slot_name)]
    weight, bias = _denormalize_affine(slot)

    weights = torch.zeros((layout.d_embed, 1), dtype=torch.float32)
    weights[col, 0] = weight
    bias_t = torch.tensor([bias], dtype=torch.float32)
    raw = Linear(
        input_vec,
        weights,
        bias_t,
        name=f"extract_raw_{token_type.name}_{slot_name}",
    )
    return _bound_slot_output(raw, slot)


def extract_type_slot(input_vec: Node, token_type: TokenType, slot_name: str) -> Node:
    """Auto-masked single-``(type, slot)`` read.

    Returns the slot value when the active token has type
    ``token_type``, else 0. Depth: 2 MLP
    sublayers (the ``compare`` in :func:`_is_type_pm1` and the
    ``cond_gate``); the raw-extract ``Linear`` folds into one of them.
    """
    raw = extract_type_slot_raw(input_vec, token_type, slot_name)
    cond = _is_type_pm1(input_vec, token_type)
    return cond_gate(cond, raw)


def extract_int_slot_raw(input_vec: Node, slot_name: str) -> Node:
    """Single fused ``Linear`` reading the flat-namespace IntSlot
    ``slot_name`` from ``input_vec``.

    Sums the raw columns across every type declaring ``slot_name`` as
    an ``IntSlot``, scaled by that slot's cardinality and offset by
    ``lo``. Because only the active type's raw column is non-zero, the
    sum equals that one type's declared slot value. Off-name (active
    type doesn't declare this slot): returns ``lo``.

    Construction-time validation:

    * At least one type declares ``slot_name`` as an ``IntSlot``.
    * All declaring types agree on ``(lo, hi)`` — same-named slots with
      different ranges would denormalize into different value
      semantics, a footgun.
    """
    return _extract_flat_raw(input_vec, slot_name, IntSlot)


def extract_int_slot(input_vec: Node, slot_name: str) -> Node:
    """Auto-masked flat-namespace IntSlot read.

    Same as :func:`extract_int_slot_raw`, but returns 0 when no
    declaring type is active (instead of ``lo``).
    """
    return _extract_flat_auto_masked(input_vec, slot_name, IntSlot)


def extract_float_slot_raw(input_vec: Node, slot_name: str) -> Node:
    """Flat-namespace FloatSlot read; see :func:`extract_int_slot_raw`.

    All declaring types must agree on ``(lo, hi, levels)``.
    """
    return _extract_flat_raw(input_vec, slot_name, FloatSlot)


def extract_float_slot(input_vec: Node, slot_name: str) -> Node:
    """Auto-masked flat-namespace FloatSlot read."""
    return _extract_flat_auto_masked(input_vec, slot_name, FloatSlot)


def extract_derived(input_vec: Node, derived_name: str) -> Node:
    """Single fused ``Linear`` reading the derived span ``derived_name``.

    Derived spans carry ``fn(slot_value)`` baked in at embed time for any
    row of a declaring type, and 0 elsewhere. No inverse / auto-mask is
    needed — a 0 read on a non-declaring type is the correct
    semantic.

    Returns a ``width``-wide ``Node`` (1-wide for scalar derivations).
    Each ``(type, slot, derived_name)`` declaration owns its own span;
    this Linear places 1.0 at every declaration's span columns and sums
    them — at inference only the active type's span is non-zero, so the
    sum recovers that value (or 0 on a non-declaring type).
    """
    layout = TOKEN_VOCAB.layout
    if len(input_vec) != layout.d_embed:
        raise ValueError(
            f"extract_derived: expected input_vec width "
            f"{layout.d_embed}, got {len(input_vec)}"
        )
    entries = layout.derived_columns_by_name.get(derived_name)
    if not entries:
        available = sorted(layout.derived_columns_by_name)
        raise ValueError(
            f"derived column {derived_name!r} is not declared on any "
            f"slot in the vocab. Available: {available[:20]}"
            + ("..." if len(available) > 20 else "")
        )
    width = entries[0][3]
    weights = torch.zeros((layout.d_embed, width), dtype=torch.float32)
    for _t_name, _slot_name, start, _w in entries:
        for off in range(width):
            weights[start + off, off] = 1.0
    return Linear(input_vec, weights, name=f"extract_derived_{derived_name}")


# ---------------------------------------------------------------------------
# Internal: flat-namespace mechanics
# ---------------------------------------------------------------------------


def _bound_slot_output(node: Node, slot: IntSlot | FloatSlot) -> Node:
    """Attach a bounded ``NodeValueType`` to a raw slot-extract ``Linear``.

    For an active row, the value lands in ``[lo, hi]`` (``IntSlot``: by
    construction; ``FloatSlot``: by the encoding's grid). For an
    off-type row the raw column is 0; the affine evaluates to ``lo``
    for ``IntSlot`` and to ``lo - span / (2·(levels - 1))`` for
    ``FloatSlot``. Returns a wrapper carrying the looser of the two so
    every input row passes the asserted bound; downstream consumers
    (``cond_gate``, in particular) can use this to size their internal
    offsets correctly.
    """
    if isinstance(slot, IntSlot):
        lo = float(slot.lo)
        hi = float(slot.hi)
    else:
        span = slot.hi - slot.lo
        off_type_residual = span / (2.0 * float(slot.levels - 1))
        lo = float(slot.lo) - off_type_residual
        hi = float(slot.hi)
    vt = NodeValueType(value_range=Range(lo, hi))
    return assert_matches_value_type(node, vt, atol=1e-3)


def _denormalize_affine(slot: IntSlot | FloatSlot) -> tuple[float, float]:
    """Affine that inverts the row encoding in :func:`_normalized_slot_column`.

    * ``IntSlot``: raw = ``q / cardinality`` where ``q = value − lo``.
      Inverse: ``value = cardinality · raw + lo``.
    * ``FloatSlot``: raw = ``(2k + 1) / (2·levels)`` and
      ``value = lo + (k / (levels − 1)) · span``.
      Solving: ``value = (levels·span / (levels − 1)) · raw +
      (lo − span / (2·(levels − 1)))``.

    Returns ``(weight, bias)`` so that ``weight · raw + bias = value``.
    """
    if isinstance(slot, IntSlot):
        cardinality = slot.hi - slot.lo
        return float(cardinality), float(slot.lo)
    span = slot.hi - slot.lo
    levels = slot.levels
    weight = float(levels) * span / float(levels - 1)
    bias = float(slot.lo) - span / (2.0 * float(levels - 1))
    return weight, bias


def _require_slot(token_type: TokenType, slot_name: str) -> IntSlot | FloatSlot:
    if slot_name not in token_type.slots:
        raise ValueError(
            f"token type {token_type.name!r} has no slot {slot_name!r}; "
            f"declared: {list(token_type.slots)}"
        )
    return token_type.slots[slot_name]


def _flat_declaring_types(
    slot_name: str, slot_kind: type[IntSlot] | type[FloatSlot]
) -> list[tuple[TokenType, IntSlot | FloatSlot]]:
    """Find every (type, slot) pair where the type declares ``slot_name``
    of kind ``slot_kind``, and check (lo, hi[, levels]) agreement across
    the resulting set.
    """
    layout = TOKEN_VOCAB.layout
    matches: list[tuple[TokenType, IntSlot | FloatSlot]] = []
    for t in layout.types:
        slot = t.slots.get(slot_name)
        if slot is None:
            continue
        if not isinstance(slot, slot_kind):
            continue
        matches.append((t, slot))
    if not matches:
        raise ValueError(
            f"no type in the vocab declares slot {slot_name!r} of kind "
            f"{slot_kind.__name__}"
        )
    first_t, first_slot = matches[0]
    for t, slot in matches[1:]:
        if isinstance(first_slot, IntSlot):
            assert isinstance(slot, IntSlot)
            if (slot.lo, slot.hi) != (first_slot.lo, first_slot.hi):
                raise ValueError(
                    f"flat-namespace IntSlot {slot_name!r} disagrees on "
                    f"(lo, hi): {first_t.name}=({first_slot.lo}, "
                    f"{first_slot.hi}) vs {t.name}=({slot.lo}, {slot.hi})"
                )
        else:
            assert isinstance(slot, FloatSlot) and isinstance(first_slot, FloatSlot)
            if (slot.lo, slot.hi, slot.levels) != (
                first_slot.lo,
                first_slot.hi,
                first_slot.levels,
            ):
                raise ValueError(
                    f"flat-namespace FloatSlot {slot_name!r} disagrees on "
                    f"(lo, hi, levels): {first_t.name}="
                    f"({first_slot.lo}, {first_slot.hi}, {first_slot.levels}) "
                    f"vs {t.name}=({slot.lo}, {slot.hi}, {slot.levels})"
                )
    return matches


def _extract_flat_raw(
    input_vec: Node,
    slot_name: str,
    slot_kind: type[IntSlot] | type[FloatSlot],
) -> Node:
    layout = TOKEN_VOCAB.layout
    if len(input_vec) != layout.d_embed:
        raise ValueError(
            f"extract flat-slot: expected input_vec width "
            f"{layout.d_embed}, got {len(input_vec)}"
        )
    matches = _flat_declaring_types(slot_name, slot_kind)
    # All declaring types share the same (weight, bias) by construction.
    weight, bias = _denormalize_affine(matches[0][1])
    weights = torch.zeros((layout.d_embed, 1), dtype=torch.float32)
    for t, _slot in matches:
        col = layout.slot_columns[(t.name, slot_name)]
        weights[col, 0] = weight
    bias_t = torch.tensor([bias], dtype=torch.float32)
    raw = Linear(
        input_vec,
        weights,
        bias_t,
        name=f"extract_flat_{slot_kind.__name__}_{slot_name}",
    )
    return _bound_slot_output(raw, matches[0][1])


def _extract_flat_auto_masked(
    input_vec: Node,
    slot_name: str,
    slot_kind: type[IntSlot] | type[FloatSlot],
) -> Node:
    raw = _extract_flat_raw(input_vec, slot_name, slot_kind)
    matches = _flat_declaring_types(slot_name, slot_kind)
    cond_pm1 = _any_type_pm1(input_vec, [t for t, _ in matches])
    return cond_gate(cond_pm1, raw)


def _any_type_pm1(input_vec: Node, types: list[TokenType]) -> Node:
    """±1 indicator that the active token's type is one of ``types``.

    Computes the per-type ±1 ``compare`` outputs independently (every
    ``compare`` is one MLP sublayer; multiple compares against
    different threshold/dot-projection pairs schedule in parallel), then
    folds the sum into a single ±1 via one fused ``Linear``.

    For ``k = len(types)``: when exactly one type in the set is active,
    its compare returns +1 and the rest return −1, so the sum is
    ``2 − k``; when no type in the set is active, every compare
    returns −1 and the sum is ``−k``. The affine
    ``cond_pm1 = sum + (k − 1)`` maps these to +1 / −1 respectively.

    Special case ``k = 1`` short-circuits to the single type's
    ``_is_type_pm1`` directly.
    """
    if not types:
        raise ValueError("_any_type_pm1 requires at least one type")
    if len(types) == 1:
        return _is_type_pm1(input_vec, types[0])
    per_type = [_is_type_pm1(input_vec, t) for t in types]
    stacked = Concatenate(per_type)
    k = len(types)
    weights = torch.ones((k, 1), dtype=torch.float32)
    bias = torch.tensor([float(k - 1)], dtype=torch.float32)
    label = "_".join(t.name for t in types)[:40]
    return Linear(stacked, weights, bias, name=f"any_type_pm1_{label}")


# ---------------------------------------------------------------------------
# Token-compatibility helpers — the surface GraphPast.input_type() /
# _input_type_matches consume.
# ---------------------------------------------------------------------------


def is_type_pm1(input_vec: Node, token_type: TokenType) -> Node:
    """Public ±1 indicator that the active token is ``token_type``.

    The +1/-1 form (the free :func:`is_type` returns 0/1). This is the
    surface ported code should use directly for boolean composition or
    attention validity. Adds depth +1 (one ``compare``).
    """
    return _is_type_pm1(input_vec, token_type)


def input_type_code(input_vec: Node) -> Node:
    """Extract the 8-wide E8 category block (cols ``[0:8]``) of ``input_vec``.

    The compact per-token type code. ``GraphPast.input_type()``
    returns a handle over this; :func:`type_matches` / :func:`type_matches_any`
    compare it against token types without re-reading the full
    ``input_vec``. Depth +1 (one fused ``Linear``).
    """
    layout = TOKEN_VOCAB.layout
    if len(input_vec) != layout.d_embed:
        raise ValueError(
            f"input_type_code: expected input_vec width "
            f"{layout.d_embed}, got {len(input_vec)}"
        )
    weights = torch.zeros((layout.d_embed, D_CATEGORY), dtype=torch.float32)
    for c in range(D_CATEGORY):
        weights[c, c] = 1.0
    return Linear(input_vec, weights, name="input_type_code")


def type_matches(type_code: Node, token_type: TokenType) -> Node:
    """±1 whether an 8-wide E8 ``type_code`` is ``token_type``.

    ``type_code`` is the output of :func:`input_type_code`. Computes
    ``type_code · E8(token_type)`` and ``compare``\\ s it against the same
    halfway threshold :func:`is_type` uses, saturating to +1 (match) /
    -1 (mismatch). Depth +1.
    """
    if len(type_code) != D_CATEGORY:
        raise ValueError(
            f"type_matches: expected an {D_CATEGORY}-wide type code, "
            f"got width {len(type_code)}"
        )
    thresh = _is_type_threshold(token_type)
    weights = _e8_code(token_type).reshape(D_CATEGORY, 1)
    dot = Linear(type_code, weights, name=f"type_matches_dot_{token_type.name}")
    return compare(dot, thresh)


def type_matches_any(type_code: Node, token_types: list[TokenType]) -> Node:
    """±1 whether an 8-wide ``type_code`` matches any of ``token_types``.

    Mirrors :func:`_any_type_pm1` but over a type code: per-type
    :func:`type_matches` outputs are folded into a single ±1 via one
    ``Linear`` — ``sum + (k - 1)`` maps "exactly one match" -> +1 and
    "none" -> -1.
    """
    if not token_types:
        raise ValueError("type_matches_any requires at least one token type")
    if len(token_types) == 1:
        return type_matches(type_code, token_types[0])
    per_type = [type_matches(type_code, t) for t in token_types]
    stacked = Concatenate(per_type)
    k = len(token_types)
    weights = torch.ones((k, 1), dtype=torch.float32)
    bias = torch.tensor([float(k - 1)], dtype=torch.float32)
    label = "_".join(t.name for t in token_types)[:40]
    return Linear(stacked, weights, bias, name=f"type_matches_any_{label}")


def indicator_to_bool(indicator_01: Node) -> Node:
    """Convert a numeric 0/1 indicator to a +1/-1 boolean (``2x - 1``).

    The inverse of the ±1 -> 0/1 fold :func:`is_type` applies on its way
    out. Elementwise over any width; depth +1 (one fused ``Linear``).
    """
    width = len(indicator_01)
    weights = torch.eye(width, dtype=torch.float32) * 2.0
    bias = torch.full((width,), -1.0, dtype=torch.float32)
    return Linear(indicator_01, weights, bias, name="indicator_to_bool")
