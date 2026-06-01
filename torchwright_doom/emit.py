"""In-graph emit helpers for token construction.

The graph's ``forward()`` produces a token at each AR step by writing
a residual row matching one row of ``W_EMBED`` — the host argmaxes
that row against ``W_EMBED.T`` to pick the next token ID. The helpers
here build that residual:

* :func:`emit_slotless` for marker types with no slots (BEGIN, DONE,
  NO_OP, …) — pure :class:`LiteralValue`, no inputs.
* :func:`emit_int_slot_token` for types whose slots are all
  :class:`IntSlot` — takes one 1-wide ``Node`` per slot carrying the
  (integer-valued) slot value.
* :func:`emit_float_slot_token` for types with a :class:`FloatSlot`
  (today only ``VALUE``) — quantizes the continuous value to the
  slot's step grid before routing through the same digit-quad payload
  builder.
* :func:`emit_token` dispatches to one of the above based on the
  token type's slot shapes.

The producer-side encoding is the mirror image of the row table in
``embedding.py``: for each slot, write the slot's raw normalized
column and a digit-quadratic payload that argmaxes against the slot's
2- or 4-wide block. Slots a row doesn't carry — and every derived
column — stay at zero on the emit side: nothing the producer puts
there could improve argmax, and a non-zero value would bias toward
rows that happen to carry those columns.

Depth, measured by compiling each helper alone with ``d=1024`` against
``compile_headless`` and reading ``compiled._n_layers``:

* Slotless: 1 layer (just the constant `LiteralValue` row).
* 1-digit slot (cardinality ≤ 256), ``slot.lo == 0``: 1 layer.
  The raw col ``q / cardinality`` and the digit-quad query
  ``[2·(q − CENTER), 1]`` are each one ``Linear`` over the input;
  both fire in parallel in the same layer.
* 1-digit slot, ``slot.lo != 0``: 2 layers (one Linear to compute
  ``q = value − lo`` plus one for the raw / digit-quad payload).
* 2-digit IntSlot (cardinality 257..65,536): 3 layers. One layer
  for ``q = value − lo`` (or 0 layers if ``lo == 0``); one layer for
  the ``thermometer_floor_div`` PWL; one layer for the centred /
  scaled outputs ``[2·(hi_q − CENTER)]`` and the fused
  ``[2·(q − BASE·hi_q − CENTER)]`` (the latter a single ``Linear``
  over ``[q, hi_q]`` so the lo recovery and the centring share an
  op).
* 2-digit FloatSlot (VALUE today): 3 layers — same path; the v→q
  affine is fused with whatever follows.

Two scheduler facts that drive the layer count, since the heuristic
scheduler does not fuse chained ``Linear`` nodes:

1. Every chain of two pure-affine ``Linear``s costs a layer per link.
   ``multiply_const(add_const(node, c), s)`` is two scheduled
   ``Linear``s; the equivalent single ``Linear(node, s·I, s·c)``
   compiles to one layer. The helpers below collapse those chains.
2. ``subtract(q, multiply_const(hi_q, BASE))`` leaves the multiply
   and the subtract as two scheduled ``Linear``s. Computing
   ``lo_q − CENTER`` and scaling it requires the same trick: one
   ``Linear`` over ``Concatenate([q, hi_q])`` with row matrix
   ``[2, −2·BASE]`` and bias ``−2·CENTER`` does both lo recovery and
   centring in a single op.

The 2-digit encoder splits into two scalar ops instead of one
vector PWL deliberately: a vector PWL that emits both the
staircase hi-byte and a sawtooth lo-byte in parallel has cycling
slopes of ±2,560 (the sawtooth wraps ~256 per byte width
~0.1 in the transition zone), and the cumulative fp32 cancellation
across 256 transitions overruns the design's 1-unit argmax gap.
Computing ``hi_q`` first and recovering ``lo_q = q − BASE·hi_q``
keeps every accumulated PWL sum bounded by the staircase
amplitude (256) and lets the affine recover ``lo_q`` exactly.
"""

from __future__ import annotations

from typing import Mapping

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import thermometer_floor_div
from torchwright.ops.inout_nodes import create_literal_value

from .embedding import (
    BASE,
    CENTER,
    D_CATEGORY,
    TOKEN_VOCAB,
    _digit_count_for_cardinality,
    _slot_levels,
)
from .tokens import FloatSlot, IntSlot, TokenType

__all__ = [
    "emit_slotless",
    "emit_int_slot_token",
    "emit_float_slot_token",
    "emit_token",
    "emit_token_head",
    "emit_derived_zero",
    "head_width",
]


def head_width() -> int:
    """Width of the emit *head* — the row minus the trailing derived region.

    Every emit row is ``[E8 code | raw slot cols | digit-quad blocks |
    derived zeros]``. The first three pieces (the *head*) are the only part
    that varies between tokens; the derived region is a constant block of
    zeros on the emit side (``n_derived_columns`` wide). The renderer's
    dispatch selects over heads and stamps one shared derived-zero tail at the
    end (see :func:`emit_derived_zero`), so the wide constant is built once
    instead of once per branch candidate.
    """
    layout = TOKEN_VOCAB.layout
    return layout.d_embed - layout.n_derived_columns


def emit_derived_zero(suffix: str = "") -> Node:
    """The shared trailing derived-region zeros (``n_derived_columns`` wide).

    Concatenated once after the dispatch's head selection to reconstitute a
    full ``d_embed``-wide emit row. Identical to the derived piece
    :func:`emit_token` appends internally, so a head + this tail argmaxes
    exactly as the full :func:`emit_token` would.
    """
    n_derived = TOKEN_VOCAB.layout.n_derived_columns
    return create_literal_value(
        torch.zeros(n_derived, dtype=torch.float32),
        name=f"emit_derived_zero{suffix}",
    )


def emit_token_head(
    token_type: TokenType,
    *,
    suffix: str = "",
    **slot_value_nodes: Node,
) -> Node:
    """Like :func:`emit_token`, but returns only the head (no derived tail).

    The dispatch folds over heads and appends one shared
    :func:`emit_derived_zero`; building each candidate at ``head_width()``
    instead of ``d_embed`` is what keeps the dispatch's live residual width
    small enough to compile.
    """
    return emit_token(
        token_type, suffix=suffix, include_derived=False, **slot_value_nodes
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def emit_slotless(
    token_type: TokenType, suffix: str = "", *, include_derived: bool = True
) -> Node:
    """Build the residual row for a slotless token type.

    The row is fully determined by the type's E8 category code; every
    other column stays at zero. Returns a single :class:`LiteralValue`
    so the compiler can keep this constant and lazy. With
    ``include_derived=False`` the row is truncated to the head (no derived
    tail) — see :func:`emit_token_head`.
    """
    if token_type.slots:
        raise ValueError(
            f"emit_slotless: {token_type.name!r} has slots "
            f"{list(token_type.slots)!r}; use emit_int_slot_token / "
            f"emit_float_slot_token instead."
        )
    layout = TOKEN_VOCAB.layout
    width = layout.d_embed if include_derived else head_width()
    row = torch.zeros(width, dtype=torch.float32)
    e8_idx = layout.e8_indices[token_type.name]
    row[0:D_CATEGORY] = index_to_vector(e8_idx)
    return create_literal_value(row, name=f"emit_{token_type.name}{suffix}")


def emit_int_slot_token(
    token_type: TokenType,
    *,
    suffix: str = "",
    include_derived: bool = True,
    **slot_value_nodes: Node,
) -> Node:
    """Build the residual row for a token whose slots are all IntSlots.

    Each ``slot_value_nodes[slot_name]`` is a 1-wide :class:`Node`
    carrying the slot's (near-)integer value in its declared range
    ``[slot.lo, slot.hi)``.
    """
    if not token_type.slots:
        raise ValueError(
            f"emit_int_slot_token: {token_type.name!r} has no slots; "
            f"use emit_slotless instead."
        )
    for slot_name, slot in token_type.slots.items():
        if not isinstance(slot, IntSlot):
            raise ValueError(
                f"emit_int_slot_token: {token_type.name}.{slot_name} is "
                f"{type(slot).__name__}, not IntSlot. Use "
                f"emit_float_slot_token for FloatSlot types."
            )
    return _emit_token_with_step_indices(
        token_type,
        _value_to_step_index_nodes(token_type, slot_value_nodes),
        suffix=suffix,
        include_derived=include_derived,
    )


def emit_float_slot_token(
    token_type: TokenType,
    *,
    suffix: str = "",
    include_derived: bool = True,
    **slot_value_nodes: Node,
) -> Node:
    """Build the residual row for a token with one or more FloatSlots.

    The FloatSlot's continuous value is quantized onto the slot's
    ``levels``-step grid by an affine ``q = (value - lo) * (levels -
    1) / (hi - lo)``; the result feeds the same digit-quad payload
    builder used by integer slots.

    IntSlots on the same type are also accepted — the per-slot mapping
    to step indices is the same affine, just with a slope of 1 and no
    quantization needed.
    """
    if not token_type.slots:
        raise ValueError(f"emit_float_slot_token: {token_type.name!r} has no slots.")
    has_float = any(isinstance(slot, FloatSlot) for slot in token_type.slots.values())
    if not has_float:
        raise ValueError(
            f"emit_float_slot_token: {token_type.name!r} has no FloatSlot; "
            f"use emit_int_slot_token instead."
        )
    return _emit_token_with_step_indices(
        token_type,
        _value_to_step_index_nodes(token_type, slot_value_nodes),
        suffix=suffix,
        include_derived=include_derived,
    )


def emit_token(
    token_type: TokenType,
    *,
    suffix: str = "",
    include_derived: bool = True,
    **slot_value_nodes: Node,
) -> Node:
    """Dispatch to the right ``emit_*`` helper for ``token_type``.

    ``include_derived=False`` returns just the head (the row minus the
    trailing derived zeros); :func:`emit_token_head` is the public shorthand.
    """
    if not token_type.slots:
        if slot_value_nodes:
            raise ValueError(
                f"emit_token: {token_type.name!r} is slotless but got "
                f"slot kwargs {sorted(slot_value_nodes)}"
            )
        return emit_slotless(token_type, suffix=suffix, include_derived=include_derived)
    has_float = any(isinstance(slot, FloatSlot) for slot in token_type.slots.values())
    if has_float:
        return emit_float_slot_token(
            token_type,
            suffix=suffix,
            include_derived=include_derived,
            **slot_value_nodes,
        )
    return emit_int_slot_token(
        token_type, suffix=suffix, include_derived=include_derived, **slot_value_nodes
    )


# ---------------------------------------------------------------------------
# Internal: residual-row assembly
# ---------------------------------------------------------------------------


def _value_to_step_index_nodes(
    token_type: TokenType,
    slot_value_nodes: Mapping[str, Node],
) -> dict[str, Node]:
    """Convert each ``slot_value_nodes[slot_name]`` from declared-range
    value to step index ``q ∈ [0, cardinality)`` via a single fused
    ``Linear``.

    ``q = scale · value + (−scale · slot.lo)`` — slope and bias folded
    into one ``Linear`` instead of an ``add_const → multiply_const``
    chain, since the compiler's heuristic scheduler treats each
    chained ``Linear`` as its own layer.

    When ``slot.lo == 0`` and the slope is 1 (IntSlot), the affine is
    pure identity — pass the value_node through unchanged so no
    Linear layer is consumed.
    """
    for k in slot_value_nodes:
        if k not in token_type.slots:
            raise ValueError(
                f"emit_token({token_type.name}): unknown slot {k!r}; "
                f"expected one of {list(token_type.slots)}"
            )
    out: dict[str, Node] = {}
    for slot_name, slot in token_type.slots.items():
        if slot_name not in slot_value_nodes:
            raise ValueError(
                f"emit_token({token_type.name}): missing slot value for "
                f"{slot_name!r}"
            )
        value_node = slot_value_nodes[slot_name]
        if len(value_node) != 1:
            raise ValueError(
                f"emit_token({token_type.name}.{slot_name}): expected a "
                f"1-wide value Node, got width {len(value_node)}"
            )
        if isinstance(slot, IntSlot):
            if slot.lo == 0:
                out[slot_name] = value_node
            else:
                out[slot_name] = _affine_1d(
                    value_node,
                    1.0,
                    -float(slot.lo),
                    name=f"q_{token_type.name}_{slot_name}",
                )
        else:
            span = slot.hi - slot.lo
            scale = (slot.levels - 1) / span
            out[slot_name] = _affine_1d(
                value_node,
                scale,
                -scale * float(slot.lo),
                name=f"q_{token_type.name}_{slot_name}",
            )
    return out


def _affine_1d(node: Node, scale: float, offset: float, *, name: str) -> Node:
    """Fused single-``Linear`` for ``scale · node + offset`` on a
    1-wide node. Used in preference to ``multiply_const(add_const(.))``
    so the chain compiles to one layer instead of two."""
    return Linear(
        node,
        torch.tensor([[scale]], dtype=torch.float32),
        torch.tensor([offset], dtype=torch.float32),
        name=name,
    )


def _emit_token_with_step_indices(
    token_type: TokenType,
    step_index_nodes: Mapping[str, Node],
    *,
    suffix: str,
    include_derived: bool = True,
) -> Node:
    """Build the residual row given step-index nodes for each slot.

    Assembles the row by concatenating pieces in column order:
    E8 code, per-(type, slot) raw cols, per-(type, slot) digit-quad
    blocks, derived cols. Pieces that don't belong to ``token_type``
    contribute zero literals at the right widths. With
    ``include_derived=False`` the trailing derived-zero block is omitted,
    yielding the head (the dispatch appends one shared
    :func:`emit_derived_zero` instead).
    """
    layout = TOKEN_VOCAB.layout
    type_name = token_type.name

    pieces: list[Node] = []

    # E8 category code (cols [0:8])
    e8_idx = layout.e8_indices[type_name]
    pieces.append(
        create_literal_value(
            index_to_vector(e8_idx).to(torch.float32),
            name=f"emit_e8_{type_name}{suffix}",
        )
    )

    # Per-(type, slot) raw cols in layout order
    for (t_name, slot_name), _col in layout.slot_columns.items():
        if t_name == type_name:
            slot = token_type.slots[slot_name]
            q_node = step_index_nodes[slot_name]
            pieces.append(
                _raw_col_from_step_index(
                    q_node,
                    slot,
                    name=f"emit_raw_{type_name}_{slot_name}{suffix}",
                )
            )
        else:
            pieces.append(
                create_literal_value(
                    torch.zeros(1, dtype=torch.float32),
                    name=f"emit_raw_zero_{t_name}_{slot_name}{suffix}",
                )
            )

    # Per-(type, slot) digit-quad blocks in layout order
    for (t_name, slot_name), (_dq_col, dq_n) in layout.digit_quad_columns.items():
        if t_name == type_name:
            slot = token_type.slots[slot_name]
            q_node = step_index_nodes[slot_name]
            pieces.append(
                _digit_quad_payload(
                    q_node,
                    slot,
                    name=f"emit_dq_{type_name}_{slot_name}{suffix}",
                )
            )
        else:
            pieces.append(
                create_literal_value(
                    torch.zeros(dq_n, dtype=torch.float32),
                    name=f"emit_dq_zero_{t_name}_{slot_name}{suffix}",
                )
            )

    # Derived columns: zero on the emit side (any non-zero value would
    # bias argmax toward rows that carry that derived value). Omitted for the
    # head; the dispatch stamps one shared `emit_derived_zero` after selection.
    n_derived = layout.n_derived_columns
    if include_derived and n_derived:
        pieces.append(
            create_literal_value(
                torch.zeros(n_derived, dtype=torch.float32),
                name=f"emit_derived_zero_{type_name}{suffix}",
            )
        )

    out = Concatenate(pieces)
    expected = layout.d_embed if include_derived else head_width()
    if len(out) != expected:
        raise RuntimeError(
            f"emit_token({type_name}): assembled width {len(out)} != "
            f"expected {expected} (include_derived={include_derived})"
        )
    return out


# ---------------------------------------------------------------------------
# Per-piece encoders
# ---------------------------------------------------------------------------


def _raw_col_from_step_index(
    q_node: Node, slot: IntSlot | FloatSlot, *, name: str
) -> Node:
    """Affine map from step index to the slot's normalized raw column.

    Matches the formula in ``_normalized_slot_column`` row-by-row:

    * IntSlot: ``raw = q / cardinality``.
    * FloatSlot: ``raw = (2q + 1) / (2 · levels) = q / levels +
      1 / (2 · levels)``.

    Both are folded into a single ``Linear`` so the raw col compiles
    into one layer instead of chaining a ``multiply_const`` after an
    ``add_const``.
    """
    if isinstance(slot, IntSlot):
        cardinality = slot.hi - slot.lo
        return _affine_1d(q_node, 1.0 / float(cardinality), 0.0, name=name)
    levels = slot.levels
    return _affine_1d(
        q_node, 1.0 / float(levels), 1.0 / (2.0 * float(levels)), name=name
    )


def _digit_quad_payload(q_node: Node, slot: IntSlot | FloatSlot, *, name: str) -> Node:
    """Build the digit-quad payload ``[..., 2·d_c, 1, ...]`` from a
    step-index node ``q``.

    For 1-digit slots (cardinality ≤ 256): pure affine,
    ``[2·(q - CENTER), 1.0]``. Two columns, depth-free
    (the affine folds into the surrounding Concatenate consumer).

    For 2-digit slots (257 ≤ cardinality ≤ 65,536): the high byte
    comes from :func:`thermometer_floor_div` (``hi_q = floor((q + 0.5)
    / BASE)`` — staircase PWL with transitions centred on the
    half-integer byte thresholds 255.5, 511.5, …; one MLP sublayer)
    and the low byte recovers via the affine ``lo_q = q - BASE·hi_q``
    that folds into the surrounding Concatenate consumer.
    Total depth: 1 MLP sublayer for the staircase, the rest is affine.
    """
    cardinality = _slot_levels(slot)
    digits = _digit_count_for_cardinality(cardinality)

    one = create_literal_value(
        torch.tensor([1.0], dtype=torch.float32),
        name=f"{name}_const1",
    )

    if digits == 1:
        # 2·(q − CENTER) = 2·q + (−2·CENTER) — one Linear.
        scaled = _affine_1d(q_node, 2.0, -2.0 * CENTER, name=f"{name}_lo_c2")
        return Concatenate([scaled, one])

    max_q = cardinality - 1
    # thermometer_floor_div places transitions at k·BASE − 0.5 for
    # k = 1..max_q//BASE; that's exactly the half-integer byte
    # threshold we want. Integer-step q never lands inside the ramp.
    hi_q = thermometer_floor_div(q_node, BASE, max_q)
    hi_c_2 = _affine_1d(hi_q, 2.0, -2.0 * CENTER, name=f"{name}_hi_c2")
    # lo_q = q − BASE·hi_q realised as a single Linear over the
    # concat of (q, hi_q). The ``subtract(q, multiply_const(hi_q,
    # BASE))`` form leaves two chained Linears the compiler will not
    # fuse, costing two extra layers.
    # 2·(lo_q − CENTER) = 2·q − 2·BASE·hi_q − 2·CENTER, also one
    # Linear over (q, hi_q) — folds the centering and scaling into
    # the same op as the lo_q recovery itself.
    lo_c_2_node = Linear(
        Concatenate([q_node, hi_q]),
        torch.tensor([[2.0], [-2.0 * float(BASE)]], dtype=torch.float32),
        torch.tensor([-2.0 * CENTER], dtype=torch.float32),
        name=f"{name}_lo_c2",
    )
    one_b = create_literal_value(
        torch.tensor([1.0], dtype=torch.float32),
        name=f"{name}_const1b",
    )
    return Concatenate([hi_c_2, one, lo_c_2_node, one_b])
