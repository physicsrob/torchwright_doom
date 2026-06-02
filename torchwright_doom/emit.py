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

from dataclasses import dataclass
from typing import Mapping

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import floor_int
from torchwright.ops.inout_nodes import create_literal_value

from .embedding import (
    BASE,
    CENTER,
    D_CATEGORY,
    TOKEN_VOCAB,
    _digit_count_for_cardinality,
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
    "ScalarEmit",
    "value_scalar",
    "angle_scalar",
]


# ---------------------------------------------------------------------------
# Route-then-emit-once: deferred numeric-carrier emission
# ---------------------------------------------------------------------------
#
# A numeric carrier (``VALUE`` / ``ANGLE_VALUE``) emits its value through a
# ~255-wide ``floor_int`` digit-quad (see :func:`_digit_quad_payload`). Each
# branch that emits one used to build its OWN ``make_token_head`` head eagerly,
# so ~24 of these 255-wide staircases sat live on the residual at once — the
# measured residual-width peak (``scripts/COST_NOTES.md``: ~39% of the d=6400
# peak is digit-quad emit, all collapsible).
#
# Instead, an owner ``after_*`` returns just its 1-wide scalar wrapped in a
# :class:`ScalarEmit`. The dispatch (``render_main._collapse_scalar_emits``)
# gathers every branch sharing a carrier, picks the active branch's scalar by
# its dispatch predicate (float-exact, exactly one predicate is +1), and runs
# ONE shared digit-quad per carrier. 24 eager quads → 2. The emitted row is
# byte-identical: the picked scalar is the winning branch's scalar bit-for-bit,
# and the shared head clamps + quantizes it exactly as the per-branch head did.


@dataclass(frozen=True)
class ScalarEmit:
    """A branch's deferred numeric-carrier emission: the carrier token type plus
    the 1-wide scalar it carries (``VALUE``: range-encoded into [-1, 1];
    ``ANGLE_VALUE``: the raw angle). Built by :func:`value_scalar` /
    :func:`angle_scalar`; collapsed into one shared head per carrier by the
    dispatch. See the module note above."""

    carrier: TokenType
    scalar: Node


def value_scalar(range_id, value: Node) -> ScalarEmit:
    """A ``VALUE``-carrier branch's scalar contribution: range-encode ``value``
    into the shared slot's [-1, 1] space and defer the digit-quad.

    Drop-in for ``make_value`` at a branch ``after_*`` emitter — the encode is
    the same producer-side affine, but the head is built once at dispatch time
    over the picked scalar instead of once per branch."""
    from .value_ranges import encode_vec
    from .vocab import VALUE

    return ScalarEmit(VALUE, encode_vec(range_id, value))


def angle_scalar(angle: Node) -> ScalarEmit:
    """An ``ANGLE_VALUE``-carrier branch's scalar contribution (the raw angle).

    Drop-in for ``make_token_head(ANGLE_VALUE, angle=...)`` at a branch
    ``after_*`` emitter; see :func:`value_scalar`."""
    from .vocab import ANGLE_VALUE

    return ScalarEmit(ANGLE_VALUE, angle)


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

    Column order: E8 code, one shared raw column per slot *position*, one shared
    digit-quad block per slot position, derived cols. The token writes its own
    slots at their declaration order (slot 0 -> position 0, …) and zeros the
    trailing positions it doesn't have. Because the columns are shared across
    types and sized for the widest slot at each position, each slot is encoded
    at its position's digit width (matching ``_build_w_embed``). With
    ``include_derived=False`` the trailing derived-zero block is omitted,
    yielding the head (the dispatch appends one shared
    :func:`emit_derived_zero` instead).
    """
    layout = TOKEN_VOCAB.layout
    type_name = token_type.name
    type_slots = list(token_type.slots.items())  # declaration order == positions

    pieces: list[Node] = []

    # E8 category code (cols [0:8])
    e8_idx = layout.e8_indices[type_name]
    pieces.append(
        create_literal_value(
            index_to_vector(e8_idx).to(torch.float32),
            name=f"emit_e8_{type_name}{suffix}",
        )
    )

    # One shared raw column per slot position (zero where this type has no slot).
    for j in range(layout.n_slot_positions):
        if j < len(type_slots):
            slot_name, slot = type_slots[j]
            pieces.append(
                _raw_col_from_step_index(
                    step_index_nodes[slot_name],
                    slot,
                    name=f"emit_raw_{type_name}_p{j}{suffix}",
                )
            )
        else:
            pieces.append(
                create_literal_value(
                    torch.zeros(1, dtype=torch.float32),
                    name=f"emit_raw_zero_p{j}{suffix}",
                )
            )

    # One shared digit-quad block per slot position, encoded at the position's
    # width (the position cardinality, not the slot's own).
    for j in range(layout.n_slot_positions):
        width = layout.shared_position_dq_width[j]
        if j < len(type_slots):
            slot_name, slot = type_slots[j]
            pieces.append(
                _digit_quad_payload(
                    step_index_nodes[slot_name],
                    layout.shared_position_cardinality[j],
                    name=f"emit_dq_{type_name}_p{j}{suffix}",
                )
            )
        else:
            pieces.append(
                create_literal_value(
                    torch.zeros(width, dtype=torch.float32),
                    name=f"emit_dq_zero_p{j}{suffix}",
                )
            )

    # Derived columns: zero on the emit side (any non-zero value would
    # bias argmax toward rows that carry that derived value). Omitted for the
    # head; the dispatch stamps one shared `emit_derived_zero` after selection.
    if include_derived and layout.n_derived_columns:
        pieces.append(emit_derived_zero(suffix=f"_{type_name}{suffix}"))

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


def _digit_quad_payload(q_node: Node, cardinality: int, *, name: str) -> Node:
    """Build the digit-quad payload ``[..., 2·d_c, 1, ...]`` from a
    step-index node ``q``, at the width implied by ``cardinality``.

    ``cardinality`` is the slot *position*'s cardinality (the widest slot at
    this position across the vocab), not the emitting slot's own — the
    digit-quad columns are shared by position, so a narrow slot is encoded at
    its position's width (its high byte is then a constant 0). The matching
    ``W_EMBED`` rows are built the same way via ``_digit_quad_block`` on the
    block width, so emit and table stay consistent.

    For 1 digit (cardinality ≤ 256): pure affine, ``[2·(q - CENTER), 1.0]``,
    depth-free. For 2 digits (257 ≤ cardinality ≤ 65,536): the high byte comes
    from :func:`floor_int` (``floor(q / BASE)``, exact on a continuous q) and the
    low byte recovers via an affine that folds into the surrounding Concatenate.
    """
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
    # hi_q = floor(q / BASE). q is a *continuous* FloatSlot step index (e.g.
    # ~62196 for v=0.898), so use floor_int — ramps at integer boundaries, exact
    # mid-bin — over the high-byte range [0, max_q // BASE]. thermometer_floor_div
    # is integer-inputs-only (half-integer thresholds) and interpolates junk on a
    # continuous q. floor_int is cancellation-free at this magnitude (the
    # saturating-step staircase); sharpness=1000 keeps the ramp well clear of the
    # value's fractional part.
    q_over_base = _affine_1d(q_node, 1.0 / float(BASE), 0.0, name=f"{name}_hi_div")
    hi_q = floor_int(q_over_base, 0, max_q // BASE, sharpness=1000)
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
