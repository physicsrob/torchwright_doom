"""Round-trip correctness tests for the in-graph extract primitives.

For every shape an extract helper handles, build an ``input_vec`` row
straight out of ``W_EMBED``, feed it through the helper's compute
graph via ``reference_eval``, and confirm the recovered value matches
what was encoded. Also exercises off-type / off-name behaviour to lock
in the wrong-type semantic.

Reference-eval (exact math) is the right substrate for correctness —
``test_extract_compiled.py`` separately confirms the same helpers
survive fp32 matmul + PWL noise after compilation.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchwright.debug.probe import reference_eval
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.inout_nodes import create_input

from torchwright_doom import extract
from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.tokens import FloatSlot, IntSlot
from torchwright_doom.vocab import (
    ANGLE_BAM,
    ANGLE_VALUE,
    BEGIN,
    DONE,
    EMIT_X2,
    NODE,
    NO_OP,
    SCREEN_WIDTH,
    SEG_CLOSED_DOOR,
    SEG_EMPTY_LINE,
    SEG_TWO_SIDED,
    VALUE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _value_slot() -> FloatSlot:
    slot = VALUE.slots["v"]
    assert isinstance(slot, FloatSlot)
    return slot


def _row_for(token_type, slot_values: dict[str, int | float]) -> torch.Tensor:
    """Pick the W_EMBED row for ``token_type`` with the given slot values.

    Mirrors how ``TokenVocab`` enumerates rows (itertools.product over
    slots in declaration order, last slot fastest). Returns a (1,
    d_embed) tensor ready to feed as ``input_vec`` at a single
    position.
    """
    start, end = TOKEN_VOCAB.type_to_row_range[token_type]
    if not token_type.slots:
        assert (
            slot_values == {}
        ), f"{token_type.name} is slotless; got slot_values {slot_values}"
        return W_EMBED[start : start + 1].clone()

    slot_names = list(token_type.slots.keys())
    slot_objs = [token_type.slots[n] for n in slot_names]

    def step_index(slot, value):
        if isinstance(slot, IntSlot):
            return int(value) - slot.lo
        span = slot.hi - slot.lo
        return round((float(value) - slot.lo) / span * (slot.levels - 1))

    sizes = [
        (slot.hi - slot.lo) if isinstance(slot, IntSlot) else slot.levels
        for slot in slot_objs
    ]
    indices = [
        step_index(slot_objs[i], slot_values[n]) for i, n in enumerate(slot_names)
    ]
    # itertools.product is C-order: last dim fastest. Compose:
    row = 0
    for i, idx in enumerate(indices):
        stride = 1
        for j in range(i + 1, len(sizes)):
            stride *= sizes[j]
        row += idx * stride
    return W_EMBED[start + row : start + row + 1].clone()


def _eval_one(node, row: torch.Tensor) -> float:
    """Evaluate ``node`` on a single 1-position input row, returning the
    scalar value (node must be 1-wide)."""
    out = reference_eval(node, {"iv": row}, 1)[node]
    assert out.shape == (1, 1)
    return out.item()


# ---------------------------------------------------------------------------
# is_type
# ---------------------------------------------------------------------------


def test_is_type_indicator_strict_zero_or_one() -> None:
    """For every type T in the vocab, embed a row of T and confirm
    is_type(input_vec, T) is exactly 1.0 and is_type(input_vec, T') is
    exactly 0.0 for at least one other T'."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)

    # Build helpers once per type — they're cheap, but reuse for speed.
    is_type_nodes = {T.name: extract.is_type(inp, T) for T in TOKEN_VOCAB.types}

    # Pick a representative "first row" per type.
    for T in TOKEN_VOCAB.types:
        start, _ = TOKEN_VOCAB.type_to_row_range[T]
        row = W_EMBED[start : start + 1].clone()

        # Self-match: exactly 1.0.
        out = _eval_one(is_type_nodes[T.name], row)
        assert out == pytest.approx(
            1.0, abs=1e-6
        ), f"is_type({T.name}) on a {T.name} row should be 1.0, got {out}"

        # Cross-match against the first non-matching type.
        other = next((Tp for Tp in TOKEN_VOCAB.types if Tp.name != T.name), None)
        assert other is not None
        out = _eval_one(is_type_nodes[other.name], row)
        assert out == pytest.approx(
            0.0, abs=1e-6
        ), f"is_type({other.name}) on a {T.name} row should be 0.0, got {out}"


# ---------------------------------------------------------------------------
# extract_type_slot / extract_type_slot_raw — IntSlot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token_type,slot_name,test_values",
    [
        (NODE, "j", [0, 1, 31, 63]),  # boundaries + interior
        (SEG_TWO_SIDED, "flag", [0, 1]),
        (EMIT_X2, "x", [0, 1, SCREEN_WIDTH // 2, SCREEN_WIDTH - 1]),
    ],
    ids=["NODE.j", "SEG_TWO_SIDED.flag", "EMIT_X2.x"],
)
def test_extract_type_slot_int_round_trip(token_type, slot_name, test_values) -> None:
    """For an IntSlot, ``extract_type_slot_raw`` recovers the integer
    value exactly; ``extract_type_slot`` agrees when the type matches."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    raw_node = extract.extract_type_slot_raw(inp, token_type, slot_name)
    masked_node = extract.extract_type_slot(inp, token_type, slot_name)

    for value in test_values:
        row = _row_for(token_type, {slot_name: value})
        # Some types have multiple slots — fill the others with 0 / lo
        if len(token_type.slots) > 1:
            slot_vals = {slot_name: value}
            for other_name in token_type.slots:
                if other_name == slot_name:
                    continue
                other_slot = token_type.slots[other_name]
                slot_vals[other_name] = (
                    other_slot.lo if isinstance(other_slot, IntSlot) else other_slot.lo
                )
            row = _row_for(token_type, slot_vals)
        raw_out = _eval_one(raw_node, row)
        masked_out = _eval_one(masked_node, row)
        assert raw_out == pytest.approx(
            float(value), abs=1e-3
        ), f"{token_type.name}.{slot_name}={value} raw mismatch: {raw_out}"
        assert masked_out == pytest.approx(
            float(value), abs=1e-3
        ), f"{token_type.name}.{slot_name}={value} masked mismatch: {masked_out}"


def test_extract_type_slot_wrong_type_returns_zero_masked() -> None:
    """Feeding a different-type row through ``extract_type_slot`` returns 0.

    The unmasked ``_raw`` form returns ``slot.lo`` (IntSlot) — that's
    the inverse of a 0 raw column. The masked form zeros it out.
    """
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    raw_node = extract.extract_type_slot_raw(inp, NODE, "j")
    masked_node = extract.extract_type_slot(inp, NODE, "j")

    # VALUE row through "NODE.j"
    row = _row_for(VALUE, {"v": 0.5})
    raw_out = _eval_one(raw_node, row)
    masked_out = _eval_one(masked_node, row)
    assert raw_out == pytest.approx(float(NODE.slots["j"].lo), abs=1e-3)
    assert masked_out == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# extract_type_slot / extract_type_slot_raw — FloatSlot
# ---------------------------------------------------------------------------


def test_extract_type_slot_float_round_trip_grid_aligned() -> None:
    """For VALUE.v (the only FloatSlot in the vocab), embedding at a
    grid-aligned step index ``k`` and extracting returns the same
    quantized value the encoder snapped to.

    FloatSlots round-trip to the encoder's grid value — the snapped
    value, not the requested float — since the raw column is computed
    from ``k``.
    """
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    raw_node = extract.extract_type_slot_raw(inp, VALUE, "v")
    masked_node = extract.extract_type_slot(inp, VALUE, "v")

    slot = _value_slot()
    span = slot.hi - slot.lo
    for k in [0, 1, 32768, slot.levels - 1]:
        snapped = slot.lo + (k / (slot.levels - 1)) * span
        row = _row_for(VALUE, {"v": snapped})
        raw_out = _eval_one(raw_node, row)
        masked_out = _eval_one(masked_node, row)
        # FloatSlot grid-snap step is ~0.125 for VALUE.v; sub-LSB
        # noise from the affine round-trip is well under 1e-3.
        assert raw_out == pytest.approx(
            snapped, abs=0.2
        ), f"VALUE.v k={k} (snapped={snapped}): raw_out={raw_out}"
        assert masked_out == pytest.approx(snapped, abs=0.2)


def test_extract_type_slot_float_wrong_type_masked_zero() -> None:
    """A non-VALUE row through ``extract_type_slot(VALUE, v)`` returns 0.

    The unmasked path returns ``lo - span / (2·(levels-1))`` ≈ -4096.0625
    (the FloatSlot off-type residual).
    """
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    raw_node = extract.extract_type_slot_raw(inp, VALUE, "v")
    masked_node = extract.extract_type_slot(inp, VALUE, "v")

    row = _row_for(NODE, {"j": 5})
    raw_out = _eval_one(raw_node, row)
    masked_out = _eval_one(masked_node, row)
    slot = _value_slot()
    off_type_residual = (slot.hi - slot.lo) / (2.0 * (slot.levels - 1))
    expected_raw = slot.lo - off_type_residual
    assert raw_out == pytest.approx(expected_raw, abs=1e-3)
    assert masked_out == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Flat namespace
# ---------------------------------------------------------------------------


def test_extract_int_slot_flat_namespace_across_types() -> None:
    """The slot name 'flag' is shared across SEG_TWO_SIDED,
    SEG_EMPTY_LINE, SEG_CLOSED_DOOR — all IntSlot(0, 2).
    ``extract_int_slot('flag')`` returns the active type's flag value.
    """
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    raw_node = extract.extract_int_slot_raw(inp, "flag")
    masked_node = extract.extract_int_slot(inp, "flag")

    for t in (SEG_TWO_SIDED, SEG_EMPTY_LINE, SEG_CLOSED_DOOR):
        for flag in [0, 1]:
            row = _row_for(t, {"flag": flag})
            raw_out = _eval_one(raw_node, row)
            masked_out = _eval_one(masked_node, row)
            assert raw_out == pytest.approx(
                float(flag), abs=1e-3
            ), f"{t.name}.flag={flag}: raw={raw_out}"
            assert masked_out == pytest.approx(
                float(flag), abs=1e-3
            ), f"{t.name}.flag={flag}: masked={masked_out}"


def test_extract_int_slot_off_name_masked_zero() -> None:
    """A row of a type that doesn't declare 'flag' returns 0 from the
    masked variant. The unmasked variant returns ``lo`` (= 0 for flag)."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    raw_node = extract.extract_int_slot_raw(inp, "flag")
    masked_node = extract.extract_int_slot(inp, "flag")

    row = _row_for(NODE, {"j": 5})  # NODE has no 'flag'
    raw_out = _eval_one(raw_node, row)
    masked_out = _eval_one(masked_node, row)
    assert raw_out == pytest.approx(0.0, abs=1e-3)
    assert masked_out == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------


def test_extract_derived_angle_sin_cos() -> None:
    """ANGLE_VALUE's sin / cos derived columns recover ``sin(angle)`` /
    ``cos(angle)`` at depth 0."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    sin_node = extract.extract_derived(inp, "sin")
    cos_node = extract.extract_derived(inp, "cos")

    for angle in [-2048, -1024, 0, 256, 1024, 2048]:
        row = _row_for(ANGLE_VALUE, {"angle": angle})
        out_sin = _eval_one(sin_node, row)
        out_cos = _eval_one(cos_node, row)
        expected_sin = math.sin(angle * 2 * math.pi / ANGLE_BAM)
        expected_cos = math.cos(angle * 2 * math.pi / ANGLE_BAM)
        assert out_sin == pytest.approx(
            expected_sin, abs=1e-5
        ), f"angle={angle}: sin {out_sin} != {expected_sin}"
        assert out_cos == pytest.approx(
            expected_cos, abs=1e-5
        ), f"angle={angle}: cos {out_cos} != {expected_cos}"


def test_extract_derived_one_hot_x_oh_NNN() -> None:
    """EMIT_X2's ``x_oh_NNN`` derived column reads 1 at the matching x,
    0 elsewhere."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    targets = [0, 7, SCREEN_WIDTH // 2, SCREEN_WIDTH - 1]
    nodes = {x: extract.extract_derived(inp, f"x_oh_{x:03d}") for x in targets}

    for x_actual in targets:
        row = _row_for(EMIT_X2, {"x": x_actual})
        for x_target, node in nodes.items():
            out = _eval_one(node, row)
            expected = 1.0 if x_target == x_actual else 0.0
            assert out == pytest.approx(
                expected, abs=1e-5
            ), f"x_actual={x_actual} x_target={x_target}: {out} != {expected}"


def test_extract_derived_off_type_zero() -> None:
    """For a type that doesn't declare the derived column, reading
    returns 0 — the column is 0 in that row."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    sin_node = extract.extract_derived(inp, "sin")
    row = _row_for(NODE, {"j": 5})
    assert _eval_one(sin_node, row) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# type_code
# ---------------------------------------------------------------------------


def test_type_code_matches_e8_index() -> None:
    """``type_code(T)`` is the 8-wide E8 code at the layout's e8 index
    for ``T`` — the constant the embedding builder writes into the
    first 8 cols of every row of type T."""
    for T in (VALUE, NODE, ANGLE_VALUE, BEGIN, DONE, NO_OP):
        node = extract.type_code(T)
        out = reference_eval(node, {}, 1)[node]
        expected = index_to_vector(TOKEN_VOCAB.layout.e8_indices[T.name]).to(
            torch.float32
        )
        assert torch.allclose(
            out.squeeze(0), expected, atol=0.0
        ), f"type_code({T.name}) mismatch"
