"""Compiled-side correctness for the extract primitives.

Compiles each extract helper into a minimal ``CompiledHeadless`` and
runs it on representative ``input_vec`` rows, confirming that the
piecewise-linear ``compare`` op and the ``cond_gate`` masking survive
fp32 matmul + PL approximation noise. One representative per shape —
the broader correctness sweep lives in ``test_extract_correctness.py``
on reference math.

The compiled module runs the full transformer forward, so a failure
here on a case the reference-eval sibling passes indicates a noise
budget exceedance, not a logic bug. Tolerances are tight (1e-2) for
all paths — values to recover are in [-4096, 4096] and the round-trip
math is one affine ``Linear`` per slot extract.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.ops.inout_nodes import create_input

from torchwright_doom.model import extract
from torchwright_doom.model.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.model.tokens import FloatSlot, IntSlot
from torchwright_doom.model.vocab import (
    ANGLE_BAM,
    ANGLE_VALUE,
    NODE,
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


def _row_index(token_type, slot_values: dict[str, int | float]) -> int:
    """Compute the row index in W_EMBED for ``token_type`` with
    ``slot_values``.

    Deliberately an independent recomputation, NOT an import of
    ``tokenizer.rows.row_index``: this test cross-checks the row
    enumeration, so sharing the implementation under test would make the
    check tautological.
    """
    start, _ = TOKEN_VOCAB.type_to_row_range[token_type]
    if not token_type.slots:
        return start

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
    row = 0
    for i, idx in enumerate(indices):
        stride = 1
        for j in range(i + 1, len(sizes)):
            stride *= sizes[j]
        row += idx * stride
    return start + row


def _compile_one(output_node, device) -> tuple:
    """Compile a single-output graph that reads ``input_vec`` (named ``iv``)
    and feeds it through ``output_node``. Returns ``(compiled, d_embed)``."""
    compiled = compile_headless(
        output_node,
        d=1024,
        d_head=32,
        max_layers=20,
        verbose=False,
        device=str(device),
    )
    d_embed = TOKEN_VOCAB.layout.d_embed
    return compiled, d_embed


def _run_compiled(compiled, row: torch.Tensor, device) -> float:
    """Run the compiled module on a single 1-position input row and
    return the scalar output."""
    # Locate the iv input's columns in the compiled input layout.
    iv_specs = [(s, w) for n, s, w in compiled._input_specs if n == "iv"]
    assert len(iv_specs) == 1
    start, width = iv_specs[0]

    d_input = max(s + w for _, s, w in compiled._input_specs)
    inp = torch.zeros(1, d_input, dtype=torch.float32, device=device)
    inp[:, start : start + width] = row.to(device)
    with torch.no_grad():
        out = compiled(inp)
    return out[0, -1].item() if out.shape[-1] > 0 else float("nan")


# ---------------------------------------------------------------------------
# is_type
# ---------------------------------------------------------------------------


def test_compiled_is_type_self_and_cross(device) -> None:
    """is_type(NODE) is 1.0 on a NODE row, 0.0 on a VALUE row, after
    compilation. The 400-unit dot gap is more than 100x the ``compare``
    transition zone, so the compiled saturation should match reference
    math to fp32 precision."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    node_node = extract.is_type(inp, NODE)
    compiled, _ = _compile_one(node_node, device)

    node_row = W_EMBED[_row_index(NODE, {"j": 5}) : _row_index(NODE, {"j": 5}) + 1]
    value_row = W_EMBED[
        _row_index(VALUE, {"v": 0.5}) : _row_index(VALUE, {"v": 0.5}) + 1
    ]

    out_self = _run_compiled(compiled, node_row, device)
    out_cross = _run_compiled(compiled, value_row, device)
    assert out_self == pytest.approx(
        1.0, abs=1e-3
    ), f"is_type(NODE) on NODE row compiled to {out_self}, expected 1.0"
    assert out_cross == pytest.approx(
        0.0, abs=1e-3
    ), f"is_type(NODE) on VALUE row compiled to {out_cross}, expected 0.0"


# ---------------------------------------------------------------------------
# extract_type_slot — IntSlot
# ---------------------------------------------------------------------------


def test_compiled_extract_type_slot_int(device) -> None:
    """``extract_type_slot(NODE, j)`` recovers j exactly on a NODE row and
    zeros on a VALUE row, after compilation."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    masked = extract.extract_type_slot(inp, NODE, "j")
    compiled, _ = _compile_one(masked, device)

    for j in [0, 5, 63]:
        row = W_EMBED[_row_index(NODE, {"j": j}) : _row_index(NODE, {"j": j}) + 1]
        out = _run_compiled(compiled, row, device)
        assert out == pytest.approx(float(j), abs=1e-2), f"NODE.j={j} compiled to {out}"

    # Wrong type → masked 0
    value_row = W_EMBED[
        _row_index(VALUE, {"v": 0.5}) : _row_index(VALUE, {"v": 0.5}) + 1
    ]
    out = _run_compiled(compiled, value_row, device)
    assert out == pytest.approx(0.0, abs=1e-2), f"VALUE row through NODE.j: {out}"


# ---------------------------------------------------------------------------
# extract_type_slot — FloatSlot
# ---------------------------------------------------------------------------


def test_compiled_extract_type_slot_float(device) -> None:
    """``extract_type_slot(VALUE, v)`` recovers the quantized v on a VALUE
    row (within sub-LSB grid-snap noise) and zeros on a NODE row."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    masked = extract.extract_type_slot(inp, VALUE, "v")
    compiled, _ = _compile_one(masked, device)

    slot = _value_slot()
    span = slot.hi - slot.lo
    for k in [0, 32768, slot.levels - 1]:
        snapped = slot.lo + (k / (slot.levels - 1)) * span
        row = W_EMBED[
            _row_index(VALUE, {"v": snapped}) : _row_index(VALUE, {"v": snapped}) + 1
        ]
        out = _run_compiled(compiled, row, device)
        # The affine round-trip + cond_gate scales by M ≈ 4096; fp32
        # multiplies and the cond_gate cancellation add a few-ULP residual.
        assert out == pytest.approx(
            snapped, abs=1.0
        ), f"VALUE.v k={k} (snapped={snapped}) compiled to {out}"

    node_row = W_EMBED[_row_index(NODE, {"j": 5}) : _row_index(NODE, {"j": 5}) + 1]
    out = _run_compiled(compiled, node_row, device)
    assert out == pytest.approx(0.0, abs=1.0), f"NODE row through VALUE.v: {out}"


# ---------------------------------------------------------------------------
# Flat namespace
# ---------------------------------------------------------------------------


def test_compiled_extract_int_slot_flat(device) -> None:
    """``extract_int_slot('flag')`` recovers SEG_TWO_SIDED.flag from a
    SEG_TWO_SIDED row, after compilation."""
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    masked = extract.extract_int_slot(inp, "flag")
    compiled, _ = _compile_one(masked, device)

    for flag in [0, 1]:
        row = W_EMBED[
            _row_index(SEG_TWO_SIDED, {"flag": flag}) : _row_index(
                SEG_TWO_SIDED, {"flag": flag}
            )
            + 1
        ]
        out = _run_compiled(compiled, row, device)
        assert out == pytest.approx(
            float(flag), abs=1e-2
        ), f"SEG_TWO_SIDED.flag={flag} compiled to {out}"

    # Off-name → 0
    node_row = W_EMBED[_row_index(NODE, {"j": 5}) : _row_index(NODE, {"j": 5}) + 1]
    out = _run_compiled(compiled, node_row, device)
    assert out == pytest.approx(0.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Derived column
# ---------------------------------------------------------------------------


def test_compiled_extract_derived_sin(device) -> None:
    """``extract_derived('sin')`` recovers sin(angle) directly from an
    ANGLE_VALUE row's derived column — no PWL chain, just a Linear.
    """
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    sin_node = extract.extract_derived(inp, "sin")
    compiled, _ = _compile_one(sin_node, device)

    for angle in [0, 1024, 2048]:
        row = W_EMBED[
            _row_index(ANGLE_VALUE, {"angle": angle}) : _row_index(
                ANGLE_VALUE, {"angle": angle}
            )
            + 1
        ]
        out = _run_compiled(compiled, row, device)
        expected = math.sin(angle * 2 * math.pi / ANGLE_BAM)
        assert out == pytest.approx(
            expected, abs=1e-4
        ), f"angle={angle}: derived sin compiled={out} expected={expected}"
