"""Plan A: value-range bank round-trips (the foundation test).

For each ``ValueRange`` R0..R9: encode a physical value into the VALUE
carrier's ``[-1, 1]`` space, embed it as a VALUE row (FloatSlot snaps to
the 65,536-step grid), then read it back through ``value_derived`` (the
``v{idx}`` decoded column). The recovered value must match within a
*per-range* grid tolerance ``(hi - lo) / (2·(levels - 1))`` — a single
global atol would either reject R0 or rubber-stamp the tight ranges.

Also checks the ``inv{idx}`` reciprocal column and the pure
encode -> grid-snap -> decode chain (the same grid emit.py quantizes onto).
"""

from __future__ import annotations

import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input

from torchwright_doom import value_ranges as vr
from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.tokens import FloatSlot
from torchwright_doom.vocab import VALUE


def _value_slot() -> FloatSlot:
    slot = VALUE.slots["v"]
    assert isinstance(slot, FloatSlot)
    return slot


def _value_row(encoded: float) -> torch.Tensor:
    slot = _value_slot()
    span = slot.hi - slot.lo
    idx = round((encoded - slot.lo) / span * (slot.levels - 1))
    idx = max(0, min(slot.levels - 1, idx))
    start, _end = TOKEN_VOCAB.type_to_row_range[VALUE]
    return W_EMBED[start + idx : start + idx + 1].clone()


def _eval_one(node, row: torch.Tensor) -> float:
    out = reference_eval(node, {"iv": row}, 1)[node]
    assert out.shape == (1, 1)
    return out.item()


def _grid_tolerance(spec: vr.ValueRangeSpec, levels: int = 65536) -> float:
    return (spec.hi - spec.lo) / (2.0 * (levels - 1))


def test_value_derived_round_trip_all_ranges() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    for rng in vr.ValueRange:
        spec = vr.VALUE_RANGES[rng]
        tol = _grid_tolerance(spec) + 1e-4
        node = vr.value_derived(inp, rng)  # -> v{idx} decoded column
        for frac in (0.0, 0.1, 0.37, 0.5, 0.83, 1.0):
            v = spec.lo + frac * (spec.hi - spec.lo)
            row = _value_row(vr.encode_float(rng, v))
            got = _eval_one(node, row)
            assert abs(got - v) <= tol, f"{rng.name} v={v}: recovered {got}, tol {tol}"


def test_value_derived_inverse_column() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    for rng in vr.ValueRange:
        spec = vr.VALUE_RANGES[rng]
        node = vr.value_derived(inp, rng, kind="inv")
        # pick a value comfortably away from 0 so 1/v is well-conditioned
        v = spec.lo + 0.7 * (spec.hi - spec.lo)
        if abs(v) < 1.0:
            v = spec.hi if abs(spec.hi) >= abs(spec.lo) else spec.lo
        row = _value_row(vr.encode_float(rng, v))
        got = _eval_one(node, row)
        # decode is grid-snapped; compare 1/decoded, not 1/v
        decoded = vr.decode_float(rng, vr.encode_float(rng, v))
        expected = vr._VALUE_INVERSE_LIMIT if decoded == 0.0 else 1.0 / decoded
        assert (
            abs(got - expected) <= abs(expected) * 1e-3 + 1e-4
        ), f"{rng.name} inv: recovered {got}, expected {expected}"


def test_encode_grid_decode_pure() -> None:
    """encode -> VALUE grid snap -> decode recovers v within the per-range
    grid step (the same grid emit.py quantizes onto for AR emission)."""
    slot = _value_slot()
    span = slot.hi - slot.lo
    for rng in vr.ValueRange:
        spec = vr.VALUE_RANGES[rng]
        tol = _grid_tolerance(spec) + 1e-6
        for frac in (0.0, 0.25, 0.5, 0.9, 1.0):
            v = spec.lo + frac * (spec.hi - spec.lo)
            encoded = vr.encode_float(rng, v)
            idx = round((encoded - slot.lo) / span * (slot.levels - 1))
            snapped = slot.lo + (idx / (slot.levels - 1)) * span
            assert (
                abs(vr.decode_float(rng, snapped) - v) <= tol
            ), f"{rng.name} v={v}: grid round-trip out of tolerance"
