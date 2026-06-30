"""Compiled numerics gate for radix digit extraction (Task 3 derisk).

The single highest numerical risk in the radix successor port is digit
extraction: turning a screen column ``v`` into ``hi(v) = v // B`` and
``lo(v) = v % B`` without an integer column ever landing inside a
piecewise-linear ramp. The original design proposed ``floor_int(v/B + 0.5/B)``;
at ``B = 13`` an integer just below a bucket boundary sits only ``0.5/13`` from
the ramp and floors to a fractional digit. The port instead uses
``thermometer_floor_div`` (ramps centred at ``k*B - 0.5``, so every integer is
0.5 away) for ``hi`` and ``mod_const`` for ``lo`` — both already measured at
"0 abs, 0 rel" noise.

This test confirms, *through an actual compile* at BOTH project scales
(``B = 8`` fixture, ``B = 13`` real), that:

* ``hi`` / ``lo`` are exact for every integer column ``0..W`` (covers all
  ``k*B-1``, ``k*B``, ``k*B+1`` boundaries);
* ``hi * B + lo == v`` (reconstruction);
* ``one_hot(hi, N_BUCKETS)`` is a clean one-hot (what H1/H2/H3 consume);
* the compiled fp32 transformer stays within 0.05 of the exact integers
  everywhere — comfortably inside the 0.5 rounding margin.

GPU compile: run with ``make test-local FILE=tests/scene/test_radix_digit_extraction.py``.
"""

from __future__ import annotations

import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.ops.arithmetic_ops import mod_const, thermometer_floor_div
from torchwright.ops.inout_nodes import create_input

from torchwright_doom.std import concat, one_hot

from .test_radix_successor_oracle import hi, lo, radix_params

# Both scales: (W, B, N_BUCKETS). B/N_BUCKETS cross-checked by the oracle module.
_SCALES = [(60, 8, 8), (160, 13, 13)]


def _columns_tensor(W: int) -> torch.Tensor:
    return torch.tensor([[float(c)] for c in range(W + 1)])


@pytest.mark.parametrize("W,B,N_BUCKETS", _SCALES)
def test_digit_extraction_exact_in_reference(W, B, N_BUCKETS):
    """Exact-math wiring: hi/lo/reconstruction/one-hot for every column."""
    assert radix_params(W) == (B, N_BUCKETS)
    v = create_input("v", 1)
    hi_node = thermometer_floor_div(v, B, W)
    lo_node = mod_const(v, B, W)
    bucket_oh = one_hot(hi_node, N_BUCKETS)
    out = concat(hi_node, lo_node, bucket_oh)

    n_pos = W + 1
    inputs = {"v": _columns_tensor(W)}
    vals = reference_eval(out, inputs, n_pos)[out]

    for c in range(n_pos):
        got_hi = vals[c, 0].item()
        got_lo = vals[c, 1].item()
        oh = vals[c, 2:]
        assert got_hi == pytest.approx(hi(c, B), abs=1e-6), (W, c, got_hi)
        assert got_lo == pytest.approx(lo(c, B), abs=1e-6), (W, c, got_lo)
        assert got_hi * B + got_lo == pytest.approx(c, abs=1e-6), (W, c)
        assert oh.sum().item() == pytest.approx(1.0, abs=1e-6), (W, c)
        assert oh.argmax().item() == hi(c, B), (W, c)


@pytest.mark.parametrize("W,B,N_BUCKETS", _SCALES)
def test_digit_extraction_exact_when_compiled(W, B, N_BUCKETS):
    """Compiled fp32: hi/lo stay within 0.05 of the exact integers everywhere,
    so round-to-nearest recovers the exact digit (0.5 margin)."""
    v = create_input("v", 1)
    hi_node = thermometer_floor_div(v, B, W)
    lo_node = mod_const(v, B, W)
    out = concat(hi_node, lo_node)

    n_pos = W + 1
    inputs = {"v": _columns_tensor(W)}

    report = probe_graph(out, inputs, n_pos, d=512, d_head=16, atol=0.05)
    assert report.first_divergent is None, report.format_short()

    # Belt-and-braces: the exact-math oracle this compiled run matched is itself
    # integer-exact (asserted in the reference test above), so a clean probe at
    # atol=0.05 means the compiled digits round to the exact integers.
