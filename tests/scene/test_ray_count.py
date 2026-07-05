"""Exact-integer gate for the swiglu ``_ray_count`` (the swiglu cutover's D6).

``render_ops._ray_count`` counts the positive components of a ray vector on
two swish-hinge FFN stages (``hinge(z) = Swish(scale·z)/scale``, equal to
``relu(z)`` exactly once ``|scale·z| >= 17``). Its docstring derives, from
the pinned kernel constants, that every per-ray step is exactly 1 or 0 and
the count is an exact fp32 integer for any ray outside the unsaturated band
``|v| < 17/(scale·s) ≈ 4.2e-6``. This file converts that derivation into
evidence at the ``_ray_count`` layer:

* exact-math (``reference_eval``): exact integer counts at the fixture's
  extreme ray magnitudes — the smallest nonzero committed-fixture ray
  (~1.5e-4, the ``_RAY_SHARPNESS`` note in ``render_ops.py``) and the
  largest (~6e3) — at the production lane width (1024, one
  ``signed_world_angle`` ray half);
* compiled fp32: the same exactness through an actual compile (the count is
  a sum of exact 0/1 lane terms, an integer far below 2^24, so GEMM
  accumulation order cannot move it).

The saturation margin itself (smallest ray clears the band by ~36x) is
asserted numerically so a retune of ``scale`` / ``_RAY_SHARPNESS`` that
erodes it fails here, not in a walkthrough.
"""

from __future__ import annotations

import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.ops.const import scale
from torchwright.ops.inout_nodes import create_input

from torchwright_doom.render_ops import _RAY_SHARPNESS, _ray_count

#: The committed-fixture extreme ray magnitudes (see ``_RAY_SHARPNESS``'s
#: note in ``render_ops.py``): the smallest nonzero |ray| across the e1m1
#: fixtures is ~1.5e-4; rays reach ~6e3 at the large end.
_SMALLEST_RAY = 1.5e-4
_LARGEST_RAY = 6000.0

#: Production lane width: ``signed_world_angle`` feeds ``_ray_count`` two
#: ANGLE_BAM/8 = 1024-wide ray halves.
_N_PRODUCTION = 1024


def _case_vectors(n: int) -> list[tuple[str, torch.Tensor, int]]:
    """(label, width-n ray vector, expected exact count) fixtures."""
    smallest = _SMALLEST_RAY
    largest = _LARGEST_RAY

    mixed = torch.zeros(n)
    # Repeating 5-group [+small, -small, 0, +large, -large]: 2 positives per
    # full group; the expected count is derived, not hard-coded, so any n works.
    for i in range(n):
        mixed[i] = [smallest, -smallest, 0.0, largest, -largest][i % 5]
    mixed_count = int((mixed > 0).sum().item())

    one_hot_small = torch.zeros(n)
    one_hot_small[0] = smallest

    return [
        ("all_zero", torch.zeros(n), 0),
        ("all_pos_smallest", torch.full((n,), smallest), n),
        ("all_neg_smallest", torch.full((n,), -smallest), 0),
        ("all_pos_largest", torch.full((n,), largest), n),
        ("all_neg_largest", torch.full((n,), -largest), 0),
        ("single_pos_smallest", one_hot_small, 1),
        ("mixed_extremes", mixed, mixed_count),
    ]


def test_saturation_margin_at_smallest_fixture_ray():
    """The smallest nonzero fixture ray sits ~36x past the hinge's
    unsaturated band — the margin the exactness argument rides on."""
    gate_arg = scale * _RAY_SHARPNESS * _SMALLEST_RAY
    assert gate_arg >= 17.0 * 36.0, gate_arg


def test_ray_count_exact_integers_in_reference():
    """Exact-math wiring at production width: every fixture-class vector
    counts to a bit-exact integer (== on floats, no tolerance)."""
    cases = _case_vectors(_N_PRODUCTION)
    rays = create_input("rays", _N_PRODUCTION)
    out = _ray_count(rays)

    n_pos = len(cases)
    inputs = {"rays": torch.stack([vec for _label, vec, _c in cases])}
    vals = reference_eval(out, inputs, n_pos)[out]

    for i, (label, _vec, expected) in enumerate(cases):
        got = vals[i, 0].item()
        assert got == float(expected), (label, got, expected)


def test_ray_count_exact_integers_when_compiled():
    """Compiled fp32: the count matches the exact-math oracle to 1e-6 at a
    reduced lane width (the construction is width-uniform; the 0/1 lane terms
    and the integer sum are exact regardless of GEMM accumulation order)."""
    n = 16
    cases = _case_vectors(n)
    rays = create_input("rays", n)
    out = _ray_count(rays)

    n_pos = len(cases)
    inputs = {"rays": torch.stack([vec for _label, vec, _c in cases])}

    report = probe_graph(out, inputs, n_pos, d=512, d_head=16, atol=1e-6)
    assert report.first_divergent is None, report.format_short()

    # The oracle leg the probe matched is itself pinned to exact integers:
    vals = reference_eval(out, inputs, n_pos)[out]
    for i, (label, _vec, expected) in enumerate(cases):
        assert vals[i, 0].item() == float(expected), (label, expected)
