"""Exact-integer gate for the SwiGLU ``_ray_count``.

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


# ---------------------------------------------------------------------------
# Two-level ray count (the width-d4096 track): the production
# ``signed_world_angle`` counts each 1,024-threshold half as 32 coarse
# crossings + 32 fine tests with runtime slope DELTAS (constant-table row
# picked by the segment one-hot, applied via the exact ± product pair).
# Every threshold it evaluates — coarse (32j − 0.5) and fine (32s + i + 0.5)
# — is a member of the flat form's half-integer threshold set, and each fine
# ray is ALGEBRAICALLY the flat form's ray (v_base + Δ·op with constant
# Δ = slope(fine) − slope(base)), so the two forms may differ only by fp32
# rounding of the delta product (~1e-5, vs the ~1.5e-4 fixture ray
# clearance). These tests pin the two forms EQUAL — including at
# segment-boundary-adjacent angles, the two-level analogue of the
# digit-quad's boundary-sliver hazard.
# ---------------------------------------------------------------------------

import math

from torchwright_doom.render_ops import (
    _HIGH_RAY_MATRIX,
    _LOW_RAY_MATRIX,
    _abs_coord,
    signed_world_angle,
)
from torchwright_doom.std import concat as _concat, linear as _linear
from torchwright_doom.vocab import ANGLE_BAM

_BAM_U = 2.0 * math.pi / ANGLE_BAM


def _flat_signed_world_angle(dx, dy):
    """The reference 1,024-wide-thermometer form of ``signed_world_angle``
    (the shape production used before the two-level rewrite), rebuilt from
    the committed reference matrices — quadrant folding matches
    ``render_ops.signed_world_angle``."""
    import torch as _t

    from torchwright.graph import Linear as _Linear

    from torchwright_doom.render_ops import _ray_count
    from torchwright_doom.std import constant as _constant, select as _select
    from torchwright_doom.render_ops import compare as _compare, neg as _neg

    abs_dx = _abs_coord(dx)
    abs_dy = _abs_coord(dy)
    abs_pair = _concat(abs_dy, abs_dx)
    low = _ray_count(_linear(abs_pair, _LOW_RAY_MATRIX))
    high = _ray_count(_linear(abs_pair, _HIGH_RAY_MATRIX))
    base = _Linear(_concat(low, high), _t.tensor([[1.0], [1.0]], dtype=_t.float32))
    one = _constant(1.0)
    q2 = _linear(_concat(base, one), [[-1.0], [float(ANGLE_BAM // 2)]])
    q3 = _linear(_concat(base, one), [[1.0], [-float(ANGLE_BAM // 2)]])
    q4 = _neg(base)
    dx_pos = _compare(dx, 0.0)
    dy_pos = _compare(dy, 0.0)
    upper = _select(dx_pos, base, q2)
    lower = _select(dx_pos, q4, q3)
    return _select(dy_pos, upper, lower)


def _angle_cases() -> list[tuple[float, float]]:
    """(dx, dy) fixtures spanning both halves, all quadrants, segment
    boundaries, and the fixture-extreme radii."""
    angles = [
        1.0,
        2.0,
        30.0,
        31.0,
        32.0,
        33.0,  # around the first coarse boundary
        511.0,
        512.0,
        513.0,  # mid-low-half boundary
        1022.0,
        1023.0,
        1024.0,
        1025.0,  # the low/high half seam
        1055.0,
        1056.0,
        1057.0,  # first high-half coarse boundary
        1536.0,
        2046.0,
        2047.0,  # deep high half
        # segment-boundary-adjacent (0.25 BAM off a 32j - 0.5 coarse
        # threshold — inside a segment, outside every ramp/band):
        31.25,
        31.75,
        1055.25,
        1055.75,
    ]
    # Radii keep dx, dy clear of the quadrant compare(·, 0) deadband (0.1
    # at default sharpness) even at the 1-BAM extreme (sin(1u) ≈ 7.7e-4):
    # world-coordinate deltas are map-unit scale, so production inputs sit
    # far outside it too.
    radii = [200.0, 1000.0, 2752.0]
    cases: list[tuple[float, float]] = []
    for a in angles:
        for r in radii:
            th = a * _BAM_U
            cases.append((r * math.cos(th), r * math.sin(th)))
    # Quadrant reflections at a couple of representative angles/radii:
    for a in (33.0, 1055.0):
        th = a * _BAM_U
        dx, dy = 300.0 * math.cos(th), 300.0 * math.sin(th)
        cases += [(-dx, dy), (dx, -dy), (-dx, -dy)]
    return cases


def test_two_level_equals_flat_thermometer():
    """The production two-level ``signed_world_angle`` computes the SAME
    fp32 angle as the reference 1,024-thermometer form on the sweep —
    bit-equal counts (both are exact integers away from ramp zones)."""
    import torch as _t

    from torchwright.ops.inout_nodes import create_input

    cases = _angle_cases()
    n_pos = len(cases)
    dx = create_input("dx", 1)
    dy = create_input("dy", 1)
    new = signed_world_angle(dx, dy)
    old = _flat_signed_world_angle(dx, dy)
    inputs = {
        "dx": _t.tensor([[c[0]] for c in cases], dtype=_t.float32),
        "dy": _t.tensor([[c[1]] for c in cases], dtype=_t.float32),
    }
    new_vals = reference_eval(new, inputs, n_pos)[new]
    old_vals = reference_eval(old, inputs, n_pos)[old]
    for i, (cdx, cdy) in enumerate(cases):
        got, want = new_vals[i, 0].item(), old_vals[i, 0].item()
        assert got == want, (
            f"case {i} (dx={cdx:.4f}, dy={cdy:.4f}): two-level {got} != " f"flat {want}"
        )


def test_two_level_exact_integer_angles():
    """At integer BAM angles (radius large enough that fp32 coordinate
    rounding stays far inside the half-BAM threshold distance) the count IS
    the angle — an absolute pin, not just old/new agreement."""
    import torch as _t

    from torchwright.ops.inout_nodes import create_input

    angles = [1.0, 31.0, 32.0, 512.0, 1023.0, 1024.0, 1056.0, 2047.0]
    n_pos = len(angles)
    dx = create_input("dx", 1)
    dy = create_input("dy", 1)
    new = signed_world_angle(dx, dy)
    inputs = {
        "dx": _t.tensor(
            [[1000.0 * math.cos(a * _BAM_U)] for a in angles], dtype=_t.float32
        ),
        "dy": _t.tensor(
            [[1000.0 * math.sin(a * _BAM_U)] for a in angles], dtype=_t.float32
        ),
    }
    vals = reference_eval(new, inputs, n_pos)[new]
    for i, a in enumerate(angles):
        assert vals[i, 0].item() == a, (a, vals[i, 0].item())
