"""Phase F / F0 de-risk gate: the R_PointToAngle BAM-atan2 octant.

``signed_world_angle`` is the chunk's single biggest numeric risk (Plan E
deferred it as E5): a ray-threshold *count* approximating ``atan2`` whose
``compare(sharpness=32000)`` deadband and ±3072 coordinate clamp both have to
hold for the projected angle to argmax to the exact BAM value the sandbox
emits. This isolates it — no forward graph, no compile — and checks the
real-graph build against the exact golden BAM angle:

1. a dense angle sweep at radii up to the clamp, covering every octant; and
2. the real ``e1m1_subset`` geometry — every seg endpoint and every BSP-node
   bbox corner, relative to every test pose — which is what the projection (F)
   and bbox-pruning (G) owners actually feed the octant.

Both are evaluated through ``reference_eval`` (exact math). The companion
``wrap_signed_angle`` one-turn wrap is checked the same way.
"""

from __future__ import annotations

import math

import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input

from torchwright_doom.render_ops import (
    _ATAN_ABS_RANGE,
    signed_world_angle,
    wrap_signed_angle,
)

from ..sandbox_support import import_sandbox, require_doom_sandbox

ANGLE_BAM = 8192


def _golden_bam(dx: float, dy: float) -> int:
    """Exact R_PointToAngle: round(atan2·8192/2π) mod 8192, signed to [-4096, 4096)."""
    a = round(math.atan2(dy, dx) * ANGLE_BAM / (2.0 * math.pi)) % ANGLE_BAM
    return a - ANGLE_BAM if a >= ANGLE_BAM // 2 else a


def _eval_octant(points: list[tuple[float, float]]) -> torch.Tensor:
    """``reference_eval`` ``signed_world_angle`` at one (dx, dy) per position."""
    dx = create_input("dx", 1)
    dy = create_input("dy", 1)
    angle = signed_world_angle(dx, dy)
    n_pos = len(points)
    xs = torch.tensor([[p[0]] for p in points], dtype=torch.float32)
    ys = torch.tensor([[p[1]] for p in points], dtype=torch.float32)
    cache = reference_eval(angle, {"dx": xs, "dy": ys}, n_pos)
    return cache[angle][:, 0]


def _bam_diff(a: float, b: float) -> float:
    """Smallest absolute difference on the BAM circle (handles the ±4096 seam)."""
    return abs(((a - b + ANGLE_BAM // 2) % ANGLE_BAM) - ANGLE_BAM // 2)


def test_octant_matches_golden_on_dense_sweep() -> None:
    # 720 angles × 4 radii spanning small offsets up to the clamp edge. Both
    # coordinates are kept clear of 0 (|coord| > 1): the quadrant ``select``
    # gates on ``compare(dx, 0)`` / ``compare(dy, 0)``, whose ±0.05 ramp would
    # produce a fractional (non-±1) gate for a coordinate sitting inside it.
    # Real geometry never lands there (poses are chosen off the axes); this
    # filter keeps the synthetic sweep on the same footing.
    points: list[tuple[float, float]] = []
    for i in range(720):
        theta = -math.pi + (2.0 * math.pi) * i / 720.0
        for r in (1.0, 37.0, 1140.0, 3000.0):
            dx, dy = r * math.cos(theta), r * math.sin(theta)
            if min(abs(dx), abs(dy)) > 1.0:
                points.append((dx, dy))
    out = _eval_octant(points)

    # Each output must be integer-valued (the count is a sum of exact 0/1 steps).
    frac = (out - out.round()).abs().max().item()
    assert frac < 1e-3, f"octant output not integer-valued: max frac {frac}"

    worst = max(
        _bam_diff(out[i].item(), _golden_bam(*points[i])) for i in range(len(points))
    )
    # Off-grid angles can sit a single BAM step from the rounded atan2 only when
    # the true angle lands within half a BAM unit of a threshold; the count is
    # otherwise exact. Real geometry (next test) is exact.
    assert worst <= 1.0, f"octant diverges from golden BAM by {worst}"


def _e1m1_octant_points() -> list[tuple[float, float]]:
    fixtures = import_sandbox("doom_sandbox.fixtures")
    scene = fixtures.load_fixture("e1m1_subset")
    md = scene.map_data
    points: list[tuple[float, float]] = []
    for pose in scene.test_poses:
        px, py = pose.x, pose.y
        for seg in md.segs:
            for vi in (seg.v1, seg.v2):
                v = md.vertices[vi]
                points.append((v.x - px, v.y - py))
        for node in md.nodes:
            for top, bot, left, right in (node.front_bbox, node.back_bbox):
                for cx in (left, right):
                    for cy in (top, bot):
                        points.append((cx - px, cy - py))
    return points


def test_octant_exact_on_e1m1_geometry() -> None:
    require_doom_sandbox()
    points = _e1m1_octant_points()

    # The clamp must cover the geometry: bbox corners reach |d| ~2752, which
    # overruns the legacy 2048 clamp and is why the unified helper uses 3072.
    max_abs = max(max(abs(dx), abs(dy)) for dx, dy in points)
    assert (
        max_abs < _ATAN_ABS_RANGE
    ), f"geometry |d|={max_abs:.1f} exceeds clamp {_ATAN_ABS_RANGE}"

    out = _eval_octant(points)
    mismatches = [
        (points[i], round(out[i].item()), _golden_bam(*points[i]))
        for i in range(len(points))
        if _bam_diff(out[i].item(), _golden_bam(*points[i])) > 0.0
    ]
    assert (
        not mismatches
    ), f"{len(mismatches)} octant/golden BAM mismatches, first 10: {mismatches[:10]}"


def test_wrap_signed_angle_matches_reference() -> None:
    def wrap_py(delta: float) -> float:
        if delta > 4095.5:
            return delta - ANGLE_BAM
        if delta <= -4096.5:  # not (delta > -4096.5)
            return delta + ANGLE_BAM
        return delta

    deltas = list(range(-ANGLE_BAM, ANGLE_BAM + 1, 7))
    delta_node = create_input("delta", 1)
    wrapped = wrap_signed_angle(delta_node)
    xs = torch.tensor([[float(d)] for d in deltas], dtype=torch.float32)
    cache = reference_eval(wrapped, {"delta": xs}, len(deltas))
    out = cache[wrapped][:, 0]

    worst = max(abs(out[i].item() - wrap_py(deltas[i])) for i in range(len(deltas)))
    assert worst < 1e-3, f"wrap_signed_angle diverges from reference by {worst}"
