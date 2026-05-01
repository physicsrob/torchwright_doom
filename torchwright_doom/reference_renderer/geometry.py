"""Standalone 2D geometry primitives.

Originally lived in the legacy ``render.py`` ray-cast renderer; moved
here when that file was deleted.  Used by the walkthrough controller
for forward-wall-distance sensing — not by the C-faithful renderer in
``doom_render.py`` (which does its own analytic clipping).
"""

from __future__ import annotations

from typing import Optional, Tuple

from torchwright_doom.reference_renderer.types import Segment


def intersect_ray_segment(
    px: float,
    py: float,
    ray_cos: float,
    ray_sin: float,
    seg: Segment,
) -> Optional[Tuple[float, float]]:
    """Compute ray-segment intersection distance and u parameter.

    Returns ``(t, u)`` where *t* is the distance along the ray and *u*
    is the fractional position along the segment (0 at A, 1 at B).
    Returns ``None`` if the ray misses.
    """
    dx = ray_cos
    dy = ray_sin
    ex = seg.bx - seg.ax
    ey = seg.by - seg.ay
    fx = seg.ax - px
    fy = seg.ay - py

    den = dx * ey - dy * ex
    num_t = fx * ey - fy * ex
    num_u = fx * dy - fy * dx

    if den == 0.0:
        return None

    if den > 0.0:
        if num_t <= 0.0:
            return None
        if num_u < 0.0 or num_u > den:
            return None
    else:
        if num_t >= 0.0:
            return None
        if num_u > 0.0 or num_u < den:
            return None

    return num_t / den, num_u / den
