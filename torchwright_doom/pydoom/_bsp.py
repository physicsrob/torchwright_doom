"""Vendored BSP plane scalars (``doom_sandbox`` ``bsp.py``).

``decode_child`` / ``make_plane`` / ``side_P`` plus the ``BspPlane`` tuple — the
handful of pure-scalar helpers the Python renderer and drafter call. The rest of
the sandbox ``bsp`` module (path coefficients, traversal ordering) became the
compiled BSP graph and has no counterpart here. Copied verbatim.
"""

from __future__ import annotations

from typing import NamedTuple

from ..prompt.types import SUBSECTOR_FLAG


class BspPlane(NamedTuple):
    """Implicit-form plane ``nx*x + ny*y + d == 0``; FRONT side iff > 0."""

    nx: float
    ny: float
    d: float


def decode_child(child_ref: int) -> tuple[bool, int]:
    """Decode a BSP child reference. Returns ``(is_subsector, index)``."""
    if child_ref & SUBSECTOR_FLAG:
        return True, child_ref & ~SUBSECTOR_FLAG
    return False, child_ref


def make_plane(node) -> BspPlane:
    """Convert a BspNode's ``(px, py, dx, dy)`` to ``(nx, ny, d)``.

    DOOM classifies a point FRONT when ``dx*(y - py) < dy*(x - px)`` ⇒
    ``(nx, ny, d) = (dy, -dx, dx*py - dy*px)``. Not unit-normalized.
    """
    return BspPlane(
        nx=float(node.dy),
        ny=float(-node.dx),
        d=float(node.dx) * float(node.py) - float(node.dy) * float(node.px),
    )


def side_P(plane: BspPlane, px: float, py: float) -> int:
    """Return ``1`` if ``(px, py)`` is on the FRONT side of ``plane``, else ``0``."""
    return 1 if (plane.nx * px + plane.ny * py + plane.d) > 0 else 0
