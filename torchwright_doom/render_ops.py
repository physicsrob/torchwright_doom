"""Forward/render PWL-adjacent ops (Plan D / D0, extended for Plan E).

Real-side mirror of the read-side + traversal subset of
``doom_sandbox/implementation/forward/constants.py`` / ``forward/ops.py``.

- Plan D (read side): ``MARKER_PRESENT`` / ``and_`` / ``one_minus``.
- Plan E (BSP traversal): ``sub`` / ``neg`` / ``add_const`` (affine glue),
  ``mul_side`` (the ``R_PointOnSide`` cross-product multiply), and the
  ``SIDE_POSITIVE`` / ``IS_SUBSECTOR`` / ``DEPTH_NONZERO`` comparators.

The projection/visplane/flat write-side of ``forward/ops.py`` (the atan2
octant tables, the y-clamp staircases, the scale products) stays out of
scope until the projection phase.
"""

from __future__ import annotations

import torch

from torchwright.graph import Linear, Node
from torchwright.ops.arithmetic_ops import compare, multiply_2d
from torchwright.ops.logic_ops import bool_all_true, bool_not

from .std import concat
from .vocab import N_NODES_MAX


# Marker and integer-slot ids are exact integers, so ``> 0.5`` cleanly
# means "different / present / nonzero". Mirrors the sandbox
# ``compare_const(0.5, input_range=(-0.5, 1.5))`` at the default
# ``step_sharpness`` (deadband 0.1, far wider than any marker noise).
def MARKER_PRESENT(marker: Node) -> Node:
    """±1 boolean: was a marker (≈1) published at the picked row?"""
    return compare(marker, 0.5)


def and_(a: Node, b: Node) -> Node:
    """Boolean conjunction over two ±1 predicates (sandbox ``and_``)."""
    return bool_all_true([a, b])


def one_minus(a: Node) -> Node:
    """Boolean negation of a ±1 predicate (sandbox ``one_minus``)."""
    return bool_not(a)


# ---------------------------------------------------------------------------
# Plan E: affine glue + the R_PointOnSide cross product
# ---------------------------------------------------------------------------


def sub(a: Node, b: Node) -> Node:
    """``a - b`` (sandbox ``sub`` -> ``linear(concat(a, b), [[1],[-1]])``)."""
    return Linear(
        concat(a, b),
        torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
        name="sub",
    )


def neg(a: Node) -> Node:
    """``-a`` (sandbox ``neg``)."""
    return Linear(a, torch.tensor([[-1.0]], dtype=torch.float32), name="neg")


def add_const(a: Node, c: float) -> Node:
    """``a + c`` as one fused ``Linear`` (sandbox ``add_const``)."""
    return Linear(
        a,
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.tensor([float(c)], dtype=torch.float32),
        name="add_const",
    )


# DOOM: R_PointOnSide cross-product sign (r_main.c). Only the SIGN feeds
# SIDE_POSITIVE. The sandbox keeps a magnitude split (MUL_SIDE_SMALL/LARGE)
# for precision near unit-normal coefficients, but (1) the BSP side test
# always feeds large node deltas (R2 = [-512, 512]) and (2) the real-side
# ``multiply_2d`` *extrapolates* out-of-range inputs (the sandbox ``multiply``
# clamps), so feeding a large coef to a small grid yields a degenerate bound.
# A single grid over the full coefficient / relative-coordinate ranges is
# therefore both correct (the sign is robust to the ~step1*step2/4 product
# noise for poses off the partition planes) and avoids the extrapolation trap.
# Node deltas are recovered within R2 = [-512, 512]; rel = view - p with both
# in R1 = [-1152, 1152], so |rel| <= 2304 — all inputs stay in-grid.
def mul_side(coef: Node, rel: Node) -> Node:
    """Product of a partition coefficient and a view-relative coordinate
    (sandbox ``mul_side``), resolved to the sign by ``SIDE_POSITIVE``."""
    return multiply_2d(
        coef,
        rel,
        max_abs1=512.0,
        max_abs2=2400.0,
        step1=8.0,
        step2=37.5,
        name="mul_side",
    )


def SIDE_POSITIVE(cross_z: Node) -> Node:
    """±1: is the R_PointOnSide cross product > 0? (sandbox ``SIDE_POSITIVE``)."""
    return compare(cross_z, 0.0)


def IS_SUBSECTOR(child_u: Node) -> Node:
    """±1: is a unified child id a subsector (>= N_NODES_MAX) vs a node?"""
    return compare(child_u, float(N_NODES_MAX) - 0.5)


def DEPTH_NONZERO(depth: Node) -> Node:
    """±1: is the BSP tree depth nonzero? (sandbox ``DEPTH_NONZERO``)."""
    return compare(depth, 0.5)
