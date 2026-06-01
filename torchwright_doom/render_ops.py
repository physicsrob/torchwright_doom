"""Forward/render PWL-adjacent ops (Plan D / D0).

Real-side mirror of the read-side subset of
``doom_sandbox/implementation/forward/constants.py`` /
``forward/ops.py`` that the ported scene/protocol views consume:
``MARKER_PRESENT`` (a marker presence comparator), ``and_`` (boolean
conjunction), and ``one_minus`` (boolean negation). The large write-side
of the sandbox ``forward/ops.py`` (projection products, atan2 tables,
y-clamps) is renderer-write-side and out of scope for the read port.
"""

from __future__ import annotations

from torchwright.graph import Node
from torchwright.ops.arithmetic_ops import compare
from torchwright.ops.logic_ops import bool_all_true, bool_not


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
