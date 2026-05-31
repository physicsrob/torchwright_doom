"""Primitive token types — ``TokenType``, ``Token``, ``IntSlot``, ``FloatSlot``.

The renderer's actual vocabulary lives in :mod:`.vocab` and uses these
primitives to declare each token type. The prompt builder produces a
sequence of :class:`Token` instances; flattening to integer vocab IDs
against the embedding table is a separate downstream stage.

``IntSlot`` / ``FloatSlot`` carry an optional ``derived`` map of
precomputed scalar functions of the slot's value. Derived columns are
width-side metadata for the embedding table — they don't affect the
slot's cardinality. Each entry maps a column name to ``fn(slot_value)``
returning a float; the embedding builder writes that float into a
dedicated column shared by all tokens of the declaring type. Consumers
read the column directly at depth 0, skipping the PWL chain that would
otherwise reconstruct the function from the raw slot value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass(frozen=True)
class IntSlot:
    """Integer range ``[lo, hi)``.

    ``derived`` declares per-slot-value precomputed columns. Each entry
    ``name -> fn`` is evaluated on the integer slot value at embed time
    and written to a column the embedding table reserves for that
    ``(type, slot, name)`` triple.
    """

    lo: int
    hi: int
    derived: Mapping[str, Callable[[int], float]] = field(default_factory=dict)


@dataclass(frozen=True)
class FloatSlot:
    """Float range ``[lo, hi]`` quantized to ``levels`` evenly-spaced steps.

    ``derived`` follows the same rules as ``IntSlot.derived``, evaluated
    on the (float) slot value.
    """

    lo: float
    hi: float
    levels: int = 65536
    derived: Mapping[str, Callable[[float], float]] = field(default_factory=dict)


@dataclass(eq=False)
class TokenType:
    """Name-based equality (Plan A / A3).

    Two instances with the same ``name`` compare equal and hash equally,
    regardless of slot definitions; names are expected to be unique
    within a vocab. This mirrors the sandbox ``TokenType`` so copied
    sandbox helpers that reconstruct a type from its name keep working
    across the ported/native boundary — removing a class of silent
    "looks like the same type but isn't" failures.
    """

    name: str
    slots: Mapping[str, IntSlot | FloatSlot] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TokenType) and self.name == other.name


@dataclass
class Token:
    type: TokenType
    values: Mapping[str, int | float] = field(default_factory=dict)
