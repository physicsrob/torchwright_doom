"""Primitive token types — ``TokenType``, ``Token``, ``IntSlot``, ``FloatSlot``.

The renderer's actual vocabulary lives in :mod:`.vocab` and uses these
primitives to declare each token type. The prompt builder produces a
sequence of :class:`Token` instances; flattening to integer vocab IDs
against the embedding table is a separate downstream stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class IntSlot:
    lo: int
    hi: int
    """Integer range ``[lo, hi)``."""


@dataclass(frozen=True)
class FloatSlot:
    lo: float
    hi: float
    levels: int = 65536


@dataclass(eq=False)
class TokenType:
    """Identity-based equality — module-level instances are unique by reference."""

    name: str
    slots: Mapping[str, IntSlot | FloatSlot] = field(default_factory=dict)


@dataclass
class Token:
    type: TokenType
    values: Mapping[str, int | float] = field(default_factory=dict)
