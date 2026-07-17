"""Primitive token types — ``TokenType``, ``Token``, ``IntSlot``, ``FloatSlot``.

This module runs once at compile time: it builds part of the computation
graph that torchwright lowers into the transformer's weights. Nothing here
executes during inference — at render time, only the compiled transformer
runs. Coined terms: see GLOSSARY.md.

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
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TypeAlias

if TYPE_CHECKING:
    from torchwright.graph import Node


DerivedFn = Callable[[Any], "float | Sequence[float]"]
DerivedLike: TypeAlias = "Derived | Callable[..., Any]"


@dataclass(frozen=True)
class Derived:
    """A precomputed function of a slot value, written into ``input_vec``
    as ``width`` consecutive columns at embed time (depth 0).

    ``width=1`` (the default) is a scalar derivation and ``fn`` returns a
    float; ``width=N`` packs a precomputed N-vector (per-screen-column
    projection rays, a one-hot mask, the 3-digit ``id_lifted_key``, …)
    into one named span and ``fn`` returns a length-``N`` sequence. The
    embedding builder evaluates ``fn`` per row; ``extract_derived`` reads
    the span back at depth 0.
    """

    fn: DerivedFn
    width: int = 1

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError("Derived.fn must be callable")
        if self.width < 1:
            raise ValueError(f"Derived.width must be >= 1, got {self.width}")


def _normalize_derived(
    slot_kind: str, derived: Mapping[str, DerivedLike] | None
) -> dict[str, "Derived"]:
    """Accept either ``Derived`` instances or bare scalar callables on a
    slot's ``derived`` map, normalizing bare callables to
    ``Derived(fn, width=1)``. Keeps existing scalar-callable declarations
    working while the multi-width ``Derived`` form is available.
    """
    if not derived:
        return {}
    out: dict[str, Derived] = {}
    for name, value in derived.items():
        if not name:
            raise ValueError(f"{slot_kind} derived names must be non-empty")
        if isinstance(value, Derived):
            out[name] = value
        elif callable(value):
            out[name] = Derived(value, width=1)
        else:
            raise TypeError(
                f"{slot_kind} derived entry {name!r} must be a Derived "
                f"instance or a callable; got {type(value).__name__}"
            )
    return out


@dataclass(frozen=True)
class IntSlot:
    """Integer range ``[lo, hi)``.

    ``derived`` declares per-slot-value precomputed columns: each entry
    is a :class:`Derived` (bare callables are normalized to
    ``Derived(fn, width=1)``), evaluated on the integer slot value at
    embed time and written to the columns the embedding table reserves
    for that ``(type, slot, name)`` declaration.
    """

    lo: int
    hi: int
    derived: Mapping[str, DerivedLike] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "derived", _normalize_derived("IntSlot", self.derived))


@dataclass(frozen=True)
class FloatSlot:
    """Float range ``[lo, hi]`` quantized to ``levels`` evenly-spaced steps.

    ``derived`` follows the same rules as ``IntSlot.derived``, evaluated
    on the (float) slot value.
    """

    lo: float
    hi: float
    levels: int = 65536
    derived: Mapping[str, DerivedLike] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "derived", _normalize_derived("FloatSlot", self.derived)
        )


@dataclass(eq=False)
class TokenType:
    """Name-based equality.

    Two instances with the same ``name`` compare equal and hash equally,
    regardless of slot definitions; names are expected to be unique
    within a vocab. A helper that reconstructs a type from its name
    therefore compares equal to the declared type — removing a class of
    silent "looks like the same type but isn't" failures.
    """

    name: str
    slots: Mapping[str, IntSlot | FloatSlot] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TokenType) and self.name == other.name

    def check(self, input_vec: "Node") -> "Node":
        """±1 indicator that ``input_vec``'s active type is this type.

        Method form of :func:`torchwright_doom.model.extract.is_type_pm1`
        (imported lazily — ``extract`` -> ``embedding`` -> ``vocab`` ->
        ``tokens`` would otherwise cycle).
        """
        from .extract import is_type_pm1

        return is_type_pm1(input_vec, self)

    def extract(self, input_vec: "Node", slot_name: str) -> "Node":
        """Read the ``(self, slot_name)`` column from ``input_vec``.

        Method form of
        :func:`torchwright_doom.model.extract.extract_type_slot` (lazy import,
        as for :meth:`check`).
        """
        from .extract import extract_type_slot

        return extract_type_slot(input_vec, self, slot_name)


@dataclass
class Token:
    type: TokenType
    values: Mapping[str, int | float] = field(default_factory=dict)
