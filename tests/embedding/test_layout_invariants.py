"""Plan A: W_EMBED layout invariants + the A2 derived-column contract.

Pins the per-declaration derived-span model: widths agree across types
for a shared name, a name is not declared on two slots of one type
(both raise ``VocabLayoutError`` at construction), and the live layout's
derived spans are disjoint and correctly sized.
"""

from __future__ import annotations

import pytest

from torchwright_doom.constants import SCREEN_WIDTH
from torchwright_doom.embedding import TOKEN_VOCAB, Layout, VocabLayoutError
from torchwright_doom.tokens import Derived, IntSlot, TokenType


def test_width_disagreement_raises() -> None:
    """Same derived name, different widths across two types -> raise."""
    t1 = TokenType("a", slots={"x": IntSlot(0, 4, derived={"d": Derived(lambda v: float(v))})})
    t2 = TokenType(
        "b",
        slots={"y": IntSlot(0, 4, derived={"d": Derived(lambda v: [float(v), 0.0], width=2)})},
    )
    with pytest.raises(VocabLayoutError):
        Layout([t1, t2])


def test_duplicate_name_within_type_raises() -> None:
    """Same derived name on two slots of one type -> raise (extract is
    name-addressed within the active type)."""
    t = TokenType(
        "a",
        slots={
            "x": IntSlot(0, 4, derived={"d": Derived(lambda v: float(v))}),
            "y": IntSlot(0, 4, derived={"d": Derived(lambda v: float(v))}),
        },
    )
    with pytest.raises(VocabLayoutError):
        Layout([t])


def test_same_name_same_width_across_types_ok() -> None:
    """A shared name with agreeing widths gets one span per declaration."""
    t1 = TokenType("a", slots={"x": IntSlot(0, 4, derived={"d": Derived(lambda v: float(v))})})
    t2 = TokenType("b", slots={"y": IntSlot(0, 4, derived={"d": Derived(lambda v: float(v))})})
    layout = Layout([t1, t2])
    entries = layout.derived_columns_by_name["d"]
    assert len(entries) == 2  # one span per declaration
    assert entries[0][3] == entries[1][3] == 1
    # distinct spans (no cross-type sharing)
    assert entries[0][2] != entries[1][2]


def test_live_derived_spans_disjoint() -> None:
    layout = TOKEN_VOCAB.layout
    spans = sorted(
        (start, start + width) for start, width in layout.derived_columns.values()
    )
    for (_s1, e1), (s2, _e2) in zip(spans, spans[1:]):
        assert e1 <= s2, f"derived spans overlap: ...{e1}) vs ({s2}..."
    # total derived region matches n_derived_columns
    total = sum(width for _start, width in layout.derived_columns.values())
    assert total == layout.n_derived_columns


def test_multiwidth_declared_widths() -> None:
    layout = TOKEN_VOCAB.layout
    assert layout.derived_columns_by_name["id_lifted_key"][0][3] == 3
    assert layout.derived_columns_by_name["u_tan_by_column"][0][3] == SCREEN_WIDTH
    # id_lifted_key is shared across several types, each its own span
    assert len(layout.derived_columns_by_name["id_lifted_key"]) >= 2
