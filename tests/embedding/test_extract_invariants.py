"""Construction-time validation tests for the extract primitives.

These pin the load-bearing checks the primitives make at construction
time so they fail loudly if the vocab or a caller violates an
assumption.

* ``_is_type_threshold(T)`` rejects a vocab where ``T``'s E8 self-dot
  minus its worst cross-dot is below the margin ``compare`` needs to
  saturate. The current vocab passes (gap = 400, margin = 200) — we
  also probe that property directly.
* The flat-namespace ``extract_int_slot`` / ``extract_float_slot``
  construct against a fabricated vocab where same-named slots disagree
  on ``(lo, hi)`` and confirm construction raises.
* ``extract_type_slot`` raises when ``(T, name)`` doesn't exist on
  ``T``.
"""

from __future__ import annotations

import pytest

from torchwright.graph.spherical_codes import index_to_vector

from torchwright_doom import extract
from torchwright_doom.embedding import TOKEN_VOCAB
from torchwright_doom.vocab import NODE

# ---------------------------------------------------------------------------
# is_type threshold margin — the load-bearing E8 sanity check.
# ---------------------------------------------------------------------------


def test_every_vocab_type_has_safe_is_type_gap() -> None:
    """For every type in the current vocab, ``_is_type_threshold(T)``
    succeeds — meaning the gap between self-dot (1600) and worst-case
    cross-dot is above the safety margin.

    If this regresses (a new vocab type lands at a too-near E8 index,
    or the codes get re-scaled), ``_is_type_threshold`` raises during
    construction. The plan called this out as the load-bearing
    condition to confirm before any forward() body wires up is_type
    against the vocab.
    """
    for T in TOKEN_VOCAB.types:
        # Should not raise.
        thresh = extract._is_type_threshold(T)
        # Threshold is exactly midway, given uniform max cross-dot = 1200
        # in the current E8 assignment.
        assert (
            1300.0 <= thresh <= 1500.0
        ), f"{T.name} threshold {thresh} far from the expected midpoint"


def test_e8_gap_directly() -> None:
    """Spot-check the underlying gap numbers the threshold derives from."""
    import torch

    layout = TOKEN_VOCAB.layout
    codes = torch.stack(
        [
            index_to_vector(layout.e8_indices[t.name]).to(torch.float32)
            for t in layout.types
        ]
    )
    # Self-dot is 1600 by construction (10x scaling on a unit sphere).
    self_dots = (codes * codes).sum(dim=1)
    assert torch.allclose(self_dots, torch.full_like(self_dots, 1600.0))

    # No row's worst cross-dot exceeds 1600 - MARGIN (= 1400).
    gram = codes @ codes.t()
    gram.fill_diagonal_(float("-inf"))
    max_cross_per_T = gram.max(dim=1).values
    assert float(max_cross_per_T.max()) <= 1600.0 - extract._IS_TYPE_GAP_MARGIN, (
        f"some type's worst cross-dot {float(max_cross_per_T.max())} crosses "
        f"the {extract._IS_TYPE_GAP_MARGIN}-unit margin"
    )


# ---------------------------------------------------------------------------
# Flat-namespace consistency
# ---------------------------------------------------------------------------


def test_flat_namespace_disagreement_raises(monkeypatch) -> None:
    """When two types declare the same slot name with different (lo, hi),
    ``extract_int_slot`` construction raises.

    We patch ``TOKEN_VOCAB.layout.types`` to simulate a divergent vocab
    rather than mutating the real one. The flat-namespace check walks
    the layout's type list looking for declaring types and validates
    (lo, hi) — that's the path the test stresses.
    """
    from torchwright_doom.tokens import IntSlot, TokenType

    fake_a = TokenType("fake_a", slots={"shared": IntSlot(0, 4)})
    fake_b = TokenType("fake_b", slots={"shared": IntSlot(0, 8)})

    # Patch the layout helpers ``_flat_declaring_types`` walks. Need
    # both ``types`` (for iteration) and ``slot_columns`` (assigned in
    # the raw-extract Linear construction — but the test should raise
    # before that point during ``_flat_declaring_types``).
    monkeypatch.setattr(TOKEN_VOCAB.layout, "types", [fake_a, fake_b], raising=True)
    monkeypatch.setattr(
        TOKEN_VOCAB.layout,
        "slot_columns",
        {("fake_a", "shared"): 0, ("fake_b", "shared"): 1},
        raising=True,
    )

    from torchwright.ops.inout_nodes import create_input

    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    with pytest.raises(ValueError, match=r"flat-namespace IntSlot.*disagrees"):
        extract.extract_int_slot_raw(inp, "shared")


def test_flat_namespace_missing_kind_raises(monkeypatch) -> None:
    """When the requested kind (IntSlot / FloatSlot) doesn't match what
    any type declares for ``name``, construction raises."""
    from torchwright.ops.inout_nodes import create_input

    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    # 'v' is a FloatSlot on VALUE; asking for it as an IntSlot should fail.
    with pytest.raises(ValueError, match=r"no type in the vocab declares slot"):
        extract.extract_int_slot_raw(inp, "v")


# ---------------------------------------------------------------------------
# extract_type_slot rejects unknown (T, name)
# ---------------------------------------------------------------------------


def test_extract_type_slot_rejects_unknown_slot() -> None:
    """``extract_type_slot(T, name)`` raises if ``T`` doesn't declare
    ``name`` — a typo at the call site shouldn't silently get a 0."""
    from torchwright.ops.inout_nodes import create_input

    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    # NODE declares 'j' but not 'flag'.
    with pytest.raises(ValueError, match=r"has no slot"):
        extract.extract_type_slot_raw(inp, NODE, "flag")


# ---------------------------------------------------------------------------
# extract_derived rejects unknown name
# ---------------------------------------------------------------------------


def test_extract_derived_rejects_unknown_name() -> None:
    """``extract_derived('nonexistent')`` raises."""
    from torchwright.ops.inout_nodes import create_input

    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    with pytest.raises(ValueError, match=r"is not declared"):
        extract.extract_derived(inp, "definitely_not_a_derived_name")
