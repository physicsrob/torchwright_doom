"""Feasibility/safety check for a *shared-slot-column* embedding layout.

Today every ``(token type, slot)`` pair gets its own private columns in
``W_EMBED``; an emitted token fills its ~3 columns and zeros the other ~225.
That per-type layout is what forces the renderer's output head to build one
near-all-zero candidate row per token type and discard all but one — the
"N×constant-zeros" the dispatch peak grows with.

The proposed alternative lays the table out the way a normal transformer would:
``[E8 type code][slot-A value][slot-B value][slot-C value]`` with **all token
types sharing the same slot columns**, the E8 code doing the type-discrimination.
The risk that decides whether that's viable: does ``argmax(emit_query · table)``
still land on the correct token when same-position slot values from *different*
types now collide in the same columns?

This test builds a faithful shared-column table (real E8 codes + the real
digit-quadratic value encoding, just relocated to shared columns) and the
matching emit queries for the whole vocabulary, and proves:

1. argmax is exact for **every** token (no collision flips the result), and
2. the worst-case margin stays comfortably positive — in particular the
   *type* margin (same value, different type → resolved purely by the E8 code)
   is large, far above the ~1-unit value-resolution margin the current design
   already depends on.

It is a design-validation test, not a test of shipping code: if it ever fails,
the shared-column layout is unsafe and must not be adopted as-is.
"""

from __future__ import annotations

import torch

from torchwright.graph.spherical_codes import index_to_vector

from torchwright_doom.embedding import (
    BASE,
    CENTER,
    TOKEN_VOCAB,
    _digit_count_for_cardinality,
    _step_index_for,
)


def _cardinality(slot) -> int:
    return (slot.hi - slot.lo) if hasattr(slot, "lo") else slot.levels


def _build_shared_layout():
    """Build the shared-column ``(table, query)`` for the whole vocab.

    table[i] is row i's embedding; query[i] is the emit query that should
    argmax back to row i. Columns: ``[E8 (8)] [digit-quad per slot position]``,
    every type's k-th slot sharing one block sized for the widest k-th slot.
    (The all-zero "derived" region is omitted — the emit query is zero there,
    so it never affects argmax.)
    """
    layout = TOKEN_VOCAB.layout
    rows = TOKEN_VOCAB.row_to_token
    n = len(rows)
    e8 = {name: index_to_vector(idx).float() for name, idx in layout.e8_indices.items()}

    n_pos = max(len(t.slots) for t, _ in rows)
    pos_card = [0] * n_pos
    for t, _ in rows:
        for j, slot in enumerate(t.slots.values()):
            pos_card[j] = max(pos_card[j], _cardinality(slot))
    pos_digits = [_digit_count_for_cardinality(c) for c in pos_card]
    pos_w = [2 * d for d in pos_digits]

    width = 8 + sum(pos_w)
    table = torch.zeros(n, width)
    query = torch.zeros(n, width)

    # Per-row E8 + per-position step index (-1 = type has no slot at position j).
    e8_idx = torch.empty(n, 8)
    steps = torch.full((n, n_pos), -1, dtype=torch.long)
    for i, (t, vals) in enumerate(rows):
        e8_idx[i] = e8[t.name]
        for j, (nm, slot) in enumerate(t.slots.items()):
            steps[i, j] = _step_index_for(slot, vals[nm])
    table[:, :8] = e8_idx
    query[:, :8] = e8_idx

    off = 8
    for j in range(n_pos):
        k = steps[:, j].clamp(min=0).float()
        present = (steps[:, j] >= 0).float()
        if pos_digits[j] == 1:
            c = k - CENTER
            blk_row = torch.stack([c, -c * c], 1)
            blk_q = torch.stack([2 * c, torch.ones_like(c)], 1)
        else:
            hi = (steps[:, j].clamp(min=0) // BASE).float() - CENTER
            lo = (steps[:, j].clamp(min=0) % BASE).float() - CENTER
            blk_row = torch.stack([hi, -hi * hi, lo, -lo * lo], 1)
            blk_q = torch.stack(
                [2 * hi, torch.ones_like(hi), 2 * lo, torch.ones_like(lo)], 1
            )
        w = pos_w[j]
        table[:, off : off + w] = blk_row * present[:, None]
        query[:, off : off + w] = blk_q * present[:, None]
        off += w
    return table, query, pos_card, pos_digits


def _argmax_report(query, table, subset=None):
    """Return (n_wrong, min_margin, worst_row) over rows in ``subset`` (or all),
    argmaxing each query against the *full* table."""
    table_t = table.t()
    n = table.shape[0]
    idx = torch.arange(n) if subset is None else torch.tensor(sorted(subset))
    n_wrong = 0
    min_margin = float("inf")
    worst = None
    B = 4096
    for s in range(0, len(idx), B):
        gold = idx[s : s + B]
        scores = query[gold] @ table_t  # (b, n)
        top2 = scores.topk(2, dim=1)
        pred = top2.indices[:, 0]
        n_wrong += int((pred != gold).sum())
        gold_score = scores[torch.arange(len(gold)), gold]
        runner_up = torch.where(pred == gold, top2.values[:, 1], top2.values[:, 0])
        m = gold_score - runner_up
        if float(m.min()) < min_margin:
            min_margin = float(m.min())
            worst = int(gold[int(m.argmin())])
    return n_wrong, min_margin, worst


def test_shared_layout_argmax_is_exact_for_every_token() -> None:
    table, query, _pc, _pd = _build_shared_layout()
    n_wrong, min_margin, worst = _argmax_report(query, table)
    name = TOKEN_VOCAB.row_to_token[worst][0].name if worst is not None else "?"
    assert (
        n_wrong == 0
    ), f"{n_wrong} tokens argmax to the wrong row in the shared layout"
    # The worst-case margin is a value-resolution margin (~1, the gap between
    # adjacent integer values) — the *same* margin the current per-type design
    # already operates at. Well above zero with room to spare.
    assert min_margin >= 0.5, f"min argmax margin {min_margin:.3f} (worst: {name})"


def test_shared_layout_type_margin_is_large() -> None:
    """Same value, different type collides in shared columns → resolved purely by
    the E8 code. That margin must dominate the ~1-unit value margin."""
    layout = TOKEN_VOCAB.layout
    e8 = torch.stack(
        [
            index_to_vector(layout.e8_indices[n]).float()
            for n in sorted(layout.e8_indices)
        ]
    )
    gram = e8 @ e8.t()
    self_dot = gram.diag().clone()
    gram.fill_diagonal_(float("-inf"))
    type_margin = (self_dot - gram.max(dim=1).values).min()
    # E8 codes are length-40 (self-dot 1600) with nearest cross-dot 1200 →
    # margin 400, ~400x the value-resolution margin the design already relies on.
    assert (
        float(type_margin) >= 100.0
    ), f"E8 type margin {float(type_margin):.1f} too small"


def test_shared_layout_emitted_types_match_current_design() -> None:
    """For the token types the renderer actually emits (the traversal spine), the
    shared layout's worst margin is at least as good as the per-type design's
    known-good ~0.95 — i.e. no regression on the tokens we emit."""
    table, query, _pc, _pd = _build_shared_layout()
    emitted = {
        "setCursorDirectionY",
        "R_PointOnSide",
        "pointOnSideResult",
        "bspFront",
        "bspReturn",
        "done",
        "noOp",
    }
    present = {
        i for i, (t, _) in enumerate(TOKEN_VOCAB.row_to_token) if t.name in emitted
    }
    assert present, "no emitted traversal types found in the vocab"
    n_wrong, min_margin, _worst = _argmax_report(query, table, subset=present)
    assert n_wrong == 0
    assert min_margin >= 0.9, f"emitted-type min margin {min_margin:.3f}"
