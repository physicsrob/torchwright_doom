"""``emit_token_head`` + ``emit_derived_zero`` reconstructs the full emit row.

The renderer's output head (``render_main.dispatch_next_token``) never builds a
full ``emit_token`` row — it selects over 236-col *heads* and concatenates one
shared ``emit_derived_zero`` tail at the end. That is only correct if
``concat(emit_token_head(t, **s), emit_derived_zero())`` is value-identical to
``emit_token(t, **s)`` for every token, so the teacher-forced oracle and the
free-run argmax see exactly the reference row. The whole-forward compile tests
exercise this end-to-end but are heavy; this pins the equivalence directly and
cheaply (exact math, no compile), for slotless, one-slot, and two-slot tokens.
"""

from __future__ import annotations

import pytest
import torch

from torchwright.debug.probe import reference_eval
from torchwright.graph import Concatenate

from torchwright_doom.model.embedding import TOKEN_VOCAB
from torchwright_doom.model.emit import (
    emit_derived_zero,
    emit_token,
    emit_token_head,
    head_width,
)
from torchwright_doom.model.std import constant
from torchwright_doom.model.vocab import NO_OP, SIDE_RECORD, THINK_SIDE, TRAVERSE_RETURN


def _slots(**kw):
    return {name: constant(float(v)) for name, v in kw.items()}


@pytest.mark.parametrize(
    "token_type, slot_kwargs",
    [
        (NO_OP, {}),  # slotless -> head is a pure literal
        (THINK_SIDE, {"node": 3}),  # one int slot
        (SIDE_RECORD, {"node": 5, "side": 1}),  # two int slots
        (
            TRAVERSE_RETURN,
            {"entity_u": 7, "depth": 2},
        ),  # two int slots, nonzero lo path
    ],
)
def test_head_plus_tail_equals_full_row(token_type, slot_kwargs) -> None:
    full = emit_token(token_type, **_slots(**slot_kwargs))
    recon = Concatenate(
        [emit_token_head(token_type, **_slots(**slot_kwargs)), emit_derived_zero()]
    )

    # Shape sanity: the head is exactly head_width() and the full row is d_embed
    # (so emit_token_head really drops only the derived tail).
    assert len(emit_token_head(token_type, **_slots(**slot_kwargs))) == head_width()
    assert len(full) == TOKEN_VOCAB.layout.d_embed
    assert len(recon) == TOKEN_VOCAB.layout.d_embed

    cache = reference_eval(recon, {}, 1)
    full_value = reference_eval(full, {}, 1)[full]
    assert torch.equal(
        cache[recon], full_value
    ), f"{token_type.name}{slot_kwargs}: head+tail diverges from full emit row"
