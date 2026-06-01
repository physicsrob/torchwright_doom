"""Shared test helpers for the Plan D renderer-reads port.

Turns a list of ``Token`` (or compact ``(type, slot_values)`` tuples) into the
stacked ``W_EMBED`` rows the compiled graph's ``input_vec`` reads, mirroring the
hand-rolled ``_row_index`` in ``tests/embedding/test_extract_compiled.py``.
This is the prompt -> token_ids -> input bridge the oracle harness needs (no
public token->row helper exists yet; Plan A would own one).
"""

from __future__ import annotations

import torch

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.tokens import IntSlot, Token, TokenType
from torchwright_doom.value_ranges import ValueRange, encode_float
from torchwright_doom.vocab import VALUE


def row_index(token_type: TokenType, slot_values: dict | None = None) -> int:
    """W_EMBED row index for ``token_type`` carrying ``slot_values``."""
    slot_values = slot_values or {}
    start, _ = TOKEN_VOCAB.type_to_row_range[token_type]
    if not token_type.slots:
        return start
    names = list(token_type.slots.keys())
    objs = [token_type.slots[n] for n in names]

    def step(slot, value):
        if isinstance(slot, IntSlot):
            return int(value) - slot.lo
        span = slot.hi - slot.lo
        return round((float(value) - slot.lo) / span * (slot.levels - 1))

    sizes = [(s.hi - s.lo) if isinstance(s, IntSlot) else s.levels for s in objs]
    idxs = [step(objs[i], slot_values[n]) for i, n in enumerate(names)]
    row = 0
    for i, idx in enumerate(idxs):
        stride = 1
        for j in range(i + 1, len(sizes)):
            stride *= sizes[j]
        row += idx * stride
    return start + row


def token_row(token_type: TokenType, slot_values: dict | None = None) -> torch.Tensor:
    """The single ``W_EMBED`` row (width ``d_embed``) for one token."""
    return W_EMBED[row_index(token_type, slot_values)].clone()


def tokens_to_input(tokens) -> torch.Tensor:
    """Stack a token sequence into an ``(n_pos, d_embed)`` input tensor.

    Accepts ``Token`` instances or compact ``(type, slot_values)`` tuples.
    """
    rows = []
    for tok in tokens:
        if isinstance(tok, Token):
            rows.append(token_row(tok.type, dict(tok.values)))
        else:
            ttype, values = tok
            rows.append(token_row(ttype, values))
    return torch.stack(rows)


def value(range_id: ValueRange, physical: float) -> tuple[TokenType, dict]:
    """A compact ``VALUE`` carrier row encoding ``physical`` in ``range_id``."""
    return (VALUE, {"v": encode_float(range_id, physical)})
