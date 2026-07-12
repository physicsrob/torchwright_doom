"""Row arithmetic of the token contract: token <-> ``W_EMBED``-row crossings.

All compute is pure host arithmetic over the static vocab; **no graph nodes are
created at import or call time** (the import-time-node-free rule, twdoom
CLAUDE.md).

The compiled artifact reads a 1-wide integer ``token_ids`` input and re-embeds
each id through its in-graph ``Embedding``; ``rows_to_input`` builds that input.
Decode is ``argmax(out @ W_EMBED.t())`` -> a row index -> ``(TokenType, values)``
via ``TOKEN_VOCAB.row_to_token``.

Native ``Token`` <-> row crossings (:func:`token_to_row`, :func:`row_to_token`)
are a direct structural round-trip with no name bridge — used by the standard
tokenizer, formatter, and special-token resolution.

:func:`carrier_delta` is vocab-derived contract math too: the canonical
encoded-value distance for a carrier row, shared by the graph-gate oracles.
This module must not acquire bundle, run, CLI, or Transformers
responsibilities.
"""

from __future__ import annotations

import torch

from ..model.embedding import TOKEN_VOCAB, W_EMBED
from ..model.tokens import IntSlot, Token, TokenType
from ..model.value_ranges import ValueRange, encode_float
from ..model.vocab import ANGLE_VALUE, VALUE

# --- row <-> (type, slot_values) ------------------------------------------


def row_index(token_type: TokenType, slot_values: dict | None = None) -> int:
    """``W_EMBED`` row index for ``token_type`` carrying ``slot_values``.

    Mirrors ``TokenVocab``'s row enumeration: slots are laid out in declaration
    order, mixed-radix, with each slot's index its quantization step.
    """
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
    """Stack a token sequence into an ``(n_pos, d_embed)`` pre-embedded input.

    Accepts ``Token`` instances or compact ``(type, slot_values)`` tuples. Used
    by the teacher-forced diagnostic (the pre-embedded ``iv`` graph). The compiled
    token-id artifact uses :func:`rows_to_input` instead.
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


def rows_to_input(rows) -> torch.Tensor:
    """``(n_new, 1)`` float token-id tensor for ``compiled.step`` (token-id input).

    The artifact's only input slot is the 1-wide ``token_ids``, so the raw tensor
    *is* full-width — no column padding needed.
    """
    return torch.tensor([[float(r)] for r in rows], dtype=torch.float32)


# --- row <-> native Token ---
#
# Token identity is by structure (same type + slot values), so the round-trip
# between a native ``Token`` and its ``W_EMBED`` row index is direct. Used by
# the standard tokenizer, formatter, and BOS/EOS resolution.


def token_to_row(tok: Token) -> int:
    """Encode a native ``Token`` to its ``W_EMBED`` row index."""
    return row_index(tok.type, dict(tok.values))


def row_to_token(row: int) -> Token:
    """Decode a ``W_EMBED`` row index back to a native ``Token``.

    Slot values round-trip through the same value-range quantization used to
    encode them.
    """
    rtype, values = TOKEN_VOCAB.row_to_token[row]
    return Token(rtype, dict(values))


# --- carrier distance ------------------------------------------------------
#
# Carrier row ranges, derived from the vocab (not hardcoded) so a layout change
# can't silently break the delta math.

_VALUE_START, _VALUE_END = TOKEN_VOCAB.type_to_row_range[VALUE]
_VALUE_LEVELS = _VALUE_END - _VALUE_START
_ANGLE_START, _ANGLE_END = TOKEN_VOCAB.type_to_row_range[ANGLE_VALUE]
_ANGLE_LEVELS = _ANGLE_END - _ANGLE_START
_ANGLE_LO = -_ANGLE_LEVELS // 2  # angle slot is centered (IntSlot(-N/2, N/2))


def carrier_delta(name: str, predicted_row: int, expected_row: int):
    """Encoded-value distance for a carrier; None if the prediction isn't even
    the right carrier type (a hard divergence).

    The canonical definition — the scene and emit graph-gate oracles import it
    (per-test copies previously hardcoded the vocab row ranges; these derive
    from ``TOKEN_VOCAB`` so a layout change can't silently break the delta
    math).
    """
    if name == "angleValue":
        if not (_ANGLE_START <= predicted_row < _ANGLE_END):
            return None
        pa = predicted_row - _ANGLE_START + _ANGLE_LO
        ea = expected_row - _ANGLE_START + _ANGLE_LO
        half = _ANGLE_LEVELS // 2
        return abs(((pa - ea + half) % _ANGLE_LEVELS) - half)
    if not (_VALUE_START <= predicted_row < _VALUE_END):
        return None
    return abs(predicted_row - expected_row) * 2.0 / (_VALUE_LEVELS - 1)
