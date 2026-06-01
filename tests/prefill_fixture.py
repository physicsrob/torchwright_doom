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
from torchwright_doom.vocab import (
    BEGIN,
    NODE,
    NODE_BACK_CHILD,
    NODE_DX,
    NODE_DY,
    NODE_FRONT_CHILD,
    NODE_PX,
    NODE_PY,
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    SEG,
    SEG_AX,
    SS,
    VALUE,
)


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


# Smallest scene that exercises the whole renderer spine: one BSP node (root=0,
# both children subsectors) and one subsector/seg, ending at BEGIN (the AR seed).
# Shared by the whole-forward compile gate and the free-running rollout gate.
TINY_BSP_SCENE: list[tuple[TokenType, dict]] = [
    (PLAYER_X_MARK, {}),
    value(ValueRange.R1, 100.0),
    (PLAYER_Y_MARK, {}),
    value(ValueRange.R1, -30.0),
    (NODE, {"j": 0}),
    (NODE_PX, {}),
    value(ValueRange.R1, 50.0),
    (NODE_PY, {}),
    value(ValueRange.R1, -20.0),
    (NODE_DX, {}),
    value(ValueRange.R2, 40.0),
    (NODE_DY, {}),
    value(ValueRange.R2, -30.0),
    (NODE_FRONT_CHILD, {"child_u": 64}),
    (NODE_BACK_CHILD, {"child_u": 65}),
    (SS, {"s": 0}),
    (SEG, {"i": 0, "is_first_of_ss": 1}),
    (SEG_AX, {}),
    value(ValueRange.R1, 10.0),
    (BEGIN, {}),
]


def pad_iv(compiled, iv_input: torch.Tensor) -> torch.Tensor:
    """Zero-pad an ``iv`` input tensor to a compiled module's full input width.

    ``CompiledHeadless.__call__`` takes a positional ``(n_pos, d_in)`` tensor and
    re-slices each declared input slot by ``_input_specs`` internally, so a test
    that builds only the ``iv`` slot must place it at its column offset in a
    full-width row. This localizes the one read of the private ``_input_specs``.
    """
    n_pos = iv_input.shape[0]
    specs = compiled._input_specs
    d_in = max(start + width for _, start, width in specs)
    start, width = next((s, w) for nm, s, w in specs if nm == "iv")
    full = torch.zeros(n_pos, d_in, dtype=iv_input.dtype)
    full[:, start : start + width] = iv_input
    return full
