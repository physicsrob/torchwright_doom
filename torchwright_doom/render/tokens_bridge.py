"""Token <-> ``W_EMBED``-row encode/decode, plus the sandbox<->real name bridge.

This is the public home for the row-index helpers that used to live only in
``tests/prefill_fixture.py`` (that file now re-exports from here). All compute is
pure host arithmetic over the static vocab; **no graph nodes are created at import
or call time** (the import-time-node-free rule, twdoom CLAUDE.md). ``doom_sandbox``
is imported lazily so ``torchwright_doom`` stays importable standalone.

The compiled artifact reads a 1-wide integer ``token_ids`` input and re-embeds
each id through its in-graph ``Embedding``; ``rows_to_input`` builds that input.
Decode is ``argmax(out @ W_EMBED.t())`` -> a row index -> ``(TokenType, values)``
via ``TOKEN_VOCAB.row_to_token``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..embedding import TOKEN_VOCAB, W_EMBED
from ..tokens import IntSlot, Token, TokenType
from ..value_ranges import ValueRange, encode_float
from ..vocab import VOCAB_TYPES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from doom_sandbox.api.tokens import Token as SandboxToken


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
    from ..vocab import VALUE

    return (VALUE, {"v": encode_float(range_id, physical)})


def rows_to_input(rows) -> torch.Tensor:
    """``(n_new, 1)`` float token-id tensor for ``compiled.step`` (token-id input).

    The artifact's only input slot is the 1-wide ``token_ids``, so the raw tensor
    *is* full-width — no column padding needed.
    """
    return torch.tensor([[float(r)] for r in rows], dtype=torch.float32)


# --- sandbox <-> real name bridge -----------------------------------------
#
# The reference drafter (doom_sandbox) emits sandbox ``Token``s; the compiled
# artifact speaks ``W_EMBED`` rows. The two vocabularies are a pinned 1:1 mirror
# (scripts/vocab_diff.py), keyed by ``TokenType.name`` (identity is by-name on
# both sides). These two functions are the only crossing points.

# real (torchwright_doom) TokenType keyed by name — built once, no graph nodes.
_REAL_BY_NAME: dict[str, TokenType] = {t.name: t for t in VOCAB_TYPES}

_SB_TYPE_BY_NAME: dict[str, Any] | None = None


def _sandbox_types() -> dict[str, Any]:
    """Sandbox ``TokenType`` keyed by name (lazy; requires the sibling checkout)."""
    global _SB_TYPE_BY_NAME
    if _SB_TYPE_BY_NAME is None:
        try:
            from doom_sandbox.implementation.setup import VOCAB
        except ImportError as e:  # pragma: no cover - exercised only standalone
            raise RuntimeError(
                "row_to_sandbox_token needs the doom_sandbox sibling checkout "
                "(its TokenType objects); not available in a standalone "
                "torchwright_doom checkout."
            ) from e
        _SB_TYPE_BY_NAME = {t.name: t for t in VOCAB.types}
    return _SB_TYPE_BY_NAME


def real_type_for_name(name: str) -> TokenType:
    """The real ``TokenType`` mirroring a sandbox token type ``name``."""
    try:
        return _REAL_BY_NAME[name]
    except KeyError as e:  # pragma: no cover - guarded by the totality test
        raise KeyError(
            f"sandbox token type {name!r} has no real (torchwright_doom) mirror; "
            f"the vocabularies are supposed to be 1:1 (scripts/vocab_diff.py)."
        ) from e


def sandbox_token_to_row(tok: "SandboxToken") -> int:
    """Encode a sandbox ``Token`` to its ``W_EMBED`` row index.

    The accept test in speculative decoding is exactly
    ``predicted_row == sandbox_token_to_row(drafted_token)`` — "bit-identical"
    means "same row id".
    """
    return row_index(real_type_for_name(tok.type.name), dict(tok.values))


def row_to_sandbox_token(row: int) -> "SandboxToken":
    """Decode a ``W_EMBED`` row index back to a sandbox ``Token``.

    Used to feed ``drafter.consume`` the model's own correction/bonus emission
    (the §3.4 re-sync). Slot values round-trip through the same value-range
    quantization the drafter encodes with; sandbox type identity is by name.
    """
    from doom_sandbox.api.tokens import Token as SandboxToken

    rtype, values = TOKEN_VOCAB.row_to_token[row]
    return SandboxToken(_sandbox_types()[rtype.name], dict(values))
