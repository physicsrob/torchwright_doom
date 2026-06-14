"""Spec-decode control flow (accept/commit/terminal/resync) with no GPU.

A scripted memoryless "model" (transition table) + a mock drafter exercise the
ported ``_spec_decode_step`` logic. The load-bearing assertion: the spec-decode
emitted stream is byte-identical to the pure-AR stream — even when the drafter
mispredicts — because the model's argmax decides every commit.
"""

from __future__ import annotations

import torch

from torchwright_doom.inference.generation import TokenRuntime
from torchwright_doom.inference.kv_cache import KVCache, commit

_TERMINAL = 999


def _model_next(r: int) -> int:
    """Memoryless next-token: 50 -> 100 -> 101 -> ... -> 110 -> 999 (terminal)."""
    if r == 50:
        return 100
    if 100 <= r < 110:
        return r + 1
    if r == 110:
        return _TERMINAL
    return _TERMINAL


class _MockDrafter:
    """Yields a fixed draft list; consume advances one position (resync = advance).

    Correctness of the emitted stream never depends on the drafts being right —
    only the model's argmax does — so a deliberately-wrong list still produces a
    bit-identical stream (it just costs mispredicts).
    """

    def __init__(self, drafts: list[int]):
        self.drafts = drafts
        self.i = 0

    def next_draft(self):
        return self.drafts[self.i] if self.i < len(self.drafts) else None

    def consume(self, actual):
        self.i += 1

    def snapshot(self):
        return self.i

    def rollback(self, snap):
        self.i = snap


class _MockCompiled(TokenRuntime):
    """``step`` over the owned-``KVCache`` contract, model = ``_model_next``.

    Mimics ``OnnxTokenRuntime``: ``empty_past(max_len)`` returns a
    preallocated ``KVCache``, and ``step`` writes each row's K/V into the cache
    tail *in place* and advances ``length`` — exactly the production delta
    write.  Drives the spec-decode in-place commit/overwrite path
    (``commit`` lowering ``length``, the dead tail overwritten next step).
    The model is memoryless, so batched decode is trivially row-wise
    bit-identical to sequential — a faithful stand-in for the
    strict-optimization property the real artifact provides.
    """

    def empty_past(self, max_len: int) -> KVCache:
        return KVCache(
            k=[torch.zeros(max_len, 1, 1)],
            v=[torch.zeros(max_len, 1, 1)],
            length=0,
            max_len=max_len,
        )

    def step(
        self,
        inputs: torch.Tensor,
        cache: KVCache,
        past_len: int | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        flat = inputs.reshape(-1)
        n = int(flat.shape[0])
        base = cache.length
        preds = torch.tensor(
            [[float(_model_next(int(round(float(flat[i])))))] for i in range(n)]
        )
        rows = flat.reshape(n, 1, 1).to(torch.float32)
        cache.k[0][base : base + n] = rows
        cache.v[0][base : base + n] = rows
        cache.length = base + n
        return preds, cache

    def max_safe_prefill_chunk(self, planned_rows: int | None = None) -> int:
        return 1 << 30  # no int32-transient limit on an in-memory stand-in


def _ident_argmax(out: torch.Tensor) -> list[int]:
    """Identity argmax over plain int 'logits' (no W_EMBED, no pydoom)."""
    return [int(round(float(x))) for x in out[:, 0]]


def _ident_token(tok: int) -> int:
    return tok


_EXPECTED = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 999]


def test_pure_ar_baseline():
    res = _MockCompiled().pure_ar_rollout(
        [50], max_positions=50, terminal_row=_TERMINAL, argmax_fn=_ident_argmax
    )
    assert res.emitted_rows == _EXPECTED
    assert res.stopped == "terminal"


def test_spec_decode_perfect_drafter_is_bit_identical():
    drafts = list(
        _EXPECTED
    )  # perfect: drafts == the trajectory (incl. seed + terminal)
    res, stats = _MockCompiled().spec_decode_rollout(
        [50],
        _MockDrafter(drafts),
        max_positions=50,
        terminal_row=_TERMINAL,
        draft_window=8,
        argmax_fn=_ident_argmax,
        token_to_row_fn=_ident_token,
        row_to_token_fn=_ident_token,
    )
    assert res.emitted_rows == _EXPECTED
    assert res.stopped == "terminal"
    assert stats["full_accepts"] >= 1
    assert stats["terminal_truncations"] >= 1
    # far fewer forward passes than the 12-token pure-AR rollout (1 prefill + few batches)
    assert res.n_forward_passes < len(_EXPECTED)


def test_spec_decode_with_mispredict_is_still_bit_identical():
    # Corrupt two interior drafts; the model corrects each and the stream is unchanged.
    drafts = [100, 101, 102, 555, 104, 105, 777, 107, 108, 109, 110, 999]
    res, stats = _MockCompiled().spec_decode_rollout(
        [50],
        _MockDrafter(drafts),
        max_positions=50,
        terminal_row=_TERMINAL,
        draft_window=8,
        argmax_fn=_ident_argmax,
        token_to_row_fn=_ident_token,
        row_to_token_fn=_ident_token,
    )
    assert res.emitted_rows == _EXPECTED
    assert stats["mispredicts"] >= 1
    assert stats["terminal_truncations"] >= 1


def test_spec_decode_matches_pure_ar_exactly():
    pure = _MockCompiled().pure_ar_rollout(
        [50], max_positions=50, terminal_row=_TERMINAL, argmax_fn=_ident_argmax
    )
    spec, _ = _MockCompiled().spec_decode_rollout(
        [50],
        _MockDrafter([100, 101, 102, 555, 104]),
        max_positions=50,
        terminal_row=_TERMINAL,
        draft_window=4,
        argmax_fn=_ident_argmax,
        token_to_row_fn=_ident_token,
        row_to_token_fn=_ident_token,
    )
    assert spec.emitted_rows == pure.emitted_rows


def test_kvcache_reject_then_overwrite():
    """The spec-decode reject primitive: lower length, then the next write
    overwrites the abandoned tail — verified on the buffer contents."""
    compiled = _MockCompiled()
    cache = compiled.empty_past(max_len=10)

    # Write a 4-row batch (e.g. one input + 3 drafts) at base 0.
    _, cache = compiled.step(torch.tensor([[10.0], [11.0], [12.0], [13.0]]), cache)
    assert cache.length == 4

    # Accept only the first 2 — the spec reject is a pure length rollback.
    cache = commit(cache, 2)
    assert cache.length == 2

    # The next batch writes from the lowered length, overwriting rows 2..3.
    _, cache = compiled.step(torch.tensor([[20.0], [21.0], [22.0]]), cache)
    assert cache.length == 5

    # Committed prefix is correct; the rejected drafts (12, 13) are gone.
    got = cache.k[0][:5, 0, 0].tolist()
    assert got == [10.0, 11.0, 20.0, 21.0, 22.0], got
