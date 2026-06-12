"""Spec-decode control flow (accept/commit/terminal/resync) with no GPU.

A scripted memoryless "model" (transition table) + a mock drafter exercise the
ported ``_spec_decode_step`` logic. The load-bearing assertion: the spec-decode
emitted stream is byte-identical to the pure-AR stream — even when the drafter
mispredicts — because the model's argmax decides every commit.
"""

from __future__ import annotations

import torch

from torchwright_doom.render import pure_ar, spec_decode
from torchwright_doom.render.inference import KVCache, _commit

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


class _MockCompiled:
    """``compiled.step`` over the owned-``KVCache`` contract, model = ``_model_next``.

    Mimics ``OnnxTokenRuntime``: ``empty_past(max_len)`` returns a
    preallocated ``KVCache``, and ``step`` writes each row's K/V into the cache
    tail *in place* and advances ``length`` — exactly the production delta
    write.  Drives the spec-decode in-place commit/overwrite path
    (``_commit`` lowering ``length``, the dead tail overwritten next step).
    The model is memoryless, so batched decode is trivially row-wise
    bit-identical to sequential — a faithful stand-in for the
    strict-optimization property the real artifact provides.
    """

    def empty_past(self, max_len):
        return KVCache(
            k=[torch.zeros(max_len, 1, 1)],
            v=[torch.zeros(max_len, 1, 1)],
            length=0,
            max_len=max_len,
        )

    def step(self, inputs, cache, past_len=None):
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


def _patch_decode(monkeypatch):
    """Make argmax/bridge identity over plain ints (no W_EMBED, no doom_sandbox)."""
    ident = lambda out: [int(round(float(x))) for x in out[:, 0]]  # noqa: E731
    monkeypatch.setattr(pure_ar, "argmax_rows", ident)
    monkeypatch.setattr(spec_decode, "argmax_rows", ident)
    monkeypatch.setattr(spec_decode, "sandbox_token_to_row", lambda tok: tok)
    monkeypatch.setattr(spec_decode, "row_to_sandbox_token", lambda row: row)


_EXPECTED = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 999]


def test_pure_ar_baseline(monkeypatch):
    _patch_decode(monkeypatch)
    res = pure_ar.pure_ar_rollout(_MockCompiled(), [50], max_positions=50, terminal_row=_TERMINAL)
    assert res.emitted_rows == _EXPECTED
    assert res.stopped == "terminal"


def test_spec_decode_perfect_drafter_is_bit_identical(monkeypatch):
    _patch_decode(monkeypatch)
    drafts = list(_EXPECTED)  # perfect: drafts == the trajectory (incl. seed + terminal)
    res, stats = spec_decode.spec_decode_rollout(
        _MockCompiled(), [50], _MockDrafter(drafts), max_positions=50,
        terminal_row=_TERMINAL, draft_window=8,
    )
    assert res.emitted_rows == _EXPECTED
    assert res.stopped == "terminal"
    assert stats["full_accepts"] >= 1
    assert stats["terminal_truncations"] >= 1
    # far fewer forward passes than the 12-token pure-AR rollout (1 prefill + few batches)
    assert res.n_forward_passes < len(_EXPECTED)


def test_spec_decode_with_mispredict_is_still_bit_identical(monkeypatch):
    _patch_decode(monkeypatch)
    # Corrupt two interior drafts; the model corrects each and the stream is unchanged.
    drafts = [100, 101, 102, 555, 104, 105, 777, 107, 108, 109, 110, 999]
    res, stats = spec_decode.spec_decode_rollout(
        _MockCompiled(), [50], _MockDrafter(drafts), max_positions=50,
        terminal_row=_TERMINAL, draft_window=8,
    )
    assert res.emitted_rows == _EXPECTED
    assert stats["mispredicts"] >= 1
    assert stats["terminal_truncations"] >= 1


def test_spec_decode_matches_pure_ar_exactly(monkeypatch):
    _patch_decode(monkeypatch)
    pure = pure_ar.pure_ar_rollout(_MockCompiled(), [50], max_positions=50, terminal_row=_TERMINAL)
    spec, _ = spec_decode.spec_decode_rollout(
        _MockCompiled(), [50], _MockDrafter([100, 101, 102, 555, 104]), max_positions=50,
        terminal_row=_TERMINAL, draft_window=4,
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
    cache = _commit(cache, 2)
    assert cache.length == 2

    # The next batch writes from the lowered length, overwriting rows 2..3.
    _, cache = compiled.step(torch.tensor([[20.0], [21.0], [22.0]]), cache)
    assert cache.length == 5

    # Committed prefix is correct; the rejected drafts (12, 13) are gone.
    got = cache.k[0][:5, 0, 0].tolist()
    assert got == [10.0, 11.0, 20.0, 21.0, 22.0], got
