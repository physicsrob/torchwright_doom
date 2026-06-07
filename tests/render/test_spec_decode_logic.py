"""Spec-decode control flow (accept/commit/terminal/resync) with no GPU.

A scripted memoryless "model" (transition table) + a mock drafter exercise the
ported ``_spec_decode_step`` logic. The load-bearing assertion: the spec-decode
emitted stream is byte-identical to the pure-AR stream — even when the drafter
mispredicts — because the model's argmax decides every commit.
"""

from __future__ import annotations

import torch

from torchwright_doom.render import pure_ar, spec_decode

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


class _MockCompiled:
    """compiled.step that returns ``model_next(input)`` per batch row.

    Batched decode is trivially row-wise bit-identical to sequential for a
    memoryless model, so this is a faithful stand-in for the strict-optimization
    property the real artifact provides.
    """

    def empty_past(self):
        z = torch.zeros(1, 0, 1)
        return ((z,), (z,))

    def step(self, inputs, past, past_len=None):
        cache_len = past[0][0].shape[1]
        n = inputs.shape[0]
        preds = torch.tensor(
            [[float(_model_next(int(round(float(inputs[i, 0])))))] for i in range(n)]
        )
        new = torch.zeros(1, cache_len + n, 1)
        return preds, ((new,), (new,))


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
