"""Windowed-cache (attention sink + sliding window) host-policy tests, no GPU.

The torchwright exporter owns the graph half of the windowed protocol
(mask + staging scatter — tested in torchwright's
test_windowed_cache_onnx.py); this file pins the HOST half that
``OnnxTokenRuntime`` layers on top:

  - the sink+ring slot mapping (prefill at identity slots, rollout rows
    wrapping through the ring),
  - persistence scheduling — and especially the speculative-decode rule
    that a multi-row rollout batch is stashed on ``cache.pending`` and
    only its ACCEPTED prefix reaches the ring at ``_commit``.  The
    unbounded cache tolerated persist-then-lower-length on a reject; in
    a ring that order would evict a live in-window row for a draft that
    never becomes part of the stream.  The regression test here fails
    against that (old) behavior by construction.

The mock mirrors ``_MockKVCompiled`` in test_spec_decode_logic.py but
routes every write through the production ``_persist_rows`` /
``_commit`` helpers, and writes K = position / V = token so the buffer
contents are checkable against the final stream.
"""

from __future__ import annotations

import pytest
import torch

from torchwright_doom.render import pure_ar, spec_decode
from torchwright_doom.render.inference import (
    KVCache,
    _commit,
    _flush_pending,
    _persist_rows,
    _window_slot,
    _window_slot_runs,
)

_TERMINAL = 999


def _model_next(r: int) -> int:
    if r == 50:
        return 100
    if 100 <= r < 110:
        return r + 1
    return _TERMINAL


def _windowed_cache(window: int, sink: int | None, staging: int = 16) -> KVCache:
    return KVCache(
        k=[torch.zeros(window + staging, 1, 1)],
        v=[torch.zeros(window + staging, 1, 1)],
        length=0,
        max_len=10_000,
        window=window,
        sink_len=sink,
        staging=staging,
    )


def _rows(values: list[float]) -> list[torch.Tensor]:
    return [torch.tensor(values, dtype=torch.float32).reshape(len(values), 1, 1)]


# ---------------------------------------------------------------------------
# Slot mapping
# ---------------------------------------------------------------------------


def test_window_slot_identity_then_wrap():
    cache = _windowed_cache(window=8, sink=4)
    # Sink positions persist at identity slots.
    assert [_window_slot(cache, p) for p in range(4)] == [0, 1, 2, 3]
    # Ring positions wrap through slots [4, 8).
    assert [_window_slot(cache, p) for p in range(4, 14)] == [
        4,
        5,
        6,
        7,
        4,
        5,
        6,
        7,
        4,
        5,
    ]


def test_window_slot_requires_sink_len():
    cache = _windowed_cache(window=8, sink=None)
    with pytest.raises(RuntimeError, match="sink_len"):
        _window_slot(cache, 5)


def test_window_slot_runs_split_at_wrap():
    cache = _windowed_cache(window=8, sink=4)
    # Positions 6..9 -> slots 6, 7, 4, 5: one wrap, two contiguous runs.
    assert _window_slot_runs(cache, 6, 4) == [(6, 0, 2), (4, 2, 2)]
    # Fully inside one revolution: a single run.
    assert _window_slot_runs(cache, 4, 3) == [(4, 0, 3)]
    # Prefill (identity): a single run.
    assert _window_slot_runs(cache, 0, 4) == [(0, 0, 4)]


# ---------------------------------------------------------------------------
# Persistence scheduling
# ---------------------------------------------------------------------------


def _fill(cache: KVCache, upto: int) -> None:
    """Commit positions [length, upto) one row at a time (K=pos, V=1000+pos)."""
    for p in range(cache.length, upto):
        _persist_rows(cache, p, 1, _rows([float(p)]), _rows([1000.0 + p]))
        cache.length = p + 1


def test_prefill_identity_and_single_row_wrap():
    cache = _windowed_cache(window=8, sink=4)
    # Prefill batch (all below sink): immediate, identity slots.
    _persist_rows(cache, 0, 4, _rows([0.0, 1.0, 2.0, 3.0]), _rows([0.0] * 4))
    cache.length = 4
    assert cache.k[0][:4, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert cache.pending is None
    # Single-row decodes: immediate, wrapping after one ring revolution.
    _fill(cache, 13)  # positions 4..12; ring slots hold the last 4
    # slot(12)=4+(12-4)%4=4, slot(9)=5, slot(10)=6, slot(11)=7
    assert cache.k[0][4:8, 0, 0].tolist() == [12.0, 9.0, 10.0, 11.0]
    assert cache.pending is None


def test_spec_batch_pends_and_commit_flushes_accepted_only():
    """THE ring regression: rejected speculative rows must never reach the
    ring.  The old persist-then-lower-length order would have evicted two
    live in-window rows here."""
    cache = _windowed_cache(window=8, sink=4)
    _fill(cache, 10)  # ring slots [4,5,6,7] hold positions [8, 9, 6, 7]
    ring_before = cache.k[0][4:8, 0, 0].tolist()
    assert ring_before == [8.0, 9.0, 6.0, 7.0]

    # A 3-row speculative batch at positions 10..12 (slots 6, 7, 4): the
    # step stashes it — the ring must be untouched until the commit.
    _persist_rows(
        cache, 10, 3, _rows([10.0, 11.0, 12.0]), _rows([1010.0, 1011.0, 1012.0])
    )
    cache.length = 13
    assert cache.pending is not None
    assert cache.k[0][4:8, 0, 0].tolist() == ring_before

    # Accept only the first row (the model rejected the drafts at 11, 12).
    _commit(cache, 11)
    assert cache.pending is None
    assert cache.length == 11
    # Position 10 landed at slot 6 (evicting position 6 — correctly out of
    # window); positions 7, 8, 9 survive.  Under the old order, slots 7 and
    # 4 would now hold the REJECTED rows 11 and 12, having evicted the
    # live rows 7 and 8.
    assert cache.k[0][4:8, 0, 0].tolist() == [8.0, 9.0, 10.0, 7.0]
    assert cache.v[0][6, 0, 0].item() == 1010.0


def test_full_accept_flushes_everything():
    cache = _windowed_cache(window=8, sink=4)
    _fill(cache, 10)
    _persist_rows(cache, 10, 3, _rows([10.0, 11.0, 12.0]), _rows([0.0] * 3))
    cache.length = 13
    _commit(cache, 13)
    # slots: 10->6, 11->7, 12->4; position 9 (slot 5) survives.
    assert cache.k[0][4:8, 0, 0].tolist() == [12.0, 9.0, 10.0, 11.0]


def test_commit_at_base_drops_whole_batch():
    cache = _windowed_cache(window=8, sink=4)
    _fill(cache, 10)
    ring_before = cache.k[0][4:8, 0, 0].tolist()
    _persist_rows(cache, 10, 2, _rows([10.0, 11.0]), _rows([0.0] * 2))
    cache.length = 12
    _commit(cache, 10)  # reject-at-0 (commit includes only the corrected row
    # in production; here: drop everything)
    assert cache.pending is None
    assert cache.k[0][4:8, 0, 0].tolist() == ring_before


def test_double_pending_raises():
    cache = _windowed_cache(window=8, sink=4)
    _fill(cache, 10)
    _persist_rows(cache, 10, 2, _rows([10.0, 11.0]), _rows([0.0] * 2))
    with pytest.raises(RuntimeError, match="uncommitted"):
        _persist_rows(cache, 12, 2, _rows([12.0, 13.0]), _rows([0.0] * 2))


def test_flush_pending_noop_without_pending():
    cache = _windowed_cache(window=8, sink=4)
    _fill(cache, 6)
    _flush_pending(cache, 6)  # no pending: a no-op, not an error
    assert cache.length == 6


# ---------------------------------------------------------------------------
# Full spec-decode control flow over a windowed mock: the emitted stream is
# bit-identical to pure AR, and the final buffer contents equal the committed
# stream (sink + last-ring-revolution positions) — drafts never leak.
# ---------------------------------------------------------------------------


class _MockWindowedKVCompiled:
    """Windowed twin of test_spec_decode_logic's _MockKVCompiled: memoryless
    transition-table model whose step persists through the production
    ``_persist_rows`` (K = position, V = fed token)."""

    def __init__(self, window: int, staging: int = 16):
        self.window = window
        self.staging = staging
        self._last_cache: KVCache | None = None

    def empty_past(self, max_len):
        self._last_cache = KVCache(
            k=[torch.zeros(self.window + self.staging, 1, 1)],
            v=[torch.zeros(self.window + self.staging, 1, 1)],
            length=0,
            max_len=max_len,
            window=self.window,
            staging=self.staging,
        )
        return self._last_cache

    def step(self, inputs, cache, past_len=None):
        flat = inputs.reshape(-1)
        n = int(flat.shape[0])
        base = cache.length
        preds = torch.tensor(
            [[float(_model_next(int(round(float(flat[i])))))] for i in range(n)]
        )
        dk = [torch.arange(base, base + n, dtype=torch.float32).reshape(n, 1, 1)]
        dv = [flat.reshape(n, 1, 1).to(torch.float32)]
        _persist_rows(cache, base, n, dk, dv)
        cache.length = base + n
        return preds, cache


def _patch_decode(monkeypatch):
    ident = lambda out: [int(round(float(x))) for x in out[:, 0]]  # noqa: E731
    monkeypatch.setattr(pure_ar, "argmax_rows", ident)
    monkeypatch.setattr(spec_decode, "argmax_rows", ident)
    monkeypatch.setattr(spec_decode, "sandbox_token_to_row", lambda tok: tok)
    monkeypatch.setattr(spec_decode, "row_to_sandbox_token", lambda row: row)


_EXPECTED = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 999]


def _assert_buffer_matches_stream(cache: KVCache, stream: list[int]):
    """Every committed slot holds the row for the LAST position mapped to it,
    with V = the token fed at that position — drafts never leak."""
    assert cache.sink_len is not None
    assert cache.pending is None
    for slot in range(cache.window):
        candidates = [p for p in range(cache.length) if _window_slot(cache, p) == slot]
        if not candidates:
            continue
        p = max(candidates)
        assert cache.k[0][slot, 0, 0].item() == float(p), (slot, p)
        assert cache.v[0][slot, 0, 0].item() == float(stream[p]), (slot, p)


def test_windowed_spec_decode_stream_and_ring_contents(monkeypatch):
    _patch_decode(monkeypatch)
    # window=6, sink=1 (the [50] prefill), ring=5: the 13-position
    # trajectory wraps the ring twice.  Mispredicted drafts force partial
    # accepts, exercising the pending->commit flush on every batch.
    drafts = [100, 101, 102, 555, 104, 105, 777, 107, 108, 109, 110, 999]
    compiled = _MockWindowedKVCompiled(window=6)
    res, stats = spec_decode.spec_decode_rollout(
        compiled,
        [50],
        _Drafter(drafts),
        max_positions=50,
        terminal_row=_TERMINAL,
        draft_window=4,
    )
    assert res.emitted_rows == _EXPECTED
    assert stats["mispredicts"] >= 1
    assert compiled._last_cache is not None
    _assert_buffer_matches_stream(compiled._last_cache, [50] + _EXPECTED)


def test_windowed_pure_ar_stream_and_ring_contents(monkeypatch):
    _patch_decode(monkeypatch)
    compiled = _MockWindowedKVCompiled(window=6)
    res = pure_ar.pure_ar_rollout(
        compiled, [50], max_positions=50, terminal_row=_TERMINAL
    )
    assert res.emitted_rows == _EXPECTED
    assert compiled._last_cache is not None
    _assert_buffer_matches_stream(compiled._last_cache, [50] + _EXPECTED)


class _Drafter:
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
