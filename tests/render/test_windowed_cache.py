"""Windowed-cache host-policy tests (permanent/expiring slots), no GPU.

The torchwright exporter owns the graph half of the windowed protocol
(writtenness mask + staging scatter — tested in torchwright's
test_windowed_cache_onnx.py); this file pins the HOST half that
``OnnxTokenRuntime`` layers on top:

  - slot allocation: strictly sequential while the buffer fills (the
    graph's ``j < base`` writtenness mask requires first-fill in slot
    order), then recycling of EXPIRED slots in cursor order — permanent
    rows are never overwritten, and saturation (all slots permanent)
    fails loud;
  - persistence scheduling: single-row decodes write through; every
    multi-row batch (speculative verify batches AND prefill chunks)
    pends on ``WindowedState.pending`` and ``commit`` allocates slots for
    exactly the accepted prefix — a rejected draft row never consumes a
    slot (the regression that motivated the pending discipline: under
    persist-then-rollback it would evict a live row for a row that
    never becomes part of the stream).

The mock mirrors ``_MockCompiled`` in test_spec_decode_logic.py but
routes every write through the production ``persist_rows`` /
``commit`` helpers with a per-row expiring tag (the body tokens
100..110 "are pixels"; 50 and the terminal are permanent), and writes
K = position / V = token so buffer contents are checkable against the
final stream.
"""

from __future__ import annotations

import pytest
import torch

from torchwright_doom.render.generation import TokenRuntime
from torchwright_doom.render.kv_cache import (
    KVCache,
    WindowedState,
    alloc_runs,
    alloc_slot,
    commit,
    flush_pending,
    persist_rows,
)

_TERMINAL = 999


def _model_next(r: int) -> int:
    if r == 50:
        return 100
    if 100 <= r < 110:
        return r + 1
    return _TERMINAL


def _is_expiring_token(token: int) -> bool:
    """The mock policy: body tokens are 'pixels'; 50/terminal permanent."""
    return 100 <= token < 999


def _windowed_cache(window: int, staging: int = 16) -> tuple[KVCache, WindowedState]:
    ws = WindowedState(window=window, staging=staging, slot_expiring=[False] * window)
    return (
        KVCache(
            k=[torch.zeros(window + staging, 1, 1)],
            v=[torch.zeros(window + staging, 1, 1)],
            length=0,
            max_len=10_000,
            windowed=ws,
        ),
        ws,
    )


def _rows(values: list[float]) -> list[torch.Tensor]:
    return [torch.tensor(values, dtype=torch.float32).reshape(len(values), 1, 1)]


# ---------------------------------------------------------------------------
# Slot allocation
# ---------------------------------------------------------------------------


def test_alloc_sequential_fill_then_recycle_skips_permanent():
    _, ws = _windowed_cache(window=6)
    # Fill: slots 0..5, alternating permanent/expiring tags.
    flags = [False, True, False, True, True, False]  # permanent at 0, 2, 5
    assert [alloc_slot(ws, f) for f in flags] == [0, 1, 2, 3, 4, 5]
    assert ws.n_permanent == 3
    # Recycling visits only the expired slots, in cursor order.
    assert alloc_slot(ws, True) == 1
    assert alloc_slot(ws, True) == 3
    assert alloc_slot(ws, True) == 4
    # Second revolution wraps back to slot 1.
    assert alloc_slot(ws, True) == 1


def test_alloc_recycled_slot_can_become_permanent():
    _, ws = _windowed_cache(window=4)
    for f in (False, True, True, False):
        alloc_slot(ws, f)
    # A permanent row recycles an expired slot — the pool shrinks.
    assert alloc_slot(ws, False) == 1
    assert ws.n_permanent == 3
    # Only slot 2 is left expiring.
    assert alloc_slot(ws, True) == 2
    assert alloc_slot(ws, True) == 2  # sole expired slot, every time


def test_alloc_saturation_raises():
    _, ws = _windowed_cache(window=3)
    for _ in range(3):
        alloc_slot(ws, False)
    with pytest.raises(RuntimeError, match="saturated"):
        alloc_slot(ws, True)


def test_alloc_runs_compress_contiguous_assignments():
    _, ws = _windowed_cache(window=8)
    # Pure fill: one run.
    assert alloc_runs(ws, [True] * 5) == [[0, 0, 5]]
    assert alloc_runs(ws, [False] * 3) == [[5, 0, 3]]
    # Recycling: slots 0..4 are expired and contiguous -> one run again.
    assert alloc_runs(ws, [True] * 3) == [[0, 0, 3]]


# ---------------------------------------------------------------------------
# Persistence scheduling
# ---------------------------------------------------------------------------


def _fill(cache: KVCache, upto: int, expiring_fn=_is_expiring_token) -> None:
    """Commit positions [length, upto) one row at a time
    (K=pos, V=1000+pos, expiring iff expiring_fn(pos) — position stands in
    for the token here)."""
    for p in range(cache.length, upto):
        persist_rows(
            cache, p, 1, _rows([float(p)]), _rows([1000.0 + p]), [expiring_fn(p)]
        )
        cache.length = p + 1


def test_single_row_writes_through_and_multi_row_pends():
    cache, ws = _windowed_cache(window=6)
    persist_rows(cache, 0, 1, _rows([0.0]), _rows([1000.0]), [False])
    cache.length = 1
    assert cache.k[0][0, 0, 0].item() == 0.0
    assert ws.pending is None
    # Multi-row: stashed, nothing written, no slots consumed.
    persist_rows(
        cache, 1, 3, _rows([1.0, 2.0, 3.0]), _rows([0.0] * 3), [True, True, True]
    )
    cache.length = 4
    assert ws.pending is not None
    assert ws.write_head == 1


def test_commit_flushes_accepted_prefix_only():
    """A rejected draft row never consumes a slot — the core discipline."""
    cache, ws = _windowed_cache(window=6)
    persist_rows(
        cache,
        0,
        3,
        _rows([0.0, 1.0, 2.0]),
        _rows([10.0, 11.0, 12.0]),
        [False, True, True],
    )
    cache.length = 3
    commit(cache, 2)  # accept rows 0..1, reject row 2
    assert ws.pending is None
    assert cache.length == 2
    assert ws.write_head == 2  # only two slots ever allocated
    assert cache.k[0][:2, 0, 0].tolist() == [0.0, 1.0]
    assert cache.k[0][2, 0, 0].item() == 0.0  # rejected row never landed


def test_commit_at_base_drops_whole_batch():
    cache, ws = _windowed_cache(window=6)
    _fill(cache, 4, expiring_fn=lambda p: False)
    persist_rows(cache, 4, 2, _rows([4.0, 5.0]), _rows([0.0] * 2), [True, True])
    cache.length = 6
    commit(cache, 4)
    assert ws.pending is None
    assert ws.write_head == 4  # no slots consumed by the dropped batch


def test_double_pending_raises():
    cache, _ws = _windowed_cache(window=6)
    persist_rows(cache, 0, 2, _rows([0.0, 1.0]), _rows([0.0] * 2), [True, True])
    with pytest.raises(RuntimeError, match="uncommitted"):
        persist_rows(cache, 2, 2, _rows([2.0, 3.0]), _rows([0.0] * 2), [True, True])


def test_windowed_persist_requires_flags():
    cache, _ws = _windowed_cache(window=6)
    with pytest.raises(RuntimeError, match="expiring tag"):
        persist_rows(cache, 0, 1, _rows([0.0]), _rows([0.0]))


def test_flush_pending_noop_without_pending():
    cache, _ws = _windowed_cache(window=6)
    _fill(cache, 3)
    flush_pending(cache, 3)
    assert cache.length == 3


def test_recycling_preserves_permanent_rows():
    """Drive past the fill boundary with mixed types: every permanent row
    stays resident, expired rows recycle oldest-first."""
    cache, _ws = _windowed_cache(window=6)
    # positions 0..5 fill the window; 0 and 3 permanent (per the token fn:
    # use explicit flags via a custom fn)
    perm = {0, 3}
    _fill(cache, 9, expiring_fn=lambda p: p not in perm)
    resident = {int(cache.k[0][s, 0, 0].item()) for s in range(6)}
    assert perm <= resident, resident
    # 9 rows over 6 slots with 2 permanent: the 4 most recent expiring
    # rows + the 2 permanent ones are resident.
    assert resident == {0, 3, 5, 6, 7, 8}, resident


# ---------------------------------------------------------------------------
# Full control flow over a windowed mock: spec-decode and pure-AR streams
# are bit-identical to the unbounded mock's, and the final buffer holds
# every permanent row + only committed values (drafts never leak).
# ---------------------------------------------------------------------------


class _MockWindowedKVCompiled(TokenRuntime):
    """Windowed twin of test_spec_decode_logic's _MockCompiled: memoryless
    transition-table model whose step persists through the production
    helpers (K = position, V = fed token, expiring = body tokens)."""

    def __init__(self, window: int, staging: int = 16):
        self.window = window
        self.staging = staging
        self._last_cache: KVCache | None = None

    def empty_past(self, max_len: int) -> KVCache:
        self._last_cache = KVCache(
            k=[torch.zeros(self.window + self.staging, 1, 1)],
            v=[torch.zeros(self.window + self.staging, 1, 1)],
            length=0,
            max_len=max_len,
            windowed=WindowedState(
                window=self.window,
                staging=self.staging,
                slot_expiring=[False] * self.window,
            ),
        )
        return self._last_cache

    def step(
        self,
        inputs: torch.Tensor,
        cache: KVCache,
        past_len: int | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        flat = inputs.reshape(-1)
        n = int(flat.shape[0])
        base = cache.length
        tokens = [int(round(float(flat[i]))) for i in range(n)]
        preds = torch.tensor([[float(_model_next(t))] for t in tokens])
        dk = [torch.arange(base, base + n, dtype=torch.float32).reshape(n, 1, 1)]
        dv = [flat.reshape(n, 1, 1).to(torch.float32)]
        persist_rows(cache, base, n, dk, dv, [_is_expiring_token(t) for t in tokens])
        cache.length = base + n
        return preds, cache

    def max_safe_prefill_chunk(self, planned_rows: int | None = None) -> int:
        return 1 << 30  # no int32-transient limit on an in-memory stand-in


def _ident_argmax(out: torch.Tensor) -> list[int]:
    return [int(round(float(x))) for x in out[:, 0]]


def _ident_token(tok: int) -> int:
    return tok


_EXPECTED = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 999]


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


def _assert_buffer_matches_stream(cache: KVCache, stream: list[int]):
    """Every slot holds a committed row (K = its position, V = the token
    fed there, expiring tag matching the token's type), and every
    PERMANENT position is resident — drafts and evicted rows never leak."""
    ws = cache.windowed
    assert ws is not None
    assert ws.pending is None
    resident: dict[int, float] = {}
    for slot in range(min(ws.window, ws.write_head)):
        p = int(cache.k[0][slot, 0, 0].item())
        v = cache.v[0][slot, 0, 0].item()
        assert 0 <= p < cache.length, (slot, p)
        assert v == float(stream[p]), (slot, p, v)
        assert ws.slot_expiring[slot] == _is_expiring_token(stream[p])
        resident[p] = v
    for p in range(cache.length):
        if not _is_expiring_token(stream[p]):
            assert p in resident, f"permanent position {p} evicted"


def test_windowed_spec_decode_stream_and_buffer():
    # window=6 over a 12-row stream (1 permanent prefill + 11 body rows):
    # the window fills and recycling runs for half the rollout, while
    # mispredicted drafts force partial accepts through pending/commit.
    drafts = [100, 101, 102, 555, 104, 105, 777, 107, 108, 109, 110, 999]
    compiled = _MockWindowedKVCompiled(window=6)
    res, stats = compiled.spec_decode_rollout(
        [50],
        _Drafter(drafts),
        max_positions=50,
        terminal_row=_TERMINAL,
        draft_window=4,
        argmax_fn=_ident_argmax,
        sandbox_token_to_row_fn=_ident_token,
        row_to_sandbox_token_fn=_ident_token,
    )
    assert res.emitted_rows == _EXPECTED
    assert stats["mispredicts"] >= 1
    assert compiled._last_cache is not None
    assert compiled._last_cache.windowed is not None
    assert compiled._last_cache.windowed.write_head == 6  # the window really filled
    _assert_buffer_matches_stream(compiled._last_cache, [50] + _EXPECTED)


def test_windowed_pure_ar_stream_and_buffer():
    compiled = _MockWindowedKVCompiled(window=6)
    res = compiled.pure_ar_rollout(
        [50], max_positions=50, terminal_row=_TERMINAL, argmax_fn=_ident_argmax
    )
    assert res.emitted_rows == _EXPECTED
    assert compiled._last_cache is not None
    _assert_buffer_matches_stream(compiled._last_cache, [50] + _EXPECTED)


def test_windowed_chunked_prefill_commits_per_chunk():
    """Prefill chunks ride the pending/commit path: run_prefill commits
    after every chunk, so multi-chunk prefill lands fully resident."""
    compiled = _MockWindowedKVCompiled(window=10)
    prefill = [50, 50, 50, 50, 50]  # permanent type
    out, past, passes = compiled.run_prefill(
        prefill, max_cache_len=100, chunk_size=2, label="test"
    )
    assert passes == 3  # 2 + 2 + 1
    ws = past.windowed
    assert ws is not None
    assert ws.pending is None
    assert past.length == 5
    assert ws.write_head == 5
    assert past.k[0][:5, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert ws.n_permanent == 5
