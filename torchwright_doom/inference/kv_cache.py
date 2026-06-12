"""Runtime-owned static KV cache + the windowed slot-placement policy.

The vanilla-transformer analog is HF's ``StaticCache``: one preallocated
buffer per layer per side, owned by the host, written in place.  Two
protocols share the dataclass:

* **Unbounded** (``windowed is None``): slot == position, the buffer is the
  full ``cache_stride`` rows, committing just raises ``length``.
* **Windowed** (``windowed`` set): a fixed C-slot window whose slots are
  recycled under the PERMANENT/EXPIRING policy — every committed row is
  either permanent (resident for the whole run) or expiring (its slot may
  be recycled once the window fills), decided purely by the row's token
  type.  Slot placement is entirely host-side; the graph's writtenness
  mask + in-graph scatter own the wire half (see torchwright's exporter).

The speculative-decode discipline lives here too: a multi-row batch is
speculative — rejected rows must never consume a slot — so windowed
multi-row writes pend on :class:`WindowedState.pending` and
:func:`commit` flushes exactly the accepted prefix.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

# Host-side mirror of the certified read-surface census
# (plan_tier1_expiry.md at the umbrella; CLAUDE.md "Windowed KV cache —
# the protocol invariant"): worst consuming-read distance, in positions,
# for each CERTIFIED expiring token type.  ``alloc_slot`` refuses to
# recycle a slot younger than the active set's worst scope — without the
# guard the invariant ("no attention read may target an expiring-type row
# at long range") breaks SILENTLY and only shows up as token divergence
# in gates.  Adding a type here requires the same census + gate bar the
# tier-1 set passed (2026-06-12).
CERTIFIED_READ_SCOPE: dict[str, int] = {
    "pixel": 3,
    "setcursorx": 164,
    "setcursory": 2,
    "setcursordirectionx": 2,
    "setcursordirectiony": 2,
    "wallcolu": 2,
    "screeny": 292,
    "wallspanmeta": 619,
    "clipupdate": 1,
    "r_mapplane.row": 128,
}


def min_recycle_distance_for(expiring_types: Iterable[str]) -> int:
    """Worst certified read scope across the active expiring set.

    An uncertified type (an ad-hoc ``TWDOOM_EXPIRING_TYPES`` experiment)
    gets the conservative max of the certified table with a warning — the
    guard is a tripwire for the resident-read invariant, not a substitute
    for the census + gate that certification requires.
    """
    types = {t.lower() for t in expiring_types}
    unknown = sorted(types - set(CERTIFIED_READ_SCOPE))
    if unknown:
        print(
            f"[kvcache] WARNING: expiring types {unknown} have no certified "
            f"read scope — recycle guard falls back to the certified max "
            f"({max(CERTIFIED_READ_SCOPE.values())}); certify before "
            f"production use (see plan_tier1_expiry.md)",
            flush=True,
        )
        return max(CERTIFIED_READ_SCOPE.values())
    return max((CERTIFIED_READ_SCOPE[t] for t in types), default=0)


@dataclass
class PendingBatch:
    """A stashed multi-row windowed write awaiting :func:`commit`.

    The tensors are VIEWS of the runtime's persistent binding buffers —
    safe because commit always runs before the next step (the runtime's
    ``step`` raises on an unflushed pending).
    """

    base: int
    n_new: int
    dk_rows: list[torch.Tensor]
    dv_rows: list[torch.Tensor]
    expiring: list[bool]


@dataclass
class WindowedState:
    """Host-side slot-placement state of a windowed (``cache_window=C``) run.

    Slots fill strictly sequentially (the graph's ``j < base`` writtenness
    mask requires first-fill in slot order); once all C slots are written,
    new rows recycle the slots of expired rows, cursor order, skipping
    permanent ones.  Prefill needs no special case — scene tokens are
    permanent because of their types.
    """

    window: int  # C, the committed slot count baked into the compile
    # Allocated staging-tail width = the widest pass this cache can bind
    # (the graph scatters new rows at slots [C, C + n_new)).
    staging: int
    slot_expiring: list[bool]  # per-slot tag of the current occupant
    write_head: int = 0  # slots written so far (== fill progress, <= C)
    recycle_cursor: int = 0  # next candidate slot for recycling
    n_permanent: int = 0  # capacity guard: permanent rows can never exceed C
    pending: PendingBatch | None = None
    # Read-distance guard: absolute committed position of each slot's
    # current occupant (-1 = never written), and the minimum age (in
    # positions) a row must reach before its slot may recycle — the worst
    # certified read scope of the active expiring set
    # (``min_recycle_distance_for``).  0 disables the guard (positions
    # strictly increase, so age <= 0 never holds).
    slot_write_pos: list[int] = field(default_factory=list)
    min_recycle_distance: int = 0

    def __post_init__(self) -> None:
        if not self.slot_write_pos:
            self.slot_write_pos = [-1] * self.window


@dataclass
class KVCache:
    """Runtime-owned static KV cache for the sequence-major ONNX protocol.

    One preallocated buffer per layer per side, shape
    ``(n_slots, n_heads, d_head)``, **zero-initialized** — the graph's
    mask gives unwritten slots softmax weight exactly 0.0, and ``0 * NaN``
    from a ``torch.empty`` tail would still be NaN.  The whole buffer is
    bound as ``past_K_i`` every step (static shape + stable address: the
    CUDA-graph replay requirements); the committed prefix is
    ``k[i][:length]`` and the graph re-injects the new rows in-graph via
    ScatterND.  The runtime persists each step's ``delta_K_i`` output via
    :func:`persist_rows` after the run.  A speculative reject just lowers
    ``length`` — rows past it are masked (exactly-zero weight) and
    overwritten by a later write.

    ``max_len`` is the run's demand cap: on the unbounded protocol it is
    the slot count (== ``cache_stride``); on the windowed protocol slots
    and positions diverge by design, so it caps absolute POSITIONS (the
    pos-encoding table) instead.

    ``length`` always counts absolute committed POSITIONS (it exceeds
    ``windowed.window`` once recycling starts).
    """

    k: list[torch.Tensor]  # each (n_slots, n_heads, d_head)
    v: list[torch.Tensor]
    length: int
    max_len: int
    # THE PLACEMENT POLICY (windowed protocol only; None = unbounded).
    # See WindowedState — every committed row is PERMANENT or EXPIRING,
    # decided by the row's token type (OnnxTokenRuntime._expiring_rows;
    # default: only PIXEL rows expire — they publish no channels and are
    # read only at offset <= 3, so evicting old ones is safe by
    # construction).
    windowed: WindowedState | None = None


_RECYCLE_POOL_WARN = 256


def alloc_slot(ws: WindowedState, expiring: bool, pos: int) -> int:
    """Allocate the committed slot for the new row at absolute position ``pos``.

    Sequential while the buffer is filling (the writtenness-mask
    contract); afterwards, recycle the next expired slot in cursor
    order.  A full revolution without an expired slot means the cache
    is saturated with permanent rows — fail loud, that's a
    "recompile with a larger cache_window" event.

    Recycling enforces the resident-read invariant: the evicted row must
    be older than the active expiring set's certified read scope
    (``min_recycle_distance``), or some read could still target it —
    fail loud instead of corrupting the stream silently.
    """
    C = ws.window
    if ws.write_head < C:
        slot = ws.write_head
        ws.write_head += 1
    else:
        cur = ws.recycle_cursor
        for _ in range(C):
            if ws.slot_expiring[cur % C]:
                break
            cur += 1
        else:
            raise RuntimeError(
                f"windowed cache saturated: all {C} slots hold permanent "
                f"rows ({ws.n_permanent} permanent committed); recompile "
                f"with a larger model.cache_window or expire more token types"
            )
        slot = cur % C
        ws.recycle_cursor = cur + 1
        prev_pos = ws.slot_write_pos[slot]
        if prev_pos >= 0 and pos - prev_pos <= ws.min_recycle_distance:
            raise RuntimeError(
                f"windowed-cache recycle would evict a row still inside its "
                f"certified read scope: slot {slot} holds position {prev_pos}, "
                f"new row at position {pos} (age {pos - prev_pos} <= "
                f"min_recycle_distance {ws.min_recycle_distance}) — the "
                f"recycle pool is too small for the expiring set; grow "
                f"model.cache_window or shrink expiring_types"
            )
    if not expiring:
        ws.n_permanent += 1
        free = C - ws.n_permanent
        if free == _RECYCLE_POOL_WARN:
            print(
                f"[kvcache] WARNING: recycle pool down to {free} slots "
                f"({ws.n_permanent}/{C} permanent) — nearing saturation",
                flush=True,
            )
    ws.slot_expiring[slot] = expiring
    ws.slot_write_pos[slot] = pos
    return slot


def alloc_runs(ws: WindowedState, expiring: list[bool], base: int) -> list[list[int]]:
    """Allocate slots for a batch of rows (in position order, starting at
    absolute position ``base``) and compress the assignments into
    contiguous [slot_start, row_start, count] runs — sequential fill is
    one run; recycling fragments only as much as the expired slots do."""
    runs: list[list[int]] = []
    for row, flag in enumerate(expiring):
        slot = alloc_slot(ws, bool(flag), base + row)
        if (
            runs
            and runs[-1][0] + runs[-1][2] == slot
            and runs[-1][1] + runs[-1][2] == row
        ):
            runs[-1][2] += 1
        else:
            runs.append([slot, row, 1])
    return runs


def _write_runs(
    cache: KVCache,
    runs: list[list[int]],
    dk_rows: list[torch.Tensor],
    dv_rows: list[torch.Tensor],
) -> None:
    n_layers = len(cache.k)
    for slot, row, count in runs:
        for i in range(n_layers):
            cache.k[i][slot : slot + count] = dk_rows[i][row : row + count]
            cache.v[i][slot : slot + count] = dv_rows[i][row : row + count]


def persist_rows(
    cache: KVCache,
    base: int,
    n_new: int,
    dk_rows: list[torch.Tensor],
    dv_rows: list[torch.Tensor],
    expiring: list[bool] | None = None,
) -> None:
    """Write a pass's delta rows into the owned cache.

    ``dk_rows[i]`` / ``dv_rows[i]`` are the layer-i ``(n_new, nh, d_head)``
    delta tensors (views of the binding buffers are fine — see
    :class:`PendingBatch`).  Unbounded protocol: slots == positions,
    exactly the historical write.  Windowed protocol: ``expiring`` carries
    the per-row policy tag; single-row decodes commit unconditionally and
    write through; every multi-row batch (speculative verify batches AND
    prefill chunks) is stashed on ``WindowedState.pending`` and flushed by
    :func:`commit` — rejected rows never consume a slot.
    """
    ws = cache.windowed
    if ws is None:
        for i in range(len(cache.k)):
            cache.k[i][base : base + n_new] = dk_rows[i]
            cache.v[i][base : base + n_new] = dv_rows[i]
        return
    if expiring is None or len(expiring) != n_new:
        raise RuntimeError(
            "windowed persist requires a per-row expiring tag for every row"
        )
    if n_new == 1:
        _write_runs(cache, alloc_runs(ws, list(expiring), base), dk_rows, dv_rows)
        return
    if ws.pending is not None:
        raise RuntimeError(
            "windowed KVCache already holds an uncommitted batch; commit "
            "must flush it before the next multi-row step"
        )
    ws.pending = PendingBatch(base, n_new, list(dk_rows), list(dv_rows), list(expiring))


def flush_pending(cache: KVCache, target: int) -> None:
    """Flush the accepted prefix of a stashed batch: rows at positions
    [pending.base, target) get slots and persist; the rejected tail is
    dropped (its positions are re-drafted by the next pass)."""
    ws = cache.windowed
    if ws is None or ws.pending is None:
        return
    pending = ws.pending
    ws.pending = None
    keep = target - pending.base
    assert 0 <= keep <= pending.n_new, (
        f"commit target {target} outside the pending batch "
        f"[{pending.base}, {pending.base + pending.n_new}]"
    )
    if keep == 0:
        return
    _write_runs(
        cache,
        alloc_runs(ws, pending.expiring[:keep], pending.base),
        pending.dk_rows,
        pending.dv_rows,
    )


def commit(cache: KVCache, target: int) -> KVCache:
    """Set the committed length to ``target``: lower the logical length in
    place (no copy), and — on a windowed cache — flush the accepted prefix
    of the pending batch into freshly-allocated slots (see
    :func:`persist_rows`; rejected rows are dropped, never written).
    """
    flush_pending(cache, target)
    cache.length = target
    return cache
