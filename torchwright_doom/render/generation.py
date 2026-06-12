"""Generation loops over a token-stepped runtime (the GenerationMixin analog).

:class:`TokenRuntime` is the base every generation loop runs on: the
abstract trio (``empty_past`` / ``step`` / ``max_safe_prefill_chunk``) is
exactly the surface the concrete loop methods call, and the loops —
chunked prefill, pure autoregression, speculative decode — are ordinary
methods, the way ``model.generate()`` is in every mainstream stack.
Production is :class:`~torchwright_doom.render.onnx_runtime.OnnxTokenRuntime`;
the render logic tests subclass with in-memory transition-table models.

``tokens_bridge`` (and through it the screen-env-dependent vocab) is
imported lazily inside the loop bodies, so importing this module never
builds ``W_EMBED`` — ``apply_screen_env`` must precede the first rollout,
not this import.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from .kv_cache import KVCache, commit

# 1024-row prefill chunks bound the per-layer (n_heads, chunk, S) logits
# transient under the static-S cache (unchunked prefill at n=3613, S=12288
# peaks ~45 GB on the widest d=4096 layer).  Chunking is semantically
# identical to a single pass; this is a memory knob, not an algorithm change.
DEFAULT_PREFILL_CHUNK_SIZE = 1024


@dataclass
class RolloutResult:
    emitted_rows: list[int]
    stopped: str  # "terminal" | "cap"
    n_forward_passes: int
    seconds: float


def argmax_rows(outputs: torch.Tensor) -> list[int]:
    """Argmax-decode compiled-step LOGITS to token row ids.

    The artifact owns the unembed — production :class:`OnnxTokenRuntime`
    and torchwright's ``OnnxDebugSession`` both return logits.  (The
    in-process ``compile_headless`` gate decodes its embedding-width
    outputs with its own ``@ W_EMBED.T`` helper:
    ``tests/scene/test_forward_ar_rollout.py``.)
    """
    return outputs.detach().argmax(dim=-1).cpu().tolist()


class Drafter(Protocol):
    """Speculative-draft source for :meth:`TokenRuntime.spec_decode_rollout`.

    Deliberately a structural Protocol, not an ABC: the production
    implementation (``doom_sandbox``'s ``ARDrafter``) lives across a repo
    boundary and cannot subclass torchwright_doom types, and the logic tests
    carry a second implementation.  Two genuine implementers plus a package
    boundary is the case structural typing is for.
    """

    def next_draft(self) -> Any: ...

    def consume(self, actual: Any) -> None: ...

    def snapshot(self) -> Any: ...

    def rollback(self, snap: Any) -> None: ...


class TokenRuntime(ABC):
    """A token-stepped transformer over a host-owned static :class:`KVCache`.

    The abstract trio below is exactly the surface the concrete generation
    loops call; production is :class:`OnnxTokenRuntime` — the one real
    runtime — and the render logic tests substitute in-memory
    transition-table models (``tests/render/test_spec_decode_logic.py`` /
    ``test_windowed_cache.py``) by subclassing, so a missing or drifted
    method is a definition-time error, not a mid-rollout AttributeError.

    Guardrails (keep it this way): the abstract trio is frozen — never add
    an abstract member only a test needs — and the base carries no state
    (no ``__init__``); loop helpers stay module-private functions.
    """

    @abstractmethod
    def empty_past(self, max_len: int) -> KVCache:
        """Allocate (or zero-reset) the run's owned cache for ``max_len`` rows."""

    @abstractmethod
    def step(
        self,
        inputs: torch.Tensor,
        cache: KVCache,
        past_len: int | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """One forward pass: token ids in, per-position outputs + cache out."""

    @abstractmethod
    def max_safe_prefill_chunk(self, planned_rows: int | None = None) -> int:
        """Largest prefill chunk whose per-layer transients stay indexable."""

    def run_prefill(
        self,
        prefill_ids: list[int],
        *,
        max_cache_len: int,
        chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        label: str,
        progress_every: int = 0,
    ) -> tuple[torch.Tensor, KVCache, int]:
        """Run prefill in large chunks and return ``(last_out, past, n_passes)``.

        Allocates the run's single owned cache (``max_cache_len`` rows) up
        front via ``self.empty_past(max_cache_len)`` and writes prefill
        straight into it.
        """
        from .tokens_bridge import rows_to_input

        if not prefill_ids:
            raise ValueError("prefill_ids must be non-empty")
        chunk_size = max(1, int(chunk_size))
        # Clamp to the runtime's int32-indexability bound.  Bucket-aware: the
        # clamp is computed at the prefix bucket the whole prefill will bind,
        # so smaller buckets allow proportionally larger (equal-cost) chunks.
        safe_chunk = self.max_safe_prefill_chunk(len(prefill_ids))
        if chunk_size > safe_chunk:
            print(
                f"[{label}] prefill chunk {chunk_size} -> {safe_chunk} "
                f"(int32 transient-indexability clamp at the prefill's bucket)",
                flush=True,
            )
            chunk_size = safe_chunk
        n_chunks = (len(prefill_ids) + chunk_size - 1) // chunk_size
        if progress_every:
            print(
                f"[{label}] prefill start rows={len(prefill_ids)} "
                f"chunk_size={chunk_size} chunks={n_chunks} max_cache_len={max_cache_len}",
                flush=True,
            )

        past = self.empty_past(max_cache_len)
        ws = past.windowed
        if ws is not None:
            if len(prefill_ids) >= ws.window:
                raise RuntimeError(
                    f"prefill ({len(prefill_ids)} rows) does not fit the "
                    f"{ws.window}-slot cache window; raise model.cache_window"
                )
            print(
                f"[{label}] windowed cache: window={ws.window} "
                f"staging={ws.staging}; prefill ({len(prefill_ids)} rows) "
                f"and every non-expiring rollout row stay resident; expiring "
                f"rows recycle slots once the window fills",
                flush=True,
            )
        out = None
        offset = 0
        passes = 0
        for chunk_idx in range(n_chunks):
            chunk = prefill_ids[offset : offset + chunk_size]
            t0 = time.time()
            out, past = self.step(rows_to_input(chunk), past, past_len=offset)
            offset += len(chunk)
            # Multi-row batches pend on a windowed cache (the speculative-batch
            # discipline; prefill rides the same path) — every chunk is fully
            # accepted, so commit it before the next step.
            past = commit(past, offset)
            passes += 1
            dt = time.time() - t0
            if progress_every and (
                n_chunks > 1 or dt >= 5.0 or chunk_idx == n_chunks - 1
            ):
                print(
                    f"[{label}] prefill chunk {chunk_idx + 1}/{n_chunks} "
                    f"rows={len(chunk)} done in {dt:.1f}s "
                    f"cache_len={past.length}",
                    flush=True,
                )

        assert out is not None
        return out, past, passes

    def pure_ar_rollout(
        self,
        prefill_ids: list[int],
        max_positions: int,
        terminal_row: int,
        *,
        progress_every: int = 0,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        argmax_fn: Callable[[torch.Tensor], list[int]] | None = None,
    ) -> RolloutResult:
        """Prefill once, then decode one token at a time until terminal or cap."""
        from .tokens_bridge import rows_to_input

        argmax_fn = argmax_fn or argmax_rows
        t0 = time.time()
        prefill_t0 = time.time()
        # Cache holds prefill + at most (max_positions - 1) decoded rows.
        max_cache_len = len(prefill_ids) + max_positions - 1
        out, past, n_passes = self.run_prefill(
            prefill_ids,
            max_cache_len=max_cache_len,
            chunk_size=prefill_chunk_size,
            label="pure_ar",
            progress_every=progress_every,
        )
        cur = argmax_fn(out[-1:])[0]
        emitted = [cur]
        if progress_every:
            print(
                f"[pure_ar] prefill done in {time.time() - prefill_t0:.1f}s "
                f"seed_row={cur} cache_len={past.length}",
                flush=True,
            )
        seq_pos = len(prefill_ids)
        last_timed_print = time.time()
        while cur != terminal_row and len(emitted) < max_positions:
            step_t0 = time.time()
            out, past = self.step(rows_to_input([cur]), past, past_len=seq_pos)
            n_passes += 1
            seq_pos += 1
            cur = argmax_fn(out[-1:])[0]
            emitted.append(cur)
            step_dt = time.time() - step_t0
            if progress_every and (
                len(emitted) % progress_every == 0
                or step_dt >= 5.0
                or time.time() - last_timed_print >= 30.0
            ):
                last_timed_print = time.time()
                print(
                    f"[pure_ar] {len(emitted)} tokens  ({time.time() - t0:.1f}s, "
                    f"{(time.time() - t0) / len(emitted) * 1000:.0f} ms/tok, "
                    f"last_forward={step_dt:.1f}s, cache_len={past.length})",
                    flush=True,
                )
        return RolloutResult(
            emitted_rows=emitted,
            stopped="terminal" if cur == terminal_row else "cap",
            n_forward_passes=n_passes,
            seconds=time.time() - t0,
        )

    def spec_decode_rollout(
        self,
        prefill_ids: list[int],
        drafter: Drafter,
        max_positions: int,
        terminal_row: int,
        *,
        draft_window: int = 8,
        enable_reuse: bool = True,
        progress_every: int = 0,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        argmax_fn: Callable[[torch.Tensor], list[int]] | None = None,
        sandbox_token_to_row_fn: Callable[[Any], int] | None = None,
        row_to_sandbox_token_fn: Callable[[int], Any] | None = None,
    ) -> tuple[RolloutResult, dict[str, Any]]:
        """Generate the rollout with speculative decoding. Returns (result, stats)."""
        from .tokens_bridge import (
            row_to_sandbox_token,
            rows_to_input,
            sandbox_token_to_row,
        )

        argmax_fn = argmax_fn or argmax_rows
        sandbox_token_to_row_fn = sandbox_token_to_row_fn or sandbox_token_to_row
        row_to_sandbox_token_fn = row_to_sandbox_token_fn or row_to_sandbox_token
        stats = _new_spec_stats()
        t0 = time.time()

        prefill_t0 = time.time()
        # Cache holds prefill + (max_positions - 1) committed rows; a speculative
        # batch transiently writes up to draft_window extra rows past cache_len
        # before committing fewer, so reserve that headroom too.
        max_cache_len = len(prefill_ids) + max_positions - 1 + max(0, draft_window)
        out, past, prefill_passes = self.run_prefill(
            prefill_ids,
            max_cache_len=max_cache_len,
            chunk_size=prefill_chunk_size,
            label="spec_decode",
            progress_every=progress_every,
        )
        stats["forward_passes"] += prefill_passes
        seed_row = argmax_fn(out[-1:])[0]
        emitted = [seed_row]
        if progress_every:
            print(
                f"[spec_decode] prefill done in {time.time() - prefill_t0:.1f}s "
                f"seed_row={seed_row} cache_len={past.length}",
                flush=True,
            )

        first = drafter.next_draft()
        if first is not None and sandbox_token_to_row_fn(first) == seed_row:
            drafter.consume(first)

        reuse_buffer: list = []
        reuse_health = _REUSE_HEALTH_INIT
        reuse_probe = 0
        drafter_done = drafter.next_draft() is None
        last_print = 1
        last_timed_print = time.time()
        stopped = "cap"

        while True:
            next_input_row = emitted[-1]
            if next_input_row == terminal_row:
                stopped = "terminal"
                break
            if len(emitted) >= max_positions:
                stopped = "cap"
                break

            remaining_capacity = max_positions - len(emitted)
            step_cache_len = past.length
            step_t0 = time.time()
            if progress_every and stats["forward_passes"] <= 3:
                print(
                    f"[spec_decode] forward start pass={stats['forward_passes'] + 1} "
                    f"emitted={len(emitted)} cache_len={step_cache_len} "
                    f"remaining={remaining_capacity}",
                    flush=True,
                )
            if (
                not drafter_done
                and remaining_capacity >= 2
                and (reuse_buffer or drafter.next_draft() is not None)
            ):
                past, reuse_buffer, reuse_health, reuse_probe = _spec_step(
                    self,
                    past,
                    next_input_row,
                    emitted,
                    drafter,
                    min(draft_window, remaining_capacity - 1),
                    terminal_row,
                    stats,
                    reuse_buffer,
                    reuse_health,
                    reuse_probe,
                    enable_reuse,
                    argmax_fn=argmax_fn,
                    sandbox_token_to_row_fn=sandbox_token_to_row_fn,
                    row_to_sandbox_token_fn=row_to_sandbox_token_fn,
                )
                if not reuse_buffer and drafter.next_draft() is None:
                    drafter_done = True
            else:
                reuse_buffer = []
                past = _single_step(
                    self, past, next_input_row, emitted, stats, argmax_fn=argmax_fn
                )
                if not drafter_done:
                    drafter.consume(row_to_sandbox_token_fn(emitted[-1]))
                    if drafter.next_draft() is None:
                        drafter_done = True

            step_dt = time.time() - step_t0
            should_print = progress_every and (
                len(emitted) - last_print >= progress_every
                or step_dt >= 5.0
                or time.time() - last_timed_print >= 30.0
            )
            if should_print:
                last_print = len(emitted)
                last_timed_print = time.time()
                print(
                    f"[spec_decode] {len(emitted)} tokens, {stats['forward_passes']} "
                    f"forward passes ({time.time() - t0:.1f}s), "
                    f"last_forward={step_dt:.1f}s, cache_len={past.length}, "
                    f"accept={stats['accepted_drafts'] / max(1, len(emitted)):.0%} per-tok "
                    f"(drafts {stats['accepted_drafts']}/{stats['attempted_drafts']})",
                    flush=True,
                )

        return (
            RolloutResult(
                emitted_rows=emitted,
                stopped=stopped,
                n_forward_passes=stats["forward_passes"],
                seconds=time.time() - t0,
            ),
            stats,
        )


_REUSE_HEALTH_INIT = 4
_REUSE_HEALTH_CAP = 8
_REUSE_HEALTH_PENALTY = 2
_REUSE_PROBE_INTERVAL = 64


def _new_spec_stats() -> dict[str, Any]:
    return {
        "batches": 0,
        "forward_passes": 0,
        "attempted_drafts": 0,
        "committed_rows": 0,
        "accepted_drafts": 0,
        "full_accepts": 0,
        "mispredicts": 0,
        "reject_at_0": 0,
        "terminal_truncations": 0,
        "fallback_single_steps": 0,
        "reused_offered": 0,
        "reused_accepted": 0,
        "accept_histogram": {},
    }


def _single_step(
    compiled: TokenRuntime,
    past: KVCache,
    input_row: int,
    emitted: list[int],
    stats: dict[str, Any],
    *,
    argmax_fn: Callable[[torch.Tensor], list[int]],
) -> KVCache:
    from .tokens_bridge import rows_to_input

    out, new_past = compiled.step(rows_to_input([input_row]), past)
    stats["forward_passes"] += 1
    stats["fallback_single_steps"] += 1
    emitted.append(argmax_fn(out[-1:])[0])
    return new_past


def _spec_step(
    compiled: TokenRuntime,
    past: KVCache,
    next_input_row: int,
    emitted: list[int],
    drafter: Drafter,
    max_drafts: int,
    terminal_row: int,
    stats: dict[str, Any],
    reuse_buffer: list,
    reuse_health: int,
    reuse_probe: int,
    enable_reuse: bool,
    *,
    argmax_fn: Callable[[torch.Tensor], list[int]],
    sandbox_token_to_row_fn: Callable[[Any], int],
    row_to_sandbox_token_fn: Callable[[int], Any],
):
    """One batched draft-verify step. Returns (new_past, reuse_tail, health, probe)."""
    from .tokens_bridge import rows_to_input

    cache_len = past.length
    snap = drafter.snapshot()

    drafts: list = []
    if enable_reuse and reuse_health > 0:
        for r in reuse_buffer:
            if len(drafts) >= max_drafts:
                break
            drafts.append(r)
            drafter.consume(r)
    n_reused = len(drafts)
    while len(drafts) < max_drafts:
        d = drafter.next_draft()
        if d is None:
            break
        drafts.append(d)
        drafter.consume(d)

    n_drafts = len(drafts)
    if n_drafts == 0:
        drafter.rollback(snap)
        new_past = _single_step(
            compiled, past, next_input_row, emitted, stats, argmax_fn=argmax_fn
        )
        drafter.consume(row_to_sandbox_token_fn(emitted[-1]))
        return new_past, [], reuse_health, reuse_probe

    draft_rows = [sandbox_token_to_row_fn(d) for d in drafts]
    batch_rows = [next_input_row] + draft_rows
    out, new_past = compiled.step(rows_to_input(batch_rows), past)
    stats["forward_passes"] += 1
    pred = argmax_fn(out)

    accept = n_drafts
    for i in range(n_drafts):
        if pred[i] != draft_rows[i]:
            accept = i
            break
    ncommit = (n_drafts + 1) if accept == n_drafts else (accept + 1)

    terminal_hit = False
    for i in range(ncommit):
        emission_row = draft_rows[i] if i < accept else pred[i]
        if emission_row == terminal_row:
            ncommit = i + 1
            terminal_hit = True
            stats["terminal_truncations"] += 1
            break

    new_past = commit(new_past, cache_len + ncommit)

    stats["batches"] += 1
    stats["attempted_drafts"] += n_drafts
    stats["committed_rows"] += ncommit
    stats["accepted_drafts"] += min(accept, ncommit)
    stats["reused_offered"] += n_reused
    stats["reused_accepted"] += min(accept, n_reused)
    if accept == n_drafts:
        stats["full_accepts"] += 1
    else:
        stats["mispredicts"] += 1
        if accept == 0:
            stats["reject_at_0"] += 1
    stats["accept_histogram"][accept] = stats["accept_histogram"].get(accept, 0) + 1

    drafter.rollback(snap)
    for i in range(ncommit):
        emission_row = draft_rows[i] if i < accept else pred[i]
        emission_tok = drafts[i] if i < accept else row_to_sandbox_token_fn(pred[i])
        drafter.consume(emission_tok)
        emitted.append(emission_row)

    if n_reused > 0:
        if accept > 0:
            reuse_health = min(_REUSE_HEALTH_CAP, reuse_health + 1)
        else:
            reuse_health = max(0, reuse_health - _REUSE_HEALTH_PENALTY)
    if reuse_health == 0:
        reuse_probe += 1
        if reuse_probe >= _REUSE_PROBE_INTERVAL:
            reuse_health = 1
            reuse_probe = 0
    else:
        reuse_probe = 0

    if terminal_hit or not enable_reuse or reuse_health == 0:
        return new_past, [], reuse_health, reuse_probe
    reuse_tail = [
        row_to_sandbox_token_fn(pred[i]) for i in range(ncommit, n_drafts + 1)
    ]
    return new_past, reuse_tail, reuse_health, reuse_probe
