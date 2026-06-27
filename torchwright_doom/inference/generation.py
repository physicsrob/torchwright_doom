"""Generation loops over a token-stepped runtime (the GenerationMixin analog).

:class:`TokenRuntime` is the base every generation loop runs on: the
abstract trio (``empty_past`` / ``step`` / ``max_safe_prefill_chunk``) is
exactly the surface the concrete loop methods call, and the loops —
chunked prefill and pure autoregression — are ordinary methods, the way
``model.generate()`` is in every mainstream stack. Production is
:class:`~torchwright_doom.inference.hf_runtime.HfTokenRuntime` (a native
HuggingFace causal LM); the render logic tests subclass with in-memory
transition-table models.

``tokens_bridge`` (and through it the screen-env-dependent vocab) is
imported lazily inside the loop bodies, so importing this module never
builds ``W_EMBED`` — ``apply_screen_env`` must precede the first rollout,
not this import.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar

import torch

from .kv_cache import commit


class _Cache(Protocol):
    """The cache surface the generation loops touch: the committed position
    count."""

    length: int


# The cache type a concrete runtime owns: the unbounded
# :class:`~torchwright_doom.inference.kv_cache.KVCache`, the production HF
# runtime's duck-typed ``HfCache``, or a test runtime's own. The loops only
# ever read ``.length`` and thread it through ``step``/``commit``.
CacheT = TypeVar("CacheT", bound=_Cache)

# 1024-row prefill chunks bound the per-layer transient activations of a
# single forward pass.  Chunking is semantically identical to a single pass;
# this is a memory knob, not an algorithm change.  Direct-API default only:
# render jobs resolve their chunk size from the config's
# ``run.prefill_chunk_size``.
DEFAULT_PREFILL_CHUNK_SIZE = 1024


@dataclass
class RolloutResult:
    emitted_rows: list[int]
    stopped: str  # "terminal" | "cap"
    n_forward_passes: int
    seconds: float


def argmax_rows(outputs: torch.Tensor) -> list[int]:
    """Argmax-decode compiled-step LOGITS to token row ids.

    The artifact owns the unembed — the production
    :class:`~torchwright_doom.inference.hf_runtime.HfTokenRuntime` returns
    logits.  (The in-process ``compile_headless`` gate decodes its
    embedding-width outputs with its own ``@ W_EMBED.T`` helper:
    ``tests/scene/test_forward_ar_rollout.py``.)
    """
    return outputs.detach().argmax(dim=-1).cpu().tolist()


class TokenRuntime(ABC, Generic[CacheT]):
    """A token-stepped transformer over a host-owned cache (``CacheT``).

    The abstract trio below is exactly the surface the concrete generation
    loops call; production is
    :class:`~torchwright_doom.inference.hf_runtime.HfTokenRuntime` — the one
    real runtime — and the render logic tests substitute in-memory
    transition-table models by subclassing, so a missing or drifted method is
    a definition-time error, not a mid-rollout AttributeError.

    Guardrails (keep it this way): the abstract trio is frozen — never add
    an abstract member only a test needs — and the base carries no state
    (no ``__init__``); loop helpers stay module-private functions.
    """

    @abstractmethod
    def empty_past(self, max_len: int) -> CacheT:
        """Allocate (or zero-reset) the run's owned cache for ``max_len`` rows."""

    @abstractmethod
    def step(
        self,
        inputs: torch.Tensor,
        cache: CacheT,
        past_len: int | None = None,
    ) -> tuple[torch.Tensor, CacheT]:
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
    ) -> tuple[torch.Tensor, CacheT, int]:
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
        out = None
        offset = 0
        passes = 0
        for chunk_idx in range(n_chunks):
            chunk = prefill_ids[offset : offset + chunk_size]
            t0 = time.time()
            out, past = self.step(rows_to_input(chunk), past, past_len=offset)
            offset += len(chunk)
            # Commit each fully-decoded chunk before the next step.
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
