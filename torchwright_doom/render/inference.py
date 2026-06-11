"""ONNX token inference runtime and autoregressive rollout loops.

This is the runtime-facing path for render inference: ONNX session ownership,
GPU I/O binding, prefill chunking, pure AR, and speculative decode. Compile/build
code belongs in ``compiled_model.py``; CLI/artifact orchestration belongs in
``cli.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from ..embedding import W_EMBED
from .tokens_bridge import row_to_sandbox_token, rows_to_input, sandbox_token_to_row

# 1024-row prefill chunks bound the per-layer (n_heads, chunk, S) logits
# transient under the static-S cache (unchunked prefill at n=3613, S=12288
# peaks ~45 GB on the widest d=4096 layer).  Chunking is semantically
# identical to a single pass; this is a memory knob, not an algorithm change.
DEFAULT_PREFILL_CHUNK_SIZE = 1024

_W_EMBED_T_BY_DEVICE: dict[str, torch.Tensor] = {}


@dataclass
class RolloutResult:
    emitted_rows: list[int]
    stopped: str  # "terminal" | "cap"
    n_forward_passes: int
    seconds: float


@dataclass
class KVCache:
    """Runtime-owned static KV cache for the sequence-major ONNX protocol.

    One preallocated FULL-S buffer per layer per side, shape
    ``(cache_stride, n_heads, d_head)``, **zero-initialized** — the graph's
    causal mask gives slots at positions > cache_position softmax weight
    exactly 0.0, and ``0 * NaN`` from a ``torch.empty`` tail would still be
    NaN.  The whole buffer is bound as ``past_K_i`` every step (static
    shape + stable address: the CUDA-graph replay requirements); the
    committed prefix is ``k[i][:length]`` and the graph re-injects the new
    rows in-graph via ScatterND.  The runtime persists each step's
    ``delta_K_i`` output into ``k[i][length:length+n_new]`` after the run.
    A speculative reject just lowers ``length`` — the rows past it are
    masked (exactly-zero weight) and overwritten by a later write.
    """

    k: list[torch.Tensor]  # each (cache_stride, n_heads, d_head)
    v: list[torch.Tensor]
    length: int
    max_len: int  # == cache_stride (the static allocation)
    # --- Windowed-cache state (permanent/expiring slot policy); all of
    # these stay at their defaults on the unbounded protocol above. ---
    # window = C, the committed slot count of a cache_window model: the
    # buffers are (C + staging, nh, d_head) and ``length`` keeps counting
    # absolute committed POSITIONS (it exceeds C once recycling starts).
    window: int | None = None
    # Allocated staging-tail width = the widest pass this cache can bind
    # (the graph scatters new rows at slots [C, C + n_new)).
    staging: int = 0
    # THE PLACEMENT POLICY: every committed row is either PERMANENT (it
    # stays resident for the whole run) or EXPIRING (its slot may be
    # recycled), decided purely by the row's token type — see
    # ``OnnxTokenRuntime._expiring_rows`` (default: only PIXEL rows
    # expire; pixels publish no channels and are only ever read at
    # offset <= 3, so evicting old ones is safe by construction).
    # Slots fill strictly sequentially (the graph's ``j < base``
    # writtenness mask requires first-fill in slot order); once all C
    # slots are written, new rows recycle the slots of expired rows,
    # cursor order, skipping permanent ones.  Prefill needs no special
    # case — scene tokens are permanent because of their types.
    slot_expiring: list | None = None  # per-slot tag of the current occupant
    write_head: int = 0  # slots written so far (== fill progress, <= C)
    recycle_cursor: int = 0  # next candidate slot for recycling
    n_permanent: int = 0  # capacity guard: permanent rows can never exceed C
    # A multi-row ROLLOUT batch is speculative: some of its rows may be
    # rejected, and a rejected row must never consume a slot (it would
    # evict a live row for a row that never becomes part of the stream).
    # So the step stashes (base, n_new, dk_rows, dv_rows, expiring) here
    # and ``_commit`` flushes exactly the accepted prefix.  The stashed
    # tensors are VIEWS of the persistent binding buffers — safe because
    # _commit always runs before the next step (step() raises on an
    # unflushed pending).  Prefill chunks ride the same path:
    # ``run_prefill`` commits after every chunk.
    pending: tuple | None = None


_RECYCLE_POOL_WARN = 256


def _alloc_slot(cache: KVCache, expiring: bool) -> int:
    """Allocate the committed slot for one new row.

    Sequential while the buffer is filling (the writtenness-mask
    contract); afterwards, recycle the next expired slot in cursor
    order.  A full revolution without an expired slot means the cache
    is saturated with permanent rows — fail loud, that's a
    "recompile with a larger cache_window" event.
    """
    C = cache.window
    assert C is not None and cache.slot_expiring is not None
    if cache.write_head < C:
        slot = cache.write_head
        cache.write_head += 1
    else:
        cur = cache.recycle_cursor
        for _ in range(C):
            if cache.slot_expiring[cur % C]:
                break
            cur += 1
        else:
            raise RuntimeError(
                f"windowed cache saturated: all {C} slots hold permanent "
                f"rows ({cache.n_permanent} permanent committed); recompile "
                f"with a larger model.cache_window or expire more token types"
            )
        slot = cur % C
        cache.recycle_cursor = cur + 1
    if not expiring:
        cache.n_permanent += 1
        free = C - cache.n_permanent
        if free == _RECYCLE_POOL_WARN:
            print(
                f"[kvcache] WARNING: recycle pool down to {free} slots "
                f"({cache.n_permanent}/{C} permanent) — nearing saturation",
                flush=True,
            )
    cache.slot_expiring[slot] = expiring
    return slot


def _alloc_runs(cache: KVCache, expiring: list) -> list[list[int]]:
    """Allocate slots for a batch of rows (in position order) and compress
    the assignments into contiguous [slot_start, row_start, count] runs —
    sequential fill is one run; recycling fragments only as much as the
    expired slots do."""
    runs: list[list[int]] = []
    for row, flag in enumerate(expiring):
        slot = _alloc_slot(cache, bool(flag))
        if (
            runs
            and runs[-1][0] + runs[-1][2] == slot
            and runs[-1][1] + runs[-1][2] == row
        ):
            runs[-1][2] += 1
        else:
            runs.append([slot, row, 1])
    return runs


def _write_runs(cache: KVCache, runs, dk_rows, dv_rows) -> None:
    n_layers = len(cache.k)
    for slot, row, count in runs:
        for i in range(n_layers):
            cache.k[i][slot : slot + count] = dk_rows[i][row : row + count]
            cache.v[i][slot : slot + count] = dv_rows[i][row : row + count]


def _persist_rows(
    cache: KVCache, base: int, n_new: int, dk_rows, dv_rows, expiring=None
) -> None:
    """Write a pass's delta rows into the owned cache.

    ``dk_rows[i]`` / ``dv_rows[i]`` are the layer-i ``(n_new, nh, d_head)``
    delta tensors (views of the binding buffers are fine — see
    ``KVCache.pending``).  Unbounded protocol: slots == positions, exactly
    the historical write.  Windowed protocol: ``expiring`` carries the
    per-row policy tag; single-row decodes commit unconditionally and
    write through; every multi-row batch (speculative verify batches AND
    prefill chunks) is stashed on ``cache.pending`` and flushed by
    ``_commit`` — rejected rows never consume a slot.
    """
    n_layers = len(cache.k)
    if cache.window is None:
        for i in range(n_layers):
            cache.k[i][base : base + n_new] = dk_rows[i]
            cache.v[i][base : base + n_new] = dv_rows[i]
        return
    if expiring is None or len(expiring) != n_new:
        raise RuntimeError(
            "windowed persist requires a per-row expiring tag for every row"
        )
    if n_new == 1:
        _write_runs(cache, _alloc_runs(cache, list(expiring)), dk_rows, dv_rows)
        return
    if cache.pending is not None:
        raise RuntimeError(
            "windowed KVCache already holds an uncommitted batch; _commit "
            "must flush it before the next multi-row step"
        )
    cache.pending = (base, n_new, list(dk_rows), list(dv_rows), list(expiring))


def _flush_pending(cache: KVCache, target: int) -> None:
    """Flush the accepted prefix of a stashed batch: rows at positions
    [pending_base, target) get slots and persist; the rejected tail is
    dropped (its positions are re-drafted by the next pass)."""
    pending = cache.pending
    if pending is None:
        return
    cache.pending = None
    pending_base, pending_n, dk_rows, dv_rows, expiring = pending
    keep = target - pending_base
    assert 0 <= keep <= pending_n, (
        f"commit target {target} outside the pending batch "
        f"[{pending_base}, {pending_base + pending_n}]"
    )
    if keep == 0:
        return
    _write_runs(cache, _alloc_runs(cache, expiring[:keep]), dk_rows, dv_rows)


def _resolve_buckets(
    attention_buckets: list[int] | None, cache_stride: int
) -> list[int]:
    """Normalize the attention-window bucket table.

    Default (None): quarters of the stride — e.g. S=65536 ->
    [16384, 32768, 49152, 65536].  Invariants enforced: sorted, unique,
    each in [1, cache_stride], and the LAST bucket equals cache_stride
    (the always-covering fallback; it is also what makes the degenerate
    table [cache_stride] reproduce the pre-bucketing behavior, ids
    "1"/"2", byte-for-byte).
    """
    if attention_buckets is None:
        buckets = sorted({max(1, (cache_stride * q) // 4) for q in (1, 2, 3, 4)})
    else:
        buckets = sorted({int(b) for b in attention_buckets})
        if not buckets:
            raise ValueError("attention_buckets must be non-empty")
        if buckets[0] < 1 or buckets[-1] > cache_stride:
            raise ValueError(
                f"attention_buckets {buckets} outside [1, cache_stride="
                f"{cache_stride}]"
            )
        if buckets[-1] != cache_stride:
            buckets.append(cache_stride)
    return buckets


def _cache_len(past) -> int:
    """Logical committed length across the two cache representations the
    generic rollouts thread: the owned :class:`KVCache` (production ONNX
    runtime) and the head-major ``(past_K_tuple, past_V_tuple)`` of the
    in-process ``CompiledHeadless`` / test mocks."""
    if isinstance(past, KVCache):
        return past.length
    return int(past[0][0].shape[1])


def _commit(past, target: int):
    """Set the committed length to ``target``.

    Owned cache: lower the logical length in place (no copy), and — on a
    windowed cache — flush the accepted prefix of the pending batch into
    freshly-allocated slots (see :func:`_persist_rows`; rejected rows are
    dropped, never written).  Head-major tuple: trim to ``target`` as the
    in-process path requires.
    """
    if isinstance(past, KVCache):
        _flush_pending(past, target)
        past.length = target
        return past
    past_k, past_v = past
    return (
        tuple(_trim_cache_tensor(k, target) for k in past_k),
        tuple(_trim_cache_tensor(v, target) for v in past_v),
    )


class OnnxTokenRuntime:
    """Token-I/O ONNX wrapper with CPU fallback and CUDA I/O binding."""

    def __init__(
        self,
        onnx_path: str | Path,
        providers=None,
        *,
        enable_profiling: bool = False,
        profile_dir: str | Path | None = None,
        attention_buckets: list[int] | None = None,
        expiring_types: tuple[str, ...] = ("pixel",),
    ) -> None:
        import os

        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        session_providers = providers or _default_ort_providers(ort)
        so = ort.SessionOptions()
        if any(
            isinstance(p, tuple) and p[1].get("enable_cuda_graph")
            for p in session_providers
        ):
            # Memory-pattern pre-allocation requests one giant pattern buffer
            # (7+ GiB at d=3072) and, when it fails on a tight GPU, the arena
            # falls back to on-demand cudaMalloc — which, inside stream
            # capture, invalidates the CUDA graph
            # (cudaErrorStreamCaptureInvalidated).  The BFC arena's on-demand
            # reuse peak is far smaller and is grown by ORT's internal
            # warm-up runs BEFORE capture, so patterns off is what makes
            # capture allocation-free.
            so.enable_mem_pattern = False
            print(
                "[onnxruntime] enable_cuda_graph on: memory-pattern "
                "optimization disabled (capture must not allocate)",
                flush=True,
            )
        if os.environ.get("TWDOOM_NO_OPT"):  # DIAGNOSTIC: disable ORT graph opt
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            print("[onnxruntime] DIAGNOSTIC: graph optimization DISABLED", flush=True)
        self._profiling = bool(enable_profiling)
        if self._profiling:
            # Opt-in measurement path (the CUDA-graph work, phase 1).  Turn on
            # ORT's per-node profiler (writes a JSON trace at end_profiling) and
            # drop the session log to INFO (severity 1) so the "N Memcpy nodes
            # added ... unable to run CUDA graph" placement messages and the
            # CPU/CUDA node assignments are surfaced to stdout.  Profiling adds
            # per-op timing overhead, so the absolute numbers are inflated; the
            # relative split and the Memcpy node names are what we read off it.
            so.enable_profiling = True
            if profile_dir is not None:
                prefix = str(Path(profile_dir) / "ort_profile")
                so.profile_file_prefix = prefix
            so.log_severity_level = 1
            try:
                ort.set_default_logger_severity(1)
            except Exception:
                pass
            print(
                "[onnxruntime] PROFILING enabled (log_severity_level=1); "
                "profile JSON written at end_profiling()",
                flush=True,
            )
        # No silent CPU retry here: an intended-GPU run degrading to CPU is
        # a multi-day hang, not a fallback (see the guard below); intentional
        # CPU inference is requested explicitly via TWDOOM_FORCE_CPU.
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=so,
            providers=session_providers,
        )
        print(
            f"[onnxruntime] version={ort.__version__} "
            f"active providers={self._session.get_providers()}",
            flush=True,
        )
        # Fail LOUD if a CUDA session was requested but didn't materialize.
        # ORT's InferenceSession internally swallows an EP init failure and
        # retries CPU-only ("EP Error ... Falling back to CPUExecutionProvider")
        # — observed on a Modal B200 whose host was broken (CUDA error 802
        # "system not yet initialized" from a fabric-manager race).  A d8192
        # render on CPU is a multi-day hang on a billed GPU container, not a
        # fallback; intentional CPU runs say so via TWDOOM_FORCE_CPU.
        if (
            any(
                (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
                for p in session_providers
            )
            and "CUDAExecutionProvider" not in self._session.get_providers()
            and not os.environ.get("TWDOOM_FORCE_CPU")
        ):
            raise RuntimeError(
                "CUDAExecutionProvider was requested but the session is "
                "CPU-only (ORT EP-init fallback — broken GPU host?). "
                "Retry on a fresh worker, or set TWDOOM_FORCE_CPU=1 if CPU "
                "inference is really intended."
            )
        self._use_cuda_io = (
            "CUDAExecutionProvider" in self._session.get_providers()
            and torch.cuda.is_available()
        )
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if self._use_cuda_io
            else torch.device("cpu")
        )

        # The render meta sidecar (model.meta.json) is loaded BEFORE cache
        # topology discovery: with the symbolic cache_slots input dim the
        # full stride S is only knowable from meta["model"]["cache_stride"].
        meta_path = self.onnx_path.with_suffix(".meta.json")
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        inputs = {inp.name: inp for inp in self._session.get_inputs()}
        self._n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
        if self._n_layers <= 0:
            raise ValueError(f"{self.onnx_path}: no past_K_* inputs")
        # past_K_i is sequence-major (cache_slots, nh, d_head) with a SYMBOLIC
        # slot dim — the bound prefix length S_eff (stride bucketing).  The
        # full stride S comes from the meta sidecar; old static-dim artifacts
        # still resolve from the dim itself (and cannot bucket — their baked
        # mask is full-width).
        from torchwright.compiler.onnx_load import discover_cache_stride

        stride_dim = inputs["past_K_0"].shape[0]
        self._symbolic_cache_dim = not isinstance(stride_dim, int)
        # Windowed-cache protocol discriminator: the committed cache is a
        # fixed C-slot host-managed window, every pass binds exactly
        # (C + n_new) rows, and the graph's mask is writtenness
        # ("committed slot j visible iff j < cache_position[0]") instead
        # of slot==position causality.  Slot PLACEMENT is entirely
        # host-side: rows whose token type is in ``expiring_types`` may
        # have their slots recycled once the buffer fills; every other
        # row is permanent.  Same sidecar nesting convention as
        # cache_stride.  Read FIRST: in windowed mode the window IS the
        # committed slot count — the render meta's model.cache_stride
        # field is the YAML knob the windowed compile IGNORED
        # (ModelConfig documents the two as exclusive), so it must not
        # feed the stride discovery below.
        sidecar_window = (self.metadata.get("model") or {}).get(
            "cache_window"
        ) or self.metadata.get("cache_window")
        self._cache_window = int(sidecar_window) if sidecar_window else None
        # Stride source: the render meta nests it under "model" (asdict of
        # ModelConfig); a bare torchwright token export carries it top-level.
        sidecar_stride = (
            self._cache_window
            if self._cache_window is not None
            else (self.metadata.get("model") or {}).get("cache_stride")
            or self.metadata.get("cache_stride")
        )
        self._cache_stride = discover_cache_stride(
            inputs, sidecar_stride, self.onnx_path
        )
        self._per_layer_n_heads = [
            int(inputs[f"past_K_{i}"].shape[1]) for i in range(self._n_layers)
        ]
        self._d_head = int(inputs["past_K_0"].shape[2])
        # Position-space cap (the pos-encoding table length): in windowed
        # mode positions outrun the slot count, so the demand guard checks
        # against this instead of the stride.  Only the render meta carries
        # it; a bare torchwright sidecar leaves it None (no upper guard).
        _msl = (self.metadata.get("model") or {}).get("max_seq_len")
        self._max_seq_len = int(_msl) if _msl else None
        if self._cache_window is not None:
            assert self._cache_stride == self._cache_window, (
                f"windowed sidecar disagrees: cache_stride "
                f"{self._cache_stride} != cache_window {self._cache_window}"
            )
            if not self._symbolic_cache_dim:
                raise RuntimeError(
                    "windowed-cache artifact with a static past_K dim — "
                    "the runtime needs the symbolic cache_slots dim to bind "
                    "(C + n_new) per pass width; recompile"
                )
        # Per-row expiry policy: token row id -> True iff the row's type is
        # in expiring_types (default: only "pixel" — pixel rows publish no
        # channels and are only read at offset <= 3, so recycling old ones
        # is safe by construction; everything else stays resident).  Built
        # from the render meta's row_to_token table; without it (a bare
        # torchwright sidecar) every row is treated as permanent — safe,
        # the cache just has to fit everything.
        self._expiring_rows: list[bool] = []
        self._expiring_types = tuple(expiring_types or ())
        if self._cache_window is not None:
            row_to_token = self.metadata.get("row_to_token")
            if row_to_token:
                expiring_set = {t.lower() for t in self._expiring_types}
                self._expiring_rows = [
                    str(entry.get("type", "")).lower() in expiring_set
                    for entry in row_to_token
                ]
            else:
                print(
                    "[kvcache] WARNING: windowed cache with no row_to_token "
                    "metadata — treating EVERY row as permanent (no recycling)",
                    flush=True,
                )
        # Widest staging tail this runtime will bind: prefill chunks (the
        # widest passes) plus the spec-verify bucket.  Sizes the windowed
        # allocation; unused on the unbounded protocol.
        self._batch_bucket_width = 9
        self._staging_max = max(self._batch_bucket_width, DEFAULT_PREFILL_CHUNK_SIZE)
        # Attention-window buckets (sorted, last == cache_stride): each pass
        # binds the smallest bucket covering committed length + pass width,
        # so early-frame steps pay small-S_eff attention while the cache
        # keeps its full capacity.  A runtime knob — the symbolic-dim graph
        # serves any table without recompiling.  Old static-dim artifacts
        # are forced to the degenerate single bucket.  The windowed
        # protocol has nothing to bucket — its attention width is the
        # CONSTANT C + pass width by construction.
        if self._cache_window is not None:
            if attention_buckets:
                print(
                    "[kvcache] WARNING: windowed cache has a constant "
                    "attention width; ignoring attention_buckets",
                    flush=True,
                )
            self._buckets = []
        else:
            self._buckets = _resolve_buckets(
                attention_buckets if self._symbolic_cache_dim else None,
                self._cache_stride,
            )
            if not self._symbolic_cache_dim and attention_buckets:
                print(
                    "[kvcache] WARNING: static-dim artifact cannot stride-bucket; "
                    "ignoring attention_buckets (recompile to bucket)",
                    flush=True,
                )
        _total_heads = sum(self._per_layer_n_heads)
        _bytes_per_tok = _total_heads * self._d_head * 2 * 4  # K+V, float32
        _alloc_rows = (
            self._cache_window + self._staging_max
            if self._cache_window is not None
            else self._cache_stride
        )
        print(
            f"[kvcache] layers={self._n_layers} total_heads={_total_heads} "
            f"d_head={self._d_head} cache_stride={self._cache_stride} "
            + (
                f"WINDOWED window={self._cache_window} "
                f"staging={self._staging_max} "
                f"expiring_types={list(self._expiring_types)} "
                f"({sum(self._expiring_rows)} of {len(self._expiring_rows)} "
                f"vocab rows expire) "
                if self._cache_window is not None
                else f"buckets={self._buckets} "
            )
            + f"per_layer_heads={self._per_layer_n_heads} "
            f"bytes/token={_bytes_per_tok} ({_bytes_per_tok / 1e6:.3f} MB); "
            f"static cache total={_alloc_rows * _bytes_per_tok / 1e9:.2f} GB",
            flush=True,
        )
        # Persistent bindings, one per (cache, S_eff, width) — width 1 is the
        # pure-decode bucket, width _batch_bucket_width the padded spec-verify
        # bucket (variable draft batches, n_new in 2..draft_window+1, are
        # PADDED to it so one captured graph serves them all; pad rows
        # scatter into slots beyond the committed length — masked to
        # exactly-zero weight for every real query, outputs sliced off,
        # deltas not persisted.  Must be >= draft_window+1 or wide batches
        # fall back to the uncaptured dynamic path).  Built lazily; each
        # (S_eff, width) pair captures its own CUDA graph on first use.
        # (_batch_bucket_width is defined above, before the staging sizing.)
        self._bindings: dict[tuple[int, int, int], dict] = {}
        # gpu_graph_id -> (S_eff, width) registry: a captured id must never
        # run with a second shape (ORT would replay the baked one — silent
        # wrong results); ORT does NOT cross-check this, the runtime does.
        self._graph_id_shapes: dict[str, tuple[int, int]] = {}
        # Session-lifetime static cache, allocated on first empty_past and
        # zero-reset (never reallocated) on later calls — captured CUDA
        # graphs bake the buffer addresses.
        self._static_cache: KVCache | None = None
        # Outputs are the per-layer KV *deltas* (new rows only); the runtime
        # writes them into its owned cache tail (no full-cache output).
        self._out_names = ["logits"]
        for i in range(self._n_layers):
            self._out_names += [f"delta_K_{i}", f"delta_V_{i}"]
        logits_shape = self._session.get_outputs()[0].shape
        self._logits_width = int(
            self.metadata.get("n_vocab_rows")
            or (
                logits_shape[1]
                if len(logits_shape) > 1 and isinstance(logits_shape[1], int)
                else 0
            )
            or W_EMBED.shape[0]
        )
        self.input_names = ["token_ids"]

    def max_safe_prefill_chunk(self, planned_rows: int | None = None) -> int:
        """Largest prefill chunk whose per-layer transients stay int32-indexable.

        The widest layer materializes (n_heads, chunk, S_eff) attention
        logits; past 2^31-1 elements, CUDA kernels that index with int32
        write through overflowed offsets — observed as an Xid 31 MMU fault
        on a B200 at S=65536, chunk=1024, nh=128 (8.6e9 elements).  Decode
        (n_new=1) is far below the limit at any realistic S.

        Bucket-aware: ``planned_rows`` (the whole prefill length) picks the
        prefix bucket the prefill will bind, and the clamp scales with that
        S_eff — e.g. 255 rows at S_eff=65536 but 1023 at 16384 (equal-cost
        chunks, ~4x fewer of them).  None = the conservative full stride.

        Windowed mode: S_eff = C + chunk (the chunk sizes its own
        binding), so solve widest * chunk * (C + chunk) <= 2^31 - 1 for
        chunk, then cap at the allocated staging width.
        """
        widest = max(self._per_layer_n_heads)
        if self._cache_window is not None:
            c = self._cache_window
            lim = (2**31 - 1) // widest  # chunk * (C + chunk) <= lim
            chunk = int((-c + (c * c + 4 * lim) ** 0.5) // 2)
            return max(1, min(chunk, self._staging_max))
        s_eff = (
            self._bucket_for(planned_rows) or self._cache_stride
            if planned_rows is not None
            else self._cache_stride
        )
        return max(1, (2**31 - 1) // (widest * s_eff))

    def _bucket_for(self, demand: int) -> int | None:
        """Smallest attention-window bucket covering ``demand`` rows.

        ``demand`` = committed length + this pass's (padded) width — the
        bound prefix must hold every row the in-graph ScatterND writes.
        None when demand exceeds the stride (callers fall back to the
        uncaptured full-stride path; the last bucket == cache_stride, so
        this only happens past the cache's own capacity guard).
        """
        for s in self._buckets:
            if demand <= s:
                return s
        return None

    def _graph_id_for(self, s_eff: int, width: int) -> str:
        """The captured-graph annotation id for a (S_eff, width) bucket.

        Bucket index b in the sorted table -> width-1 (pure decode) gets
        str(1 + 2b), the padded spec-verify width gets str(2 + 2b); the
        degenerate table [cache_stride] reproduces the pre-bucketing ids
        {"1", "2"} exactly.  Ids 0 (ORT-internal, capture-eligible) and
        "-1" (uncaptured) are never produced.

        Windowed mode degenerates to exactly two captured shapes for the
        whole frame — (C+1, 1) and (C+W, W) — so the pre-bucketing ids
        {"1", "2"} come back; the _graph_id_shapes registry still pins
        each id to its one shape.
        """
        if self._cache_window is not None:
            return "1" if width == 1 else "2"
        b = self._buckets.index(s_eff)
        return str((1 if width == 1 else 2) + 2 * b)

    def _windowed_s_eff(self, width: int) -> int:
        """Binding width for a windowed pass: the constant C committed
        slots plus this pass's staging tail — must be EXACT (the in-graph
        mask is C + n_new wide; any other binding fails loudly at the
        mask broadcast)."""
        assert self._cache_window is not None
        if width > self._staging_max:
            raise RuntimeError(
                f"pass width {width} exceeds the allocated staging tail "
                f"{self._staging_max}; lower the prefill chunk size"
            )
        return self._cache_window + width

    def empty_past(self, max_len: int) -> KVCache:
        """Allocate the single runtime-owned static cache for a planned run.

        ``max_len`` is the run's DEMAND (size it from ``len(prefill_ids) +
        max_positions - 1`` plus any speculative-draft headroom); the actual
        allocation is the model's full ``cache_stride`` — or, on a windowed
        model, the constant ``cache_window + staging`` (positions outrun
        slots by design there, so the demand checks against the
        pos-encoding table instead).  Zero-initialized (load-bearing:
        masked slots are read with weight exactly 0.0, and
        ``0 * NaN = NaN``).
        """
        if max_len is None:
            raise ValueError(
                "OnnxTokenRuntime.empty_past requires an explicit max_len "
                "(size from len(prefill_ids) + max_positions - 1 [+ draft_window])"
            )
        if self._cache_window is not None:
            # Windowed protocol: positions outrun the slot count by design,
            # so the demand is bounded by the POSITION space (the
            # pos-encoding table), not the allocation.
            if self._max_seq_len is not None and max_len > self._max_seq_len:
                raise RuntimeError(
                    f"run demands up to {max_len} positions but the compiled "
                    f"model's pos-encoding table is {self._max_seq_len} long; "
                    f"recompile with a larger model.max_seq_len or lower "
                    f"max_positions"
                )
        elif max_len > self._cache_stride:
            raise RuntimeError(
                f"run demands up to {max_len} cache rows but the compiled "
                f"model's static cache_stride is {self._cache_stride}; "
                f"recompile with a larger model.cache_stride or lower "
                f"max_positions"
            )
        # The runtime owns ONE session-lifetime static cache (the vanilla
        # HF-StaticCache discipline), zeroed in place per run.  This is
        # capture-correctness, not just an allocation saving: the captured
        # CUDA graph bakes the cache buffer ADDRESSES at capture time, and a
        # replay ignores any rebinding — a second rollout on a freshly
        # allocated cache would replay against the first rollout's (freed)
        # buffers.  Overwriting CONTENTS between replays is allowed; moving
        # buffers is not.
        windowed = self._cache_window is not None
        if self._static_cache is not None:
            cache = self._static_cache
            for t in cache.k:
                t.zero_()
            for t in cache.v:
                t.zero_()
            cache.length = 0
            cache.max_len = max_len if windowed else cache.max_len
            cache.pending = None
            if windowed:
                # A new run starts a fresh fill: all slots free, no
                # permanent rows yet (the prefill re-establishes them).
                cache.slot_expiring = [False] * self._cache_window
                cache.write_head = 0
                cache.recycle_cursor = 0
                cache.n_permanent = 0
            return cache
        device = self._device
        if windowed:
            # C committed slots + the widest staging tail any pass binds.
            S_alloc = self._cache_window + self._staging_max
            cache_max_len = max_len  # demand cap in POSITIONS, not slots
        else:
            S_alloc = self._cache_stride
            cache_max_len = S_alloc
        k = [
            torch.zeros(S_alloc, nh, self._d_head, device=device)
            for nh in self._per_layer_n_heads
        ]
        v = [
            torch.zeros(S_alloc, nh, self._d_head, device=device)
            for nh in self._per_layer_n_heads
        ]
        self._static_cache = KVCache(
            k=k,
            v=v,
            length=0,
            max_len=cache_max_len,
            window=self._cache_window,
            staging=self._staging_max if windowed else 0,
            slot_expiring=[False] * self._cache_window if windowed else None,
        )
        return self._static_cache

    def step(self, inputs: torch.Tensor, cache: KVCache, past_len: int | None = None):
        if not isinstance(cache, KVCache):
            raise TypeError(
                "OnnxTokenRuntime.step requires a KVCache from empty_past(max_len)"
            )
        n_new = int(inputs.reshape(-1).shape[0])
        base = cache.length
        if cache.pending is not None:
            raise RuntimeError(
                "windowed KVCache holds an uncommitted speculative batch — "
                "_commit must run between a multi-row rollout step and the "
                "next step (the pending deltas are views of binding buffers "
                "the next pass overwrites)"
            )
        if base + n_new > cache.max_len:
            raise RuntimeError(
                f"KV cache overrun: cache_len {base} + n_new {n_new} exceeds "
                f"allocated max_len {cache.max_len}. Size empty_past() with "
                f"draft headroom."
            )
        if past_len is None:
            past_len = base
        # Length invariant: the absolute query position must equal the
        # committed count.  Unbounded protocol: the bound past view is
        # cache[:base], so the graph's mask geometry reads seq-dim == base
        # (a host-trimmed cache with a lying past_len was rejected as the
        # old "sliding window" non-option).  Windowed protocol: the mask is
        # derived in-graph from cache_position[0] == committed count — the
        # SUPERSESSION of that retirement (ring_idea.md): the window lives
        # in the graph's writtenness mask + the host's slot placement, with
        # nothing lying about positions, so the invariant is the same.
        assert past_len == base, (
            f"past_len {past_len} != cache_len {base}; the committed count "
            f"drives the mask (bound-view length / windowed writtenness), so "
            f"the mask and pos-encoding would disagree"
        )
        # Per-row expiry tags for the windowed placement policy (token row
        # id -> expiring type?).  Computed once per pass; the CPU<->GPU
        # sync this costs on the decode hot path is subsumed by the stream
        # synchronize every _run_iobinding already performs.
        expiring = None
        if cache.window is not None:
            rows = inputs.detach().reshape(-1).to(torch.int64).cpu().tolist()
            lut = self._expiring_rows
            expiring = [bool(lut[r]) if r < len(lut) else False for r in rows]
        if self._use_cuda_io:
            return self._step_cuda_io(inputs, cache, past_len, n_new, base, expiring)
        return self._step_cpu(inputs, cache, past_len, n_new, base, expiring)

    def _step_cpu(
        self, inputs, cache: KVCache, past_len: int, n_new: int, base: int, expiring
    ):
        import numpy as np

        token_ids = (
            inputs.detach().cpu().reshape(-1).to(torch.int64).numpy().astype("int64")
        )
        # Unbounded: bind the full S-row buffer.  Windowed: the mask is
        # exactly C + n_new wide, so bind that prefix of the C + staging
        # allocation.
        bind_rows = (
            self._windowed_s_eff(n_new)
            if cache.window is not None
            else cache.k[0].shape[0]
        )
        feeds: dict[str, Any] = {
            "token_ids": token_ids,
            "cache_position": np.arange(base, base + n_new, dtype=np.int64),
        }
        for i in range(self._n_layers):
            feeds[f"past_K_{i}"] = (
                cache.k[i][:bind_rows]
                .detach()
                .cpu()
                .numpy()
                .astype("float32", copy=False)
            )
            feeds[f"past_V_{i}"] = (
                cache.v[i][:bind_rows]
                .detach()
                .cpu()
                .numpy()
                .astype("float32", copy=False)
            )
        results = self._session.run(self._out_names, feeds)
        logits = torch.from_numpy(results[0])
        dk_rows = [
            torch.from_numpy(results[1 + 2 * i]).to(cache.k[i].dtype)
            for i in range(self._n_layers)
        ]
        dv_rows = [
            torch.from_numpy(results[1 + 2 * i + 1]).to(cache.v[i].dtype)
            for i in range(self._n_layers)
        ]
        _persist_rows(cache, base, n_new, dk_rows, dv_rows, expiring)
        cache.length = base + n_new
        return logits, cache

    def _bind_cuda(self, io, name, tensor, kind, device_id, dtype):
        import numpy as np

        bind = io.bind_input if kind == "in" else io.bind_output
        bind(
            name,
            device_type="cuda",
            device_id=device_id,
            element_type=dtype if dtype is not None else np.float32,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    def _binding_for(self, cache: KVCache, s_eff: int, width: int):
        """Build (once per (cache, S_eff, width)) a persistent io-binding.

        Every tensor the pass touches — token_ids (width,), cache_position
        (width,), the past prefix views, the delta outputs, logits — is
        allocated once and bound once; later passes only overwrite buffer
        CONTENTS.  Stable addresses + one frozen shape per gpu_graph_id are
        the CUDA-graph replay requirements; rebuilding the io-binding per
        step (the old pattern) breaks them and costs ~ms of Python per step.

        The past bindings are CONTIGUOUS PREFIX VIEWS ``cache.k[i][:s_eff]``
        of the one session-lifetime cache (row-major, so a prefix view
        shares the base pointer): every bucket's captured graph reads and
        scatters the same underlying buffer, only the attention width
        differs.  width > 1 is the padded spec-verify bucket; width == 1
        the pure-decode bucket.
        """
        import numpy as np

        key = (id(cache), s_eff, width)
        cached = self._bindings.get(key)
        if cached is not None:
            return cached
        if any(k[0] != id(cache) for k in self._bindings):
            # A different cache object would invalidate every captured
            # graph (baked addresses).  empty_past's session-lifetime
            # singleton makes this unreachable; fail loud if it ever isn't.
            raise RuntimeError(
                "io-bindings exist for a different cache object — captured "
                "graphs bake buffer addresses; the static cache must be the "
                "session-lifetime singleton from empty_past()"
            )

        device_id = int(self._device.index or 0)
        io = self._session.io_binding()
        token_ids = torch.zeros(width, dtype=torch.int64, device=self._device)
        cache_position = torch.zeros(width, dtype=torch.int64, device=self._device)
        self._bind_cuda(io, "token_ids", token_ids, "in", device_id, np.int64)
        self._bind_cuda(io, "cache_position", cache_position, "in", device_id, np.int64)
        delta_k: list[torch.Tensor] = []
        delta_v: list[torch.Tensor] = []
        for i in range(self._n_layers):
            self._bind_cuda(
                io, f"past_K_{i}", cache.k[i][:s_eff], "in", device_id, np.float32
            )
            self._bind_cuda(
                io, f"past_V_{i}", cache.v[i][:s_eff], "in", device_id, np.float32
            )
            nh = cache.k[i].shape[1]
            dk = torch.empty(width, nh, self._d_head, device=self._device)
            dv = torch.empty(width, nh, self._d_head, device=self._device)
            delta_k.append(dk)
            delta_v.append(dv)
            self._bind_cuda(io, f"delta_K_{i}", dk, "out", device_id, np.float32)
            self._bind_cuda(io, f"delta_V_{i}", dv, "out", device_id, np.float32)
        logits = torch.empty(width, self._logits_width, device=self._device)
        self._bind_cuda(io, "logits", logits, "out", device_id, np.float32)

        binding = {
            "io": io,
            # Keep the cache alive: the io-binding holds RAW pointers into
            # cache.k/cache.v; without this reference a GC'd cache + id()
            # reuse would alias freed device memory.
            "cache": cache,
            "token_ids": token_ids,
            "cache_position": cache_position,
            "arange_W": torch.arange(width, dtype=torch.int64, device=self._device),
            "delta_k": delta_k,
            "delta_v": delta_v,
            "logits": logits,
        }
        self._bindings[key] = binding
        return binding

    def _run_iobinding(
        self, io, gpu_graph_id: str, shape: tuple[int, int] | None = None
    ) -> None:
        """All bound-buffer writes ordered, then run with an EXPLICIT bucket.

        Stream ordering: ORT's io-binding Run does NOT synchronize inputs
        and its compute stream is cudaStreamNonBlocking — no implicit
        ordering with torch's stream.  The previous step's delta->cache
        persists and this step's token/cache_position writes are async torch
        ops, so an explicit stream sync before the run is REQUIRED or the
        run (and the captured-graph replay especially) can read a cache slot
        mid-copy: intermittent stale tokens.  (Sharing one stream via
        user_compute_stream is the later optimization; unexercised with
        capture as of 1.26.)

        gpu_graph_id is MANDATORY on every run: an optionless run() on an
        enable_cuda_graph session defaults to annotation id 0, which IS
        capture-eligible — one stray call permanently consumes graph-pool
        memory with whatever shape it had.  "-1" = run uncaptured; ids >= 1
        are the (S_eff, width) buckets (each captures on its first annotated
        calls via ORT-internal warm-up, replays afterwards).  ``shape`` is
        the (S_eff, width) this call binds — registered on first use and
        asserted ever after, because ORT does NOT cross-check it and a
        reused id would silently replay the baked shape.
        """
        import onnxruntime as ort

        if gpu_graph_id != "-1":
            assert shape is not None, "captured runs must declare (S_eff, width)"
            known = self._graph_id_shapes.setdefault(gpu_graph_id, shape)
            assert known == shape, (
                f"gpu_graph_id {gpu_graph_id} captured at (S_eff, width)="
                f"{known} but asked to run {shape} — a captured id replays "
                f"its baked shape; ids must be one-to-one with shapes"
            )
        torch.cuda.current_stream().synchronize()
        ro = ort.RunOptions()
        ro.add_run_config_entry("gpu_graph_id", gpu_graph_id)
        self._session.run_with_iobinding(io, ro)

    def _step_cuda_io(
        self, inputs, cache: KVCache, past_len: int, n_new: int, base: int, expiring
    ):
        import numpy as np

        windowed = cache.window is not None
        if n_new == 1:
            # Decode: persistent binding at the smallest covering bucket
            # (windowed: the constant C+1), contents-only updates.
            # (base+1 <= max_len is guaranteed by step()'s overrun guard,
            # and the last bucket == cache_stride, so a bucket always
            # exists here.)
            s_eff = self._windowed_s_eff(1) if windowed else self._bucket_for(base + 1)
            assert s_eff is not None
            b = self._binding_for(cache, s_eff, 1)
            b["token_ids"].copy_(
                inputs.detach().reshape(-1).to(device=self._device, dtype=torch.int64)
            )
            b["cache_position"].fill_(base)
            self._run_iobinding(
                b["io"], gpu_graph_id=self._graph_id_for(s_eff, 1), shape=(s_eff, 1)
            )
            _persist_rows(cache, base, 1, b["delta_k"], b["delta_v"], expiring)
            cache.length = base + 1
            # NOTE: the returned logits tensor is the persistent buffer — it
            # is overwritten by the NEXT decode step.  Both rollout loops
            # argmax it before stepping again.
            return b["logits"], cache

        W = self._batch_bucket_width
        if 2 <= n_new <= W:
            s_eff = self._windowed_s_eff(W) if windowed else self._bucket_for(base + W)
        else:
            s_eff = None
        if s_eff is not None:
            # Speculative verify batch, PADDED to the fixed bucket width so
            # one captured graph per stride bucket replays every step.
            # Pad rows repeat the last real token at positions
            # [base+n_new .. base+W): their in-graph scatter lands in slots
            # beyond the committed length but inside the bound prefix
            # (base + W <= S_eff — the bucket rule), with exactly-zero mask
            # weight for every real query (the static-tail theorem); their
            # outputs are sliced off and their deltas never persisted.
            # When base + W exceeds even the last bucket (== cache_stride,
            # the near-cache-end case), s_eff is None and the batch falls
            # through to the uncaptured dynamic path below — the same
            # fallback the pre-bucketing code had via base+W <= max_len.
            b = self._binding_for(cache, s_eff, W)
            ids = inputs.detach().reshape(-1).to(device=self._device, dtype=torch.int64)
            b["token_ids"][:n_new].copy_(ids)
            b["token_ids"][n_new:] = ids[-1]
            torch.add(b["arange_W"], base, out=b["cache_position"])
            self._run_iobinding(
                b["io"], gpu_graph_id=self._graph_id_for(s_eff, W), shape=(s_eff, W)
            )
            # Windowed: this is the speculative path — _persist_rows stashes
            # the rows on cache.pending and the rollout's _commit flushes
            # only the accepted prefix (rejected rows never consume a
            # slot).  Unbounded: direct write, rejection just lowers
            # length as before.
            _persist_rows(
                cache,
                base,
                n_new,
                [dk[:n_new] for dk in b["delta_k"]],
                [dv[:n_new] for dv in b["delta_v"]],
                expiring,
            )
            cache.length = base + n_new
            # Sliced view of the persistent buffer — overwritten by the next
            # batched step; callers argmax before stepping again.
            return b["logits"][:n_new], cache

        # Prefill / oversized / near-cache-end batches (variable n_new):
        # per-call binding, uncaptured path.  The past binds the smallest
        # prefix bucket covering base + n_new (always exists — step()'s
        # overrun guard caps base + n_new at max_len == the last bucket);
        # windowed mode binds the exact C + n_new the mask demands.
        s_eff_dyn = (
            self._windowed_s_eff(n_new) if windowed else self._bucket_for(base + n_new)
        )
        assert s_eff_dyn is not None
        token_ids = (
            inputs.detach()
            .reshape(-1)
            .to(device=self._device, dtype=torch.int64)
            .contiguous()
        )
        cache_position = torch.arange(
            base, base + n_new, dtype=torch.int64, device=self._device
        )
        io = self._session.io_binding()
        device_id = int(self._device.index or 0)
        self._bind_cuda(io, "token_ids", token_ids, "in", device_id, np.int64)
        self._bind_cuda(io, "cache_position", cache_position, "in", device_id, np.int64)

        delta_k_outs: list[torch.Tensor] = []
        delta_v_outs: list[torch.Tensor] = []
        for i in range(self._n_layers):
            self._bind_cuda(
                io, f"past_K_{i}", cache.k[i][:s_eff_dyn], "in", device_id, np.float32
            )
            self._bind_cuda(
                io, f"past_V_{i}", cache.v[i][:s_eff_dyn], "in", device_id, np.float32
            )
            # The delta outputs go to SEPARATE per-step buffers and are copied
            # into the cache slots after the run.  Binding an output into a
            # slice of the SAME allocation that backs the past_K/past_V inputs
            # is input/output aliasing, which the CUDA EP does not handle — it
            # corrupts the past read (degenerate output).  The copy is
            # O(n_new), not O(cache_len).
            nh = cache.k[i].shape[1]
            dk = torch.empty(
                n_new, nh, self._d_head, dtype=torch.float32, device=self._device
            )
            dv = torch.empty(
                n_new, nh, self._d_head, dtype=torch.float32, device=self._device
            )
            delta_k_outs.append(dk)
            delta_v_outs.append(dv)
            self._bind_cuda(io, f"delta_K_{i}", dk, "out", device_id, np.float32)
            self._bind_cuda(io, f"delta_V_{i}", dv, "out", device_id, np.float32)

        logits = torch.empty(
            (n_new, self._logits_width),
            dtype=torch.float32,
            device=self._device,
        )
        self._bind_cuda(io, "logits", logits, "out", device_id, np.float32)

        self._run_iobinding(io, gpu_graph_id="-1")
        # Copy the freshly computed deltas into the owned cache slots (the run
        # is complete, so there is no input/output aliasing during execution).
        _persist_rows(cache, base, n_new, delta_k_outs, delta_v_outs, expiring)
        cache.length = base + n_new
        return logits, cache

    def end_profiling(self) -> str | None:
        """Flush ORT's profile trace and return the JSON path (None if profiling
        was never enabled).  Safe to call once; ORT closes the trace file here."""
        if not getattr(self, "_profiling", False):
            return None
        path = self._session.end_profiling()
        self._profiling = False
        print(f"[onnxruntime] profile trace written to {path}", flush=True)
        return str(path)

    def eval(self) -> "OnnxTokenRuntime":
        return self


def _default_ort_providers(ort) -> list:
    import os

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        # The DOOM graph's content-addressed attention resolves unit-score
        # logit gaps in full fp32; TF32 (the ORT CUDA default on Ampere+)
        # collapses them and the softmax stops concentrating -> garbage tokens.
        # The whole graph is designed around "TF32 off" (see torchwright
        # attention_ops.py); disable it on the CUDA EP.
        cuda_opts = {"use_tf32": "0"}
        if not os.environ.get("TWDOOM_NO_CUDA_GRAPH"):
            # CUDA-graph capture for the n_new=1 decode step: each decode
            # replays one captured graph instead of ~1,750 kernel launches
            # (the measured ~55%-of-wall dispatch overhead).  enable_cuda_graph
            # and use_tf32 are independent provider options and compose.
            # With this flag on, a surviving Memcpy node is a HARD ERROR at
            # session creation ("This session cannot use the graph capture
            # feature"); runs opt in/out per call via the gpu_graph_id run
            # option ("-1" = uncaptured: prefill + variable-width spec
            # batches; "1" = the captured decode bucket).  Capture lands on
            # the first annotated call via ORT-internal warm-up reruns
            # (~3x latency on that call; ORT 1.26 mechanics).
            cuda_opts["enable_cuda_graph"] = "1"
        return [
            ("CUDAExecutionProvider", cuda_opts),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def _w_embed_t(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _W_EMBED_T_BY_DEVICE.get(key)
    if t is None:
        t = W_EMBED.t().contiguous().to(device)
        _W_EMBED_T_BY_DEVICE[key] = t
    return t


def argmax_rows(outputs: torch.Tensor) -> list[int]:
    """Argmax-decode compiled-step outputs to token row ids."""
    o = outputs.detach()
    if o.shape[-1] != W_EMBED.shape[1] and o.shape[-1] == W_EMBED.shape[0]:
        return o.argmax(dim=-1).cpu().tolist()
    wt = _w_embed_t(o.device).to(o.dtype)
    return (o @ wt).argmax(dim=-1).cpu().tolist()


def run_prefill(
    compiled,
    prefill_ids: list[int],
    *,
    max_cache_len: int,
    chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
    label: str,
    progress_every: int = 0,
):
    """Run prefill in large chunks and return ``(last_out, past, n_passes)``.

    Allocates the run's single owned cache (``max_cache_len`` rows) up front
    via ``compiled.empty_past(max_cache_len)`` and writes prefill straight into
    it; the in-process reference runtime ignores the cap and grows tuples.
    """
    if not prefill_ids:
        raise ValueError("prefill_ids must be non-empty")
    chunk_size = max(1, int(chunk_size))
    # Clamp to the runtime's int32-indexability bound (ONNX runtimes expose
    # it; the in-process reference has no such limit).  Bucket-aware: the
    # clamp is computed at the prefix bucket the whole prefill will bind,
    # so smaller buckets allow proportionally larger (equal-cost) chunks.
    safe_chunk_fn = getattr(compiled, "max_safe_prefill_chunk", None)
    safe_chunk = (
        safe_chunk_fn(len(prefill_ids)) if callable(safe_chunk_fn) else safe_chunk_fn
    )
    if safe_chunk is not None and chunk_size > safe_chunk:
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

    past = compiled.empty_past(max_cache_len)
    if isinstance(past, KVCache) and past.window is not None:
        if len(prefill_ids) >= past.window:
            raise RuntimeError(
                f"prefill ({len(prefill_ids)} rows) does not fit the "
                f"{past.window}-slot cache window; raise model.cache_window"
            )
        print(
            f"[{label}] windowed cache: window={past.window} "
            f"staging={past.staging}; prefill ({len(prefill_ids)} rows) "
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
        out, past = compiled.step(rows_to_input(chunk), past, past_len=offset)
        offset += len(chunk)
        # Multi-row batches pend on a windowed cache (the speculative-batch
        # discipline; prefill rides the same path) — every chunk is fully
        # accepted, so commit it before the next step.
        past = _commit(past, offset)
        passes += 1
        dt = time.time() - t0
        if progress_every and (n_chunks > 1 or dt >= 5.0 or chunk_idx == n_chunks - 1):
            print(
                f"[{label}] prefill chunk {chunk_idx + 1}/{n_chunks} "
                f"rows={len(chunk)} done in {dt:.1f}s "
                f"cache_len={_cache_len(past)}",
                flush=True,
            )

    assert out is not None
    return out, past, passes


def pure_ar_rollout(
    compiled,
    prefill_ids: list[int],
    max_positions: int,
    terminal_row: int,
    *,
    progress_every: int = 0,
    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
    argmax_fn: Callable[[torch.Tensor], list[int]] | None = None,
) -> RolloutResult:
    """Prefill once, then decode one token at a time until terminal or cap."""
    argmax_fn = argmax_fn or argmax_rows
    t0 = time.time()
    prefill_t0 = time.time()
    # Cache holds prefill + at most (max_positions - 1) decoded rows.
    max_cache_len = len(prefill_ids) + max_positions - 1
    out, past, n_passes = run_prefill(
        compiled,
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
            f"seed_row={cur} cache_len={_cache_len(past)}",
            flush=True,
        )
    seq_pos = len(prefill_ids)
    last_timed_print = time.time()
    while cur != terminal_row and len(emitted) < max_positions:
        step_t0 = time.time()
        out, past = compiled.step(rows_to_input([cur]), past, past_len=seq_pos)
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
                f"last_forward={step_dt:.1f}s, cache_len={_cache_len(past)})",
                flush=True,
            )
    return RolloutResult(
        emitted_rows=emitted,
        stopped="terminal" if cur == terminal_row else "cap",
        n_forward_passes=n_passes,
        seconds=time.time() - t0,
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


def _trim_cache_tensor(t, target: int):
    # Head-major in-process / mock tuple cache: seq axis is 1.
    if t.shape[1] == target:
        return t
    return t[:, :target].contiguous()


def _single_step(
    compiled,
    past,
    input_row: int,
    emitted: list[int],
    stats: dict[str, Any],
    *,
    argmax_fn: Callable[[torch.Tensor], list[int]],
):
    out, new_past = compiled.step(rows_to_input([input_row]), past)
    stats["forward_passes"] += 1
    stats["fallback_single_steps"] += 1
    emitted.append(argmax_fn(out[-1:])[0])
    return new_past


def _spec_step(
    compiled,
    past,
    next_input_row: int,
    emitted: list[int],
    drafter,
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
    cache_len = _cache_len(past)
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
    commit = (n_drafts + 1) if accept == n_drafts else (accept + 1)

    terminal_hit = False
    for i in range(commit):
        emission_row = draft_rows[i] if i < accept else pred[i]
        if emission_row == terminal_row:
            commit = i + 1
            terminal_hit = True
            stats["terminal_truncations"] += 1
            break

    new_past = _commit(new_past, cache_len + commit)

    stats["batches"] += 1
    stats["attempted_drafts"] += n_drafts
    stats["committed_rows"] += commit
    stats["accepted_drafts"] += min(accept, commit)
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
    for i in range(commit):
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
    reuse_tail = [row_to_sandbox_token_fn(pred[i]) for i in range(commit, n_drafts + 1)]
    return new_past, reuse_tail, reuse_health, reuse_probe


def spec_decode_rollout(
    compiled,
    prefill_ids: list[int],
    drafter,
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
    out, past, prefill_passes = run_prefill(
        compiled,
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
            f"seed_row={seed_row} cache_len={_cache_len(past)}",
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
        step_cache_len = _cache_len(past)
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
                compiled,
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
                compiled, past, next_input_row, emitted, stats, argmax_fn=argmax_fn
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
                f"last_forward={step_dt:.1f}s, cache_len={_cache_len(past)}, "
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
