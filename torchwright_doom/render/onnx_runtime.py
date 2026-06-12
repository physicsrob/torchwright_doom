"""The production ONNX inference engine: session ownership + IO binding.

:class:`OnnxTokenRuntime` is the one production
:class:`~torchwright_doom.render.generation.TokenRuntime`: it owns the
onnxruntime session over the cached compiled artifact, the persistent
CUDA-graph-captured IO bindings (per attention-window bucket), the
prefill/decode step paths, and the per-row expiry tagging that the
windowed :class:`~torchwright_doom.render.kv_cache.KVCache` placement
policy needs.  The optimum/TRT-LLM analog: the engine that loads an
exported model and serves ``step``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .generation import DEFAULT_PREFILL_CHUNK_SIZE, TokenRuntime
from .kv_cache import KVCache, WindowedState, persist_rows


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


class OnnxTokenRuntime(TokenRuntime):
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
        )
        if not self._logits_width:
            # Last-resort width for sidecar-less artifacts; imported lazily —
            # W_EMBED is built at import from screen-env-dependent constants.
            from ..embedding import W_EMBED

            self._logits_width = int(W_EMBED.shape[0])
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
        window = self._cache_window
        if self._static_cache is not None:
            cache = self._static_cache
            for t in cache.k:
                t.zero_()
            for t in cache.v:
                t.zero_()
            cache.length = 0
            if window is not None:
                cache.max_len = max_len
                # A new run starts a fresh fill: all slots free, no
                # permanent rows yet (the prefill re-establishes them).
                cache.windowed = WindowedState(
                    window=window,
                    staging=self._staging_max,
                    slot_expiring=[False] * window,
                )
            return cache
        device = self._device
        if window is not None:
            # C committed slots + the widest staging tail any pass binds.
            S_alloc = window + self._staging_max
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
            windowed=(
                WindowedState(
                    window=window,
                    staging=self._staging_max,
                    slot_expiring=[False] * window,
                )
                if window is not None
                else None
            ),
        )
        return self._static_cache

    def step(
        self,
        inputs: torch.Tensor,
        cache: KVCache,
        past_len: int | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        if not isinstance(cache, KVCache):
            raise TypeError(
                "OnnxTokenRuntime.step requires a KVCache from empty_past(max_len)"
            )
        n_new = int(inputs.reshape(-1).shape[0])
        base = cache.length
        if cache.windowed is not None and cache.windowed.pending is not None:
            raise RuntimeError(
                "windowed KVCache holds an uncommitted speculative batch — "
                "commit must run between a multi-row rollout step and the "
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
        if cache.windowed is not None:
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
            if cache.windowed is not None
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
        persist_rows(cache, base, n_new, dk_rows, dv_rows, expiring)
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

        windowed = cache.windowed is not None
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
            persist_rows(cache, base, 1, b["delta_k"], b["delta_v"], expiring)
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
            # Windowed: this is the speculative path — persist_rows stashes
            # the rows on cache.pending and the rollout's commit flushes
            # only the accepted prefix (rejected rows never consume a
            # slot).  Unbounded: direct write, rejection just lowers
            # length as before.
            persist_rows(
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
        persist_rows(cache, base, n_new, delta_k_outs, delta_v_outs, expiring)
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
