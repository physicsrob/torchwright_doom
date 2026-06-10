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

    Owned cache: lower the logical length in place (no copy).  Head-major
    tuple: trim to ``target`` as the in-process path requires.
    """
    if isinstance(past, KVCache):
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
        try:
            self._session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=so,
                providers=session_providers,
            )
        except Exception:
            if (
                providers is not None
                or "CUDAExecutionProvider" not in session_providers
            ):
                raise
            print(
                "[onnxruntime] CUDAExecutionProvider failed; retrying with CPUExecutionProvider",
                flush=True,
            )
            self._session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
        print(
            f"[onnxruntime] version={ort.__version__} "
            f"active providers={self._session.get_providers()}",
            flush=True,
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

        inputs = {inp.name: inp for inp in self._session.get_inputs()}
        self._n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
        if self._n_layers <= 0:
            raise ValueError(f"{self.onnx_path}: no past_K_* inputs")
        # past_K_i is sequence-major with a STATIC slot count: (S, nh, d_head).
        stride_dim = inputs["past_K_0"].shape[0]
        if not isinstance(stride_dim, int):
            raise ValueError(
                f"{self.onnx_path}: past_K_0 first dim is {stride_dim!r}, "
                f"expected a static int — this is a pre-static-cache "
                f"(past_len/Concat) artifact; bust the compile-cache entry and "
                f"recompile with the current exporter"
            )
        self._cache_stride = int(stride_dim)
        self._per_layer_n_heads = [
            int(inputs[f"past_K_{i}"].shape[1]) for i in range(self._n_layers)
        ]
        self._d_head = int(inputs["past_K_0"].shape[2])
        _total_heads = sum(self._per_layer_n_heads)
        _bytes_per_tok = _total_heads * self._d_head * 2 * 4  # K+V, float32
        print(
            f"[kvcache] layers={self._n_layers} total_heads={_total_heads} "
            f"d_head={self._d_head} cache_stride={self._cache_stride} "
            f"per_layer_heads={self._per_layer_n_heads} "
            f"bytes/token={_bytes_per_tok} ({_bytes_per_tok / 1e6:.3f} MB); "
            f"static cache total={self._cache_stride * _bytes_per_tok / 1e9:.2f} GB",
            flush=True,
        )
        # Persistent n_new=1 decode binding (built lazily per cache object);
        # see _decode_binding_for.
        self._decode_binding: tuple[int, Any] | None = None
        # Session-lifetime static cache, allocated on first empty_past and
        # zero-reset (never reallocated) on later calls — captured CUDA
        # graphs bake the buffer addresses.
        self._static_cache: KVCache | None = None
        # Outputs are the per-layer KV *deltas* (new rows only); the runtime
        # writes them into its owned cache tail (no full-cache output).
        self._out_names = ["logits"]
        for i in range(self._n_layers):
            self._out_names += [f"delta_K_{i}", f"delta_V_{i}"]

        meta_path = self.onnx_path.with_suffix(".meta.json")
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
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

    @property
    def max_safe_prefill_chunk(self) -> int:
        """Largest prefill chunk whose per-layer transients stay int32-indexable.

        The widest layer materializes (n_heads, chunk, S) attention logits;
        past 2^31-1 elements, CUDA kernels that index with int32 write
        through overflowed offsets — observed as an Xid 31 MMU fault on a
        B200 at S=65536, chunk=1024, nh=128 (8.6e9 elements).  Decode
        (n_new=1) is far below the limit at any realistic S.
        """
        widest = max(self._per_layer_n_heads)
        return max(1, (2**31 - 1) // (widest * self._cache_stride))

    def empty_past(self, max_len: int) -> KVCache:
        """Allocate the single runtime-owned static cache for a planned run.

        ``max_len`` is the run's DEMAND (size it from ``len(prefill_ids) +
        max_positions - 1`` plus any speculative-draft headroom); the actual
        allocation is always the model's full ``cache_stride`` — the static
        ``past_K_i`` input shapes reject anything else.  Zero-initialized
        (load-bearing: masked slots are read with weight exactly 0.0, and
        ``0 * NaN = NaN``).
        """
        if max_len is None:
            raise ValueError(
                "OnnxTokenRuntime.empty_past requires an explicit max_len "
                "(size from len(prefill_ids) + max_positions - 1 [+ draft_window])"
            )
        if max_len > self._cache_stride:
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
        if self._static_cache is not None:
            cache = self._static_cache
            for t in cache.k:
                t.zero_()
            for t in cache.v:
                t.zero_()
            cache.length = 0
            return cache
        device = self._device
        S = self._cache_stride
        k = [
            torch.zeros(S, nh, self._d_head, device=device)
            for nh in self._per_layer_n_heads
        ]
        v = [
            torch.zeros(S, nh, self._d_head, device=device)
            for nh in self._per_layer_n_heads
        ]
        self._static_cache = KVCache(k=k, v=v, length=0, max_len=S)
        return self._static_cache

    def step(self, inputs: torch.Tensor, cache: KVCache, past_len: int | None = None):
        if not isinstance(cache, KVCache):
            raise TypeError(
                "OnnxTokenRuntime.step requires a KVCache from empty_past(max_len)"
            )
        n_new = int(inputs.reshape(-1).shape[0])
        base = cache.length
        if base + n_new > cache.max_len:
            raise RuntimeError(
                f"KV cache overrun: cache_len {base} + n_new {n_new} exceeds "
                f"allocated max_len {cache.max_len}. Size empty_past() with "
                f"draft headroom."
            )
        if past_len is None:
            past_len = base
        # Length invariant: the bound past view is cache[:base], so the graph's
        # mask geometry reads seq-dim == base.  The non-sliding-window contract
        # requires the absolute query position to equal it.  (A future
        # sliding-window runtime that hands a trimmed cache with a larger
        # past_len would relax this.)
        assert past_len == base, (
            f"past_len {past_len} != cache_len {base}; the in-place cache binds "
            f"past_K=cache[:cache_len], so the mask and pos-encoding would disagree"
        )
        if self._use_cuda_io:
            return self._step_cuda_io(inputs, cache, past_len, n_new, base)
        return self._step_cpu(inputs, cache, past_len, n_new, base)

    def _step_cpu(self, inputs, cache: KVCache, past_len: int, n_new: int, base: int):
        import numpy as np

        token_ids = (
            inputs.detach().cpu().reshape(-1).to(torch.int64).numpy().astype("int64")
        )
        feeds: dict[str, Any] = {
            "token_ids": token_ids,
            "cache_position": np.arange(base, base + n_new, dtype=np.int64),
        }
        for i in range(self._n_layers):
            # The static past_K_i input shape requires the FULL S-row buffer.
            feeds[f"past_K_{i}"] = (
                cache.k[i].detach().cpu().numpy().astype("float32", copy=False)
            )
            feeds[f"past_V_{i}"] = (
                cache.v[i].detach().cpu().numpy().astype("float32", copy=False)
            )
        results = self._session.run(self._out_names, feeds)
        logits = torch.from_numpy(results[0])
        for i in range(self._n_layers):
            dk = torch.from_numpy(results[1 + 2 * i])
            dv = torch.from_numpy(results[1 + 2 * i + 1])
            cache.k[i][base : base + n_new] = dk.to(cache.k[i].dtype)
            cache.v[i][base : base + n_new] = dv.to(cache.v[i].dtype)
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

    def _decode_binding_for(self, cache: KVCache):
        """Build (once per cache) the persistent n_new=1 decode binding.

        Every tensor the decode step touches — token_ids (1,),
        cache_position (1,), the full static past buffers, the delta
        outputs, logits — is allocated once and bound once; later steps
        only overwrite buffer CONTENTS.  Stable addresses + static shapes
        are the CUDA-graph replay requirements; rebuilding the io-binding
        per step (the old pattern) breaks them and costs ~ms of Python
        per step.
        """
        import numpy as np

        if self._decode_binding is not None and self._decode_binding[0] == id(cache):
            return self._decode_binding[1]

        device_id = int(self._device.index or 0)
        io = self._session.io_binding()
        token_ids = torch.zeros(1, dtype=torch.int64, device=self._device)
        cache_position = torch.zeros(1, dtype=torch.int64, device=self._device)
        self._bind_cuda(io, "token_ids", token_ids, "in", device_id, np.int64)
        self._bind_cuda(io, "cache_position", cache_position, "in", device_id, np.int64)
        delta_k: list[torch.Tensor] = []
        delta_v: list[torch.Tensor] = []
        for i in range(self._n_layers):
            self._bind_cuda(io, f"past_K_{i}", cache.k[i], "in", device_id, np.float32)
            self._bind_cuda(io, f"past_V_{i}", cache.v[i], "in", device_id, np.float32)
            nh = cache.k[i].shape[1]
            dk = torch.empty(1, nh, self._d_head, device=self._device)
            dv = torch.empty(1, nh, self._d_head, device=self._device)
            delta_k.append(dk)
            delta_v.append(dv)
            self._bind_cuda(io, f"delta_K_{i}", dk, "out", device_id, np.float32)
            self._bind_cuda(io, f"delta_V_{i}", dv, "out", device_id, np.float32)
        logits = torch.empty(1, self._logits_width, device=self._device)
        self._bind_cuda(io, "logits", logits, "out", device_id, np.float32)

        binding = {
            "io": io,
            # Keep the cache alive: the io-binding holds RAW pointers into
            # cache.k/cache.v; without this reference a GC'd cache + id()
            # reuse would alias freed device memory.
            "cache": cache,
            "token_ids": token_ids,
            "cache_position": cache_position,
            "delta_k": delta_k,
            "delta_v": delta_v,
            "logits": logits,
        }
        self._decode_binding = (id(cache), binding)
        return binding

    def _run_iobinding(self, io, gpu_graph_id: str) -> None:
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
        memory with whatever shape it had.  "-1" = run uncaptured;
        "1" = the n_new=1 decode bucket (captures on its first annotated
        call via ORT-internal warm-up reruns, replays afterwards).
        """
        import onnxruntime as ort

        torch.cuda.current_stream().synchronize()
        ro = ort.RunOptions()
        ro.add_run_config_entry("gpu_graph_id", gpu_graph_id)
        self._session.run_with_iobinding(io, ro)

    def _step_cuda_io(self, inputs, cache: KVCache, past_len: int, n_new: int, base: int):
        import numpy as np

        if n_new == 1:
            # Decode: persistent binding, contents-only updates.
            b = self._decode_binding_for(cache)
            b["token_ids"].copy_(
                inputs.detach().reshape(-1).to(device=self._device, dtype=torch.int64)
            )
            b["cache_position"].fill_(base)
            self._run_iobinding(b["io"], gpu_graph_id="1")
            for i in range(self._n_layers):
                cache.k[i][base : base + 1] = b["delta_k"][i]
                cache.v[i][base : base + 1] = b["delta_v"][i]
            cache.length = base + 1
            # NOTE: the returned logits tensor is the persistent buffer — it
            # is overwritten by the NEXT decode step.  Both rollout loops
            # argmax it before stepping again.
            return b["logits"], cache

        # Prefill / speculative batches (variable n_new): per-call binding,
        # uncaptured path.  The past binds are still the FULL static buffers.
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
            self._bind_cuda(io, f"past_K_{i}", cache.k[i], "in", device_id, np.float32)
            self._bind_cuda(io, f"past_V_{i}", cache.v[i], "in", device_id, np.float32)
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
        for i in range(self._n_layers):
            cache.k[i][base : base + n_new] = delta_k_outs[i]
            cache.v[i][base : base + n_new] = delta_v_outs[i]
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
    # it; the in-process reference has no such limit).
    safe_chunk = getattr(compiled, "max_safe_prefill_chunk", None)
    if safe_chunk is not None and chunk_size > safe_chunk:
        print(
            f"[{label}] prefill chunk {chunk_size} -> {safe_chunk} "
            f"(int32 transient-indexability clamp at this cache_stride)",
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
    out = None
    offset = 0
    passes = 0
    for chunk_idx in range(n_chunks):
        chunk = prefill_ids[offset : offset + chunk_size]
        t0 = time.time()
        out, past = compiled.step(rows_to_input(chunk), past, past_len=offset)
        offset += len(chunk)
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
