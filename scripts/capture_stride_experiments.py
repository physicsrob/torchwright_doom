"""Stride-bucketing de-risk: can ONE ORT session capture MULTIPLE CUDA graphs
over prefix windows (S_eff) of a single static KV cache, with the attention
width per bucket coming from a SYMBOLIC first dim on past_K_i?

Structurally-faithful toy of the production cached graph (the exact node
pattern of torchwright/compiler/export.py::_emit_cached_preamble +
_emit_cached_layer_nodes: int64 embedding/pos Gather, arange mask chain,
sequence-major ScatterND, head-major Transpose+MatMul attention, Where(-1e6),
FFN), at adder scale (3 layers, d=64) so every experiment runs locally on the
L4 in seconds.

Three graph variants:

  static   — the CURRENT production contract (control): past_K_i first dim =
             literal S, mask from the baked full arange_S.
  dyn      — the candidate: past_K_i first dim = symbolic "cache_slots";
             mask = Greater(Slice(arange_S, 0, Shape(past_K_0)[0:1]),
             cache_position).  Introduces the graph's first Shape op — the
             load-bearing question is whether ORT places the shape chain on
             CPU WITHOUT Memcpy nodes (a single Memcpy = hard error at
             session creation under enable_cuda_graph).
  slotids  — the fallback: past_K_i first dim symbolic, and the slot
             enumeration arrives as a graph INPUT slot_ids (int64,
             ("cache_slots",)) bound per bucket by the runtime (a trivial
             integer range, same class as cache_position).  Zero Shape ops.

Experiments (run each as its own process so ORT's stderr log is attributable):

  --exp build               build + save all variants and the fp32 torch oracle
  --exp memcpy  --variant V session under enable_cuda_graph + severity-1 log;
                            the Memcpy/placement lines ARE the result
  --exp rollout --variant V prefill (uncaptured) + decode rollout that crosses
                            bucket boundaries; per-bucket capture; per-step
                            captured-replay vs uncaptured-run compare (same
                            binding, same cache state -> expect bit-identical);
                            full-stream argmax vs the torch oracle; interleaved
                            width-1/width-W buckets; nvidia-smi deltas per
                            capture; profile-event collapse as the replay
                            signature.

Usage (local L4, workspace venv):
  python scripts/capture_stride_experiments.py --exp build
  python scripts/capture_stride_experiments.py --exp memcpy --variant dyn
  python scripts/capture_stride_experiments.py --exp rollout --variant dyn
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

WORK = Path("/tmp/stride_exp")
SENTINEL = -1e6  # CAUSAL_MASK_SENTINEL (torchwright/graph/attn.py)

# Toy scale — mirrors the production shape FAMILY (non-uniform trimmed heads).
N_LAYERS = 3
D = 64
D_HEAD = 16
PER_LAYER_NH = [4, 4, 2]
VOCAB = 32
MAX_SEQ_LEN = 256
S_MAX = 256  # cache_stride (arange_S size, cache alloc)
BUCKETS = [64, 128, 256]  # S_eff prefix windows
PREFILL = 20
WIDTH_BUCKET = 4  # spec-verify-style padded width (prod: 9)
SEED = 7


# --------------------------------------------------------------------------
# Graph builders (mirror export.py node-for-node at toy scale)
# --------------------------------------------------------------------------


def _weights():
    rng = np.random.RandomState(SEED)

    def f32(a, scale):
        return (a * scale).astype(np.float32)

    w = {
        "W_embed": f32(rng.randn(VOCAB, D), 0.5),
        "pos_encoding_full": f32(rng.randn(MAX_SEQ_LEN, D), 0.1),
        "W_out": f32(rng.randn(D, VOCAB), 0.5),
    }
    for i, nh in enumerate(PER_LAYER_NH):
        hd = nh * D_HEAD
        w[f"l{i}_WQ"] = f32(rng.randn(D, hd), 0.3 / np.sqrt(D))
        w[f"l{i}_WK"] = f32(rng.randn(D, hd), 0.3 / np.sqrt(D))
        w[f"l{i}_WV"] = f32(rng.randn(D, hd), 0.3 / np.sqrt(D))
        w[f"l{i}_WO"] = f32(rng.randn(hd, D), 0.3 / np.sqrt(hd))
        w[f"l{i}_W1"] = f32(rng.randn(D, 2 * D), 0.3 / np.sqrt(D))
        w[f"l{i}_b1"] = f32(rng.randn(2 * D), 0.01)
        w[f"l{i}_W2"] = f32(rng.randn(2 * D, D), 0.3 / np.sqrt(2 * D))
        w[f"l{i}_b2"] = f32(rng.randn(D), 0.01)
    return w


def build_variant(variant: str, out_path: Path) -> None:
    from onnx import TensorProto, helper, numpy_helper

    w = _weights()
    inits = [numpy_helper.from_array(v, name=k) for k, v in w.items()]
    inits += [
        numpy_helper.from_array(np.arange(S_MAX, dtype=np.int64), name="arange_S"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="_axes0_1d"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="_axes1_1d"),
        numpy_helper.from_array(
            np.array(SENTINEL, dtype=np.float32), name="_f32_causal_sentinel_s"
        ),
    ]
    if variant == "dyn":
        inits += [
            numpy_helper.from_array(np.array([0], dtype=np.int64), name="_zeros_1d"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="_ones_1d"),
        ]
    for i, nh in enumerate(PER_LAYER_NH):
        inits += [
            numpy_helper.from_array(
                np.array([0, nh, D_HEAD], dtype=np.int64), name=f"l{i}_qkv_view_shape"
            ),
            numpy_helper.from_array(
                np.array([0, nh * D_HEAD], dtype=np.int64), name=f"l{i}_ctx_flat_shape"
            ),
        ]

    nodes = []

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # ---- preamble (export.py::_emit_cached_preamble) ----
    if variant == "static":
        add("Unsqueeze", ["arange_S", "_axes0_1d"], ["_slots_row"])
    elif variant == "dyn":
        # The candidate: derive S_eff from the BOUND past shape, slice the
        # baked arange to it.  Shape/shape-slice are CPU-native; the
        # arange-Slice keeps GPU data with CPU scalar bounds.
        add("Shape", ["past_K_0"], ["_pastK0_shape"])
        add(
            "Slice",
            ["_pastK0_shape", "_zeros_1d", "_ones_1d", "_axes0_1d"],
            ["_s_eff_1d"],
        )
        add("Slice", ["arange_S", "_zeros_1d", "_s_eff_1d", "_axes0_1d"], ["_slots"])
        add("Unsqueeze", ["_slots", "_axes0_1d"], ["_slots_row"])
    elif variant == "slotids":
        add("Unsqueeze", ["slot_ids", "_axes0_1d"], ["_slots_row"])
    else:
        raise ValueError(variant)
    add("Unsqueeze", ["cache_position", "_axes1_1d"], ["_cache_pos_col"])
    add("Greater", ["_slots_row", "_cache_pos_col"], ["_mask_bool"])
    add("Unsqueeze", ["_mask_bool", "_axes0_1d"], ["mask_bool_3d"])
    add("Gather", ["pos_encoding_full", "cache_position"], ["pos"], axis=0)

    # ---- embedding (token graph: Gather + pos add) ----
    add("Gather", ["W_embed", "token_ids"], ["emb"], axis=0)
    add("Add", ["emb", "pos"], ["res0"])

    # ---- layers (export.py::_emit_cached_layer_nodes, node-for-node) ----
    cur = "res0"
    for i in range(N_LAYERS):
        p = f"l{i}"
        add("MatMul", [cur, f"{p}_WQ"], [f"{p}_Q_flat"])
        add("Reshape", [f"{p}_Q_flat", f"{p}_qkv_view_shape"], [f"{p}_Q_sm"])
        add("MatMul", [cur, f"{p}_WK"], [f"{p}_K_flat"])
        add("Reshape", [f"{p}_K_flat", f"{p}_qkv_view_shape"], [f"delta_K_{i}"])
        add("MatMul", [cur, f"{p}_WV"], [f"{p}_V_flat"])
        add("Reshape", [f"{p}_V_flat", f"{p}_qkv_view_shape"], [f"delta_V_{i}"])
        add("Transpose", [f"{p}_Q_sm"], [f"{p}_Q"], perm=[1, 0, 2])
        add(
            "ScatterND",
            [f"past_K_{i}", "_cache_pos_col", f"delta_K_{i}"],
            [f"{p}_K_static"],
        )
        add(
            "ScatterND",
            [f"past_V_{i}", "_cache_pos_col", f"delta_V_{i}"],
            [f"{p}_V_static"],
        )
        add("Transpose", [f"{p}_K_static"], [f"{p}_K_full"], perm=[1, 0, 2])
        add("Transpose", [f"{p}_V_static"], [f"{p}_V_full"], perm=[1, 0, 2])
        add("Transpose", [f"{p}_K_full"], [f"{p}_K_T"], perm=[0, 2, 1])
        add("MatMul", [f"{p}_Q", f"{p}_K_T"], [f"{p}_logits"])
        add(
            "Where",
            ["mask_bool_3d", "_f32_causal_sentinel_s", f"{p}_logits"],
            [f"{p}_logits_masked"],
        )
        add("Softmax", [f"{p}_logits_masked"], [f"{p}_weights"], axis=-1)
        add("MatMul", [f"{p}_weights", f"{p}_V_full"], [f"{p}_ctx"])
        add("Transpose", [f"{p}_ctx"], [f"{p}_ctx_t"], perm=[1, 0, 2])
        add("Reshape", [f"{p}_ctx_t", f"{p}_ctx_flat_shape"], [f"{p}_ctx_flat"])
        add("MatMul", [f"{p}_ctx_flat", f"{p}_WO"], [f"{p}_attn_sum"])
        add("Add", [cur, f"{p}_attn_sum"], [f"{p}_res_attn"])
        add("MatMul", [f"{p}_res_attn", f"{p}_W1"], [f"{p}_l1_m"])
        add("Add", [f"{p}_l1_m", f"{p}_b1"], [f"{p}_l1_b"])
        add("Relu", [f"{p}_l1_b"], [f"{p}_l1_r"])
        add("MatMul", [f"{p}_l1_r", f"{p}_W2"], [f"{p}_l2_m"])
        add("Add", [f"{p}_l2_m", f"{p}_b2"], [f"{p}_l2_b"])
        add("Add", [f"{p}_res_attn", f"{p}_l2_b"], [f"{p}_res_next"])
        cur = f"{p}_res_next"

    add("MatMul", [cur, "W_out"], ["logits"])

    # ---- I/O value infos ----
    s_dim = S_MAX if variant == "static" else "cache_slots"
    graph_inputs = [
        helper.make_tensor_value_info("token_ids", TensorProto.INT64, ["n_new"]),
        helper.make_tensor_value_info("cache_position", TensorProto.INT64, ["n_new"]),
    ]
    if variant == "slotids":
        graph_inputs.append(
            helper.make_tensor_value_info(
                "slot_ids", TensorProto.INT64, ["cache_slots"]
            )
        )
    graph_outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n_new", VOCAB]),
    ]
    for i, nh in enumerate(PER_LAYER_NH):
        graph_inputs += [
            helper.make_tensor_value_info(
                f"past_K_{i}", TensorProto.FLOAT, [s_dim, nh, D_HEAD]
            ),
            helper.make_tensor_value_info(
                f"past_V_{i}", TensorProto.FLOAT, [s_dim, nh, D_HEAD]
            ),
        ]
        graph_outputs += [
            helper.make_tensor_value_info(
                f"delta_K_{i}", TensorProto.FLOAT, ["n_new", nh, D_HEAD]
            ),
            helper.make_tensor_value_info(
                f"delta_V_{i}", TensorProto.FLOAT, ["n_new", nh, D_HEAD]
            ),
        ]

    import onnx

    graph = helper.make_graph(
        nodes, f"stride_toy_{variant}", graph_inputs, graph_outputs, initializer=inits
    )
    # opset 14 / IR 8 — the SAME opset the production exporters emit
    # (export.py:813 / export.py:1063); kernel registration and optimizer
    # transforms are opset-versioned, so the de-risk must run what the
    # exporter will actually produce.
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 14)],
        producer_name="capture_stride_experiments",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    print(f"[build] {variant}: {out_path} ({out_path.stat().st_size} bytes)")


# --------------------------------------------------------------------------
# fp32 torch oracle (CPU) — same math, dynamic length (no padding at all)
# --------------------------------------------------------------------------


class Oracle:
    def __init__(self):
        import torch

        self.t = torch
        self.w = {k: torch.from_numpy(v) for k, v in _weights().items()}
        self.k = [torch.zeros(0, nh, D_HEAD) for nh in PER_LAYER_NH]
        self.v = [torch.zeros(0, nh, D_HEAD) for nh in PER_LAYER_NH]

    def step(self, token_ids: list[int]):
        t, w = self.t, self.w
        n = len(token_ids)
        base = self.k[0].shape[0]
        ids = t.tensor(token_ids, dtype=t.int64)
        pos = w["pos_encoding_full"][base : base + n]
        res = w["W_embed"][ids] + pos
        total = base + n
        # causal mask over the DYNAMIC length (j > p blocked)
        j = t.arange(total)[None, :]
        p = t.arange(base, base + n)[:, None]
        blocked = j > p
        for i in range(N_LAYERS):
            q = (res @ w[f"l{i}_WQ"]).reshape(n, PER_LAYER_NH[i], D_HEAD)
            dk = (res @ w[f"l{i}_WK"]).reshape(n, PER_LAYER_NH[i], D_HEAD)
            dv = (res @ w[f"l{i}_WV"]).reshape(n, PER_LAYER_NH[i], D_HEAD)
            kf = t.cat([self.k[i], dk], 0)  # (total, nh, dh)
            vf = t.cat([self.v[i], dv], 0)
            self.k[i], self.v[i] = kf, vf
            logits = t.einsum("qhd,khd->hqk", q, kf)
            logits = t.where(blocked[None], t.tensor(SENTINEL), logits)
            wts = t.softmax(logits, -1)
            ctx = t.einsum("hqk,khd->qhd", wts, vf)
            res = res + ctx.reshape(n, -1) @ w[f"l{i}_WO"]
            h = t.relu(res @ w[f"l{i}_W1"] + w[f"l{i}_b1"])
            res = res + h @ w[f"l{i}_W2"] + w[f"l{i}_b2"]
        return res @ w["W_out"]  # (n, VOCAB)


# --------------------------------------------------------------------------
# Runtime harness (mirrors render/onnx_runtime.py binding + run discipline)
# --------------------------------------------------------------------------


def _gpu_used_mb() -> int:
    out = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    return int(out)


def _make_session(
    path: Path, *, capture: bool, profile: bool = False, severity: int | None = None
):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if capture:
        so.enable_mem_pattern = False
    if profile:
        so.enable_profiling = True
        so.profile_file_prefix = str(WORK / "ort_profile")
    if severity is not None:
        so.log_severity_level = severity
        ort.set_default_logger_severity(severity)
    cuda_opts = {"use_tf32": "0"}
    if capture:
        cuda_opts["enable_cuda_graph"] = "1"
    return ort.InferenceSession(
        str(path),
        sess_options=so,
        providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"],
    )


class BucketRunner:
    """Per-(S_eff, width) persistent bindings over ONE session-lifetime cache.

    The toy mirror of OnnxTokenRuntime: full-S_MAX zero cache buffers, prefix
    views cache[i][:S_eff] bound per bucket (contiguous, same base address),
    mandatory gpu_graph_id per run, stream sync before each run.
    """

    def __init__(self, session, variant: str):
        import torch

        self.t = torch
        self.sess = session
        self.variant = variant
        self.dev = torch.device("cuda", 0)
        self.k = [
            torch.zeros(S_MAX, nh, D_HEAD, device=self.dev) for nh in PER_LAYER_NH
        ]
        self.v = [
            torch.zeros(S_MAX, nh, D_HEAD, device=self.dev) for nh in PER_LAYER_NH
        ]
        self.length = 0
        self.bindings: dict[tuple[int, int], dict] = {}
        self.out_names = ["logits"] + [
            n for i in range(N_LAYERS) for n in (f"delta_K_{i}", f"delta_V_{i}")
        ]

    def _bind(self, io, name, tensor, kind, np_dtype):
        b = io.bind_input if kind == "in" else io.bind_output
        b(
            name,
            device_type="cuda",
            device_id=0,
            element_type=np_dtype,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    def binding_for(self, s_eff: int, width: int) -> dict:
        key = (s_eff, width)
        if key in self.bindings:
            return self.bindings[key]
        t = self.t
        io = self.sess.io_binding()
        token_ids = t.zeros(width, dtype=t.int64, device=self.dev)
        cache_position = t.zeros(width, dtype=t.int64, device=self.dev)
        self._bind(io, "token_ids", token_ids, "in", np.int64)
        self._bind(io, "cache_position", cache_position, "in", np.int64)
        slot_ids = None
        if self.variant == "slotids":
            slot_ids = t.arange(s_eff, dtype=t.int64, device=self.dev)
            self._bind(io, "slot_ids", slot_ids, "in", np.int64)
        dks, dvs = [], []
        for i, nh in enumerate(PER_LAYER_NH):
            # contiguous prefix view: same base data_ptr, S_eff rows
            self._bind(io, f"past_K_{i}", self.k[i][:s_eff], "in", np.float32)
            self._bind(io, f"past_V_{i}", self.v[i][:s_eff], "in", np.float32)
            dk = t.empty(width, nh, D_HEAD, device=self.dev)
            dv = t.empty(width, nh, D_HEAD, device=self.dev)
            dks.append(dk)
            dvs.append(dv)
            self._bind(io, f"delta_K_{i}", dk, "out", np.float32)
            self._bind(io, f"delta_V_{i}", dv, "out", np.float32)
        logits = t.empty(width, VOCAB, device=self.dev)
        self._bind(io, "logits", logits, "out", np.float32)
        b = {
            "io": io,
            "token_ids": token_ids,
            "cache_position": cache_position,
            "slot_ids": slot_ids,
            "dk": dks,
            "dv": dvs,
            "logits": logits,
            "captured": False,
        }
        self.bindings[key] = b
        return b

    def run(self, io, gpu_graph_id: str):
        import onnxruntime as ort

        self.t.cuda.current_stream().synchronize()
        ro = ort.RunOptions()
        ro.add_run_config_entry("gpu_graph_id", gpu_graph_id)
        self.sess.run_with_iobinding(io, ro)
        self.t.cuda.synchronize()

    def step(
        self,
        rows: list[int],
        s_eff: int,
        width: int,
        gpu_graph_id: str,
        *,
        persist: bool = True,
        compare_uncaptured: bool = False,
    ) -> tuple:
        """One pass at (s_eff, width); rows padded by repeating the last row.

        compare_uncaptured: run id "-1" first on the SAME binding/cache state,
        snapshot outputs, then the bucket run — returns the max |diff| between
        uncaptured and captured outputs (expect exactly 0.0: same kernels,
        same shapes, same inputs).
        """
        t = self.t
        n = len(rows)
        base = self.length
        assert base + width <= s_eff, (base, width, s_eff)
        b = self.binding_for(s_eff, width)
        padded = rows + [rows[-1]] * (width - n)
        b["token_ids"].copy_(t.tensor(padded, dtype=t.int64))
        b["cache_position"].copy_(t.arange(base, base + width, dtype=t.int64))
        max_diff = None
        if compare_uncaptured:
            self.run(b["io"], "-1")
            ref = (
                b["logits"].clone(),
                [d.clone() for d in b["dk"]],
                [d.clone() for d in b["dv"]],
            )
        self.run(b["io"], gpu_graph_id)
        if not b["captured"]:
            b["captured"] = True
        if compare_uncaptured:
            diffs = [(b["logits"] - ref[0]).abs().max().item()]
            diffs += [
                (b["dk"][i] - ref[1][i]).abs().max().item() for i in range(N_LAYERS)
            ]
            diffs += [
                (b["dv"][i] - ref[2][i]).abs().max().item() for i in range(N_LAYERS)
            ]
            max_diff = max(diffs)
        if persist:
            for i in range(N_LAYERS):
                self.k[i][base : base + n] = b["dk"][i][:n]
                self.v[i][base : base + n] = b["dv"][i][:n]
            self.length = base + n
        return b["logits"][:n].clone(), max_diff


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def exp_build():
    WORK.mkdir(exist_ok=True)
    for v in ("static", "dyn", "slotids"):
        build_variant(v, WORK / f"{v}.onnx")


def exp_memcpy(variant: str):
    """Session creation under enable_cuda_graph at severity 1.

    The verdicts to grep from stderr: 'MemcpyTransformer modified: 0' (or a
    'N Memcpy nodes are added' warning = FAIL), per-node placement lines, and
    whether session creation throws (a surviving Memcpy is a hard error)."""
    print(
        f"[memcpy] variant={variant} creating capture session "
        f"(severity-1 log follows on stderr)",
        flush=True,
    )
    sess = _make_session(WORK / f"{variant}.onnx", capture=True, severity=1)
    print(
        f"[memcpy] variant={variant} session created OK; providers="
        f"{sess.get_providers()}",
        flush=True,
    )


def exp_rollout(variant: str, compare_each: bool = True):
    import torch  # noqa: F401

    buckets = BUCKETS if variant != "static" else [S_MAX]
    print(
        f"[rollout] variant={variant} buckets={buckets} "
        f"prefill={PREFILL} S_MAX={S_MAX} compare_each={compare_each}"
    )
    sess = _make_session(
        WORK / f"{variant}.onnx", capture=True, profile=True, severity=2
    )
    r = BucketRunner(sess, variant)
    oracle = Oracle()
    rng = np.random.RandomState(123)
    prefill_rows = [int(x) for x in rng.randint(0, VOCAB, PREFILL)]

    mem0 = _gpu_used_mb()

    # --- prefill: uncaptured (id -1), bound at the smallest covering bucket
    s_pre = min(b for b in buckets if b >= PREFILL + 1)
    out, _ = r.step(prefill_rows, s_pre, PREFILL, "-1")
    o_out = oracle.step(prefill_rows)
    d_pre = float((out.cpu() - o_out).abs().max())
    print(f"[rollout] prefill bucket={s_pre} max|onnx-oracle|={d_pre:.3e}")

    # --- decode: width-1 stream crossing every bucket boundary; per-bucket
    # gpu_graph_id = 10+bucket_index; every step checks replay==uncaptured.
    cur = int(out[-1].argmax())
    o_cur = int(o_out[-1].argmax())
    assert cur == o_cur, f"prefill argmax diverged: {cur} vs {o_cur}"
    stream, o_stream = [cur], [o_cur]
    captures = []  # (step, bucket, mem_delta)
    argmax_diffs = []
    per_bucket_steps: dict[int, int] = {}
    max_replay_diff = 0.0
    max_oracle_diff = 0.0
    n_steps = S_MAX - PREFILL - 1  # run to one short of cache capacity
    for s in range(n_steps):
        base = r.length
        s_eff = min(b for b in buckets if b >= base + 1)
        bidx = buckets.index(s_eff)
        first = not r.binding_for(s_eff, 1)["captured"]
        if first:
            m_before = _gpu_used_mb()
        out, rep_diff = r.step(
            [cur], s_eff, 1, str(10 + bidx), compare_uncaptured=compare_each
        )
        if first:
            captures.append((s, s_eff, _gpu_used_mb() - m_before))
        per_bucket_steps[s_eff] = per_bucket_steps.get(s_eff, 0) + 1
        if rep_diff is not None:
            max_replay_diff = max(max_replay_diff, rep_diff)
        o_out = oracle.step([o_cur])
        max_oracle_diff = max(
            max_oracle_diff, float((out[-1].cpu() - o_out[-1]).abs().max())
        )
        cur = int(out[-1].argmax())
        o_cur = int(o_out[-1].argmax())
        stream.append(cur)
        o_stream.append(o_cur)
        if cur != o_cur:
            srt = np.sort(o_out[-1].numpy())
            argmax_diffs.append((s, cur, o_cur, float(srt[-1] - srt[-2])))
            o_cur = cur  # resync on the onnx stream (toy margin tie)
            oracle.k = [k[: r.length] for k in oracle.k]  # no-op, lengths equal
    print(
        f"[rollout] width-1 decode: {n_steps} steps " f"per_bucket={per_bucket_steps}"
    )
    print(f"[rollout] captures (step, S_eff, nvidia-smi delta MB): {captures}")
    if compare_each:
        print(
            f"[rollout] max |replay - uncaptured| over all steps: "
            f"{max_replay_diff:.3e}  (expect 0.0)"
        )
    else:
        print(
            "[rollout] pure-replay run (no per-step uncaptured compare); "
            "correctness rests on the oracle stream below"
        )
    print(f"[rollout] max |onnx - oracle| logits: {max_oracle_diff:.3e}")
    print(
        f"[rollout] argmax mismatches vs oracle: {len(argmax_diffs)} "
        f"{argmax_diffs[:5]}"
    )
    tok_id = "IDENTICAL" if stream == o_stream else "DIVERGED"
    print(f"[rollout] token stream vs oracle: {tok_id} ({len(stream)} tokens)")

    # --- width-bucket interleave with LIVE pad semantics: width-4 passes
    # carry only n_real=2 real rows (pad rows repeat the last real row,
    # scatter into slots beyond the committed length, outputs sliced off,
    # deltas not persisted — the production width-9 spec-bucket semantics),
    # validated per step against the oracle, which never pads anything.
    if variant != "static":
        print(
            "[rollout] width-interleave: reset cache, prefill, alternate "
            "width-1 / width-4(n_real=2) passes across buckets"
        )
        for i in range(N_LAYERS):
            r.k[i].zero_()
            r.v[i].zero_()
        r.length = 0
        oracle2 = Oracle()
        out, _ = r.step(prefill_rows, s_pre, PREFILL, "-1")
        o_out = oracle2.step(prefill_rows)
        cur = int(out[-1].argmax())
        o_cur = int(o_out[-1].argmax())
        assert cur == o_cur
        max_il_diff = 0.0
        max_il_oracle = 0.0
        il_argmax_mismatch = 0
        wb_captures = []
        for s in range(60):
            base = r.length
            width = 1 if s % 2 == 0 else WIDTH_BUCKET
            n_real = 1 if width == 1 else 2  # pad branch LIVE: n_real < width
            s_eff = min(b for b in buckets if b >= base + width)
            bidx = buckets.index(s_eff)
            gid = str(10 + bidx) if width == 1 else str(20 + bidx)
            first = not r.binding_for(s_eff, width)["captured"]
            if first:
                m_before = _gpu_used_mb()
            # rows: current token, then (for n_real=2) the oracle's argmax of
            # it — a draft-like continuation, content otherwise arbitrary.
            rows = [cur] * n_real
            out, rep_diff = r.step(
                rows, s_eff, width, gid, compare_uncaptured=compare_each
            )
            if first:
                wb_captures.append((s, s_eff, width, _gpu_used_mb() - m_before))
            if rep_diff is not None:
                max_il_diff = max(max_il_diff, rep_diff)
            o_out = oracle2.step(rows)  # oracle steps ONLY the real rows
            max_il_oracle = max(max_il_oracle, float((out.cpu() - o_out).abs().max()))
            if int(out[-1].argmax()) != int(o_out[-1].argmax()):
                il_argmax_mismatch += 1
            cur = int(out[-1].argmax())
        print(f"[rollout] interleave captures (step, S_eff, W, MB): " f"{wb_captures}")
        print(
            f"[rollout] interleave max |replay - uncaptured|: "
            f"{max_il_diff:.3e}  (expect 0.0; vacuous if compare_each off)"
        )
        print(
            f"[rollout] interleave max |onnx - oracle| (sliced real rows): "
            f"{max_il_oracle:.3e}; argmax mismatches: {il_argmax_mismatch}"
        )

    print(
        f"[rollout] total nvidia-smi delta since start: " f"{_gpu_used_mb() - mem0} MB"
    )

    # --- replay signature: per-run profile event counts
    prof = sess.end_profiling()
    events = json.loads(Path(prof).read_text())
    runs, cur_run = [], 0
    for e in events:
        if e.get("cat") == "Node":
            cur_run += 1
        if e.get("cat") == "Session" and e.get("name") == "model_run":
            runs.append(cur_run)
            cur_run = 0
    print(f"[rollout] node-event count per run (first 12): {runs[:12]}")
    print(f"[rollout] node-event count per run (last 12): {runs[-12:]}")
    print(f"[rollout] profile: {prof}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["build", "memcpy", "rollout"])
    ap.add_argument("--variant", default="dyn", choices=["static", "dyn", "slotids"])
    ap.add_argument(
        "--no-compare",
        action="store_true",
        help="pure replay stream (no per-step uncaptured "
        "double-run) — the production-shaped run pattern",
    )
    args = ap.parse_args()
    if args.exp == "build":
        exp_build()
    elif args.exp == "memcpy":
        exp_memcpy(args.variant)
    else:
        exp_rollout(args.variant, compare_each=not args.no_compare)


if __name__ == "__main__":
    main()
