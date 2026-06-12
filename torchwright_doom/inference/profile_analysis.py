"""Parse an ONNX Runtime profile trace and report the decode-pass cost split.

The CUDA-graph work needs three things off a profiled decode (phase 1):

  1. the exact Memcpy nodes ORT inserts (the CPU<->GPU copies that block
     CUDA-graph capture — ORT prints "N Memcpy nodes added ... unable to run
     CUDA graph"),
  2. the dispatch-vs-compute split (how much per-pass wall time is kernel
     launch / scheduling overhead vs actual GPU compute), and
  3. the KV-bandwidth share (the attention ops whose cost scales with the
     cached sequence length).

ORT writes a chrome-tracing JSON array (one event per element, ``dur`` in
microseconds).  Each executed node emits a ``<node>_kernel_time`` event whose
``args`` carry ``op_name`` and ``provider``; each inference call is closed by a
``cat="Session"`` ``model_run`` event whose ``dur`` is the wall time of that
run.  We bucket node events into runs using those ``model_run`` delimiters,
treat the single largest run as prefill, and aggregate the rest as decode.

Run standalone on a downloaded trace::

    python -m torchwright_doom.inference.profile_analysis out/render/ort_profile_*.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Per-layer node-name prefixes (see torchwright/compiler/export.py
# _emit_cached_layer_nodes).  Attention nodes carry ``l<i>_`` names; the subset
# that scales with the cached sequence length (the KV-bandwidth component) is
# the full-K/V build + the QK^T / softmax / weights*V matmuls.
_KV_SCALING = re.compile(
    r"^l\d+_(K_full|V_full|K_T|logits|logits_masked|weights|ctx|"
    r"past_K_hm|past_V_hm|K_new|V_new)\b"
)
_ATTN = re.compile(
    r"^l\d+_(Q_flat|Q_sm|Q|K_flat|V_flat|K_new|V_new|past_K_hm|past_V_hm|"
    r"K_full|V_full|K_T|logits|logits_masked|weights|ctx|ctx_t|ctx_flat|"
    r"attn_sum|res_attn)\b"
)
_FFN = re.compile(r"^l\d+_(l1_m|l1_b|l1_r|l2_m|l2_b|res_ffn|res_mlp)\b")
# delta_K_i / delta_V_i are the KV projections written out per layer.
_DELTA = re.compile(r"^delta_[KV]_\d+\b")

_MEMCPY_OPS = {"MemcpyToHost", "MemcpyFromHost", "MemcpyHostToDevice", "Memcpy"}


@dataclass
class NodeEvent:
    name: str
    op: str
    provider: str
    dur_us: float


@dataclass
class Run:
    wall_us: float
    nodes: list[NodeEvent] = field(default_factory=list)

    @property
    def kernel_us(self) -> float:
        return sum(n.dur_us for n in self.nodes)


def _provider_short(p: str) -> str:
    return p.replace("ExecutionProvider", "") or "?"


def _parse_runs(events: list[dict]) -> list[Run]:
    """Bucket node ``_kernel_time`` events into runs delimited by ``model_run``."""
    runs: list[Run] = []
    pending: list[NodeEvent] = []
    for ev in events:
        cat = ev.get("cat")
        name = ev.get("name", "")
        if cat == "Node" and name.endswith("_kernel_time"):
            args = ev.get("args", {}) or {}
            pending.append(
                NodeEvent(
                    name=name[: -len("_kernel_time")],
                    op=args.get("op_name", "?"),
                    provider=args.get("provider", "?"),
                    dur_us=float(ev.get("dur", 0) or 0),
                )
            )
        elif cat == "Session" and name == "model_run":
            runs.append(Run(wall_us=float(ev.get("dur", 0) or 0), nodes=pending))
            pending = []
    if pending:  # trailing nodes with no closing model_run (shouldn't happen)
        runs.append(Run(wall_us=sum(n.dur_us for n in pending), nodes=pending))
    return runs


def _bucket(name: str, op: str, provider: str) -> str:
    if op in _MEMCPY_OPS:
        return "memcpy"
    if provider.startswith("CPU"):
        return "cpu_control"
    if _FFN.match(name):
        return "ffn"
    if _ATTN.match(name) or _DELTA.match(name):
        return "attention"
    return "embed_other"  # token embed, unembed, residual adds, mask/pos on GPU


def summarize_profile(
    json_path: str | Path,
    *,
    max_top_ops: int = 18,
) -> str:
    path = Path(json_path)
    events = json.loads(path.read_text())
    if not isinstance(events, list):
        events = events.get("traceEvents", [])  # some ORT builds wrap it
    runs = _parse_runs(events)
    if not runs:
        return f"[profile] no model_run events found in {path}"

    # Largest run by wall time = prefill; the rest are decode passes.
    prefill_idx = max(range(len(runs)), key=lambda i: runs[i].wall_us)
    decode = [r for i, r in enumerate(runs) if i != prefill_idx]
    if not decode:  # only one run profiled — analyze it directly
        decode = runs

    n = len(decode)
    wall = sum(r.wall_us for r in decode)
    kernel = sum(r.kernel_us for r in decode)

    # Aggregates over the decode passes.
    by_op: dict[str, list[float]] = {}  # op -> [count, dur_us]
    by_provider: dict[str, float] = {}
    by_bucket: dict[str, float] = {}
    kv_scaling_us = 0.0
    memcpy: dict[tuple[str, str], list[float]] = {}  # (name, op) -> [count, dur]
    for r in decode:
        for nd in r.nodes:
            o = by_op.setdefault(nd.op, [0.0, 0.0])
            o[0] += 1
            o[1] += nd.dur_us
            by_provider[_provider_short(nd.provider)] = (
                by_provider.get(_provider_short(nd.provider), 0.0) + nd.dur_us
            )
            b = _bucket(nd.name, nd.op, nd.provider)
            by_bucket[b] = by_bucket.get(b, 0.0) + nd.dur_us
            if _KV_SCALING.match(nd.name):
                kv_scaling_us += nd.dur_us
            if nd.op in _MEMCPY_OPS:
                m = memcpy.setdefault((nd.name, nd.op), [0.0, 0.0])
                m[0] += 1
                m[1] += nd.dur_us

    def ms(us: float) -> float:
        return us / 1000.0

    def per_pass(us: float) -> float:
        return us / n / 1000.0  # ms/pass

    lines: list[str] = []
    lines.append(f"=== ORT profile: {path.name} ===")
    lines.append(
        f"runs: {len(runs)} total | prefill=run#{prefill_idx} "
        f"({ms(runs[prefill_idx].wall_us):.1f} ms wall) | decode passes={n}"
    )
    lines.append("")
    lines.append("--- per decode pass (avg) ---")
    lines.append(f"  wall:            {per_pass(wall):8.3f} ms")
    lines.append(
        f"  GPU kernel time: {per_pass(kernel):8.3f} ms "
        f"({100 * kernel / wall:5.1f}% of wall)"
    )
    dispatch = wall - kernel
    lines.append(
        f"  dispatch/sched:  {per_pass(dispatch):8.3f} ms "
        f"({100 * dispatch / wall:5.1f}% of wall)   <- launch overhead, the CUDA-graph target"
    )
    lines.append("")
    lines.append("--- kernel time by provider (per pass) ---")
    for prov, us in sorted(by_provider.items(), key=lambda kv: -kv[1]):
        lines.append(
            f"  {prov:12s} {per_pass(us):8.3f} ms ({100 * us / kernel:5.1f}% of kernel)"
        )
    lines.append("")
    lines.append("--- kernel time by stage (per pass) ---")
    bucket_order = ["attention", "ffn", "embed_other", "cpu_control", "memcpy"]
    for b in bucket_order:
        us = by_bucket.get(b, 0.0)
        if us:
            lines.append(
                f"  {b:12s} {per_pass(us):8.3f} ms ({100 * us / kernel:5.1f}% of kernel)"
            )
    lines.append(
        f"  (of attention, KV-bandwidth-scaling ops: {per_pass(kv_scaling_us):.3f} "
        f"ms/pass, {100 * kv_scaling_us / kernel:.1f}% of kernel)"
    )
    lines.append("")
    lines.append("--- Memcpy nodes (the CUDA-graph blockers) ---")
    if memcpy:
        for (name, op), (cnt, us) in sorted(memcpy.items(), key=lambda kv: -kv[1][1]):
            lines.append(
                f"  {op:16s} {name:40s} x{int(cnt / n):<3d}/pass  "
                f"{per_pass(us):.4f} ms/pass"
            )
    else:
        lines.append(
            "  (none found in trace — see the session-creation log for "
            "the 'N Memcpy nodes added' message)"
        )
    lines.append("")
    lines.append(f"--- top {max_top_ops} op types by kernel time (per pass) ---")
    for op, (cnt, us) in sorted(by_op.items(), key=lambda kv: -kv[1][1])[:max_top_ops]:
        lines.append(
            f"  {op:18s} {per_pass(us):8.3f} ms  (x{int(cnt / n)}/pass, "
            f"{100 * us / kernel:4.1f}%)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "usage: python -m torchwright_doom.inference.profile_analysis "
            "<ort_profile.json>"
        )
        return 2
    print(summarize_profile(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
