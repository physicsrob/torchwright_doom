"""Attribute the compiled doom ``forward()`` graph's DEPTH (layers) and WIDTH
(residual peak) to subsystems.

Depth and width are properties of the *graph*, not the scene: the compiler
schedules the per-position graph topology, so the layer count and the residual
peak are fixed regardless of which fixture flows through. This tool answers
"what drives the 77 layers / the residual width, and how much is each
subsystem (E8 type-coding, the digit-quad VALUE emit, attention picks, the
geometry PWLs, the dispatch selects)?".

Usage (CPU, in-process):
    python -m scripts.analyze_forward_cost            # d=6400, heuristic
    D=4800 OPT=0 python -m scripts.analyze_forward_cost

It captures two things during one compile:
  * ``node -> layer`` via ``forward_compile(on_node_scheduled=...)`` — for the
    critical-path (depth) trace.
  * the residual peak + the live node-set at the peak via an ``allocate`` hook
    on ``ResidualStreamMap`` — for the width breakdown.
Both are then bucketed by subsystem from the node names.

This deliberately runs the compiler in-process: it measures compile-time
scheduling, which only exists during a compile — unlike inference, it cannot
be served by the cached ONNX artifact.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torchwright.compiler.export as _exp
import torchwright.compiler.forward.residual_map as _rm
from torchwright.ops.inout_nodes import create_pos_encoding
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

# --- subsystem classification (by node name; first match wins) --------------
_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("E8 type-code", ("emit_e8", "e8_", "type_code", "input_type")),
    (
        "digit-quad emit",
        (
            "emit_dq",
            "emit_raw",
            "floor_int",
            "in_range",
            "q_value",
            "_dq",
            "thermometer",
        ),
    ),
    ("emit other", ("emit_", "make_token", "derived_zero")),
    (
        "attention/pick",
        (
            "attn",
            "pick",
            "attend",
            "softmax",
            "_qk",
            "score",
            "most_recent",
            "argmax",
            "argmin",
            "mean_where",
        ),
    ),
    (
        "geometry PWL",
        (
            "mul",
            "multiply",
            "atan",
            "ray",
            "abs",
            "cos",
            "sin",
            "cross",
            "side",
            "signed",
            "wrap",
            "piecewise",
            "sqrt",
            "reciprocal",
            "scale",
            "proj",
        ),
    ),
    (
        "compare/clamp",
        ("compare", "clamp", "gt_", "_gt", "marker_present", "step", "bool"),
    ),
    ("select/dispatch", ("select", "cond_gate", "broadcast_select", "switch", "gate")),
    (
        "affine glue",
        (
            "add_const",
            "sub",
            "neg",
            "add",
            "linear",
            "concat",
            "split",
            "sum",
            "literal",
            "zero",
            "one_hot",
        ),
    ),
    ("embedding", ("embedding", "embed", "lifted")),
]


def bucket(name: str) -> str:
    low = (name or "").lower()
    for label, pats in _BUCKETS:
        if any(p in low for p in pats):
            return label
    return f"OTHER:{name.split('_')[0] if name else '?'}"


def _node_label(n) -> str:
    return getattr(n, "name", "") or type(n).__name__


def compile_capture(
    output_node,
    pos,
    d: int,
    d_head: int,
    optimize: int,
    d_hidden=None,
    run_optimize_graph: bool = False,
):
    """Compile once; return (compiled, node_to_layer, peak_width, peak_live).

    ``run_optimize_graph`` applies ``optimize_graph`` (``fuse_consecutive_linears``)
    in place BEFORE compiling — the pre-pass the compile pipeline never runs.
    ``d_hidden`` caps MLP weight memory (must be >= the widest single-layer MLP op).
    """
    n_fused = None
    if run_optimize_graph:
        from torchwright.graph.optimize import fuse_consecutive_linears

        n_fused = fuse_consecutive_linears({output_node}, verbose=False)

    node_to_layer: dict[int, int] = {}
    id_to_node: dict[int, object] = {}

    def _on_sched(node, layer):
        node_to_layer[node.node_id] = layer
        id_to_node[node.node_id] = node

    peak = {"used": 0, "live": []}
    _orig_alloc = _rm.ResidualStreamMap.allocate

    def _alloc(self, node):
        res = _orig_alloc(self, node)
        used = self.d - len(self._free)
        if used > peak["used"]:
            peak["used"] = used
            peak["live"] = [(_node_label(n), len(n)) for n in self._node_to_indices]
        return res

    _orig_fc = _exp.forward_compile

    def _fc(*a, **k):
        k.setdefault("optimize", optimize)
        k.setdefault("on_node_scheduled", _on_sched)
        return _orig_fc(*a, **k)

    _rm.ResidualStreamMap.allocate = _alloc
    _exp.forward_compile = _fc
    try:
        t0 = time.perf_counter()
        compiled = _exp.compile_headless(
            output_node,
            pos,
            d=d,
            d_head=d_head,
            max_layers=400,
            verbose=False,
            device="cpu",
            d_hidden=d_hidden,
        )
        t = time.perf_counter() - t0
    finally:
        _rm.ResidualStreamMap.allocate = _orig_alloc
        _exp.forward_compile = _orig_fc
    return compiled, node_to_layer, id_to_node, peak, t, n_fused


def schedule_only_capture(
    output_node,
    pos,
    d: int,
    d_head: int,
    run_optimize_graph: bool = False,
    max_layers: int = 600,
    d_hidden=None,
):
    """Memory-safe: run ONLY the heuristic scheduler (no weight tensors, no
    HeadlessTransformer) and return (n_layers, node_to_layer, id_to_node,
    peak_width, live_at_peak, n_fused).

    Replicates forward_compile's pre-loop setup, then calls the schedule-only
    warm-start. Peak residual width is computed from the per-node birth layer +
    reclaim (cancel) layer. This is the TRUE depth/width of the heuristic
    schedule at the given d, without ever allocating a d×d weight matrix.
    """
    from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
    from torchwright.compiler.forward.residual_map import ResidualStreamMap
    from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
    from torchwright.compiler.forward.compile import _run_heuristic_warm_start

    n_fused = None
    if run_optimize_graph:
        from torchwright.graph.optimize import fuse_consecutive_linears

        n_fused = fuse_consecutive_linears({output_node}, verbose=False)

    graph = GraphAnalyzer(output_node)
    output_node = graph.get_output_node()
    input_nodes = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]
    rmap = ResidualStreamMap(d)
    rmap.allocate(pos)
    rmap.mark_clean(rmap.get_indices(pos))
    for n in input_nodes:
        if n is pos:
            continue
        rmap.allocate(n)
        rmap.mark_clean(rmap.get_indices(n))
    computed = set(input_nodes)
    policy = SchedulingPolicy()

    # True residual peak: hook allocate during the warm-start (the schedule-only
    # _TrackingResidualStreamMap doesn't override allocate, so the base hook fires).
    peak = {"used": 0, "live": []}
    _orig_alloc = _rm.ResidualStreamMap.allocate

    def _alloc(self, node):
        res = _orig_alloc(self, node)
        used = self.d - len(self._free)
        if used > peak["used"]:
            peak["used"] = used
            peak["live"] = [(_node_label(n), len(n)) for n in self._node_to_indices]
        return res

    _rm.ResidualStreamMap.allocate = _alloc
    try:
        layers, _routing, cancel, n_layers = _run_heuristic_warm_start(
            graph=graph,
            d=d,
            d_head=d_head,
            pos_encoding=pos,
            d_hidden=(d_hidden if d_hidden else d),
            residual_map=rmap,
            computed=computed,
            clusters=None,
            admission_budget_fraction=0.4,
            policy=policy,
            overlay_pinned_inputs=set(),
            output_node=output_node,
            max_layers=max_layers,
        )
    finally:
        _rm.ResidualStreamMap.allocate = _orig_alloc
    if n_layers == 0:
        raise RuntimeError(f"heuristic deadlocked at d={d} (schedule-only)")

    id_to_node = {n.node_id: n for n in graph.get_all_nodes()}
    return n_layers, layers, id_to_node, peak["used"], peak["live"], n_fused, -1


def scheduled_preds(node, n2l, _memo):
    """Data-dependency predecessors that were scheduled (recurse transparently
    through Concatenate / Assert / other unscheduled intermediates)."""
    key = id(node)
    if key in _memo:
        return _memo[key]
    out = []
    for inp in getattr(node, "inputs", []) or []:
        if inp.node_id in n2l:
            out.append(inp)
        else:
            out.extend(scheduled_preds(inp, n2l, _memo))
    _memo[key] = out
    return out


def critical_path(output_node, n2l, id_to_node):
    """Trace the deepest dependency chain (the depth-determining critical path)."""
    memo: dict = {}
    # Seed from the deepest scheduled node reachable from the output.
    start = max(
        scheduled_preds(output_node, n2l, memo) or [output_node],
        key=lambda n: n2l.get(n.node_id, -1),
    )
    chain = [start]
    cur = start
    while True:
        preds = scheduled_preds(cur, n2l, memo)
        if not preds:
            break
        cur = max(preds, key=lambda n: n2l.get(n.node_id, -1))
        chain.append(cur)
    chain.reverse()
    return chain


def main():
    d = int(os.environ.get("D", "6400"))
    d_head = int(os.environ.get("D_HEAD", "160"))
    optimize = int(os.environ.get("OPT", "0"))

    emb = build_doom_embedding("token_ids")
    pos = create_pos_encoding()
    fanout_env = os.environ.get("FANOUT")
    if fanout_env is None:
        nt = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)
    else:
        # Rebuild forward() with a custom dispatch max_fanout to probe the
        # depth/width tradeoff of the type_switch fold (FANOUT=none for a full
        # tree reduction). Mirrors render_main.forward / dispatch_next_token.
        from torchwright_doom.protocol_tokens import ProtocolTokenView
        from torchwright_doom.scene_index import SceneIndex
        from torchwright_doom.render_main import (
            publish_runtime_protocols,
            build_branch_outputs,
            _distinct_head_pairs,
        )
        from torchwright_doom.emit import emit_derived_zero
        from torchwright_doom.std import concat as _concat, type_switch as _ts
        from torchwright_doom.past import PastHandleScope

        fanout = (
            None if fanout_env.lower() in ("none", "0", "full") else int(fanout_env)
        )
        gp = GraphPast(input_vec=emb, pos_encoding=pos)
        scene = SceneIndex.build(emb, gp, pos)
        scope = PastHandleScope(gp)
        inp = ProtocolTokenView(
            emb,
            scope.attend_to_offset(scope.input_type(), delta_pos=-1),
            scope.attend_to_offset(scope.input_type(), delta_pos=-2),
        )
        protocols = publish_runtime_protocols(emb, scope, inp, scene, pos)
        branches = build_branch_outputs(inp, protocols)
        head = _ts(*_distinct_head_pairs(inp, branches), max_fanout=fanout)
        nt = _concat(head, emit_derived_zero())
        print(f"[dispatch rebuilt with max_fanout={fanout}]")

    d_hidden = int(os.environ["DH"]) if os.environ.get("DH") else None
    run_og = os.environ.get("OPT_GRAPH", "0") == "1"
    sched_only = os.environ.get("SCHED_ONLY", "0") == "1"
    if sched_only:
        import time as _t

        t0 = _t.perf_counter()
        n_layers, n2l, id_to_node, peak_used, live, n_fused, pk_layer = (
            schedule_only_capture(
                nt, pos, d, d_head, run_optimize_graph=run_og, d_hidden=d_hidden
            )
        )
        t = _t.perf_counter() - t0
        peak = {"used": peak_used, "live": live}
        print(
            f"=== SCHEDULE-ONLY (no weights): d={d} d_head={d_head} "
            f"optimize_graph={run_og}(fused={n_fused}) "
            f"layers={n_layers} peak_width={peak_used}@L{pk_layer} t={t:.1f}s ==="
        )
    else:
        compiled, n2l, id_to_node, peak, t, n_fused = compile_capture(
            nt, pos, d, d_head, optimize, d_hidden=d_hidden, run_optimize_graph=run_og
        )
        n_layers = compiled._n_layers
        print(
            f"=== compiled: optimize={optimize} d={d} d_head={d_head} dh={d_hidden} "
            f"optimize_graph={run_og}(fused={n_fused}) "
            f"layers={n_layers} peak_width={peak['used']} compile={t:.1f}s ==="
        )

    # ---- DEPTH: critical path by subsystem ----
    path = critical_path(nt, n2l, id_to_node)
    print(
        f"\n--- critical path (depth driver): {len(path)} scheduled nodes "
        f"spanning layers 0..{n2l.get(path[-1].node_id, '?')} ---"
    )
    path_buckets = Counter(bucket(_node_label(n)) for n in path)
    for b, c in path_buckets.most_common():
        print(f"  {c:4d}  {b}")
    print("  (full critical-path chain):")
    for n in path:
        print(
            f"    L{n2l.get(n.node_id,'?'):>3}  {bucket(_node_label(n)):18s} {_node_label(n)}"
        )

    # ---- DEPTH: nodes-per-bucket and max-layer-per-bucket ----
    by_bucket_count: Counter = Counter()
    by_bucket_maxlayer: dict = defaultdict(int)
    for nid, layer in n2l.items():
        b = bucket(_node_label(id_to_node[nid]))
        by_bucket_count[b] += 1
        by_bucket_maxlayer[b] = max(by_bucket_maxlayer[b], layer)
    print(f"\n--- all {len(n2l)} scheduled nodes by subsystem ---")
    print(f"  {'count':>6} {'maxL':>5}  subsystem")
    for b, c in by_bucket_count.most_common():
        print(f"  {c:6d} {by_bucket_maxlayer[b]:5d}  {b}")

    # ---- WIDTH: peak live-set by subsystem ----
    width_by_bucket: Counter = Counter()
    for name, w in peak["live"]:
        width_by_bucket[bucket(name)] += w
    print(f"\n--- residual peak {peak['used']} cols, live-set by subsystem ---")
    for b, w in width_by_bucket.most_common():
        print(f"  {w:6d}  {b}")


if __name__ == "__main__":
    main()
