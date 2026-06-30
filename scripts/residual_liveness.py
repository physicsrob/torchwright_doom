"""Layer-by-layer residual-stream liveness for the doom ``forward()`` graph.

Answers "what is alive on the residual stream at each compiled layer, and how
wide is it?" — the transparency behind the residual-bound / min-d question. The
residual width (``d`` / d_model) carries node *outputs* between layers; this is
distinct from the MLP-hidden ReLU width (see ``audit_relu.py``).

Mechanism (memory-safe, no weights): runs the heuristic scheduler's schedule-only
warm-start (same path ``analyze_forward_cost`` uses). The scheduler advances one
layer at a time (compile.py ``for hi in range(max_layers)``) and the tracking
residual-map carries ``current_layer``; we hook ``ResidualStreamMap.allocate`` to
snapshot the live set (``_node_to_indices``) at the moment of peak occupancy
*within each layer*, tagged by that layer. So the per-layer live set is the
scheduler's own bookkeeping, not a reconstruction.

At an unconstrained ``d`` (default 12000) this shows the graph's intrinsic
simultaneous liveness; at a constrained ``d`` near the deadlock floor it shows
what the scheduler is forced to hold co-resident (why it can't go narrower).

Usage:
    python -m scripts.residual_liveness                 # d=12000 (unconstrained)
    D=3000 python -m scripts.residual_liveness          # near the deadlock floor
    python -m scripts.residual_liveness --top 12        # widest live nodes per layer
    python -m scripts.residual_liveness --peak-detail 40
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

import torchwright.compiler.forward.residual_map as _rm
from torchwright.compiler.forward.compile import _run_heuristic_warm_start
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.graph.misc import LiteralValue
from torchwright.graph.node import reserve_node_id_above
from torchwright.ops.inout_nodes import create_rope_config
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

from scripts.analyze_forward_cost import bucket, _node_label


def capture_per_layer_liveness(
    output_node, rope, d, d_head, d_hidden=None, max_layers=600
):
    """Run the schedule-only warm-start; return {layer: (peak_width, live)} where
    ``live`` is the list of (name, width) resident at that layer's peak
    occupancy, plus n_layers. Hooks ``allocate`` to read the tracking map's
    ``current_layer`` (set per layer in compile.py)."""
    graph = GraphAnalyzer(output_node)
    output_node = graph.get_output_node()
    input_nodes = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]
    rmap = ResidualStreamMap(d)
    # Mirror forward_compile's RoPE-era residual seed (torchwright compile.py):
    # position is a rotation inside attention, not a residual node, so instead of
    # allocating a pos column we reserve the never-freed const-1 column the Δ=0
    # self-match heads read. The scheduler's position argument threads through as
    # None (RoPE reads no position substrate from the residual stream).
    reserve_node_id_above(graph.get_all_nodes())
    const_one = LiteralValue(torch.ones(1), name="rope_self_match_const_one")
    rmap.allocate(const_one)
    rmap.mark_clean(rmap.get_indices(const_one))
    for n in input_nodes:
        rmap.allocate(n)
        rmap.mark_clean(rmap.get_indices(n))
    computed = set(input_nodes)
    policy = SchedulingPolicy()

    per_layer: dict[int, tuple[int, list]] = {}
    _orig_alloc = _rm.ResidualStreamMap.allocate

    def _alloc(self, node):
        res = _orig_alloc(self, node)
        layer = int(getattr(self, "current_layer", 0))
        used = self.d - len(self._free)
        best = per_layer.get(layer)
        if best is None or used > best[0]:
            per_layer[layer] = (
                used,
                [(_node_label(n), len(n)) for n in self._node_to_indices],
            )
        return res

    _rm.ResidualStreamMap.allocate = _alloc
    try:
        layers, _routing, _cancel, n_layers = _run_heuristic_warm_start(
            graph=graph,
            d=d,
            d_head=d_head,
            pos_encoding=None,
            d_hidden=(d_hidden if d_hidden else d),
            residual_map=rmap,
            computed=computed,
            clusters=None,
            admission_budget_fraction=0.4,
            policy=policy,
            output_node=output_node,
            max_layers=max_layers,
        )
    finally:
        _rm.ResidualStreamMap.allocate = _orig_alloc
    if n_layers == 0:
        raise RuntimeError(f"heuristic deadlocked at d={d} (schedule-only)")
    return per_layer, n_layers


def _print_layer_line(layer: int, width: int, live: list, top: int) -> None:
    widest = sorted(live, key=lambda nw: nw[1], reverse=True)[:top]
    tiny = sum(1 for _n, w in live if w <= 4)
    tag = "  ".join(f"{nm}({w})" for nm, w in widest)
    print(
        f"  L{layer:>3} {width:6d} cols  {len(live):4d} nodes "
        f"({tiny:3d}≤4w) | {tag}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8, help="widest live nodes per layer")
    ap.add_argument(
        "--peak-detail",
        type=int,
        default=30,
        help="widest live nodes to list at the peak layer",
    )
    args = ap.parse_args()

    d = int(os.environ.get("D", "12000"))
    d_head = int(os.environ.get("D_HEAD", "160"))
    d_hidden = int(os.environ["DH"]) if os.environ.get("DH") else None

    emb = build_doom_embedding("token_ids")
    # rope d_head MUST match the scheduled d_head (d_rot = d_head // 2, the
    # production 128/64 ratio).
    rope = create_rope_config(d_head=d_head, max_positions=65536, d_rot=d_head // 2)
    nt = forward(emb, GraphPast(input_vec=emb, rope=rope))

    per_layer, n_layers = capture_per_layer_liveness(nt, rope, d, d_head, d_hidden)
    peak_layer = max(per_layer, key=lambda L: per_layer[L][0])
    peak_w, peak_live = per_layer[peak_layer]
    print(
        f"residual liveness: d={d} d_head={d_head} dh={d_hidden or d} "
        f"layers={n_layers} | peak {peak_w} cols @ L{peak_layer}"
    )

    print(f"\n=== per-layer residual occupancy (widest {args.top} live nodes) ===")
    for layer in sorted(per_layer):
        w, live = per_layer[layer]
        _print_layer_line(layer, w, live, args.top)

    # Peak layer: who is co-resident, by subsystem and by individual node.
    by_bucket: Counter = Counter()
    for nm, w in peak_live:
        by_bucket[bucket(nm)] += w
    print(
        f"\n=== PEAK layer L{peak_layer}: {peak_w} cols, {len(peak_live)} live "
        f"nodes — by subsystem ==="
    )
    for b, w in by_bucket.most_common():
        print(f"  {w:6d} {100*w/peak_w:5.1f}%  {b}")

    # Width distribution of the co-resident set: few-wide vs many-narrow?
    edges = [(1, 4), (5, 16), (17, 64), (65, 256), (257, 1024), (1025, 1 << 30)]
    labels = ["1-4", "5-16", "17-64", "65-256", "257-1024", "1025+"]
    print(f"\n=== PEAK layer L{peak_layer}: width distribution of live nodes ===")
    print(f"  {'node width':>10} {'cols':>7} {'%peak':>6} {'nodes':>6}")
    for (lo, hi), lab in zip(edges, labels):
        cols = sum(w for _n, w in peak_live if lo <= w <= hi)
        cnt = sum(1 for _n, w in peak_live if lo <= w <= hi)
        if cnt:
            print(f"  {lab:>10} {cols:7d} {100*cols/peak_w:5.1f}% {cnt:6d}")

    print(f"\n=== PEAK layer L{peak_layer}: widest {args.peak_detail} live nodes ===")
    for nm, w in sorted(peak_live, key=lambda nw: nw[1], reverse=True)[
        : args.peak_detail
    ]:
        print(f"  {w:6d}  {bucket(nm):18s} {nm}")


if __name__ == "__main__":
    main()
