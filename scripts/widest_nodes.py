"""Enumerate the WIDEST nodes in the doom ``forward()`` graph by ``d_output``.

Node width (``d_output`` = number of residual columns a node's output occupies)
is the per-node analogue of the residual peak that ``residual_liveness.py`` and
``analyze_forward_cost.py`` measure. This tool answers the simpler, intrinsic
question: "which individual graph nodes are the widest, what are they, and which
subsystem do they belong to?" — independent of scheduling/liveness.

Pure graph construction: no compile, no weights, no GPU, scene-independent
(topology is fixed regardless of fixture). Walks ``.inputs`` from the output
node to collect every reachable node, then ranks by ``d_output``.

Usage:
    python -m scripts.widest_nodes               # top 40 widest nodes
    python -m scripts.widest_nodes --top 80
    python -m scripts.widest_nodes --by-annotation
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from torchwright.ops.inout_nodes import create_pos_encoding
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward


def _build():
    emb = build_doom_embedding("token_ids")
    pos = create_pos_encoding()
    out = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)
    return out


def _collect(output_node):
    """BFS over ``.inputs`` from the output node; return all reachable nodes."""
    seen: dict[int, object] = {}
    stack = [output_node]
    while stack:
        n = stack.pop()
        if n.node_id in seen:
            continue
        seen[n.node_id] = n
        for inp in n.inputs:
            if inp.node_id not in seen:
                stack.append(inp)
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument(
        "--by-annotation",
        action="store_true",
        help="also print total width grouped by annotation subtree",
    )
    args = ap.parse_args()

    out = _build()
    nodes = _collect(out)

    total_nodes = len(nodes)
    total_width = sum(n.d_output for n in nodes)
    print(f"reachable nodes: {total_nodes}   sum(d_output): {total_width}")
    print(f"output node: {out!r}\n")

    ranked = sorted(nodes, key=lambda n: n.d_output, reverse=True)

    print(f"=== top {args.top} widest nodes (by d_output) ===")
    print(f"{'d_output':>9}  {'type':<22} {'annotation':<40} name")
    for n in ranked[: args.top]:
        ann = (n.annotation or "")[-40:]
        nm = (n.name or "")[:50]
        print(f"{n.d_output:>9}  {n.node_type():<22} {ann:<40} {nm}")

    # distribution of widths
    print("\n=== width distribution (count of nodes >= threshold) ===")
    for thr in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        c = sum(1 for n in nodes if n.d_output >= thr)
        if c:
            print(f"  >= {thr:>5}: {c:>6} nodes")

    # by node_type
    print("\n=== total residual width by node type (top 15) ===")
    by_type_w: Counter = Counter()
    by_type_c: Counter = Counter()
    for n in nodes:
        by_type_w[n.node_type()] += n.d_output
        by_type_c[n.node_type()] += 1
    for t, w in by_type_w.most_common(15):
        print(f"  {w:>9}  ({by_type_c[t]:>5} nodes)  {t}")

    # dedup widest by name (distinct width-drivers + how many copies)
    print("\n=== distinct width-drivers (by name), widest first (top 30) ===")
    by_name: dict[tuple, list] = defaultdict(list)
    for n in nodes:
        by_name[(n.name or f"<{n.node_type()}>", n.node_type(), n.d_output)].append(n)
    rows = sorted(by_name.items(), key=lambda kv: (kv[0][2], len(kv[1])), reverse=True)
    print(f"{'d_output':>9}  {'count':>5}  {'type':<14} name")
    for (nm, typ, w), insts in rows[:30]:
        print(f"{w:>9}  {len(insts):>5}  {typ:<14} {nm}")

    # split: ReLU nodes are MLP-hidden; everything else is residual-resident
    print("\n=== widest MLP-hidden (ReLU nodes), by name (top 15) ===")
    relu_rows = [(k, v) for k, v in by_name.items() if k[1] == "ReLU"]
    relu_rows.sort(key=lambda kv: kv[0][2], reverse=True)
    for (nm, typ, w), insts in relu_rows[:15]:
        print(f"{w:>9}  x{len(insts):<4} {nm}")

    print("\n=== widest NON-ReLU (residual-resident candidates), by name (top 20) ===")
    other_rows = [(k, v) for k, v in by_name.items() if k[1] != "ReLU"]
    other_rows.sort(key=lambda kv: kv[0][2], reverse=True)
    for (nm, typ, w), insts in other_rows[:20]:
        print(f"{w:>9}  x{len(insts):<4} {typ:<14} {nm}")

    if args.by_annotation:
        print("\n=== total width by annotation prefix (top-level, top 25) ===")
        by_ann_w: Counter = Counter()
        by_ann_c: Counter = Counter()
        for n in nodes:
            top = (n.annotation or "<none>").split("/")[0]
            by_ann_w[top] += n.d_output
            by_ann_c[top] += 1
        for a, w in by_ann_w.most_common(25):
            print(f"  {w:>9}  ({by_ann_c[a]:>5} nodes)  {a}")


if __name__ == "__main__":
    main()
