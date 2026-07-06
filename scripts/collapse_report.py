"""Univariate-collapse report for the production doom graph.

Builds the graph a committed config describes (screen env applied
*before* graph imports — the screen-env trap, see
``scripts/critical_chain.py``), runs ``lower()`` with the collapse pass
on at the production lane cap (``d_hidden // 4``), and prints:

* the pass's own collapse/decline log (every univariate subgraph:
  collapsed with lane counts, or the decline reason);
* a per-source table (name, type, annotation, value range, members,
  chain depth) for every subgraph at chain >= 2 — the working view for
  placing ``assert_integer`` at winner sites
  (docs/univariate_collapse_plan.md, rollout phase "doom").

GPU-free but heavy (full graph build + lower), so run it on Modal:

    make modal-run MODULE=scripts.collapse_report \\
        ARGS="--config configs/e1m1_lowres.yaml" CPU_ONLY=1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="configs/e1m1.yaml",
        help="committed render config (the production graph shape)",
    )
    ap.add_argument(
        "--lane-cap",
        type=int,
        default=None,
        help="collapse lane cap (default: the config's d_hidden//4 — "
        "forward_compile's own formula)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=40,
        help="rows in the per-source table (chain >= 2, deepest first)",
    )
    ap.add_argument(
        "--v2",
        action="store_true",
        help="run the v2 continuous-source analysis (report-only): "
        "composed-PL certificates, S1/S2 shape models, modeled layer "
        "floor, S2 residual-column liveness (Phase A of the v2 plan)",
    )
    args = ap.parse_args()

    # Screen env BEFORE any graph-module import, or this silently
    # reports the 60x50 hud-off graph.
    from torchwright_doom.inference.config import (
        apply_screen_env,
        load_render_config,
    )

    cfg = load_render_config(args.config)
    apply_screen_env(cfg)

    from torchwright.compiler.collapse import scalar_sources
    from torchwright.compiler.graph_clone import topological_order
    from torchwright.compiler.lower import lower
    from torchwright.graph import Add, Attn, Concatenate, Linear
    from torchwright.graph.ffn import FFN
    from torchwright_doom.inference.compiled_model import build_graph

    m = cfg.model
    lane_cap = args.lane_cap
    if lane_cap is None:
        lane_cap = (m.d_hidden if m.d_hidden else m.d) // 4

    print(
        f"config {args.config}: d={m.d} d_hidden={m.d_hidden} "
        f"d_head={m.d_head} d_rot={m.d_rot} scale={m.scale} "
        f"detail={m.detail} hud={m.hud} -> lane cap {lane_cap}"
    )
    out, _rope, _emb, _banks = build_graph(
        d_head=m.d_head,
        max_positions=m.max_seq_len,
        d_rot=m.d_rot,
        wad_path=cfg.wad,
    )

    lowered = lower(out, collapse_univariate=True, collapse_lane_cap=lane_cap)
    report = lowered.collapse_report
    print("\n=== collapse report ===")
    if report is None or not report.outcomes:
        print("(no univariate subgraphs found)")
    else:
        print(report.format())
        print(
            f"-> {report.n_collapsed} collapsed, "
            f"{len(report.outcomes) - report.n_collapsed} declined"
        )

    if args.v2:
        # Phase A of the v2 plan: certify the composed PL of every
        # subgraph v1 leaves (the texel chains), model S1/S2 shapes,
        # the layer floor, and the S2 stage-1 residual-column liveness.
        # Report-only; the lowered copy is throwaway (the floor model
        # rewires it in place).
        from torchwright.compiler.collapse_analysis import analyze_collapse_v2

        print("\n=== v2 analysis (continuous-source collapse, Phase A) ===")
        v2 = analyze_collapse_v2(lowered.output_node, lane_cap=lane_cap, verbose=True)
        print(v2.format())
        print(
            f"liveness check: {v2.total_stage1_cols} strict / "
            f"{v2.total_stage1_cols_banded} banded S2 stage-1 columns "
            f"on top of the existing residual peak at d={m.d} "
            f"(structural gate re-bracketed at d=4608 in the v1 rollout)"
        )
        detailed = [s for s in v2.subgraphs if s.members]
        if detailed:
            print("--- member detail (subgraphs with synthesized members) ---")
            for s in detailed:
                print(s.format_detail())

    # Per-source table on the PRE-collapse lowering (the sites where an
    # assert_integer would go live in the source graph).
    pre = lower(out)
    order = topological_order(pre.output_node)
    src = scalar_sources(order)
    costly = (Attn, FFN, Linear, Add)
    by_src: dict = {}
    for n in order:
        s = src[n]
        if s is not None and s is not n:
            by_src.setdefault(s, []).append(n)

    # Effective consumers (through virtual Concatenates) — for boundary
    # member identification.
    from collections import Counter

    direct: dict = {n: [] for n in order}
    for n in order:
        for u in n.inputs:
            direct[u].append(n)
    eff: dict = {}
    for n in reversed(order):
        acc = []
        for c in direct[n]:
            if isinstance(c, Concatenate):
                acc.extend(eff[c])
            else:
                acc.append(c)
        eff[n] = acc

    def _ann(n) -> str:
        return "/".join(((getattr(n, "annotation", None) or "-").split("/"))[:4])

    from typing import Any

    rows: list[dict[str, Any]] = []
    for s, members in by_src.items():
        mset = set(members)
        depth: dict = {}
        for n in order:
            if n not in mset:
                continue
            base = max((depth.get(u, 0) for u in n.inputs), default=0)
            depth[n] = base + (1 if isinstance(n, costly) else 0)
        chain = max(depth.values(), default=0)
        if chain < 2:
            continue
        vr = getattr(getattr(s, "value_type", None), "value_range", None)
        rng = f"[{vr.lo:.6g}, {vr.hi:.6g}]" if vr is not None else "?"
        boundary = [
            m
            for m in members
            if not isinstance(m, Concatenate) and any(c not in mset for c in eff[m])
        ]
        rows.append(
            {
                "name": s.name or type(s).__name__,
                "type": type(s).__name__,
                "ann": _ann(s),
                "range": rng,
                "members": len(members),
                "chain": chain,
                "integer_claim": bool(getattr(s, "integer_claim", False)),
                "ops": Counter(
                    m.name or type(m).__name__
                    for m in members
                    if not isinstance(m, Concatenate)
                ),
                "boundary": Counter(
                    f"{m.name or type(m).__name__}@{_ann(m)}" for m in boundary
                ),
            }
        )
    rows.sort(key=lambda r: r["chain"], reverse=True)
    print(f"\n=== univariate sources at chain >= 2 (top {args.top}) ===")
    for r in rows[: args.top]:
        claim = "claimed" if r["integer_claim"] else "no-claim"
        print(
            f"  chain {r['chain']:>2}  {r['members']:>3} members  {claim:>8}  "
            f"{r['name']} ({r['type']})  range {r['range']}  ann {r['ann']}"
        )
        top_ops = ", ".join(f"{k}x{v}" for k, v in r["ops"].most_common(6))
        print(f"        ops: {top_ops}")
        top_b = "; ".join(f"{k}x{v}" for k, v in r["boundary"].most_common(4))
        print(f"        boundary: {top_b}")
    if len(rows) > args.top:
        print(f"  ... {len(rows) - args.top} more")


if __name__ == "__main__":
    main()
