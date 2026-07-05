"""Attribute the production DAG depth floor to actual node chains + slack.

Mirrors ``critical_path_layers``' model exactly (the production
always-on fusion pre-pass, lower(), ``build_graph_model``, mode-aware
layer bounds), then reports:

* the **zero-slack set** — with ``max_layers`` pinned to the floor,
  nodes whose earliest and latest feasible layers coincide are exactly
  the union of ALL critical chains — bucketed by name / annotation and
  laid out per layer;
* a **per-site slack table** for the wide op families (a floor/table
  rewrite that adds k sublayers to its own chain is depth-free only
  where slack >= k);
* one **witness chain** traced back from a deepest node.

Depth semantics note: the CP-SAT optimum EQUALS this floor on the DOOM
graph at production width (measured 49 = 49; see
``swiglu_opportunities_findings.md``), so "on the zero-slack set" means
"a layer saved here is a compiled layer saved".

**Screen-env trap**: run under the production env or you measure the
60x50 hud-off graph:

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \\
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \\
    TORCHWRIGHT_DOOM_HUD=1 python -m scripts.critical_chain
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from torchwright.ops.inout_nodes import create_rope_config
from torchwright.graph.optimize import fuse_consecutive_linears
from torchwright.compiler.lower import lower
from torchwright.compiler.forward.cpsat_scheduler import (
    build_graph_model,
    _compute_layer_bounds,
    LEGACY_POLICY,
)
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

# Families whose per-site slack the width/depth plans track.
_SLACK_FAMILIES = (
    "floor_int",
    "table_lookup_2d",
    "colormap_row",
    "in_range",
    "sawtooth",
    "ray_",
)


def main():
    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=128, max_positions=65536, d_rot=64)
    nt = forward(emb, GraphPast(input_vec=emb, rope=rope))
    n_fused = fuse_consecutive_linears({nt}, verbose=False)
    out = lower(nt).output_node
    gm = build_graph_model(out, None)
    es, _ = _compute_layer_bounds(gm, LEGACY_POLICY, True, max_layers=1 << 20)
    floor = max(es.values()) + 1
    print(f"fused={n_fused} floor={floor}")

    es2, ls2 = _compute_layer_bounds(gm, LEGACY_POLICY, True, max_layers=floor)
    id2node = {n.node_id: n for n in gm.schedulable}
    crit = [i for i in es2 if es2[i] == ls2[i]]
    print(f"zero-slack nodes: {len(crit)} of {len(es2)}")

    def label(n) -> str:
        return getattr(n, "name", "") or type(n).__name__

    def ann(n) -> str:
        return "/".join((getattr(n, "annotation", "") or "<none>").split("/")[:2])

    by_name: Counter = Counter()
    by_ann: Counter = Counter()
    for i in crit:
        n = id2node[i]
        by_name[label(n)] += 1
        by_ann[ann(n)] += 1
    print("\n=== zero-slack nodes by name (top 25) ===")
    for nm, c in by_name.most_common(25):
        print(f"  {c:>5}  {nm}")
    print("\n=== zero-slack nodes by annotation (depth 2, top 25) ===")
    for a, c in by_ann.most_common(25):
        print(f"  {c:>5}  {a}")

    per_layer = defaultdict(list)
    for i in crit:
        per_layer[es2[i]].append(i)
    print("\n=== critical nodes per layer (layer: count, sample names) ===")
    for layer in sorted(per_layer):
        ids = per_layer[layer]
        names = Counter(label(id2node[i]) for i in ids)
        sample = ", ".join(f"{nm} x{c}" for nm, c in names.most_common(4))
        print(f"  L{layer:>2}: {len(ids):>5}  {sample}")

    print("\n=== slack (latest - earliest) for the wide op families, by site ===")
    fam_rows = defaultdict(list)
    for i in es2:
        n = id2node[i]
        nm = label(n)
        if any(p in nm for p in _SLACK_FAMILIES):
            fam_rows[ann(n)].append(ls2[i] - es2[i])
    for a in sorted(fam_rows, key=lambda a: min(fam_rows[a])):
        sl = sorted(fam_rows[a])
        print(
            f"  min={sl[0]:>3} p50={sl[len(sl)//2]:>3} max={sl[-1]:>3} "
            f"n={len(sl):>4}  {a}"
        )

    preds = defaultdict(list)
    for u, v in gm.edges:
        preds[v.node_id].append(u.node_id)
    cur = max(es, key=lambda i: es[i])
    chain = [cur]
    while preds[cur]:
        ps = [p for p in preds[cur] if p in es]
        if not ps:
            break
        cur = max(ps, key=lambda p: es[p])
        chain.append(cur)
    chain.reverse()
    print(f"\n=== one witness chain ({len(chain)} nodes) ===")
    for i in chain:
        n = id2node.get(i)
        if n is None:
            continue
        print(f"  L{es[i]:>2}  {ann(n):<36} {label(n)}")


if __name__ == "__main__":
    main()
