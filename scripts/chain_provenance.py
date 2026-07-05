"""Pin every witness-chain node to a torchwright_doom source line, and
decompose each witness attention read into Q/K/V binding depths.

Productized merge of the paint-cascade session's P0/P1 scratch probes
(``paint_cascade_plan.md``, execution record). Runs
``scripts/critical_chain.py``'s exact pipeline (forward -> always-on
fusion -> lower -> ``build_graph_model`` -> layer bounds), plus:

* a ``Node.__init__`` patch recording each node's creation stack, with
  ``lower()``'s source->clone ``node_map`` inverted so scheduled clones
  attribute back to user code;
* the witness chain printed with per-node provenance AND each chain
  node's direct inputs (their layers and creation sites);
* for each Attn on the chain, the deepest *scheduled* dependency of
  each direct input — order ``query_in`` / ``key_in`` / ``value`` —
  reached through unscheduled glue (Concatenate, literals). The input
  whose subtree binds deepest is what places the read: query-bound
  reads serialize on their query; value-bound reads (same-pass
  publish->read) move up 1:1 as the published value's depth drops.

Two provenance caveats (measured in the paint session): an op wrapped
in an ``Assert`` can shadow the wrapped node's true creation site in
the inverted map (recover from its inputs), and nodes created inside
shared helpers attribute to the helper's call frame chain.

**Screen-env trap**: run under the production env or you measure the
60x50 hud-off graph:

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \\
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \\
    TORCHWRIGHT_DOOM_HUD=1 python -m scripts.chain_provenance
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---- provenance capture: patch BEFORE any node is created (the doom
# modules follow the no-import-time-nodes rule, but patching ahead of
# their import keeps that assumption out of this script) ----
from torchwright.graph import node as node_mod

_orig_init = node_mod.Node.__init__
_sites: dict[int, str] = {}  # id(node) -> "file:line (func)  <=  caller ..."


def _capturing_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    doom_frames = []
    fallback = None
    for fr in traceback.extract_stack()[:-1]:
        fn = fr.filename
        if "/torchwright_doom/" in fn and "/scripts/" not in fn:
            doom_frames.append(f"{os.path.basename(fn)}:{fr.lineno} ({fr.name})")
        elif "/torchwright/torchwright/" in fn:
            fallback = f"[tw] {os.path.basename(fn)}:{fr.lineno} ({fr.name})"
    if doom_frames:
        _sites[id(self)] = "  <=  ".join(reversed(doom_frames[-4:]))
    else:
        _sites[id(self)] = fallback or "<unknown>"


node_mod.Node.__init__ = _capturing_init  # type: ignore[method-assign]

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


def main():
    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=128, max_positions=65536, d_rot=64)
    nt = forward(emb, GraphPast(input_vec=emb, rope=rope))
    n_fused = fuse_consecutive_linears({nt}, verbose=False)
    lowered = lower(nt)
    out = lowered.output_node
    clone2src = {id(clone): src for src, clone in lowered.node_map.items()}
    gm = build_graph_model(out, None)
    es, _ = _compute_layer_bounds(gm, LEGACY_POLICY, True, max_layers=1 << 20)
    floor = max(es.values()) + 1
    print(f"fused={n_fused} floor={floor}")

    id2node = {n.node_id: n for n in gm.schedulable}
    preds = defaultdict(list)
    for u, v in gm.edges:
        preds[v.node_id].append(u.node_id)

    def label(n) -> str:
        return getattr(n, "name", "") or type(n).__name__

    def site(n) -> str:
        s = _sites.get(id(n))
        if s is not None:
            return s
        src = clone2src.get(id(n))
        if src is not None:
            return _sites.get(id(src), "<src not captured>")
        return "<no source mapping>"

    # Witness chain: from a deepest node, greedily follow the deepest
    # predecessor (same walk as critical_chain.py).
    cur = max(es, key=lambda i: es[i])
    chain = [cur]
    while preds[cur]:
        ps = [p for p in preds[cur] if p in es]
        if not ps:
            break
        cur = max(ps, key=lambda p: es[p])
        chain.append(cur)
    chain.reverse()

    print(f"\n=== witness chain with provenance ({len(chain)} nodes) ===")
    for i in chain:
        n = id2node.get(i)
        if n is None:
            continue
        ann = getattr(n, "annotation", "") or "<none>"
        print(f"\nL{es[i]:>2}  {label(n)}   [{ann}]")
        print(f"     created at: {site(n)}")
        for inp in getattr(n, "inputs", []) or []:
            in_layer = es.get(inp.node_id)
            in_layer_s = f"L{in_layer}" if in_layer is not None else "--"
            print(
                f"     <- {in_layer_s:>3} {label(inp)} "
                f"[{getattr(inp, 'annotation', '') or ''}] @ {site(inp)}"
            )

    def deepest_scheduled(root):
        """Max-es scheduled descendants reachable through unscheduled glue."""
        best = []  # (es, node)
        seen = set()
        stack = [root]
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            if n.node_id in es:
                best.append((es[n.node_id], n))
                continue  # scheduled: stop descending
            for inp in getattr(n, "inputs", []) or []:
                stack.append(inp)
        best.sort(key=lambda t: -t[0])
        return best

    roles = ["Q(query_in)", "K(key_in)", "V(value)"]
    print("\n=== per-Attn Q/K/V binding depths (witness chain) ===")
    for i in chain:
        n = id2node.get(i)
        if n is None or type(n).__name__ != "Attn":
            continue
        print(f"\n--- Attn at L{es[i]}  [{getattr(n, 'annotation', '')}] ---")
        print(f"    created: {site(n)}")
        for k, inp in enumerate(list(getattr(n, "inputs", []) or [])):
            role = roles[k] if k < len(roles) else f"input[{k}]"
            top = deepest_scheduled(inp)[:3]
            print(f"  {role}:")
            for e, d in top:
                print(f"     L{e:>2}  {label(d)}  @ {site(d)}")
            if not top:
                print("     (no scheduled deps — constants/embedding only)")


if __name__ == "__main__":
    main()
