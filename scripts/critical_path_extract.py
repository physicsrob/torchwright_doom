"""Extract the DEPTH FLOOR and the actual mode-aware critical path of the
PRODUCTION doom forward graph.

Why this exists: production depth == ``critical_path_layers`` + {0,1} (the
CP-SAT optimum equals the dependency-DAG floor when the residual stream has
slack, which it does at d=8192).  So the only way to make the model shallower
is to shorten the *longest mode-aware path* through the post-fusion graph.
This tool reconstructs that exact path so we can see WHICH ops/subsystems make
up the floor and where the reduction opportunities are.

It is faithful to production: it builds the graph through
``inference.compiled_model.build_graph`` (which passes ``asset_index``, unlike
``analyze_forward_cost``) and applies the same width-safe linear-fusion gate
(``skip_relu_ejecting=True``).  Width is irrelevant here — this is a pure DAG
property, memory-safe, CPU, seconds.

Env:
  FUSE=1|0          apply the width-safe fusion gate (default 1, = production)
  COLLAPSE_SCAN=1   print per-subsystem / per-op-type savings ceilings instead
                    of the critical-path dump
  TOPK=N            how many of the deepest es-levels to enumerate (default 9)
  Scale envs (set to match production 320x200 low-detail):
    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low
    TORCHWRIGHT_DOOM_HUD=1

Pin PYTHONHASHSEED=0 for a reproducible floor (the fusion gate's pair
selection iterates a hash-ordered set).

Usage:
  TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \
  TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \
  TORCHWRIGHT_DOOM_HUD=1 uv run python -m scripts.critical_path_extract
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

# Import the subsystem bucketer from the sibling cost script (D8: reuse).
from scripts.analyze_forward_cost import bucket, _node_label  # noqa: E402

from torchwright.compiler.forward.cpsat_scheduler import (  # noqa: E402
    ATTN,
    MLP,
    build_graph_model,
    is_flex,
    routing,
)
from torchwright.compiler.forward.scheduling_policy import LEGACY_POLICY  # noqa: E402


def _modes(n, gm, flex_routing=True, usable_slots=None):
    if flex_routing and is_flex(n, gm):
        return (ATTN, MLP)
    # usable_slots is only read for a standalone Linear, and under
    # flex_routing=True those took the branch above; routing() raises rather
    # than guess if that ever stops being true.
    return (routing(n, gm, LEGACY_POLICY, usable_slots),)


def _gap(a, b):
    return 0 if (a == ATTN and b == MLP) else 1


def per_mode_bounds(gm, max_layers=1 << 20):
    """Re-run the mode-aware longest-path propagation, KEEPING per-mode dicts
    so we can reconstruct the tight path.  Mirrors
    cpsat_scheduler._compute_layer_bounds exactly."""
    input_ids = {n.node_id for n in gm.input_nodes}
    node_modes = {n.node_id: _modes(n, gm) for n in gm.schedulable}
    edges = []
    for u, v in gm.edges:
        if u.node_id in input_ids:
            continue
        if (
            u in gm.node_to_chain
            and v in gm.node_to_chain
            and gm.node_to_chain[u] is gm.node_to_chain[v]
        ):
            continue
        edges.append((u.node_id, v.node_id))
    ids = [n.node_id for n in gm.schedulable]
    es = {i: {m: 0 for m in node_modes[i]} for i in ids}
    chain_groups = [(c.l1.node_id, c.relu.node_id, c.l2.node_id) for c in gm.chains]

    def _converge_es():
        for _ in range(4000):
            changed = False
            for u, v in edges:
                for b in node_modes[v]:
                    lo = min(es[u][a] + _gap(a, b) for a in node_modes[u])
                    if lo > es[v][b]:
                        es[v][b] = lo
                        changed = True
            for g in chain_groups:
                m_es = max(min(es[i].values()) for i in g)
                for i in g:
                    for m in node_modes[i]:
                        if es[i][m] < m_es:
                            es[i][m] = m_es
                            changed = True
            if not changed:
                return
        raise RuntimeError("es did not converge")

    _converge_es()
    floor_minus1 = max(min(d.values()) for d in es.values())

    # Seed every node's ls to (floor-1); the backward sweep pulls non-critical
    # nodes earlier.  slack = ls - es; slack==0 means the node lies on SOME
    # critical path (no scheduling freedom relative to the floor).
    ls = {i: {m: floor_minus1 for m in node_modes[i]} for i in ids}
    for _ in range(4000):
        changed = False
        for u, v in reversed(edges):
            for a in node_modes[u]:
                hi = max(ls[v][b] - _gap(a, b) for b in node_modes[v])
                if hi < ls[u][a]:
                    ls[u][a] = hi
                    changed = True
        for g in chain_groups:
            m_ls = min(max(ls[i].values()) for i in g)
            for i in g:
                for m in node_modes[i]:
                    if ls[i][m] > m_ls:
                        ls[i][m] = m_ls
                        changed = True
        if not changed:
            break
    else:
        raise RuntimeError("ls did not converge")
    return es, ls, edges, node_modes


def floor_with_free(gm, free_ids):
    """Recompute the depth floor with every node in ``free_ids`` made
    DEPTH-FREE (all its incident edges cost gap 0).  The drop vs the baseline
    floor is an UPPER BOUND on how many layers that subsystem could ever save
    (it pretends those ops cost nothing) — a rigorous savings ceiling."""
    input_ids = {n.node_id for n in gm.input_nodes}
    node_modes = {n.node_id: _modes(n, gm) for n in gm.schedulable}
    edges = []
    for u, v in gm.edges:
        if u.node_id in input_ids:
            continue
        if (
            u in gm.node_to_chain
            and v in gm.node_to_chain
            and gm.node_to_chain[u] is gm.node_to_chain[v]
        ):
            continue
        edges.append((u.node_id, v.node_id))
    ids = [n.node_id for n in gm.schedulable]
    es = {i: {m: 0 for m in node_modes[i]} for i in ids}
    chain_groups = [(c.l1.node_id, c.relu.node_id, c.l2.node_id) for c in gm.chains]

    def gap(u, v, a, b):
        if u in free_ids or v in free_ids:
            return 0
        return _gap(a, b)

    for _ in range(4000):
        changed = False
        for u, v in edges:
            for b in node_modes[v]:
                lo = min(es[u][a] + gap(u, v, a, b) for a in node_modes[u])
                if lo > es[v][b]:
                    es[v][b] = lo
                    changed = True
        for g in chain_groups:
            m_es = max(min(es[i].values()) for i in g)
            for i in g:
                for m in node_modes[i]:
                    if es[i][m] < m_es:
                        es[i][m] = m_es
                        changed = True
        if not changed:
            break
    return max(min(d.values()) for d in es.values()) + 1


def reconstruct_path(gm, es, edges, node_modes):
    """Backward-walk the longest path from the deepest node, CHAIN-AWARE.

    Intra-chain edges (L1->ReLU->L2) are removed from ``edges`` and the chain
    shares one es value, so when we land on any chain member we must look at the
    predecessors of the WHOLE chain (i.e. the edges feeding L1).  We pick the
    determining predecessor = the incoming node with the largest es (the one
    that pushed cur's depth up), allowing gap-0 (same-level) hops, and stop on
    revisit or es 0."""
    preds = defaultdict(list)
    for u, v in edges:
        preds[v].append(u)
    es_min = {i: min(d.values()) for i, d in es.items()}

    chain_members = {}  # node_id -> tuple(member ids)
    for c in gm.chains:
        g = (c.l1.node_id, c.relu.node_id, c.l2.node_id)
        for m in g:
            chain_members[m] = g

    def incoming(nid):
        targets = chain_members.get(nid, (nid,))
        us = []
        for t in targets:
            us.extend(preds.get(t, []))
        return us

    end = max(es_min, key=lambda i: es_min[i])
    path = [end]
    cur = end
    seen = {end}
    # also collapse chain members so we don't print L1/ReLU/L2 separately
    while True:
        cands = [u for u in incoming(cur) if u not in seen]
        if not cands:
            break
        u = max(cands, key=lambda x: es_min[x])
        # must not move forward in depth
        if es_min[u] > es_min[cur]:
            break
        path.append(u)
        seen.add(u)
        # mark all chain members seen
        for m in chain_members.get(u, ()):
            seen.add(m)
        cur = u
        if es_min[u] == 0:
            break
    path.reverse()
    return path, es_min


def main():
    fuse = os.environ.get("FUSE", "1") == "1"

    from torchwright_doom.inference.compiled_model import build_graph
    from torchwright_doom import constants as C

    next_token, pos, emb, _banks = build_graph()

    n_fused = 0
    if fuse:
        from torchwright.graph.optimize import fuse_consecutive_linears

        n_fused = fuse_consecutive_linears(
            {next_token}, verbose=False, skip_relu_ejecting=True
        )

    gm = build_graph_model(next_token, pos)
    id2node = {n.node_id: n for n in gm.schedulable}
    es, ls, edges, node_modes = per_mode_bounds(gm)
    floor = max(min(d.values()) for d in es.values()) + 1

    if os.environ.get("COLLAPSE_SCAN") == "1":
        # Savings-ceiling scan: for each subsystem, recompute the floor with
        # that subsystem made depth-free. floor - ceiling = max layers it could
        # ever save. ReLU-collapse and Linear-collapse by op type too.
        print(f"BASELINE FLOOR = {floor}  (fused={n_fused})")
        subs = sorted({bucket(_node_label(n)) for n in gm.schedulable})
        print("\n--- per-SUBSYSTEM savings ceiling (floor if subsystem were free) ---")
        rows = []
        for s in subs:
            ids_s = {n.node_id for n in gm.schedulable if bucket(_node_label(n)) == s}
            f = floor_with_free(gm, ids_s)
            rows.append((floor - f, f, len(ids_s), s))
        for save, f, cnt, s in sorted(rows, reverse=True):
            print(f"  ceiling -{save:<3} -> floor {f:<3} ({cnt:5d} nodes)  {s}")
        # op-type collapses
        print("\n--- per-OP-TYPE savings ceiling ---")
        for opname in ("ReLU", "Linear", "Attn", "Add"):
            ids_o = {n.node_id for n in gm.schedulable if type(n).__name__ == opname}
            f = floor_with_free(gm, ids_o)
            print(
                f"  ceiling -{floor - f:<3} -> floor {f:<3} ({len(ids_o):5d} nodes)  {opname}"
            )
        return

    print("=" * 78)
    print(
        f"SCREEN={C.SCREEN_WIDTH}x{C.SCREEN_HEIGHT} detail={'low' if C.PIXEL_WIDTH==2 else 'high'} "
        f"COLUMN_COUNT={C.COLUMN_COUNT} HUD={C.HUD_ENABLED} "
        f"fuse={fuse}(fused={n_fused})"
    )
    print(
        f"DEPTH FLOOR (critical_path_layers) = {floor}   "
        f"[schedulable nodes={len(gm.schedulable)} chains={len(gm.chains)} edges={len(edges)}]"
    )
    print("=" * 78)

    chain, es_min = reconstruct_path(gm, es, edges, node_modes)

    print(f"\n--- CRITICAL PATH: {len(chain)} nodes, es 0..{es_min[chain[-1]]} ---")
    by_sub = Counter()
    by_op = Counter()
    by_mode = Counter()
    for nid in chain:
        n = id2node[nid]
        sub = bucket(_node_label(n))
        op = type(n).__name__
        b = min(node_modes[nid], key=lambda m: es[nid][m])
        by_sub[sub] += 1
        by_op[op] += 1
        by_mode[b] += 1
        print(f"  es{es_min[nid]:>3} [{b:>4}] {op:<14} {sub:<16} {_node_label(n)}")

    print("\n--- critical path by SUBSYSTEM ---")
    for s, c in by_sub.most_common():
        print(f"  {c:4d}  {s}")
    print("\n--- critical path by OP TYPE ---")
    for o, c in by_op.most_common():
        print(f"  {c:4d}  {o}")
    print("\n--- critical path by MODE (attn vs mlp sublayer) ---")
    for m, c in by_mode.most_common():
        print(f"  {c:4d}  {m}")

    # How many nodes sit at each earliest-start level? (a fat level = many ops
    # competing for the same depth; a thin chain = serial bottleneck)
    level_hist = Counter(es_min[i] for i in es_min)
    deep = sorted(level_hist)[-10:]
    print("\n--- node count at the 10 deepest es-levels (width of the floor) ---")
    for lvl in deep:
        print(f"  es{lvl:>3}: {level_hist[lvl]} nodes")

    # Enumerate EVERY node at the deepest es-levels (the serial tail that sets
    # the floor). Robust and complete — no path-reconstruction guessing.
    topk = int(os.environ.get("TOPK", "9"))
    cut = floor - 1 - topk
    print(f"\n--- ALL nodes at the {topk} deepest es-levels (es>{cut}) ---")
    deep_nodes = sorted(
        (i for i in es_min if es_min[i] > cut), key=lambda i: (es_min[i],)
    )
    for nid in deep_nodes:
        n = id2node[nid]
        b = min(node_modes[nid], key=lambda m: es[nid][m])
        flx = "flex" if is_flex(n, gm) else "    "
        print(
            f"  es{es_min[nid]:>3} [{b:>4}|{flx}] {type(n).__name__:<14} "
            f"{bucket(_node_label(n)):<16} {_node_label(n)}"
        )

    # Subsystem composition of the WHOLE deep region (es > cut): what dominates
    # the serial tail?
    tail_sub = Counter(bucket(_node_label(id2node[i])) for i in deep_nodes)
    tail_op = Counter(type(id2node[i]).__name__ for i in deep_nodes)
    print(f"\n--- serial tail (es>{cut}, {len(deep_nodes)} nodes) by SUBSYSTEM ---")
    for s, c in tail_sub.most_common():
        print(f"  {c:4d}  {s}")
    print("--- serial tail by OP TYPE ---")
    for o, c in tail_op.most_common():
        print(f"  {c:4d}  {o}")

    # ---- SLACK / BLAME census: every node on SOME critical path (slack==0) ----
    # A node is critical iff it has zero scheduling freedom relative to the floor
    # (es == ls). Bucketing the critical set shows which subsystems the depth
    # could be blamed on across ALL parallel critical paths, not just one.
    ls_min = {i: max(d.values()) for i, d in ls.items()}
    crit = [i for i in es_min if es_min[i] >= ls_min[i]]
    crit_sub = Counter(bucket(_node_label(id2node[i])) for i in crit)
    crit_op = Counter(type(id2node[i]).__name__ for i in crit)
    print(f"\n--- CRITICAL SET (slack==0): {len(crit)} nodes by SUBSYSTEM ---")
    for s, c in crit_sub.most_common():
        print(f"  {c:5d}  {s}")
    print("--- critical set by OP TYPE ---")
    for o, c in crit_op.most_common():
        print(f"  {c:5d}  {o}")
    # how many DISTINCT nodes are critical at each es level (parallel crit paths)
    crit_level = Counter(es_min[i] for i in crit)
    print("--- critical-set node count per es level (parallelism of the floor) ---")
    width_line = " ".join(f"{lvl}:{crit_level[lvl]}" for lvl in sorted(crit_level))
    print(f"  {width_line}")


if __name__ == "__main__":
    main()
