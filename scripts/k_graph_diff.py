"""Node-by-node exact-math diff of the iv vs token_ids graphs at the K divergence.

Plan-K unknown #1: the token_ids graph (in-graph Embedding) and the iv graph
(pre-embedded W_EMBED rows) feed *bit-identical* input vectors downstream, yet
reference_eval gives a ~500-logit common-mode difference and flips the
bspCheckBack parent (node3 -> node1) at pos 2451. Since create_input and
build_doom_embedding each create exactly one parentless node (id 0), the two
graphs' node ids align downstream, so we can compare the two reference_eval
caches *node-by-node by node_id* and find the FIRST node whose value diverges.

    python -m scripts.k_graph_diff [--window 2452]

Run from the torchwright_doom/ directory.
"""

from __future__ import annotations

import argparse
import os


POS = 2451
ROW_NODE3 = 80317  # reference: bspCheckBack(node=3, depth=8)
ROW_NODE1 = 80285  # float32 wrong pick: bspCheckBack(node=1, depth=8)


def _build_and_eval(graph_kind, full_rows, n):
    import torch

    import torchwright.graph.misc as _misc
    import torchwright.graph.node as _node_module
    from torchwright.debug.probe import reference_eval
    from torchwright.ops.inout_nodes import create_input, create_pos_encoding

    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED, build_doom_embedding
    from torchwright_doom.past import GraphPast
    from torchwright_doom.render_main import forward
    from torchwright_doom.render.tokens_bridge import rows_to_input

    _node_module.global_node_id = 0
    if graph_kind == "iv":
        in_node = create_input("iv", TOKEN_VOCAB.layout.d_embed)
        inputs = {"iv": W_EMBED[full_rows[:n]]}
    else:
        in_node = build_doom_embedding("token_ids")
        inputs = {"token_ids": rows_to_input(full_rows[:n])}
    next_token = forward(
        in_node,
        GraphPast(input_vec=in_node, pos_encoding=create_pos_encoding()),
        create_pos_encoding(),
    )
    _orig = _misc.Assert._check
    _misc.Assert._check = lambda self, x: None
    try:
        cache = reference_eval(next_token, inputs, n)
    finally:
        _misc.Assert._check = _orig
    # Snapshot by node_id -> (typename, name, tensor) and node_id -> node object.
    snap = {}
    nodes = {}
    for node, val in cache.items():
        snap[node.node_id] = (type(node).__name__, getattr(node, "name", ""), val)
        nodes[node.node_id] = node
    return next_token.node_id, snap, nodes


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", default="e1m1_subset_textured")
    p.add_argument("--pose", type=int, default=0)
    p.add_argument("--window", type=int, default=POS + 1)
    args = p.parse_args(argv)

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from torchwright_doom.render.wad_scene import ensure_doom_sandbox

    ensure_doom_sandbox()

    import torch

    from doom_sandbox import fixtures
    from doom_sandbox.implementation import prefill as sb_prefill
    from doom_sandbox.implementation import reference_drafter as drafter

    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
    from torchwright_doom.render.tokens_bridge import sandbox_token_to_row

    scene = fixtures.load_fixture(args.fixture)
    pose = scene.test_poses[args.pose]
    prefill_rows = [sandbox_token_to_row(t) for t in sb_prefill.get_prefill(scene, pose)]
    ar_rows = [sandbox_token_to_row(t) for t in drafter.expected_ar_tokens(scene, pose)]
    full_rows = prefill_rows + ar_rows
    n = min(args.window, len(full_rows))
    assert n > POS, f"window {n} must exceed POS {POS}"

    print(f"[graph-diff] building+evaluating iv graph (n={n})...")
    out_iv, snap_iv, nodes_iv = _build_and_eval("iv", full_rows, n)
    print(f"[graph-diff] building+evaluating token_ids graph (n={n})...")
    out_tid, snap_tid, nodes_tid = _build_and_eval("token_ids", full_rows, n)

    ids_iv = set(snap_iv)
    ids_tid = set(snap_tid)
    print(f"\n[graph-diff] node count: iv={len(ids_iv)} token_ids={len(ids_tid)} "
          f"shared={len(ids_iv & ids_tid)} only_iv={len(ids_iv - ids_tid)} "
          f"only_tid={len(ids_tid - ids_iv)}")
    print(f"[graph-diff] output node id: iv={out_iv} token_ids={out_tid}")

    # Per-node divergence over all positions, ANY non-zero diff (find the seed).
    def node_diff(nid):
        _t, _nm, v_iv = snap_iv[nid]
        _t2, _nm2, v_tid = snap_tid[nid]
        if v_iv.shape != v_tid.shape:
            return float("inf")
        return float((v_iv.to(torch.float64) - v_tid.to(torch.float64)).abs().max().item())

    diffs = []  # (node_id, typename, name, max_all, max_at_pos)
    for nid in sorted(ids_iv & ids_tid):
        t_iv, name_iv, v_iv = snap_iv[nid]
        _t2, _nm2, v_tid = snap_tid[nid]
        if v_iv.shape != v_tid.shape:
            diffs.append((nid, t_iv, name_iv, float("inf"), float("inf")))
            continue
        d = (v_iv.to(torch.float64) - v_tid.to(torch.float64)).abs()
        max_all = float(d.max().item())
        max_at_pos = float(d[POS].max().item()) if d.shape[0] > POS else 0.0
        diffs.append((nid, t_iv, name_iv, max_all, max_at_pos))

    for THRESH in (1e-3, 1e-6, 1e-9, 0.0):
        nd = [d for d in diffs if d[3] > THRESH]
        print(f"[graph-diff] {len(nd):5d}/{len(diffs)} shared nodes diverge "
              f"(max-abs-diff over all positions > {THRESH:g})")

    diverging = [d for d in diffs if d[3] > 0.0]
    print("\n[graph-diff] FIRST 20 diverging nodes by node_id "
          "(node_id  type  name  max_all  max_at_pos2451):")
    for nid, tn, nm, ma, mp in diverging[:20]:
        print(f"  id={nid:5d}  {tn:18s}  {nm[:28]:28s}  all={ma:12.4g}  pos={mp:12.4g}")

    # Trace the input chain of the TRUE first diverging node back to its seed.
    if diverging:
        first_id = diverging[0][0]
        print(f"\n[graph-diff] INPUT-CHAIN TRACE from first diverging node id={first_id}:")
        seen = set()
        frontier = [first_id]
        while frontier:
            nid = frontier.pop(0)
            if nid in seen or nid not in nodes_iv:
                continue
            seen.add(nid)
            node = nodes_iv[nid]
            tn = type(node).__name__
            nm = getattr(node, "name", "")
            d = node_diff(nid)
            ins = [getattr(i, "node_id", -1) for i in getattr(node, "inputs", [])]
            in_diffs = [(i, round(node_diff(i), 9) if i in snap_iv else "n/a") for i in ins]
            print(f"  id={nid:5d} {tn:16s} {nm[:24]:24s} diff={d:.6g}  inputs={in_diffs}")
            # Recurse only into inputs that themselves diverge (find the seed root).
            for i in ins:
                if i in snap_iv and node_diff(i) > 0.0:
                    frontier.append(i)
        if not any(node_diff(i) > 0.0 for i in
                   [getattr(x, "node_id", -1) for x in getattr(nodes_iv[first_id], "inputs", [])]):
            n0 = nodes_iv[first_id]
            print(f"  -> id={first_id} ({type(n0).__name__}) diverges but ALL its inputs are "
                  f"bit-identical: this is a pure float32 rounding difference (op-internal).")

    # The input node (id 0) and PE, explicitly, incl. dtype/contiguity.
    print("\n[graph-diff] leaf nodes:")
    for nid in (0, 1, 2):
        if nid in (ids_iv & ids_tid):
            tn, nm, v = snap_iv[nid]
            _, _, v2 = snap_tid[nid]
            same = bool(torch.equal(v, v2)) if v.shape == v2.shape else False
            print(f"  id={nid} {tn} '{nm}' shape={tuple(v.shape)} torch.equal={same} "
                  f"iv[dtype={v.dtype},contig={v.is_contiguous()}] "
                  f"tid[dtype={v2.dtype},contig={v2.is_contiguous()}]")

    # Output-embedding column diff at POS, decoded against the layout.
    w_iv = snap_iv[out_iv][2][POS].to(torch.float64)
    w_tid = snap_tid[out_tid][2][POS].to(torch.float64)
    col_d = (w_iv - w_tid).abs()
    layout = TOKEN_VOCAB.layout
    print(f"\n[graph-diff] output-embedding diff at pos {POS}: "
          f"max={col_d.max().item():.6g}  L2={float((col_d**2).sum().item())**0.5:.6g}  "
          f"n_cols_diff(>1e-4)={int((col_d > 1e-4).sum().item())}/{len(col_d)}")
    # Decode region for the most-divergent columns.
    region = _column_region_map(layout)
    top = torch.topk(col_d, min(12, len(col_d)))
    print("  most-divergent output columns (col  |diff|  iv  tid  region):")
    for v, c in zip(top.values.tolist(), top.indices.tolist()):
        if v < 1e-6:
            continue
        print(f"    col {c:4d}  d={v:11.5g}  iv={w_iv[c].item():11.5g}  "
              f"tid={w_tid[c].item():11.5g}  {region.get(c, '?')}")

    # Logits at the two candidate rows.
    w_embed_t = W_EMBED.t().to(torch.float64)
    for label, w in (("iv", w_iv), ("token_ids", w_tid)):
        logits = w @ w_embed_t
        am = int(logits.argmax().item())
        rt, rv = TOKEN_VOCAB.row_to_token[am]
        print(f"\n[graph-diff] {label}: argmax row {am} ({rt.name}{rv})  "
              f"logit[node3 r{ROW_NODE3}]={logits[ROW_NODE3].item():.6f}  "
              f"logit[node1 r{ROW_NODE1}]={logits[ROW_NODE1].item():.6f}  "
              f"margin(3-1)={(logits[ROW_NODE3]-logits[ROW_NODE1]).item():.6f}")

    return 0


def _column_region_map(layout) -> dict:
    """Map each W_EMBED column index to a human-readable region label."""
    region = {}
    for c in range(8):
        region[c] = "E8_category"
    for j, c in enumerate(getattr(layout, "shared_raw_col", [])):
        region[c] = f"raw_slot_pos{j}"
    for j, c in enumerate(getattr(layout, "shared_dq_col", [])):
        w = layout.shared_position_dq_width[j]
        for k in range(w):
            region[c + k] = f"digitquad_pos{j}[{k}]"
    for name, entries in getattr(layout, "derived_columns_by_name", {}).items():
        for (_t, _s, start, width) in entries:
            for k in range(width):
                region[start + k] = f"derived:{name}[{k}]"
    return region


if __name__ == "__main__":
    raise SystemExit(main())
