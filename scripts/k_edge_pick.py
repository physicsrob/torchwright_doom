"""Inspect the traversal-edge recency attention at the K divergence (pos 2451).

Plan-K unknowns #2 (collision + softmax weights) and #4 (q^2-inflation
arithmetic). Builds the token_ids graph, runs reference_eval once, locates the
``after_return`` recency pick (``attend_most_recent_matching`` with
match_gain=MATCH_GAIN_LONG and d_v=2), and at query pos 2451 reports:

  * the query (entity_u, tree_depth) decoded from the query vector;
  * every causal key position that is an active published edge, its decoded
    (child, depth), and whether it matches the query (collision check);
  * the per-key pre-softmax logits split into content vs recency terms, the
    softmax weights, and the magnitude of the content dot (q^2 inflation);
  * the recovered parent value (is it a clean integer or a blend?);
  * the float32 ULP at the logit scale vs the recency gap of _QUERY_GAIN.

    python -m scripts.k_edge_pick

Run from the torchwright_doom/ directory.
"""

from __future__ import annotations

import os
import sys

POS = 2451


def main() -> int:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from torchwright_doom.inference.wad_scene import ensure_doom_sandbox

    ensure_doom_sandbox()

    import torch

    import torchwright.graph.misc as _misc
    import torchwright.graph.node as _node_module
    from torchwright.graph import Attn
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.debug.probe import reference_eval
    from torchwright.ops.inout_nodes import create_pos_encoding

    from doom_sandbox import fixtures
    from doom_sandbox.implementation import prefill as sb_prefill
    from doom_sandbox.implementation import reference_drafter as drafter

    from torchwright.ops.inout_nodes import create_input
    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED, build_doom_embedding
    from torchwright_doom.past import GraphPast
    from torchwright_doom.render_constants import MATCH_GAIN_LONG
    from torchwright_doom.render_main import forward
    from torchwright_doom.inference.tokens_bridge import (
        rows_to_input,
        sandbox_token_to_row,
    )

    graph_kind = "token_ids"
    for a in sys.argv[1:]:
        if a in ("iv", "token_ids"):
            graph_kind = a
    print(f"[edge-pick] graph={graph_kind}")

    scene = fixtures.load_fixture("e1m1_subset_textured")
    pose = scene.test_poses[0]
    prefill_rows = [
        sandbox_token_to_row(t) for t in sb_prefill.get_prefill(scene, pose)
    ]
    ar_rows = [sandbox_token_to_row(t) for t in drafter.expected_ar_tokens(scene, pose)]
    full_rows = prefill_rows + ar_rows
    n = POS + 1

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

    # Locate candidate recency picks: Attn with query_matrix[0,0] == MATCH_GAIN_LONG.
    anc = get_ancestor_nodes({next_token})
    cands = []
    for nd in anc:
        if isinstance(nd, Attn) and nd.query_matrix.shape[0] >= 1:
            if abs(float(nd.query_matrix[0, 0].item()) - MATCH_GAIN_LONG) < 1.0:
                cands.append(nd)
    print(
        f"[edge-pick] recency-pick Attn candidates (match_gain={MATCH_GAIN_LONG}): "
        f"{len(cands)}"
    )
    for nd in cands:
        vin = nd.inputs[2]
        print(
            f"  id={nd.node_id} d_qk={nd.d_qk} d_v={nd.d_v} "
            f"d_query_in={nd.d_query_in} d_key_in={nd.d_key_in} "
            f"value_in={type(vin).__name__}('{getattr(vin,'name','')}')"
        )

    # The traversal edge pick is uniquely d_v==2 AND d_query_in==20
    # (W=19: lifted child[3] + depth one-hot[16], plus the recency col).
    edge_attns = [nd for nd in cands if nd.d_v == 2 and nd.d_query_in == 20]
    print(
        f"\n[edge-pick] traversal-edge pick (d_v==2, d_query_in==20): "
        f"{[nd.node_id for nd in edge_attns]}"
    )

    _orig = _misc.Assert._check
    _misc.Assert._check = lambda self, x: None
    try:
        cache = reference_eval(next_token, inputs, n)
    finally:
        _misc.Assert._check = _orig

    for attn in edge_attns:
        _inspect(attn, cache, n, MATCH_GAIN_LONG)

    return 0


def _inspect(attn, cache, n, MATCH_GAIN_LONG) -> None:
    import torch

    qm = attn.query_matrix.to(torch.float64)
    km = attn.key_matrix.to(torch.float64)
    W = qm.shape[1] - 1  # d_qk = W + 1; last col is recency
    query_in = cache[attn.inputs[0]].to(torch.float64)  # (n, d_query_in)
    key_in = cache[attn.inputs[1]].to(torch.float64)  # (n, d_key_in)
    value_in = cache[attn.inputs[2]].to(torch.float64)  # (n, d_v)
    out = cache[attn]  # (n, d_output)

    qv = query_in[POS] @ qm  # (d_qk,)
    kvals = key_in @ km  # (n, d_qk)

    # The lifted child query is [2q, 1, 1] in cols 0..2 (scaled by match_gain in qv).
    # Recover q from the query_vector (col 0 of query_in[POS] = 2q before scaling).
    two_q = float(query_in[POS][0].item())
    q = two_q / 2.0
    # tree_depth one-hot is query_in[POS][3:3+16]; argmax = depth.
    depth_oh = query_in[POS][3 : 3 + 16]
    q_depth = int(torch.argmax(depth_oh).item())
    print(f"\n[edge-pick] ===== Attn id={attn.node_id} (d_v={attn.d_v}) =====")
    print(
        f"[edge-pick] query at pos {POS}: entity_u(q)={q:.3f}  tree_depth={q_depth}  "
        f"(reference: child=68, depth=9)"
    )

    # Decode every causal active edge from key_in[:POS+1].
    # key_vector layout: [child, -child^2, 1, onehot_depth(16)] gated by edge_active.
    matches = []
    actives = []
    for i in range(POS + 1):
        kv = key_in[i]
        child = float(kv[0].item())
        present = float(kv[2].item())  # the lifted "1" (0 when gated/inactive)
        if present < 0.5:
            continue
        d_oh = kv[3 : 3 + 16]
        depth = int(torch.argmax(d_oh).item())
        actives.append((i, round(child, 3), depth))
        if abs(child - q) < 0.5 and depth == q_depth:
            matches.append((i, round(child, 3), depth))

    print(f"[edge-pick] active published edges (causal, <= {POS}): {len(actives)}")
    print(
        f"[edge-pick] edges matching (child~={q:.0f}, depth=={q_depth}): "
        f"{len(matches)}  -> COLLISION needs recency tiebreak: {len(matches) > 1}"
    )
    if matches:
        print("  matching edge positions (pos, child, depth):")
        for m in matches[-8:]:
            print(f"    {m}")

    # Per-key pre-softmax logits, split content vs recency.
    # content = match_gain*(query·key) over cols 0..W-1; recency = _QUERY_GAIN*pos.
    logits = (qv.unsqueeze(0) * kvals).sum(dim=1)[: POS + 1]  # full logit
    content = (qv[:W].unsqueeze(0) * kvals[: POS + 1, :W]).sum(dim=1)
    recency = qv[W] * kvals[: POS + 1, W]
    softmax = torch.softmax(logits, dim=0)

    # Report the top keys by softmax weight and the matched keys specifically.
    topk = torch.topk(softmax, min(6, POS + 1))
    print(
        "\n[edge-pick] top keys by softmax weight (pos  weight  logit  content  recency  "
        "parent_val):"
    )
    for w, i in zip(topk.values.tolist(), topk.indices.tolist()):
        print(
            f"    pos {i:5d}  w={w:.6f}  logit={logits[i].item():.4f}  "
            f"content={content[i].item():.4f}  recency={recency[i].item():.4f}  "
            f"parent={value_in[i][0].item():.4f}"
        )

    if matches:
        print("\n[edge-pick] matched keys (the collision set), most-recent last:")
        for i, child, depth in matches[-6:]:
            print(
                f"    pos {i:5d}  w={softmax[i].item():.6f}  logit={logits[i].item():.4f}  "
                f"content={content[i].item():.4f}  recency={recency[i].item():.4f}  "
                f"parent={value_in[i][0].item():.4f}  child={child} depth={depth}"
            )

    # The recovered parent (Attn output) — clean integer or blended?
    parent_recovered = float(out[POS][0].item())
    is_enter_recovered = float(out[POS][1].item())
    print(
        f"\n[edge-pick] recovered parent (Attn out[{POS}][0]) = {parent_recovered:.6f}  "
        f"is_enter = {is_enter_recovered:.6f}"
    )
    print(
        f"[edge-pick]   nearest int parent = {round(parent_recovered)}  "
        f"fractional part = {parent_recovered - round(parent_recovered):+.6f}"
    )

    # q^2 inflation vs recency gap.
    import math

    max_content = float(content.abs().max().item())
    # float32 ULP at the content scale.
    ulp = 2.0 ** (math.floor(math.log2(max(max_content, 1.0))) - 23)
    print(
        f"\n[edge-pick] q^2 inflation: max|content logit| = {max_content:.4g}  "
        f"(match_gain*q^2 ~ {MATCH_GAIN_LONG * q * q:.4g})"
    )
    print(
        f"[edge-pick]   float32 ULP at that scale ~= {ulp:.4g}  vs recency gap "
        f"_QUERY_GAIN=8.0  -> recency gap is {'BELOW' if ulp > 8 else 'above'} the ULP"
    )

    # ---- S1 causal validation (q^2-cancel the lifted edge query) ----------
    # Current query lifted cols (scaled) are qv = match_gain*[2q, 1, 1].  The
    # "1" in col 2 makes the dot 1 + q^2 - (child-q)^2 (the q^2 inflation).
    # S1: replace the col-2 query constant 1 -> (1 - q^2) so the dot becomes
    # 1 - (child-q)^2 (no inflation).  We re-derive the logits IN FLOAT32 from
    # the cached (float32) key values to mimic exactly what the graph would do.
    print("\n[edge-pick] ---- S1 PROBE: q^2-cancel the lifted edge query ----")
    qv32 = query_in[POS].to(torch.float32) @ attn.query_matrix.to(torch.float32)
    kvals32 = key_in.to(torch.float32) @ attn.key_matrix.to(torch.float32)
    mg = float(attn.query_matrix[0, 0].item())  # match_gain
    for label, qvec in (
        ("current [2q,1,1]", qv32.clone()),
        ("S1 q^2-cancel [2q,1,1-q^2]", _s1_query(qv32, mg, q)),
    ):
        lg = (qvec.unsqueeze(0).to(torch.float32) * kvals32[: POS + 1]).sum(dim=1)
        sm = torch.softmax(lg.to(torch.float32), dim=0)
        win = int(torch.argmax(lg).item())
        parent_win = float(value_in[win][0].item())
        print(
            f"  {label:30s}: winner pos {win}  parent={parent_win:.3f}  "
            f"w={sm[win].item():.6f}  logit_scale~{float(lg.abs().max().item()):.4g}  "
            f"logit[2060]={lg[2060].item():.2f} logit[2291]={lg[2291].item():.2f}"
        )

    # Exact key vectors for the two contesting positions (reconcile by hand).
    print("\n[edge-pick] exact key vectors (cols 0..2 = [child, -child^2, present]):")
    for pos in (2060, 2291):
        kin = key_in[pos]
        print(
            f"  pos {pos}: child={kin[0].item():.5f}  -child^2={kin[1].item():.5f}  "
            f"present={kin[2].item():.5f}  depth_oh_argmax="
            f"{int(torch.argmax(kin[3:3+16]).item())}  parent={value_in[pos][0].item():.3f}"
        )


def _s1_query(qv32, match_gain, q):
    """qv with col-2 lifted constant 1 -> (1 - q^2) (the S1 q^2-cancel)."""
    import torch

    new = qv32.clone()
    new[2] = match_gain * (1.0 - q * q)
    return new


if __name__ == "__main__":
    raise SystemExit(main())
