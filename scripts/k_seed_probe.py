"""Build-time inspection of the value_type/M seed on the traversal-edge key path.

No reference_eval (build-time only -> fast). For both the iv and token_ids graphs,
finds the edge-pick Attn (id 6168 region), walks its KEY input chain, and prints
each node's scalar value_range plus, for every cond_gate, the offset M baked into
its output bias (M = -output_bias[0]). This shows WHERE the wide M comes from and
how iv (narrow input value_type) vs token_ids (wide Embedding value_type) differ.

    python -m scripts.k_seed_probe [iv|token_ids]

Run from the torchwright_doom/ directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_doom_sandbox() -> None:
    try:
        import doom_sandbox  # noqa: F401

        return
    except ImportError:
        pass
    umbrella = Path(__file__).resolve().parents[2]
    if (umbrella / "doom_sandbox").is_dir():
        sys.path.insert(0, str(umbrella))
    import doom_sandbox  # noqa: F401


def _rng(node):
    try:
        r = node.value_type.value_range
        return f"[{r.lo:.4g}, {r.hi:.4g}]"
    except Exception:
        return "?"


def main() -> int:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    _ensure_doom_sandbox()

    import torch

    import torchwright.graph.node as _node_module
    from torchwright.graph import Attn
    from torchwright.graph.linear import Linear
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.ops.inout_nodes import create_input, create_pos_encoding

    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED, build_doom_embedding
    from torchwright_doom.past import GraphPast
    from torchwright_doom.render_constants import MATCH_GAIN_LONG
    from torchwright_doom.render_main import forward

    graph_kind = "token_ids"
    compare = "--compare" in sys.argv
    for a in sys.argv[1:]:
        if a in ("iv", "token_ids"):
            graph_kind = a

    # Which W_EMBED column drives the wide Embedding scalar range?
    col_max = W_EMBED.abs().max(dim=0).values
    top = torch.topk(col_max, 6)
    layout = TOKEN_VOCAB.layout
    print(f"[seed] W_EMBED widest columns (|value| max): "
          f"{[(int(c), round(float(v),1)) for v, c in zip(top.values, top.indices)]}")
    print(f"[seed] d_embed={layout.d_embed}  n_derived_cols={layout.n_derived_columns}")

    def build(kind):
        _node_module.global_node_id = 0
        if kind == "iv":
            inn = create_input("iv", layout.d_embed)
        else:
            inn = build_doom_embedding("token_ids")
        nt = forward(
            inn,
            GraphPast(input_vec=inn, pos_encoding=create_pos_encoding()),
            create_pos_encoding(),
        )
        return inn, nt

    def gate_M_map(nt):
        """node_id -> (M, input_scalar_range, width) for every cond_gate/select output Linear."""
        out = {}
        for nd in get_ancestor_nodes({nt}):
            if isinstance(nd, Linear) and ("cond_gate" in getattr(nd, "name", "")
                                            or "select" in getattr(nd, "name", "")):
                b = getattr(nd, "output_bias", None)
                if b is not None and b.numel() > 0:
                    out[nd.node_id] = (float(-b[0].item()),
                                       _rng(nd.inputs[0]) if nd.inputs else "?", len(nd))
        return out

    if compare:
        _, nt_iv = build("iv")
        m_iv = gate_M_map(nt_iv)
        _, nt_tid = build("token_ids")
        m_tid = gate_M_map(nt_tid)
        shared = sorted(set(m_iv) & set(m_tid))
        diffs = [(nid, m_iv[nid][0], m_tid[nid][0], m_iv[nid][1], m_tid[nid][1], m_iv[nid][2])
                 for nid in shared
                 if abs(m_iv[nid][0] - m_tid[nid][0]) > 1e-6]
        print(f"\n[seed] gates with DIFFERENT M between iv and token_ids: "
              f"{len(diffs)}/{len(shared)} (first 25 by node_id):")
        print(f"  {'id':>5} {'M_iv':>12} {'M_tid':>12} {'ratio':>7}  range_iv -> range_tid  d")
        for nid, miv, mtid, riv, rtid, w in diffs[:25]:
            ratio = mtid / miv if miv else float('inf')
            print(f"  {nid:5d} {miv:12.5g} {mtid:12.5g} {ratio:7.3f}  {riv} -> {rtid}  d={w}")
        return 0

    _node_module.global_node_id = 0
    if graph_kind == "iv":
        in_node = create_input("iv", layout.d_embed)
    else:
        in_node = build_doom_embedding("token_ids")
    print(f"\n[seed] graph={graph_kind}  input value_range={_rng(in_node)}")
    next_token = forward(
        in_node,
        GraphPast(input_vec=in_node, pos_encoding=create_pos_encoding()),
        create_pos_encoding(),
    )

    # Locate the traversal-edge pick: Attn, d_v==2, d_query_in==20.
    anc = get_ancestor_nodes({next_token})
    edge = None
    for nd in anc:
        if isinstance(nd, Attn) and nd.d_v == 2 and nd.d_query_in == 20:
            if abs(float(nd.query_matrix[0, 0].item()) - MATCH_GAIN_LONG) < 1.0:
                edge = nd
                break
    assert edge is not None, "edge-pick Attn not found"
    key_node = edge.inputs[1].inputs[0]  # key_in = Concatenate([key_vector, pos_enc])
    print(f"[seed] edge Attn id={edge.node_id}; key_vector node id={key_node.node_id} "
          f"({type(key_node).__name__}) value_range={_rng(key_node)} width={len(key_node)}")

    # Walk the key chain; report cond_gate M and the widest-range inputs.
    def cond_gate_M(node):
        # cond_gate's output Linear has output_bias = full(-M); M = -bias[0].
        if isinstance(node, Linear) and "cond_gate" in getattr(node, "name", ""):
            b = getattr(node, "output_bias", None)
            if b is not None and b.numel() > 0:
                return float(-b[0].item())
        return None

    key_anc = get_ancestor_nodes({key_node})
    gates = []
    for nd in key_anc:
        m = cond_gate_M(nd)
        if m is not None:
            inp_rng = _rng(nd.inputs[0]) if nd.inputs else "?"
            gates.append((nd.node_id, m, inp_rng, len(nd)))
    gates.sort()
    print(f"\n[seed] cond_gate offsets M on the edge-KEY ancestor chain "
          f"({len(gates)} gates):")
    for nid, m, inp_rng, w in gates[:40]:
        flag = "  <-- WIDE" if m > 1e5 else ""
        print(f"  cond_gate id={nid:5d}  M={m:12.4g}  input_range={inp_rng:24s} d={w}{flag}")

    # Per-column intervals of the WIDE gates' inputs (V-A feasibility check).
    print("\n[seed] PER-COLUMN affine intervals of the WIDE (M>1e5) gate inputs "
          "(d=3 lift = [child, -child^2, present]):")
    seen_inp = set()
    for nd in key_anc:
        m = cond_gate_M(nd)
        if m is not None and m > 1e4 and nd.inputs:
            inp = nd.inputs[0]
            if inp.node_id in seen_inp:
                continue
            seen_inp.add(inp.node_id)
            try:
                ivs = inp.affine_bound.to_interval()
                cols = ", ".join(f"[{r.lo:.4g},{r.hi:.4g}]" for r in ivs)
                print(f"  gate id={nd.node_id} input id={inp.node_id} ({type(inp).__name__}) "
                      f"per-col: {cols}")
            except Exception as e:
                print(f"  gate id={nd.node_id} input id={inp.node_id}: to_interval failed: {e}")
            if len(seen_inp) >= 4:
                break

    # The child/entity nodes feeding the lift: find Linears named for child extraction.
    print(f"\n[seed] nodes on key chain with widest value_range (top 12):")
    ranked = sorted(
        key_anc,
        key=lambda n: -(abs(n.value_type.value_range.lo) + abs(n.value_type.value_range.hi))
        if _has_range(n) else 0,
    )
    for nd in ranked[:12]:
        if _has_range(nd):
            print(f"  id={nd.node_id:5d} {type(nd).__name__:14s} "
                  f"{getattr(nd,'name','')[:26]:26s} range={_rng(nd)} d={len(nd)}")
    return 0


def _has_range(n):
    try:
        import math
        r = n.value_type.value_range
        return math.isfinite(r.lo) and math.isfinite(r.hi)
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
