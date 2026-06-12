"""Prove the K divergence is a float32 near-tie (CPU, no compile, no GPU).

In one process: evaluate the token_ids graph's exact-math output at the divergence
position at float32 vs a float64-built graph, and the iv graph for reference.
If float32 gives node=1 but float64 gives node=3 (the reference), the divergence
is a float32 round-off at a near-tie, not a logic error.

    .venv/bin/python -m torchwright_doom.scripts.k_dtype_ab
"""

from __future__ import annotations

import os

POS = 2451


def _decode_at(graph_kind: str, dtype, full_rows, expected_row, pred_row):
    import torch

    import torchwright.graph.misc as _misc
    import torchwright.graph.node as _node_module
    from torchwright.debug.probe import reference_eval
    from torchwright.ops.inout_nodes import create_input, create_pos_encoding

    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED, build_doom_embedding
    from torchwright_doom.past import GraphPast
    from torchwright_doom.render_main import forward
    from torchwright_doom.render.tokens_bridge import rows_to_input

    torch.set_default_dtype(dtype)  # set BEFORE building so weights are this dtype
    _node_module.global_node_id = 0
    n = POS + 1
    if graph_kind == "iv":
        in_node = create_input("iv", TOKEN_VOCAB.layout.d_embed)
        inputs = {"iv": W_EMBED[full_rows[:n]].to(dtype)}
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
    emb = cache[next_token][POS]
    wt = W_EMBED.t().to(emb.dtype)
    logits = emb @ wt
    top = torch.topk(logits, 3)
    decoded = [TOKEN_VOCAB.row_to_token[r] for r in top.indices.tolist()]
    print(
        f"  [{graph_kind:9s} {str(dtype).split('.')[-1]:8s}] argmax={top.indices[0].item()} "
        f"({decoded[0][0].name}{decoded[0][1]})  "
        f"logit[node3 row {expected_row}]={logits[expected_row].item():.6g}  "
        f"logit[node1 row {pred_row}]={logits[pred_row].item():.6g}  "
        f"margin(3-1)={(logits[expected_row]-logits[pred_row]).item():.4g}"
    )
    return top.indices[0].item()


def main() -> int:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from torchwright_doom.render.wad_scene import ensure_doom_sandbox

    ensure_doom_sandbox()
    import torch

    from doom_sandbox import fixtures
    from doom_sandbox.implementation import prefill as sb_prefill
    from doom_sandbox.implementation import reference_drafter as drafter
    from torchwright_doom.render.tokens_bridge import sandbox_token_to_row

    scene = fixtures.load_fixture("e1m1_subset_textured")
    pose = scene.test_poses[0]
    prefill_rows = [
        sandbox_token_to_row(t) for t in sb_prefill.get_prefill(scene, pose)
    ]
    ar_rows = [sandbox_token_to_row(t) for t in drafter.expected_ar_tokens(scene, pose)]
    full_rows = prefill_rows + ar_rows
    expected_row = full_rows[POS + 1]  # bspCheckBack node=3 (reference)
    pred_row = 80285  # bspCheckBack node=1 (the float32 wrong pick)

    print(f"divergence at pos {POS}: reference next = row {expected_row} (node=3)")
    f32 = torch.float32
    f64 = torch.float64
    r_iv = _decode_at("iv", f32, full_rows, expected_row, pred_row)
    r_tid32 = _decode_at("token_ids", f32, full_rows, expected_row, pred_row)
    r_tid64 = _decode_at("token_ids", f64, full_rows, expected_row, pred_row)
    print(
        f"\niv@f32 argmax row {r_iv}; token_ids@f32 row {r_tid32}; token_ids@f64 row {r_tid64}"
    )
    print(f"expected (reference) row {expected_row}")
    if r_tid32 == pred_row and r_tid64 == expected_row:
        print(
            "PROVEN: float32 near-tie. token_ids resolves wrong at f32, correct at f64."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
