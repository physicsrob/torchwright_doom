"""Exact-math (no compile, no GPU) graph-vs-reference check on e1m1_subset_textured.

Phase-3 of the probe showed the compiled output equals the exact-math
(reference_eval) output bit-for-bit at the divergence, so the divergence lives in
the GRAPH, not the compiler. This confirms it independently on the *iv* graph
(pre-embedded input, exactly like the J2 oracle — no in-graph Embedding), finds
the first marker mismatch vs the sandbox reference, and is the GPU-free reproducer.

    .venv/bin/python -m torchwright_doom.scripts.k_exact_check --window 2460
"""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", default="e1m1_subset_textured")
    p.add_argument("--pose", type=int, default=0)
    p.add_argument("--window", type=int, default=2460)
    p.add_argument(
        "--graph",
        choices=["iv", "token_ids"],
        default="iv",
        help="iv = pre-embedded input (J2 setup); token_ids = in-graph Embedding",
    )
    p.add_argument(
        "--float64",
        action="store_true",
        help="run reference_eval at float64 (causal test of float32 near-tie)",
    )
    args = p.parse_args(argv)

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from torchwright_doom.inference.wad_scene import ensure_doom_sandbox

    ensure_doom_sandbox()

    import torch

    if args.float64:
        torch.set_default_dtype(torch.float64)

    import torchwright.graph.misc as _misc
    import torchwright.graph.node as _node_module
    from torchwright.debug.probe import reference_eval
    from torchwright.ops.inout_nodes import create_input, create_pos_encoding

    from doom_sandbox import fixtures
    from doom_sandbox.implementation import prefill as sb_prefill
    from doom_sandbox.implementation import reference_drafter as drafter

    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED, build_doom_embedding
    from torchwright_doom.past import GraphPast
    from torchwright_doom.render_main import forward
    from torchwright_doom.inference.tokens_bridge import (
        rows_to_input,
        sandbox_token_to_row,
    )

    scene = fixtures.load_fixture(args.fixture)
    pose = scene.test_poses[args.pose]
    prefill_rows = [
        sandbox_token_to_row(t) for t in sb_prefill.get_prefill(scene, pose)
    ]
    ar_rows = [sandbox_token_to_row(t) for t in drafter.expected_ar_tokens(scene, pose)]
    full_rows = prefill_rows + ar_rows
    begin = len(prefill_rows) - 1
    n = min(args.window, len(full_rows))

    _node_module.global_node_id = 0
    if args.graph == "iv":
        # Pre-embedded W_EMBED rows in, no in-graph Embedding (J2 setup).
        in_node = create_input("iv", TOKEN_VOCAB.layout.d_embed)
        inputs = {"iv": W_EMBED[full_rows[:n]]}
    else:
        # In-graph token-id Embedding (the compiled K artifact's input path).
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
    print(f"[exact-check] graph={args.graph} dtype={torch.get_default_dtype()}")
    emitted = cache[next_token]
    w_embed_t = W_EMBED.t().to(emitted.dtype)

    carriers = {"value", "angleValue", "pixel", "wallColU"}
    first = None
    n_marker_mismatch = 0
    for i in range(begin, n - 1):
        exp_row = full_rows[i + 1]
        exp_type = TOKEN_VOCAB.row_to_token[exp_row][0].name
        if exp_type in carriers:
            continue
        pred_row = int(torch.argmax(emitted[i] @ w_embed_t).item())
        if pred_row != exp_row:
            n_marker_mismatch += 1
            pred_type, pred_v = TOKEN_VOCAB.row_to_token[pred_row]
            exp_v = TOKEN_VOCAB.row_to_token[exp_row][1]
            if first is None:
                first = (i, exp_type, exp_v, exp_row, pred_type.name, pred_v, pred_row)
            if n_marker_mismatch <= 6:
                print(
                    f"  marker mismatch pos {i} (rollout {i - begin}): expected "
                    f"{exp_type}{exp_v} (row {exp_row}) -> exact-math graph "
                    f"{pred_type.name}{pred_v} (row {pred_row})"
                )

    print(
        f"\n[exact-check] {args.fixture} pose={args.pose} window={n} "
        f"marker mismatches in exact math: {n_marker_mismatch}"
    )
    if first:
        i, et, ev, er, pt, pv, pr = first
        print(
            f"[exact-check] FIRST marker mismatch at pos {i} (rollout {i - begin}): "
            f"expected {et}{ev} -> exact-math graph {pt}{pv}.  "
            f"This is a GRAPH-vs-reference divergence in exact math (no compile)."
        )
    else:
        print(
            "[exact-check] no marker mismatches in window — exact-math graph matches "
            "the reference here."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
