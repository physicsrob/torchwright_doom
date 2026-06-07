"""Float64 confirmation: does the token_ids graph resolve to node3 at full precision?

Plan-K unknown #3. The graph is BUILT under float32 (so every M baked into a
cond_gate / select by `_max_abs_or_raise(value_type)` is the same value the real
float32 graph uses), then every weight tensor is up-cast to float64 and
reference_eval runs in double. If the float32 graph picks node1 but the
identical-logic float64 graph picks node3, the divergence is purely float32
precision at the q^2-inflated edge-pick scale.

    python -m scripts.k_float64

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


POS = 2451
ROW_NODE3 = 80317
ROW_NODE1 = 80285
_WEIGHT_ATTRS = (
    "output_matrix", "output_bias", "value", "table",
    "query_matrix", "key_matrix", "value_matrix",
)


def main() -> int:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    _ensure_doom_sandbox()

    import torch

    import torchwright.graph.misc as _misc
    import torchwright.graph.node as _node_module
    from torchwright.compiler.utils import get_ancestor_nodes
    from torchwright.debug.probe import reference_eval
    from torchwright.ops.inout_nodes import create_pos_encoding

    from doom_sandbox import fixtures
    from doom_sandbox.implementation import prefill as sb_prefill
    from doom_sandbox.implementation import reference_drafter as drafter

    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED, build_doom_embedding
    from torchwright_doom.past import GraphPast
    from torchwright_doom.render_main import forward
    from torchwright_doom.render.tokens_bridge import rows_to_input, sandbox_token_to_row

    scene = fixtures.load_fixture("e1m1_subset_textured")
    pose = scene.test_poses[0]
    prefill_rows = [sandbox_token_to_row(t) for t in sb_prefill.get_prefill(scene, pose)]
    ar_rows = [sandbox_token_to_row(t) for t in drafter.expected_ar_tokens(scene, pose)]
    full_rows = prefill_rows + ar_rows
    n = POS + 1

    # Build under float32 so the baked M / weights match the real float32 graph.
    _node_module.global_node_id = 0
    in_node = build_doom_embedding("token_ids")
    next_token = forward(
        in_node,
        GraphPast(input_vec=in_node, pos_encoding=create_pos_encoding()),
        create_pos_encoding(),
    )

    def decode(cache, label):
        emb = cache[next_token][POS].to(torch.float64)
        logits = emb @ W_EMBED.t().to(torch.float64)
        am = int(logits.argmax().item())
        rt, rv = TOKEN_VOCAB.row_to_token[am]
        print(f"[float64] {label}: argmax row {am} ({rt.name}{rv})  "
              f"logit[node3 r{ROW_NODE3}]={logits[ROW_NODE3].item():.6f}  "
              f"logit[node1 r{ROW_NODE1}]={logits[ROW_NODE1].item():.6f}  "
              f"margin(3-1)={(logits[ROW_NODE3]-logits[ROW_NODE1]).item():.6f}")
        return am

    inputs = {"token_ids": rows_to_input(full_rows[:n])}
    _orig = _misc.Assert._check
    _misc.Assert._check = lambda self, x: None
    try:
        # Float32 baseline (same process, same graph).
        print("[float64] evaluating float32 baseline...")
        cache32 = reference_eval(next_token, inputs, n)
        am32 = decode(cache32, "float32")
        del cache32

        # Up-cast every weight tensor to float64, then re-evaluate in double.
        print("[float64] casting all weight tensors to float64...")
        torch.set_default_dtype(torch.float64)
        for nd in get_ancestor_nodes({next_token}):
            for attr in _WEIGHT_ATTRS:
                t = getattr(nd, attr, None)
                if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
                    setattr(nd, attr, t.to(torch.float64))
        print("[float64] evaluating float64...")
        cache64 = reference_eval(next_token, inputs, n)
        am64 = decode(cache64, "float64")
    finally:
        _misc.Assert._check = _orig

    print(f"\n[float64] float32 -> row {am32} ({'node3' if am32==ROW_NODE3 else 'node1' if am32==ROW_NODE1 else '?'}); "
          f"float64 -> row {am64} ({'node3' if am64==ROW_NODE3 else 'node1' if am64==ROW_NODE1 else '?'})")
    if am32 == ROW_NODE1 and am64 == ROW_NODE3:
        print("[float64] CONFIRMED: token_ids logic is correct (node3); float32 precision "
              "alone flips it to node1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
