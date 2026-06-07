"""Measure where the compiled forward departs exact math at the K divergence.

Builds the graph ONCE (so the oracle and the compiled module share node identity),
teacher-forces the reference stream up to the first hard divergence via the KV
cache, then:

  Phase 1 (cheap): replay the divergence position with ``debug=True`` — re-checks
  every Assert/DebugWatch on the compiled values. A firing assert names the exact
  node + invariant (CLAUDE.md triage step 1).

  Phase 2 (if Phase 1 is clean): compare every graph node's compiled value to the
  exact-math oracle (reference_eval) AT the divergence position, and report the
  first divergent node in topological order — the origin of the fp32 departure.

    .venv/bin/python -m torchwright_doom.scripts.k_probe_divergence --pos 2450
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", default="e1m1_subset_textured")
    p.add_argument("--pose", type=int, default=0)
    p.add_argument("--pos", type=int, default=2450,
                   help="input stream position whose prediction diverges")
    p.add_argument("--d", type=int, default=4096)
    p.add_argument("--d-head", type=int, default=32, dest="d_head")
    p.add_argument("--phase2", action="store_true",
                   help="run the per-node oracle compare even if no assert fires")
    p.add_argument("--atol", type=float, default=1e-2)
    args = p.parse_args(argv)

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    _ensure_doom_sandbox()

    import torch

    import torchwright.graph.node as _node_module
    from torchwright.compiler.export import compile_headless
    from torchwright.debug.probe import reference_eval
    from torchwright.ops.inout_nodes import create_pos_encoding

    from doom_sandbox.implementation import prefill as sb_prefill
    from doom_sandbox.implementation import reference_drafter as drafter
    from doom_sandbox import fixtures

    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED, build_doom_embedding
    from torchwright_doom.past import GraphPast
    from torchwright_doom.render_main import forward
    from torchwright_doom.render.tokens_bridge import rows_to_input, sandbox_token_to_row

    scene = fixtures.load_fixture(args.fixture)
    pose = scene.test_poses[args.pose]
    prefill_rows = [sandbox_token_to_row(t) for t in sb_prefill.get_prefill(scene, pose)]
    ar_rows = [sandbox_token_to_row(t) for t in drafter.expected_ar_tokens(scene, pose)]
    full_rows = prefill_rows + ar_rows
    POS = args.pos
    expected_next = full_rows[POS + 1]
    exp_t, exp_v = TOKEN_VOCAB.row_to_token[expected_next]
    print(f"[probe] {args.fixture} pose={args.pose} pos={POS} "
          f"expected next = {exp_t.name} {exp_v} (row {expected_next})", flush=True)

    # Build the graph ONCE so oracle and compiled share node identity.
    _node_module.global_node_id = 0
    emb = build_doom_embedding("token_ids")
    pe = create_pos_encoding()
    next_token = forward(emb, GraphPast(input_vec=emb, pos_encoding=pe), pe)
    compiled = compile_headless(next_token, pe, d=args.d, d_head=args.d_head,
                                max_layers=200, verbose=False, device="cuda")
    w_embed_t = W_EMBED.t().cuda()

    # Teacher-force to POS via the KV cache: prefill 0..POS-1, then a single debug
    # step at POS (cheap: only one position's residual is captured).
    past = compiled.empty_past()
    _, past = compiled.step(rows_to_input(full_rows[:POS]), past, past_len=0)

    print("\n[probe] Phase 1: debug=True step at POS (checks all Asserts) ...", flush=True)
    fired = None
    try:
        out, _ = compiled.step(rows_to_input([full_rows[POS]]), past, past_len=POS, debug=True)
    except AssertionError as e:
        fired = str(e)
        print(f"[probe] ASSERT FIRED at POS {POS}:\n{fired}", flush=True)
        # Re-run without debug to get the prediction for context.
        out, _ = compiled.step(rows_to_input([full_rows[POS]]), past, past_len=POS)
    pred_row = int(torch.argmax(out[-1] @ w_embed_t).item())
    pred_t, pred_v = TOKEN_VOCAB.row_to_token[pred_row]
    print(f"[probe] compiled prediction at POS {POS}: {pred_t.name} {pred_v} (row {pred_row})"
          f"  {'== expected' if pred_row == expected_next else '!= expected (DIVERGES)'}",
          flush=True)

    if fired is not None and not args.phase2:
        print("[probe] localized via assert (Phase 1). Pass --phase2 for the node compare.")
        return 0

    print("\n[probe] Phase 2: per-node compiled-vs-oracle compare at POS ...", flush=True)
    import torchwright.graph.misc as _misc
    _orig = _misc.Assert._check
    _misc.Assert._check = lambda self, x: None
    try:
        oracle = reference_eval(next_token, {"token_ids": rows_to_input(full_rows[:POS + 1])}, POS + 1)
    finally:
        _misc.Assert._check = _orig

    # Phase 3: is the flip in the unembed argmax, or upstream in the graph?
    # Compare the compiled output embedding to the exact one at POS, and look at
    # both argmax margins over the two contending rows.
    comp_emb = out[-1].detach().float().cuda()
    exact_emb = oracle[next_token][POS].detach().float().cuda()
    emb_linf = (comp_emb - exact_emb).abs().max().item()
    print(f"\n[probe] Phase 3: output-embedding compiled-vs-exact  L_inf={emb_linf:.4g}")
    for label, emb in (("compiled", comp_emb), ("exact   ", exact_emb)):
        logits = emb @ w_embed_t
        top = torch.topk(logits, 3)
        rows = top.indices.tolist()
        vals = top.values.tolist()
        decoded = [f"{TOKEN_VOCAB.row_to_token[r][0].name}{TOKEN_VOCAB.row_to_token[r][1]}" for r in rows]
        print(f"  {label} top-3: " + ", ".join(
            f"{d}(row {r}, logit {v:.6g})" for d, r, v in zip(decoded, rows, vals)))
    # Margin between the expected (node=3) and predicted (node=1) rows, both ways.
    for label, emb in (("compiled", comp_emb), ("exact   ", exact_emb)):
        l = emb @ w_embed_t
        print(f"  {label} logit[expected row {expected_next}]={l[expected_next].item():.6g}  "
              f"logit[predicted row {pred_row}]={l[pred_row].item():.6g}  "
              f"margin(exp-pred)={(l[expected_next]-l[pred_row]).item():.6g}")

    # Topological order: reference_eval inserts nodes as it computes them, so dict
    # order is a valid topological order (inputs before consumers).
    diffs = []
    for node in oracle:
        try:
            cv = compiled.debug_value(node)
        except Exception:
            cv = None
        if cv is None:
            continue
        ov = oracle[node]
        if ov.shape[0] <= POS:
            continue
        c = cv[-1].detach().cpu().float()  # the single debug-step position == POS
        o = ov[POS].detach().cpu().float()
        if c.shape != o.shape:
            continue
        err = (c - o).abs().max().item()
        if err > args.atol:
            diffs.append((node, err, c, o))

    print(f"[probe] nodes diverging > {args.atol} at POS {POS}: {len(diffs)}", flush=True)
    for node, err, c, o in diffs[:15]:
        name = getattr(node, "name", None) or type(node).__name__
        print(f"  {type(node).__name__:20s} {str(name)[:40]:40s} err={err:.4g} "
              f"compiled={_short(c)} oracle={_short(o)}")
    if diffs:
        node, err, c, o = diffs[0]
        print(f"\n[probe] FIRST divergent node (topo): {type(node).__name__} "
              f"{getattr(node,'name',None)} err={err:.4g}")
    return 0


def _short(t):
    v = t.flatten().tolist()
    if len(v) > 6:
        return f"[{', '.join(f'{x:.3g}' for x in v[:6])}, ...]"
    return "[" + ", ".join(f"{x:.3g}" for x in v) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
