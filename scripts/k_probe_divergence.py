"""Measure where the compiled artifact departs exact math at the K divergence.

Rebuilds the graph ONCE via ``inference.compiled_model.build_graph`` (so the
exact-math oracle and the artifact's debug session share node identity — the
session's fingerprint check guarantees the rebuild matches the compiled graph),
opens an :class:`OnnxDebugSession` over the cached production ONNX,
teacher-forces the reference stream up to the first hard divergence via the KV
cache, then:

  Phase 1 (cheap): replay the divergence position with ``debug=True`` — re-checks
  every Assert/DebugWatch on the compiled values. A firing assert names the exact
  node + invariant (CLAUDE.md triage step 1).

  Phase 2 (if Phase 1 is clean): compare every graph node's compiled value to the
  exact-math oracle (reference_eval) AT the divergence position, and report the
  first divergent node in topological order — the origin of the fp32 departure.

    .venv/bin/python -m scripts.k_probe_divergence --pos 2450

Needs the config's compiled cache entry to exist already (this never compiles —
build it via ``python -m torchwright_doom.inference compile --config <yaml>``).
The wide prefill pass doesn't fit the local L4 (promoted debug outputs disable
ORT's memory planning) — run locally on CPU via ``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO / "configs" / "e1m1.yaml"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(_DEFAULT_CONFIG), dest="config_path")
    p.add_argument(
        "--cache-dir",
        default=None,
        dest="cache_dir",
        help="compiled cache entry holding model.onnx "
        "(default: the config's own cache key)",
    )
    p.add_argument(
        "--pos",
        type=int,
        default=2450,
        help="input stream position whose prediction diverges",
    )
    p.add_argument("--x", type=float, help="world pose (default: config default pose)")
    p.add_argument("--y", type=float)
    p.add_argument("--angle", type=int)
    p.add_argument("--viewz", type=float)
    p.add_argument(
        "--phase2",
        action="store_true",
        help="run the per-node oracle compare even if no assert fires",
    )
    p.add_argument("--atol", type=float, default=1e-2)
    args = p.parse_args(argv)

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    # Screen env BEFORE any graph/sandbox module imports — constants.py bakes
    # the dims at import and the artifact was compiled at the config's dims.
    from torchwright_doom.inference.config import (
        apply_screen_env,
        compile_cache_dir,
        load_render_config,
        resolve_wad_path,
    )

    config_path = Path(args.config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)

    import torch

    from torchwright.debug.onnx_debug import OnnxDebugSession
    from torchwright.debug.probe import reference_eval

    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
    from torchwright_doom.inference.compiled_model import build_graph
    from torchwright_doom.inference.tokens_bridge import rows_to_input
    from torchwright_doom.inference.wad_scene import reference_stream

    _sb_scene, sb_pose, prefill_rows, full_rows = reference_stream(
        config,
        base_dir=config_path.parent,
        x=args.x,
        y=args.y,
        angle=args.angle,
        viewz=args.viewz,
    )
    POS = args.pos
    expected_next = full_rows[POS + 1]
    exp_t, exp_v = TOKEN_VOCAB.row_to_token[expected_next]
    print(
        f"[probe] {config_path.name} pose=({sb_pose.x:g}, {sb_pose.y:g}, "
        f"a{sb_pose.angle}) prefill={len(prefill_rows)} pos={POS} "
        f"expected next = {exp_t.name} {exp_v} (row {expected_next})",
        flush=True,
    )

    # Locate the cached artifact (never compile — that's a Modal-scale job).
    wad_path = resolve_wad_path(config, base_dir=config_path.parent)
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else compile_cache_dir(config, wad_path)
    )
    onnx_path = cache_dir / "model.onnx"
    if not onnx_path.exists() or not (cache_dir / "model.debug.json").exists():
        raise SystemExit(
            f"[probe] no debuggable artifact at {cache_dir} (model.onnx + "
            f"model.debug.json required) — recompile via `python -m "
            f"torchwright_doom.inference compile --config {config_path}`"
        )

    # Build the graph ONCE so the oracle and the debug session share node
    # identity; the session's fingerprint check fails loud if this rebuild
    # differs from the graph the artifact was compiled from.
    next_token, pos_enc, _emb, _banks = build_graph(
        asset_config=config.asset_config(), wad_path=wad_path
    )
    providers = None
    if torch.cuda.is_available():
        # fp32 only: TF32 collapses the content-addressed attention's
        # unit-score logit gaps (see inference._default_ort_providers).
        providers = [
            ("CUDAExecutionProvider", {"use_tf32": "0"}),
            "CPUExecutionProvider",
        ]
    session = OnnxDebugSession(str(onnx_path), next_token, pos_enc, providers=providers)

    # Teacher-force to POS via the KV cache: prefill 0..POS-1, then a single debug
    # step at POS (cheap: only one position's residual is captured).
    past = session.empty_past()
    _, past = session.step(rows_to_input(full_rows[:POS]), past, past_len=0)

    print(
        "\n[probe] Phase 1: debug=True step at POS (checks all Asserts) ...", flush=True
    )
    fired = None
    try:
        out, _ = session.step(
            rows_to_input([full_rows[POS]]), past, past_len=POS, debug=True
        )
    except AssertionError as e:
        fired = str(e)
        print(f"[probe] ASSERT FIRED at POS {POS}:\n{fired}", flush=True)
        # Re-run without debug to get the prediction for context.
        out, _ = session.step(rows_to_input([full_rows[POS]]), past, past_len=POS)
    # The session returns LOGITS (the artifact's own unembed) — argmax directly.
    compiled_logits = out[-1].detach().float()
    pred_row = int(torch.argmax(compiled_logits).item())
    pred_t, pred_v = TOKEN_VOCAB.row_to_token[pred_row]
    print(
        f"[probe] compiled prediction at POS {POS}: {pred_t.name} {pred_v} (row {pred_row})"
        f"  {'== expected' if pred_row == expected_next else '!= expected (DIVERGES)'}",
        flush=True,
    )

    if fired is not None and not args.phase2:
        print(
            "[probe] localized via assert (Phase 1). Pass --phase2 for the node compare."
        )
        return 0

    print(
        "\n[probe] Phase 2: per-node compiled-vs-oracle compare at POS ...", flush=True
    )
    from torchwright_doom.graph_debug import silenced_graph_asserts

    with silenced_graph_asserts():
        oracle = reference_eval(
            next_token, {"token_ids": rows_to_input(full_rows[: POS + 1])}, POS + 1
        )

    # Phase 3: is the flip in the unembed argmax, or upstream in the graph?
    # Compare the compiled logits row to the exact one at POS (exact side via
    # the oracle's output embedding @ W_EMBED.T), and look at both argmax
    # margins over the two contending rows.
    exact_logits = oracle[next_token][POS].detach().float() @ W_EMBED.t().float()
    logit_linf = (compiled_logits - exact_logits).abs().max().item()
    print(f"\n[probe] Phase 3: logits compiled-vs-exact  L_inf={logit_linf:.4g}")
    for label, logits in (("compiled", compiled_logits), ("exact   ", exact_logits)):
        top = torch.topk(logits, 3)
        rows = top.indices.tolist()
        vals = top.values.tolist()
        decoded = [
            f"{TOKEN_VOCAB.row_to_token[r][0].name}{TOKEN_VOCAB.row_to_token[r][1]}"
            for r in rows
        ]
        print(
            f"  {label} top-3: "
            + ", ".join(
                f"{d}(row {r}, logit {v:.6g})" for d, r, v in zip(decoded, rows, vals)
            )
        )
    # Margin between the expected and predicted rows, both ways.
    for label, l in (("compiled", compiled_logits), ("exact   ", exact_logits)):
        print(
            f"  {label} logit[expected row {expected_next}]={l[expected_next].item():.6g}  "
            f"logit[predicted row {pred_row}]={l[pred_row].item():.6g}  "
            f"margin(exp-pred)={(l[expected_next]-l[pred_row]).item():.6g}"
        )

    # Topological order: reference_eval inserts nodes as it computes them, so dict
    # order is a valid topological order (inputs before consumers).
    diffs = []
    for node in oracle:
        try:
            cv = session.debug_value(node)
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

    print(
        f"[probe] nodes diverging > {args.atol} at POS {POS}: {len(diffs)}", flush=True
    )
    for node, err, c, o in diffs[:15]:
        name = getattr(node, "name", None) or type(node).__name__
        print(
            f"  {type(node).__name__:20s} {str(name)[:40]:40s} err={err:.4g} "
            f"compiled={_short(c)} oracle={_short(o)}"
        )
    if diffs:
        node, err, c, o = diffs[0]
        print(
            f"\n[probe] FIRST divergent node (topo): {type(node).__name__} "
            f"{getattr(node,'name',None)} err={err:.4g}"
        )
    return 0


def _short(t):
    v = t.flatten().tolist()
    if len(v) > 6:
        return f"[{', '.join(f'{x:.3g}' for x in v[:6])}, ...]"
    return "[" + ", ".join(f"{x:.3g}" for x in v) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
