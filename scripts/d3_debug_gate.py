"""D3 gate: debug=True step + probe_compiled over the bias=False artifact.

The direct check on the folded arithmetic (swiglu_cutover_plan.md, D3 gate
3): open the cached swish+bias=False ONNX artifact in an OnnxDebugSession,
teacher-force the real e1m1 stream through it, and

1. run a ``debug=True`` step over an AR window — the residual
   self-consistency check (structural compiler integrity on the folded
   artifact) plus the graph's Assert predicates on compiled values;
2. run ``probe_compiled`` over a prefill window — node-by-node compiled
   vs exact-math oracle, ``atol=500`` (the doom graph's empirical floor,
   see CLAUDE.md "Debugging compiled graphs").

The Assert leg runs twice: once live, once with asserts silenced (the
renderer's discarded garbage rows can legally land conds inside comparator
ramps — graph_debug.silenced_graph_asserts exists for exactly this), so a
live-assert failure can be told apart from a garbage-row artifact.

Run on Modal (the artifact lives in the compile-cache volume):

    make modal-run MODULE=scripts.d3_debug_gate \\
        ARGS="--cache-dir /root/.cache/torchwright_doom/compiled/<key>"
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e1m1_lowres.yaml")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--ar-window", type=int, default=64)
    parser.add_argument("--probe-positions", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--debug-first-chunk",
        action="store_true",
        help="run the debug=True step on the FIRST chunk (positions 0..chunk) "
        "instead of the last — separates position-depth effects from the "
        "debug feed itself",
    )
    args = parser.parse_args()

    from torchwright_doom.inference.config import apply_screen_env, load_render_config

    config = load_render_config(args.config)
    apply_screen_env(config)  # BEFORE any graph-module import (the env trap)

    # Stash the global-position node from the session's own graph build so
    # debug_value can read its compiled value (the 148352-plateau diagnosis).
    import torchwright_doom.past as _past_mod

    _gp_nodes: list = []
    _orig_gp = _past_mod.GraphPast.global_position

    def _gp_patched(self):
        node = _orig_gp(self)
        if node not in _gp_nodes:
            _gp_nodes.append(node)
        return node

    _past_mod.GraphPast.global_position = _gp_patched  # type: ignore[method-assign]

    import torchwright_doom.pydoom as pydoom
    from torchwright_doom.graph_debug import silenced_graph_asserts
    from torchwright_doom.inference.compile_cache import load_debug_session
    from torchwright_doom.inference.tokens_bridge import row_index, rows_to_input
    from torchwright_doom.inference.wad_scene import (
        load_render_scene,
        pose_from_world,
        pydoom_scene_for,
    )
    from torchwright_doom.prompt.build import build_prompt

    scene = load_render_scene(config, base_dir=Path.cwd())
    pose = pose_from_world(scene)
    py_scene = pydoom_scene_for(scene, pose)
    py_pose = py_scene.test_poses[0]
    prefill = list(build_prompt(scene.map_data, pose, asset_config=scene.asset_config))
    golden = list(pydoom.expected_ar_tokens(py_scene, py_pose))
    full = prefill + golden
    begin = len(prefill) - 1
    rows = [row_index(t.type, dict(t.values)) for t in full]
    print(f"[gate] stream: prefill={len(prefill)} full={len(full)} begin={begin}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    compiled = load_debug_session(args.cache_dir, config, providers=providers)
    print(f"[gate] debug session open over {args.cache_dir or '<config cache>'}")

    # ---- Leg 1: debug=True step over an AR window ----------------------
    n_dbg = begin + args.ar_window

    def _feed(debug_last: bool, silence: bool) -> None:
        past = compiled.empty_past()
        offset = 0
        while offset < n_dbg:
            chunk = rows[offset : offset + args.chunk_size]
            if args.debug_first_chunk:
                is_dbg = offset == 0
            else:
                is_dbg = offset + len(chunk) >= n_dbg
            if is_dbg and debug_last:
                if silence:
                    with silenced_graph_asserts():
                        _out, past = compiled.step(
                            rows_to_input(chunk), past, past_len=offset, debug=True
                        )
                else:
                    _out, past = compiled.step(
                        rows_to_input(chunk), past, past_len=offset, debug=True
                    )
            else:
                _out, past = compiled.step(rows_to_input(chunk), past, past_len=offset)
            offset += len(chunk)

    print("[gate] leg 1a: debug=True step (asserts SILENCED — consistency only)")
    _feed(debug_last=True, silence=True)
    print("[gate] leg 1a PASS: residual self-consistency holds on the folded artifact")

    if _gp_nodes:
        gp_val = compiled.debug_value(_gp_nodes[0])
        if gp_val is not None:
            lo, hi = float(gp_val.min()), float(gp_val.max())
            print(
                f"[gate] global-position compiled values over the debug chunk: "
                f"min={lo:.1f} max={hi:.1f} "
                f"(chunk positions ~{0 if args.debug_first_chunk else n_dbg - args.chunk_size}"
                f"..{args.chunk_size if args.debug_first_chunk else n_dbg})"
            )

    print("[gate] leg 1b: debug=True step (asserts LIVE)")
    try:
        _feed(debug_last=True, silence=False)
        print("[gate] leg 1b PASS: every Assert predicate holds on compiled values")
    except AssertionError as exc:
        print(f"[gate] leg 1b: Assert fired (judge garbage-row vs real): {exc}")

    # ---- Leg 2: probe_compiled over a prefill window --------------------
    from torchwright.debug.probe import probe_compiled
    from torchwright_doom.inference.compiled_model import build_graph
    from torchwright_doom.inference.config import resolve_wad_path

    n_probe = min(args.probe_positions, len(rows))
    wad_path = resolve_wad_path(config, base_dir=Path.cwd())
    next_token, _rope, _emb, _banks = build_graph(
        d_head=config.model.d_head,
        max_positions=config.model.max_seq_len,
        d_rot=config.model.d_rot,
        asset_config=config.asset_config(),
        wad_path=wad_path,
    )
    print(f"[gate] leg 2: probe_compiled over positions [0, {n_probe}) atol=500")
    with silenced_graph_asserts():
        report = probe_compiled(
            compiled,
            next_token,
            {"token_ids": rows_to_input(rows[:n_probe])},
            n_probe,
            atol=500.0,
        )
    if report.first_divergent is None:
        print("[gate] leg 2 PASS: compiled matches the exact-math oracle everywhere")
    else:
        print("[gate] leg 2 FAIL:")
        print(report.format_short())
        raise SystemExit(1)

    print("[gate] D3 debug gate complete")


if __name__ == "__main__":
    main()
