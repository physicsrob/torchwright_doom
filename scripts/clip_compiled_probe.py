"""CLAUDE.md triage on the wallSpanMeta y=0 flip: in-process compiled probe.

Compiles forward() headless at PRODUCTION widths (d=8192, d_head=128,
d_hidden=16384, bias=False), teacher-forces the golden stream to just
past the failing position (4400: golden wallSpanMeta(y=31), artifact
emits y=0), then:
  1. prediction check at the failing position on the compiled module;
  2. probe_compiled first-divergent (compiled vs exact oracle, atol=500);
  3. probe_attention at the ClipMemory pick for the failing query pos.

    MODAL_RUN_CPU=64 MODAL_RUN_MEMORY=262144 MODAL_RUN_TIMEOUT=7200 \
        make modal-run MODULE=scripts.clip_compiled_probe CPU_ONLY=1
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e1m1_lowres.yaml")
    ap.add_argument("--n-pos", type=int, default=4402)
    ap.add_argument("--fail-pos", type=int, default=4400)
    ap.add_argument("--d", type=int, default=8192)
    ap.add_argument("--d-head", type=int, default=128)
    ap.add_argument("--d-hidden", type=int, default=16384)
    ap.add_argument("--skip-probe-compiled", action="store_true")
    args = ap.parse_args()

    from torchwright_doom.inference.config import apply_screen_env, load_render_config

    config = load_render_config(args.config)
    apply_screen_env(config)  # BEFORE any graph-module import (the env trap)

    # Creation-site capture so the ClipMemory pick can be found by site.
    from torchwright.graph import node as node_mod

    _orig_init = node_mod.Node.__init__
    _sites: dict[int, str] = {}

    def _capturing_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        frames = []
        for fr in traceback.extract_stack()[:-1]:
            if "/torchwright_doom/" in fr.filename and "/scripts/" not in fr.filename:
                frames.append(
                    f"{os.path.basename(fr.filename)}:{fr.lineno} ({fr.name})"
                )
        _sites[id(self)] = "  <=  ".join(reversed(frames[-3:])) if frames else "?"

    node_mod.Node.__init__ = _capturing_init  # type: ignore[method-assign]

    import torch

    import torchwright_doom.pydoom as pydoom
    from torchwright.compiler.export import compile_headless
    from torchwright.debug.probe import probe_attention
    from torchwright.ops.inout_nodes import create_input, create_rope_config
    from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
    from torchwright_doom.graph_debug import silenced_graph_asserts
    from torchwright_doom.inference.tokens_bridge import row_index, tokens_to_input
    from torchwright_doom.inference.wad_scene import (
        load_render_scene,
        pose_from_world,
        pydoom_scene_for,
    )
    from torchwright_doom.past import GraphPast
    from torchwright_doom.prompt.build import build_prompt
    from torchwright_doom.render_main import forward

    scene = load_render_scene(config, base_dir=Path.cwd())
    pose = pose_from_world(scene)
    py_scene = pydoom_scene_for(scene, pose)
    py_pose = py_scene.test_poses[0]
    prefill = list(build_prompt(scene.map_data, pose, asset_config=scene.asset_config))
    golden = list(pydoom.expected_ar_tokens(py_scene, py_pose))
    full = prefill + golden
    rows = [row_index(t.type, dict(t.values)) for t in full]
    n_pos = min(args.n_pos, len(rows))
    print(f"[cprobe] prefill={len(prefill)} n_pos={n_pos} fail_pos={args.fail_pos}")

    def text(row: int) -> str:
        rtype, values = TOKEN_VOCAB.row_to_token[row]
        if values:
            parts = [
                f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in values.items()
            ]
            return f"{rtype.name}({','.join(parts)})"
        return rtype.name

    real_pairs = [(t.type, dict(t.values)) for t in full]
    inputs = {"iv": tokens_to_input(real_pairs[:n_pos])}
    d_embed = TOKEN_VOCAB.layout.d_embed
    iv = create_input("iv", d_embed)
    rope = create_rope_config(d_head=128, max_positions=65536, d_rot=64)
    past = GraphPast(input_vec=iv, rope=rope)
    next_token = forward(iv, past)

    print(
        f"[cprobe] compiling headless d={args.d} d_head={args.d_head} "
        f"d_hidden={args.d_hidden} bias=False optimize=0 ...",
        flush=True,
    )
    compiled = compile_headless(
        next_token,
        d=args.d,
        d_head=args.d_head,
        d_hidden=args.d_hidden,
        bias=False,
        optimize=0,
        max_layers=400,
        verbose=True,
    )
    print("[cprobe] compiled", flush=True)

    with silenced_graph_asserts():
        out = compiled(inputs["iv"], debug=True)
    w_embed_t = torch.as_tensor(W_EMBED, dtype=out.dtype).T
    pred = torch.argmax(out @ w_embed_t, dim=-1)
    p = args.fail_pos
    print(
        f"[cprobe] prediction at pos {p}: want {text(rows[p + 1])} "
        f"got {text(int(pred[p]))}"
    )
    for q in range(p - 3, min(p + 2, n_pos - 1)):
        print(f"    pos {q:>5}: want {text(rows[q + 1]):<40} got {text(int(pred[q]))}")

    # ClipMemory pick: Attn created under wall_column_state ClipMemory.publish.
    clip_attns = []
    seen = set()
    stack = [next_token]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        site = _sites.get(id(n), "")
        if (
            type(n).__name__ == "Attn"
            and "wall_column_state" in site
            and ("ClipMemory" in site or "publish" in site)
        ):
            clip_attns.append((site, n))
        stack.extend(getattr(n, "inputs", []) or [])
    print(f"[cprobe] candidate wall_column_state Attn nodes: {len(clip_attns)}")
    for site, n in clip_attns:
        print(f"    {site}")

    for site, n in clip_attns:
        try:
            apr = probe_attention(compiled, inputs["iv"], n, query_pos=p)
            top = apr.top(k=5, head=0)
            print(f"[cprobe] attention at pos {p} for {site}:")
            print(f"    {top}")
        except Exception as exc:
            print(f"[cprobe] probe_attention failed for {site}: {exc}")

    # Compiled-vs-exact of the span-state publish at the MARKED rows the
    # reads actually recover (junk-row divergence is by-design; only
    # marked rows matter). Table over positions whose input row is a
    # wallSpanMeta / screenY token near the failure.
    from torchwright.debug.probe import reference_eval

    print("[cprobe] reference_eval ...", flush=True)
    with silenced_graph_asserts():
        cache = reference_eval(next_token, inputs, n_pos)

    # the finish-publish concat (y1, y2-ish, ...) and the read output
    targets = []
    seen2 = set()
    stack = [next_token]
    while stack:
        n = stack.pop()
        if id(n) in seen2:
            continue
        seen2.add(id(n))
        stack.extend(getattr(n, "inputs", []) or [])
        site = _sites.get(id(n), "")
        if getattr(n, "d_output", 999) > 8 or n not in cache:
            continue
        if "wall_column_state.py:977" in site and type(n).__name__ == "Concatenate":
            targets.append(("publish@977", n))
        if "wall_column_state.py:629" in site and type(n).__name__ == "Attn":
            targets.append(("read@629", n))
    print(f"[cprobe] target nodes: {[t[0] for t in targets]}")

    meta_positions = [
        q
        for q in range(3614, n_pos)
        if TOKEN_VOCAB.row_to_token[rows[q]][0].name in ("wallSpanMeta", "screenY")
    ]
    for label, n in targets[:4]:
        cv = compiled.debug_value(n)
        ev = cache[n]
        print(f"[cprobe] {label} at marked rows (pos: exact -> compiled):")
        for q in meta_positions[-30:]:
            e = [round(float(x), 2) for x in ev[q]]
            c = [round(float(x), 2) for x in cv[q]]
            flag = "  <== DIFF" if max(abs(a - b) for a, b in zip(e, c)) > 0.5 else ""
            print(
                f"    {q:>5} {TOKEN_VOCAB.row_to_token[rows[q]][0].name:<13} {e} -> {c}{flag}"
            )


if __name__ == "__main__":
    main()
