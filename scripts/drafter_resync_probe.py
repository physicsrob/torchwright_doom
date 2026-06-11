"""Replay a recorded model token-stream through the ARDrafter and measure resync.

Spec-decode acceptance is exactly ``predicted_row == sandbox_token_to_row(draft)``;
the drafter is fed the model's *actual* emission via ``consume`` after each step,
so a robust (consume-driven) drafter should absorb a structural divergence with a
single mispredict and re-converge.  This probe drives the ARDrafter with a
recorded model stream (a render ``token_dump.json``) and reports, per position,
whether the drafter's draft matched — and crucially the *run-length* of
consecutive mispredicts (isolated points = resync works; long runs = the drafter
cannot recover from that divergence).

No model / GPU / Modal needed: it replays the already-emitted stream, so it
isolates the drafter's recovery behavior from the model's per-pass numerics.

    make modal-run is NOT needed — this is CPU-only:
    TORCHWRIGHT_DOOM_SCREEN_WIDTH=80 TORCHWRIGHT_DOOM_SCREEN_HEIGHT=50 \
    TORCHWRIGHT_DOOM_RENDER_SCALE=4 DOOM_SANDBOX_SCREEN_WIDTH=80 \
    DOOM_SANDBOX_SCREEN_HEIGHT=50 uv run python -m scripts.drafter_resync_probe \
        --config configs/e1m1.yaml \
        --dump out/render_d8192/token_dump.json \
        --x 1056 --y -3616 --angle 64 --viewz 41
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def _run_lengths(flags: list[bool]) -> list[int]:
    runs: list[int] = []
    cur = 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dump", required=True, help="token_dump.json from a render run")
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--angle", type=int, required=True)
    ap.add_argument("--viewz", type=float, required=True)
    args = ap.parse_args()

    from torchwright_doom.render.config import apply_screen_env, load_render_config
    from torchwright_doom.render.wad_scene import (
        load_render_scene,
        pose_from_world,
        sandbox_scene_for,
    )

    cfg = load_render_config(args.config)
    apply_screen_env(cfg)
    scene = load_render_scene(cfg, base_dir=str(Path(args.config).parent))
    pose = pose_from_world(scene, x=args.x, y=args.y, angle=args.angle, viewz=args.viewz)
    sb_scene = sandbox_scene_for(scene, pose)
    sb_pose = sb_scene.test_poses[0]

    from doom_sandbox.api.tokens import Token
    from doom_sandbox.implementation.reference_drafter import ARDrafter

    from torchwright_doom.render.tokens_bridge import _sandbox_types, sandbox_token_to_row

    sb_types = _sandbox_types()
    drafter = ARDrafter(sb_scene, sb_pose)

    stream = json.loads(Path(args.dump).read_text())["cases"][0]["predicted_next_tokens"]

    def to_tok(rec: dict) -> Token:
        return Token(sb_types[rec["type"]], dict(rec["values"]))

    mispredict: list[bool] = []
    types: list[str] = []
    drafter_none = 0
    for rec in stream:
        actual = to_tok(rec)
        types.append(rec["type"])
        draft = drafter.next_draft()
        if draft is None:
            drafter_none += 1
            mispredict.append(True)  # no draft offered == a forced single-step
        else:
            try:
                matched = sandbox_token_to_row(draft) == sandbox_token_to_row(actual)
            except Exception:
                matched = False
            mispredict.append(not matched)
        drafter.consume(actual)

    # Token-type breakdown of the longest mispredict runs (>=20): what structure
    # the drafter cannot recover within.
    from collections import Counter

    long_run_types: Counter = Counter()
    i = 0
    while i < len(mispredict):
        if mispredict[i]:
            j = i
            while j < len(mispredict) and mispredict[j]:
                j += 1
            if j - i >= 20:
                for k in range(i, j):
                    long_run_types[types[k]] += 1
            i = j
        else:
            i += 1
    print("token types inside long mispredict runs (>=20):")
    for t, c in long_run_types.most_common(8):
        print(f"    {t:24} {c}")

    n = len(mispredict)
    runs = _run_lengths(mispredict)
    total_mis = sum(mispredict)
    print(f"positions={n}  mispredicts={total_mis} ({total_mis/n:.1%})  "
          f"accept={1 - total_mis/n:.1%}  drafter_returned_None={drafter_none}")
    if runs:
        print(f"mispredict runs: count={len(runs)}  max_run={max(run := runs)}  "
              f"median={st.median(runs)}  isolated(len==1)={sum(1 for r in runs if r == 1)}  "
              f">=10={sum(1 for r in runs if r >= 10)}  >=50={sum(1 for r in runs if r >= 50)}")
    # Per-quarter accept rate to locate where (if anywhere) it collapses.
    q = max(1, n // 4)
    for i in range(0, n, q):
        seg = mispredict[i : i + q]
        if seg:
            print(f"  pos[{i:5}-{i+len(seg):5}] accept={1 - sum(seg)/len(seg):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
