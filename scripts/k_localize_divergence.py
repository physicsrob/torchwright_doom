"""Localize the first teacher-forced divergence of the compiled free-run (Plan K).

THE manual faithfulness gate for a cached artifact (it replaced
``tests/render/test_compiled_divergence.py``): feeds the *reference* token
stream — the config scene's prefill plus the sandbox reference renderer's
expected rollout — through the cached production ONNX (via
``render.cache.load_debug_session``, the same bits production renders with) as
a KV-cached chunked teacher-forced pass (no AR rollout), and reports every
position whose compiled argmax hard-diverges from the reference under the J2
bars.  Exits nonzero on any hard divergence.  Distinguishes a per-position
compiled error (teacher-forced hard-diverges -> an op exceeds its noise
budget, localizable with ``k_probe_divergence``) from pure AR-feedback
amplification (teacher-forced stays clean).

    .venv/bin/python -m scripts.k_localize_divergence --window 3700

Needs the config's compiled cache entry to exist already (this never compiles —
build it via ``python -m torchwright_doom.render compile --config <yaml>``).
``--window`` caps the fed prefix; the AR region starts past the prefill
(~3613 rows on the e1m1 scene), so a useful gate run needs a window beyond
that.  Debug-session passes don't fit the local L4 (promoted outputs disable
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
    p.add_argument("--window", type=int, default=3000)
    p.add_argument("--x", type=float, help="world pose (default: config default pose)")
    p.add_argument("--y", type=float)
    p.add_argument("--angle", type=int)
    p.add_argument("--viewz", type=float)
    args = p.parse_args(argv)

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    # Screen env BEFORE any graph/sandbox module imports — constants.py bakes
    # the dims at import and the artifact was compiled at the config's dims.
    from torchwright_doom.render.config import apply_screen_env, load_render_config

    config_path = Path(args.config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)

    from torchwright_doom.render import compare
    from torchwright_doom.render.cache import load_debug_session
    from torchwright_doom.render.diagnostic import teacher_forced_scan
    from torchwright_doom.render.wad_scene import reference_stream

    sb_scene, sb_pose, prefill_rows, full_rows = reference_stream(
        config,
        base_dir=config_path.parent,
        x=args.x,
        y=args.y,
        angle=args.angle,
        viewz=args.viewz,
    )
    begin = len(prefill_rows) - 1
    options = compare.reference_options(sb_scene, sb_pose)

    print(
        f"[localize] {config_path.name} pose=({sb_pose.x:g}, {sb_pose.y:g}, "
        f"a{sb_pose.angle}) prefill={len(prefill_rows)} "
        f"reference_rollout={len(full_rows) - len(prefill_rows)} "
        f"full={len(full_rows)} window={args.window}",
        flush=True,
    )

    import torch

    providers = None
    if torch.cuda.is_available():
        # fp32 only: TF32 collapses the content-addressed attention's
        # unit-score logit gaps (see inference._default_ort_providers).
        providers = [
            ("CUDAExecutionProvider", {"use_tf32": "0"}),
            "CPUExecutionProvider",
        ]
    session = load_debug_session(
        args.cache_dir, config, base_dir=config_path.parent, providers=providers
    )
    divs = teacher_forced_scan(session, full_rows, begin, options, window=args.window)

    fed = min(args.window, len(full_rows))
    n_ar = max(0, fed - begin - 1)
    print(f"[localize] teacher-forced {fed} positions ({n_ar} AR predictions checked)")
    if n_ar == 0:
        print(
            f"[localize] WARNING: window {args.window} does not reach past the "
            f"prefill ({len(prefill_rows)} rows) — no AR predictions were "
            f"checked; raise --window above {begin + 2}",
            flush=True,
        )
    print(f"[localize] hard divergences: {len(divs)}")
    if not divs:
        print("[localize] CLEAN: teacher-forced compiled matches the reference under the "
              "J2 bars over the whole window -> AR failure is feedback amplification, "
              "not a per-position op error.")
        return 0
    for d in divs[:12]:
        roll = d.pos - begin  # 0-based position in the AR rollout
        print(
            f"  pos {d.pos} (rollout {roll}): expected {d.expected_type}"
            f"(row {d.expected_row}) -> predicted {d.predicted_type}(row {d.predicted_row})"
            f"  [{d.kind}: {d.detail}]"
        )
    first = divs[0]
    print(
        f"\n[localize] FIRST hard divergence at stream pos {first.pos} "
        f"(rollout token {first.pos - begin}): {first.kind} — "
        f"expected {first.expected_type}, predicted {first.predicted_type}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
