"""Localize the first teacher-forced divergence of the compiled free-run (Plan K).

Feeds the *reference* token stream through the cached production ONNX artifact
(via ``render.cache.load_debug_session`` — the same bits production renders
with) in one wide forward (no AR rollout) and reports the first position whose
compiled argmax hard-diverges from the reference under the J2 bars.
Distinguishes a per-position compiled error (teacher-forced hard-diverges -> an
op exceeds its noise budget, localizable with ``k_probe_divergence``) from pure
AR-feedback amplification (teacher-forced stays clean).

    .venv/bin/python -m scripts.k_localize_divergence --window 3000

Needs the config's compiled cache entry to exist already (this never compiles —
build it via ``python -m torchwright_doom.render compile --config <yaml>``).
The single wide forward is O(n_pos^2) in attention memory, so ``--window`` caps
the fed prefix (3000 covers the observed AR onset at ~2448).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO / "configs" / "e1m1.yaml"


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
    p.add_argument("--window", type=int, default=3000)
    p.add_argument("--config", default=str(_DEFAULT_CONFIG), dest="config_path")
    p.add_argument(
        "--cache-dir",
        default=None,
        dest="cache_dir",
        help="compiled cache entry holding model.onnx "
        "(default: the config's own cache key)",
    )
    args = p.parse_args(argv)

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    # Screen env BEFORE any graph/sandbox module imports — constants.py bakes
    # the dims at import and the artifact was compiled at the config's dims.
    from torchwright_doom.render.config import apply_screen_env, load_render_config

    config_path = Path(args.config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)

    _ensure_doom_sandbox()
    from doom_sandbox import fixtures
    from doom_sandbox.implementation import prefill as sb_prefill
    from doom_sandbox.implementation import reference_drafter as drafter

    from torchwright_doom.render import compare
    from torchwright_doom.render.cache import load_debug_session
    from torchwright_doom.render.diagnostic import teacher_forced_scan
    from torchwright_doom.render.tokens_bridge import sandbox_token_to_row

    scene = fixtures.load_fixture(args.fixture)
    pose = scene.test_poses[args.pose]
    prefill_rows = [sandbox_token_to_row(t) for t in sb_prefill.get_prefill(scene, pose)]
    ar_rows = [sandbox_token_to_row(t) for t in drafter.expected_ar_tokens(scene, pose)]
    full_rows = prefill_rows + ar_rows
    begin = len(prefill_rows) - 1
    options = compare.reference_options(scene, pose)

    print(
        f"[localize] {args.fixture} pose={args.pose} prefill={len(prefill_rows)} "
        f"reference_rollout={len(ar_rows)} full={len(full_rows)} window={args.window}",
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
    n_ar = fed - begin - 1
    print(f"[localize] teacher-forced {fed} positions ({n_ar} AR predictions checked)")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
