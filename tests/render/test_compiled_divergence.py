"""Targeted reproducer: per-position faithfulness of the cached production artifact.

Plan K Step 1 found the compiled autoregressive free-run can render the first
visible subsector correctly and then diverge in the BSP/bbox traversal. This test
isolates any such failure as a single per-position compiled error, with **one wide
teacher-forced forward** (not an AR rollout): feed the *reference* token stream
through the cached production ONNX artifact (``configs/e1m1.yaml``) and check
each position's argmax under the J2 bars (markers exact, carriers in tolerance,
pixels in option set). Because the context is teacher-forced (correct), a hard
divergence here is a genuine per-position compiled error (an op over its noise
budget), not AR-feedback amplification.

The model is the artifact itself, run through ``OnnxDebugSession`` (via
``render.cache.load_debug_session``) — the same bits production renders with, so
this also covers ONNX-emission and execution-provider faults that an in-process
recompile is structurally blind to. The test NEVER compiles: it skips unless the
config's cache entry already exists locally (a d=8192 compile is a Modal-scale
job). Localize a failure with ``scripts/k_probe_divergence.py`` /
``scripts/k_localize_divergence.py`` (CLAUDE.md triage).

Fast relative to the free-run: one forward over ~2.5k positions vs thousands of
sequential steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "e1m1.yaml"

# Just past the first hard divergence Plan K observed (stream pos 2451). Keeps
# the single wide forward's O(n_pos^2) attention affordable while still reaching
# deep into the BSP traversal.
_WINDOW = 2470


def _session_or_skip(device):
    """Load the e1m1 config and open a debug session over its cache entry.

    All torchwright_doom graph modules import AFTER apply_screen_env — the
    artifact is compiled at the config's screen dims (scale 2 = 160x100), and
    constants.py bakes the dims at import.
    """
    from torchwright_doom.render.config import (
        apply_screen_env,
        compile_cache_dir,
        load_render_config,
        resolve_wad_path,
    )

    if not _CONFIG_PATH.exists():
        pytest.skip(
            f"no committed config at {_CONFIG_PATH} (containerized run "
            f"without configs/)"
        )
    config = load_render_config(_CONFIG_PATH)
    apply_screen_env(config)
    try:
        wad_path = resolve_wad_path(config, base_dir=_CONFIG_PATH.parent)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    cache_dir = compile_cache_dir(config, wad_path)
    if not (cache_dir / "model.onnx").exists():
        pytest.skip(
            f"no cached e1m1 artifact at {cache_dir} — this test never "
            f"compiles; build it via `python -m torchwright_doom.render "
            f"compile --config {_CONFIG_PATH}`"
        )

    from torchwright_doom.render.cache import load_debug_session

    providers = None
    if device.type == "cuda":
        # fp32 only: the content-addressed attention's unit-score logit gaps
        # collapse under TF32 (see inference._default_ort_providers).
        providers = [
            ("CUDAExecutionProvider", {"use_tf32": "0"}),
            "CPUExecutionProvider",
        ]
    return config, load_debug_session(
        cache_dir, config, base_dir=_CONFIG_PATH.parent, providers=providers
    )


def _reference_stream(config):
    """Reference token stream for the config's default pose: prefill + the
    sandbox reference renderer's expected AR rollout."""
    drafter = pytest.importorskip("doom_sandbox.implementation.reference_drafter")

    from torchwright_doom.render.tokens_bridge import sandbox_token_to_row
    from torchwright_doom.render.wad_scene import (
        load_render_scene,
        pose_from_world,
        prefill_rows_for,
        sandbox_scene_for,
    )

    render_scene = load_render_scene(config, base_dir=_CONFIG_PATH.parent)
    pose = pose_from_world(render_scene)
    prefill_rows = prefill_rows_for(render_scene, pose)
    sb_scene = sandbox_scene_for(render_scene, pose)
    sb_pose = sb_scene.test_poses[0]
    ar_rows = [
        sandbox_token_to_row(t)
        for t in drafter.expected_ar_tokens(sb_scene, sb_pose)
    ]
    return sb_scene, sb_pose, prefill_rows, prefill_rows + ar_rows


def test_compiled_teacher_forced_free_run_is_faithful(device):
    """The cached artifact, fed the correct context, must predict the reference
    next token at every position (under the J2 bars)."""
    if device.type != "cuda":
        pytest.skip("the wide teacher-forced forward over the artifact needs a GPU")

    config, session = _session_or_skip(device)
    scene, pose, prefill_rows, full_rows = _reference_stream(config)
    begin = len(prefill_rows) - 1

    from torchwright_doom.render import compare
    from torchwright_doom.render.diagnostic import teacher_forced_scan

    options = compare.reference_options(scene, pose)
    divs = teacher_forced_scan(session, full_rows, begin, options, window=_WINDOW)

    assert not divs, (
        f"compiled teacher-forced free-run hard-diverges from the reference at "
        f"{len(divs)} position(s) within the first {_WINDOW}:\n"
        + "\n".join(
            f"  rollout {d.pos - begin} (pos {d.pos}): expected {d.expected_type}"
            f"(row {d.expected_row}) -> predicted {d.predicted_type}"
            f"(row {d.predicted_row})  [{d.kind}: {d.detail}]"
            for d in divs[:10]
        )
    )
