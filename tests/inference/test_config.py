"""Render-config contract tests (batch 2 of the tech-debt cleanup).

Two load-bearing guarantees:

1. **Runtime knobs never enter the compile-cache key.**  The ``run:``
   section and ``expiring_types`` are runtime knobs; if a refactor ever
   routes them into ``canonical_compile_payload`` (e.g. by switching to
   ``asdict(config)``), every runtime-knob edit would silently recompile
   the production artifact.  The key-set pin makes adding a payload field
   a deliberate, reviewed act.

2. **Flag-over-config resolution is correct in both directions.**  A
   ``None`` flag must fall back to the config's ``run:`` value, and an
   explicit flag — including a falsy one like ``--draft-window 0`` —
   must win over the config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import torchwright_doom.inference.config as config_mod
from torchwright_doom.inference.config import (
    PoseConfig,
    RenderConfig,
    RunConfig,
    canonical_compile_payload,
    load_render_config,
    resolve_run_args,
    resolve_wad_path,
)
from torchwright_doom.inference.wad_scene import default_pose_world

_RUN_VARIANT = RunConfig(
    mode="pure_ar",
    max_positions=123,
    draft_window=0,
    prefill_chunk_size=7,
    pose=PoseConfig(x=1.0, y=2.0, angle=3, viewz=4.0),
)


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "variant.yaml"
    path.write_text(body)
    return path


def test_run_section_defaults_when_absent(tmp_path: Path) -> None:
    cfg = load_render_config(_write_config(tmp_path, "wad: doom1.wad\n"))
    assert cfg.run == RunConfig()


def test_run_section_parses(tmp_path: Path) -> None:
    cfg = load_render_config(
        _write_config(
            tmp_path,
            "wad: doom1.wad\n"
            "run:\n"
            "  mode: pure_ar\n"
            "  max_positions: 123\n"
            "  draft_window: 0\n"
            "  prefill_chunk_size: 7\n"
            "  pose: {x: 1.0, y: 2.0, angle: 3, viewz: 4.0}\n",
        )
    )
    assert cfg.run == _RUN_VARIANT


def test_run_section_partial_keeps_other_defaults(tmp_path: Path) -> None:
    cfg = load_render_config(
        _write_config(tmp_path, "wad: doom1.wad\nrun:\n  draft_window: 2\n")
    )
    assert cfg.run.draft_window == 2
    assert cfg.run.mode == RunConfig().mode
    assert cfg.run.pose == PoseConfig()


def test_bad_run_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run.mode"):
        load_render_config(
            _write_config(tmp_path, "wad: doom1.wad\nrun:\n  mode: bogus\n")
        )


def test_resolve_config_wins_when_flag_absent() -> None:
    cfg = RenderConfig(run=_RUN_VARIANT)
    args = resolve_run_args(cfg)
    assert (args.x, args.y, args.angle, args.viewz) == (1.0, 2.0, 3, 4.0)
    assert args.mode == "pure_ar"
    assert args.max_positions == 123
    assert args.draft_window == 0
    assert args.prefill_chunk_size == 7


def test_resolve_flag_wins_when_set() -> None:
    cfg = RenderConfig(run=_RUN_VARIANT)
    args = resolve_run_args(
        cfg,
        x=10.0,
        y=20.0,
        angle=30,
        viewz=40.0,
        mode="both",
        max_positions=999,
        # Explicit zero is a real value, not "unset" — the None sentinel
        # is what keeps --draft-window 0 working.
        draft_window=0,
        prefill_chunk_size=1,
    )
    assert (args.x, args.y, args.angle, args.viewz) == (10.0, 20.0, 30, 40.0)
    assert args.mode == "both"
    assert args.max_positions == 999
    assert args.draft_window == 0
    assert args.prefill_chunk_size == 1


def test_default_pose_world_reads_config_run_pose() -> None:
    cfg = RenderConfig(run=_RUN_VARIANT)
    assert default_pose_world(cfg) == (1.0, 2.0, 3, 4.0)


def test_runtime_knobs_stay_out_of_compile_payload(monkeypatch) -> None:
    # The payload embeds git SHAs (unresolvable in a gitless container) —
    # pin them; this test is about the payload's FIELD selection.
    monkeypatch.setattr(config_mod, "_git_sha", lambda repo: "pinned-test-sha")

    base = RenderConfig()
    wad = resolve_wad_path(base)
    runtime_variant = RenderConfig(
        expiring_types=("pixel", "setCursorX"),
        run=_RUN_VARIANT,
    )
    assert canonical_compile_payload(base, wad) == canonical_compile_payload(
        runtime_variant, wad
    )

    # Key-set pin: adding a payload field busts every compile-cache key —
    # make that a deliberate, reviewed change, not a refactor side effect.
    assert set(canonical_compile_payload(base, wad)) == {
        "wad",
        "wad_sha256",
        "map",
        "region",
        "wall_names",
        "flat_names",
        "model",
        "screen",
        "git",
    }

    # And a model-section change MUST move the payload (the inverse
    # guarantee — compile parameters stay in the key).
    model_variant = RenderConfig(model=config_mod.ModelConfig(d=2048))
    assert canonical_compile_payload(model_variant, wad) != canonical_compile_payload(
        base, wad
    )
