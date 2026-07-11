"""Render-config contract tests.

Two load-bearing guarantees:

1. **Runtime knobs never enter the compile-cache key.**  The ``run:``
   section is a runtime knob; if a refactor ever routes it into
   ``canonical_compile_payload`` (e.g. by switching to ``asdict(config)``),
   every runtime-knob edit would silently recompile the production
   artifact.  The key-set pin makes adding a payload field a deliberate,
   reviewed act.

2. **Flag-over-config resolution is correct in both directions.**  A
   ``None`` flag must fall back to the config's ``run:`` value, and an
   explicit flag (e.g. ``--max-positions``) must win over the config.
"""

from __future__ import annotations

from pathlib import Path

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
    max_positions=123,
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
            "  max_positions: 123\n"
            "  prefill_chunk_size: 7\n"
            "  pose: {x: 1.0, y: 2.0, angle: 3, viewz: 4.0}\n",
        )
    )
    assert cfg.run == _RUN_VARIANT


def test_run_section_partial_keeps_other_defaults(tmp_path: Path) -> None:
    cfg = load_render_config(
        _write_config(tmp_path, "wad: doom1.wad\nrun:\n  max_positions: 99\n")
    )
    assert cfg.run.max_positions == 99
    assert cfg.run.prefill_chunk_size == RunConfig().prefill_chunk_size
    assert cfg.run.pose == PoseConfig()


def test_resolve_config_wins_when_flag_absent() -> None:
    cfg = RenderConfig(run=_RUN_VARIANT)
    args = resolve_run_args(cfg)
    assert (args.x, args.y, args.angle, args.viewz) == (1.0, 2.0, 3, 4.0)
    assert args.max_positions == 123
    assert args.prefill_chunk_size == 7


def test_resolve_flag_wins_when_set() -> None:
    cfg = RenderConfig(run=_RUN_VARIANT)
    args = resolve_run_args(
        cfg,
        x=10.0,
        y=20.0,
        angle=30,
        viewz=40.0,
        max_positions=999,
        prefill_chunk_size=1,
    )
    assert (args.x, args.y, args.angle, args.viewz) == (10.0, 20.0, 30, 40.0)
    assert args.max_positions == 999
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
    runtime_variant = RenderConfig(run=_RUN_VARIANT)
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
