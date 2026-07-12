"""Render-config contract tests (the root job spec).

Load-bearing guarantees:

1. **Flag-over-config resolution is correct in both directions.**  A
   ``None`` flag must fall back to the config's ``run:`` value, and an
   explicit flag (e.g. ``--max-new-tokens``) must win over the config.
2. **The two committed configs stay in lockstep** — only ``scale`` differs.
3. **The repository-root WAD fallback survives the module's move to the
   package root** (``__file__``-relative depth), including from a working
   directory outside the repository.

Cache identity (payload field selection, key stability) lives in
``tests/test_identity.py``.
"""

from __future__ import annotations

from pathlib import Path

from torchwright_doom.config import (
    PoseConfig,
    RenderConfig,
    RunConfig,
    load_render_config,
    resolve_run_args,
    resolve_wad_path,
)
from torchwright_doom.prompt.scene import default_pose_world

ROOT = Path(__file__).resolve().parents[1]

_RUN_VARIANT = RunConfig(
    max_new_tokens=123,
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
            "  max_new_tokens: 123\n"
            "  pose: {x: 1.0, y: 2.0, angle: 3, viewz: 4.0}\n",
        )
    )
    assert cfg.run == _RUN_VARIANT


def test_run_section_partial_keeps_other_defaults(tmp_path: Path) -> None:
    cfg = load_render_config(
        _write_config(tmp_path, "wad: doom1.wad\nrun:\n  max_new_tokens: 99\n")
    )
    assert cfg.run.max_new_tokens == 99
    assert cfg.run.pose == PoseConfig()


def test_resolve_config_wins_when_flag_absent() -> None:
    cfg = RenderConfig(run=_RUN_VARIANT)
    args = resolve_run_args(cfg)
    assert (args.x, args.y, args.angle, args.viewz) == (1.0, 2.0, 3, 4.0)
    assert args.max_new_tokens == 123


def test_resolve_flag_wins_when_set() -> None:
    cfg = RenderConfig(run=_RUN_VARIANT)
    args = resolve_run_args(
        cfg,
        x=10.0,
        y=20.0,
        angle=30,
        viewz=40.0,
        max_new_tokens=999,
    )
    assert (args.x, args.y, args.angle, args.viewz) == (10.0, 20.0, 30, 40.0)
    assert args.max_new_tokens == 999


def test_default_pose_world_reads_config_run_pose() -> None:
    cfg = RenderConfig(run=_RUN_VARIANT)
    assert default_pose_world(cfg) == (1.0, 2.0, 3, 4.0)


def test_committed_configs_differ_only_in_scale() -> None:
    from dataclasses import replace

    production = load_render_config(ROOT / "configs" / "e1m1.yaml")
    lowres = load_render_config(ROOT / "configs" / "e1m1_lowres.yaml")
    assert production.model.scale == 1
    assert lowres.model.scale == 2
    # Lockstep rule (CLAUDE.md "Configurations"): everything except scale is
    # identical between the two committed configs.
    assert (
        replace(production, model=replace(production.model, scale=lowres.model.scale))
        == lowres
    )


def test_wad_fallback_resolves_from_outside_the_repo(tmp_path: Path, monkeypatch):
    """resolve_wad_path's last candidate is the repository root, derived from
    the config module's ``__file__`` — it must survive both the module's move
    to the package root and a foreign working directory."""
    monkeypatch.chdir(tmp_path)
    resolved = resolve_wad_path(RenderConfig())
    assert resolved == (ROOT / "doom1.wad").resolve()


def test_unknown_config_keys_fail_loudly(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "wad: doom1.wad\nmodel: {bias: false}\n")
    try:
        load_render_config(path)
    except ValueError as exc:
        assert "model: unknown key(s): bias" in str(exc)
    else:
        raise AssertionError("removed production bias setting was silently accepted")
