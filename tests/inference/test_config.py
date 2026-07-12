"""Render-config contract tests.

Two load-bearing guarantees:

1. **Content-bearing run defaults enter the complete-bundle key.** The prompt
   pose and default new-token bound change files shipped in the exact bundle.
   The key-set pin keeps that boundary deliberate.

2. **Flag-over-config resolution is correct in both directions.**  A
   ``None`` flag must fall back to the config's ``run:`` value, and an
   explicit flag (e.g. ``--max-new-tokens``) must win over the config.
"""

from __future__ import annotations

from pathlib import Path

import torchwright_doom.inference.config as config_mod
from torchwright_doom.inference.config import (
    PoseConfig,
    RenderConfig,
    RunConfig,
    canonical_compile_payload,
    validate_compile_payload,
    load_render_config,
    resolve_run_args,
    resolve_wad_path,
)
from torchwright_doom.inference.wad_scene import default_pose_world

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


def test_only_content_bearing_run_fields_enter_bundle_payload(monkeypatch) -> None:
    # The payload embeds git SHAs (unresolvable in a gitless container) —
    # pin them; this test is about the payload's FIELD selection.
    monkeypatch.setattr(config_mod, "_git_sha", lambda repo: "pinned-test-sha")

    base = RenderConfig()
    wad = resolve_wad_path(base)
    # Key-set pin: adding a payload field busts every compile-cache key —
    # make that a deliberate, reviewed change, not a refactor side effect.
    assert set(canonical_compile_payload(base, wad)) == {
        "artifact",
        "wad",
        "wad_sha256",
        "map",
        "region",
        "wall_names",
        "flat_names",
        "model",
        "screen",
        "bundle",
        "git",
    }
    assert canonical_compile_payload(base, wad)["artifact"] == {
        "kind": "hf_phi3_bundle",
        "format": 1,
        "architecture": "phi3",
    }
    assert canonical_compile_payload(base, wad)["wad"] == base.wad
    assert canonical_compile_payload(base, wad)["bundle"] == {
        "prompt_pose": {
            "x": base.run.pose.x,
            "y": base.run.pose.y,
            "angle": base.run.pose.angle,
            "viewz": base.run.pose.viewz,
        },
        "max_new_tokens": base.run.max_new_tokens,
    }
    assert canonical_compile_payload(RenderConfig(run=_RUN_VARIANT), wad) != (
        canonical_compile_payload(base, wad)
    ), "content-bearing bundled prompt/generation defaults must move the key"

    # And a model-section change MUST move the payload (the inverse
    # guarantee — compile parameters stay in the key).
    model_variant = RenderConfig(model=config_mod.ModelConfig(d=2048))
    assert canonical_compile_payload(model_variant, wad) != canonical_compile_payload(
        base, wad
    )


def test_unknown_config_keys_fail_loudly(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "wad: doom1.wad\nmodel: {bias: false}\n")
    try:
        load_render_config(path)
    except ValueError as exc:
        assert "model: unknown key(s): bias" in str(exc)
    else:
        raise AssertionError("removed production bias setting was silently accepted")


def test_handed_payload_is_revalidated_against_config_and_wad(monkeypatch) -> None:
    monkeypatch.setattr(config_mod, "_git_sha", lambda repo: "pinned-test-sha")
    config = RenderConfig()
    wad = resolve_wad_path(config)
    payload = canonical_compile_payload(config, wad)
    validate_compile_payload(payload, config, wad)
    payload["screen"]["width"] += 1
    try:
        validate_compile_payload(payload, config, wad)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched handed compile payload was trusted")
