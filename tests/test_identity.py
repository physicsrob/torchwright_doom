"""Compile-cache identity tests (root ``identity.py``).

Load-bearing guarantees:

1. **Content-bearing run defaults enter the complete-bundle key.** The prompt
   pose and default new-token bound change files shipped in the exact bundle.
   The key-set pin keeps that boundary deliberate.
2. **Moving the code did not move the keys.** The canonical cache keys for
   both committed configs (with the git identity pinned) are pinned to the
   values the pre-move ``inference/config.py`` produced.
3. **Both git repository roots resolve from a foreign working directory** —
   Doom resolves from ``__file__`` and Torchwright from its active import.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import torchwright_doom.identity as identity_mod
from torchwright_doom.config import (
    ModelConfig,
    PoseConfig,
    RenderConfig,
    RunConfig,
    load_render_config,
    resolve_wad_path,
)
from torchwright_doom.identity import (
    cache_key_from_payload,
    canonical_compile_payload,
    validate_compile_payload,
)

ROOT = Path(__file__).resolve().parents[1]

_RUN_VARIANT = RunConfig(
    max_new_tokens=123,
    pose=PoseConfig(x=1.0, y=2.0, angle=3, viewz=4.0),
)

# Computed with _git_sha pinned to "pinned-test-sha" — the gate: cache keys
# for unchanged config content do not change because code moved. Re-pinned the
# low-res key 2026-08-08 when that config became the distinct 80x50 consumer
# contract; the production key did not move.
_PINNED_KEYS = {
    "e1m1.yaml": "bfc3cfa363e4d069785acb6139e6c16c52cc9b44dfc4b0317c2c5c1a0e8b1b1a",
    "e1m1_lowres.yaml": (
        "8117f2c3cf40fcbd026cb935c55a67735efeac095ecd8d528720352ae21e0e07"
    ),
}


def test_only_content_bearing_run_fields_enter_bundle_payload(monkeypatch) -> None:
    # The payload embeds git SHAs (unresolvable in a gitless container) —
    # pin them; this test is about the payload's FIELD selection.
    monkeypatch.setattr(identity_mod, "_git_sha", lambda repo: "pinned-test-sha")

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
        "format": 2,
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
    model_variant = RenderConfig(model=ModelConfig(d=2048))
    assert canonical_compile_payload(model_variant, wad) != canonical_compile_payload(
        base, wad
    )


def test_canonical_cache_keys_survived_the_move_to_root(monkeypatch) -> None:
    monkeypatch.setattr(identity_mod, "_git_sha", lambda repo: "pinned-test-sha")
    for name, expected in _PINNED_KEYS.items():
        config = load_render_config(ROOT / "configs" / name)
        wad = resolve_wad_path(config, base_dir=ROOT / "configs")
        payload = canonical_compile_payload(config, wad)
        assert cache_key_from_payload(payload) == expected, name


def test_git_roots_resolve_from_outside_the_repo(tmp_path: Path, monkeypatch) -> None:
    """Absolute Doom/import-resolved Torchwright roots survive cwd changes."""
    if identity_mod._git_sha(ROOT) == "unknown":
        pytest.skip("no .git available here (e.g. a Modal test container)")
    monkeypatch.chdir(tmp_path)
    config = RenderConfig()
    wad = resolve_wad_path(config)
    git_shas = canonical_compile_payload(config, wad)["git"]
    assert set(git_shas) == {"torchwright", "torchwright_doom"}
    for sha in git_shas.values():
        assert not sha.endswith("unknown")
    # Ground truth: the Doom entry matches `git -C <repo> rev-parse HEAD`
    # (modulo the -dirty working-tree digest suffix).
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    assert git_shas["torchwright_doom"].startswith(head)


def test_torchwright_identity_follows_the_active_import(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "release-compiler" / "torchwright" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("")
    roots: list[Path] = []

    monkeypatch.setattr(
        identity_mod.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(package)),
    )

    def record_git_root(root: Path) -> str:
        roots.append(Path(root))
        return "pinned-test-sha"

    monkeypatch.setattr(identity_mod, "_git_sha", record_git_root)
    config = RenderConfig()
    canonical_compile_payload(config, resolve_wad_path(config))
    assert roots[1] == tmp_path / "release-compiler"


def test_handed_payload_is_revalidated_against_config_and_wad(monkeypatch) -> None:
    monkeypatch.setattr(identity_mod, "_git_sha", lambda repo: "pinned-test-sha")
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
