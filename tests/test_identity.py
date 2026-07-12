"""Compile-cache identity tests (root ``identity.py``).

Load-bearing guarantees:

1. **Content-bearing run defaults enter the complete-bundle key.** The prompt
   pose and default new-token bound change files shipped in the exact bundle.
   The key-set pin keeps that boundary deliberate.
2. **Moving the code did not move the keys.** The canonical cache keys for
   both committed configs (with the git identity pinned) are pinned to the
   values the pre-move ``inference/config.py`` produced.
3. **Both git repository roots resolve from a foreign working directory** —
   ``_git_sha`` addresses the repos by ``__file__``-derived absolute paths.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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

# Computed by the pre-move inference/config.py implementation with
# _git_sha pinned to "pinned-test-sha" (plan_cleanup_v2 Workstream 5 gate:
# cache keys for unchanged config content do not change because code moved).
_PINNED_KEYS = {
    "e1m1.yaml": "81e93524781aec313568275ae30538bb2c9559624b4f8f17d919caf2fc24c80b",
    "e1m1_lowres.yaml": (
        "98767b7444ecb150c910f5b3d8c8745ec5179fea23e16920076470bc7b51d3eb"
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
    """_git_sha addresses the Doom repo (parents[1] of identity.py) and the
    umbrella's torchwright checkout (parents[2]) by absolute path, so a
    foreign working directory must not degrade the payload to "unknown"."""
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
