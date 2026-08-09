"""Compile-cache identity: canonical payloads, cache keys, git/WAD hashing.

Root sibling of the job spec (``config.py``); both are shared authorities
owned by no consumer package. Distinct from
``tokenizer/identity.py`` (the vocab fingerprint) — same basename, different
concern; grep with paths.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import RenderConfig


def hf_bundle_cache_dir(config: RenderConfig, wad_path: str | Path) -> Path:
    """Production direct-HF bundle directory."""
    return (
        Path.home()
        / ".cache"
        / "torchwright_doom"
        / "hf_phi3"
        / cache_key(config, wad_path)
    )


def cache_key(config: RenderConfig, wad_path: str | Path) -> str:
    return cache_key_from_payload(canonical_compile_payload(config, wad_path))


def cache_key_from_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_compile_payload(
    config: RenderConfig, wad_path: str | Path
) -> dict[str, Any]:
    # Resolve Torchwright through the active import path. Release builds may
    # deliberately select a detached compiler worktree through PYTHONPATH;
    # deriving this as a fixed umbrella sibling would then stamp one checkout
    # while Modal mounted another.
    torchwright_spec = importlib.util.find_spec("torchwright")
    if torchwright_spec is None or torchwright_spec.origin is None:
        raise RuntimeError("torchwright is not importable for compile identity")
    torchwright_repo = Path(torchwright_spec.origin).resolve().parents[1]
    git_shas = {
        "torchwright_doom": _git_sha(Path(__file__).resolve().parents[1]),
        "torchwright": _git_sha(torchwright_repo),
    }
    # Enforce "the container must never derive its own key": a git-less caller
    # would mint a fixed
    # "unknown"-keyed payload that never changes on a code edit, so the
    # cache would silently serve artifacts compiled from other code
    # states.  endswith also catches "<head>-dirty.unknown" (rev-parse
    # worked but diff/status failed) — that key would be blind to
    # working-tree edits, the same stale-artifact hazard.  Compute the
    # payload where .git is available and hand it over explicitly
    # (compile_payload=...).
    unresolved = {k: v for k, v in git_shas.items() if v.endswith("unknown")}
    if unresolved:
        raise RuntimeError(
            f"cannot derive a compile-cache key here: git sha unresolvable "
            f"for {sorted(unresolved)} — compute canonical_compile_payload "
            f"where .git is available and pass it through compile_payload=."
        )
    return {**compile_payload_domain(config, wad_path), "git": git_shas}


def compile_payload_domain(
    config: RenderConfig, wad_path: str | Path
) -> dict[str, Any]:
    """Portable, remotely re-verifiable portion of the production cache key."""
    wad_path = Path(wad_path)
    return {
        "artifact": {
            "kind": "hf_phi3_bundle",
            # format 2 = bundle layout v2 (root infer.py + tools/, sized and
            # hash-verified manifest); bumped together with the manifest's
            # validation.format_version.
            "format": 2,
            "architecture": "phi3",
        },
        "wad": config.wad,
        "wad_sha256": _file_sha256(wad_path),
        "map": config.map,
        "region": asdict(config.region),
        "wall_names": list(config.textures.wall),
        "flat_names": list(config.textures.flat),
        "model": asdict(config.model),
        "screen": {"width": config.screen[0], "height": config.screen[1]},
        # These do not alter weights, but they do alter files inside the exact
        # complete bundle (the executable prompt and example generation bound).
        "bundle": {
            "prompt_pose": asdict(config.run.pose),
            "max_new_tokens": config.run.max_new_tokens,
        },
    }


def validate_compile_payload(
    payload: dict[str, Any], config: RenderConfig, wad_path: str | Path
) -> None:
    handed_domain = {key: value for key, value in payload.items() if key != "git"}
    expected = compile_payload_domain(config, wad_path)
    if handed_domain != expected:
        raise ValueError(
            "handed compile payload does not match the loaded config/WAD; "
            "refusing to publish an artifact under a false cache identity"
        )
    git = payload.get("git")
    if not isinstance(git, dict) or set(git) != {"torchwright", "torchwright_doom"}:
        raise ValueError("compile payload has no complete source-revision identity")


def _git_sha(repo: Path) -> str:
    """Committed HEAD sha, extended with a working-tree digest when dirty.

    The compile cache key embeds this. Without the dirty suffix, an
    uncommitted edit ships to the compile (``add_local_python_source`` sends
    the working tree) while the key stays pinned at HEAD — every edit-run
    cycle silently HITs the artifact compiled from the *first* iteration and
    the gate "validates" code that was never compiled.  sha256 of
    ``git diff HEAD`` covers tracked edits; untracked non-ignored file names and
    contents are hashed too, so iterating on a newly added bundle module cannot
    silently reuse the first artifact compiled from that path.
    """
    git = ["git", "-C", str(repo)]
    try:
        head = subprocess.check_output(
            [*git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
    try:
        diff = subprocess.check_output(
            [*git, "diff", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        status = subprocess.check_output(
            [*git, "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return f"{head}-dirty.unknown"
    if not diff and not status:
        return head
    dirty = hashlib.sha256((diff + "\0" + status).encode("utf-8"))
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        path = repo / line[3:]
        if path.is_file():
            dirty.update(line[3:].encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    dirty.update(chunk)
    digest = dirty.hexdigest()[:12]
    return f"{head}-dirty.{digest}"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
