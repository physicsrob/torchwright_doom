"""Bundle manifest: format constants, hashing, construction, validation.

The publication contract's schema side. A normal completeness probe
(:func:`validate_bundle_manifest` / :func:`is_complete_hf_bundle`) verifies
the layout version, every declared file's size, and the hash of every
hash-bearing file. Safetensor shards are the only hash-exempt files — a
deliberate cost trade-off against a ~98 GB checkpoint: for shards, only
their sizes and the index's declared name->shard map are checked; shard
*contents* are not hashed by these probes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import RenderConfig

MANIFEST_NAME = "doom_bundle_manifest.json"
BUNDLE_FORMAT = "torchwright_doom.phi3.v1"
ARTIFACT_KIND = "hf_phi3_bundle"
# Bundle layout version: 2 = root infer.py + tools/, size checks for every
# declared file, hash checks for every hash-bearing file. Bumped together
# with the compile payload's artifact.format (identity.compile_payload_domain);
# validators fail closed on older or missing versions.
LAYOUT_VERSION = 2


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_payload_sha256(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload))


def file_manifest(bundle: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != MANIFEST_NAME:
            relative = path.relative_to(bundle).as_posix()
            facts: dict[str, Any] = {"size": path.stat().st_size}
            # Shards are size-checked but not content-hashed (the index maps
            # tensor names to shards; it carries no content digest) — a
            # deliberate trade-off to keep probes cheap against ~98 GB. Hash
            # all Doom-owned data, configs, tools, and the index itself.
            if path.suffix != ".safetensors":
                facts["sha256"] = sha256_file(path)
            files[relative] = facts
    return files


def aggregate_hash(files: dict[str, dict[str, Any]], names: list[str]) -> str:
    return sha256_bytes(canonical_json({name: files[name]["sha256"] for name in names}))


def candidate_manifest(
    *,
    bundle: Path,
    config: RenderConfig,
    wad_path: Path,
    compile_payload: dict[str, Any],
    report,
    prompt_rows: list[int],
    prompt_path: Path,
    vocab_size: int,
    row_vocab_fingerprint: str,
    tokenizer_vocab_sha256: str,
    bos_row: int,
    eos_row: int,
    origin: tuple[float, float],
) -> dict[str, Any]:
    files = file_manifest(bundle)
    prompt_bytes = prompt_path.read_bytes()
    prompt_ids = canonical_json(prompt_rows)
    payload_digest = compile_payload_sha256(compile_payload)
    return {
        "format": BUNDLE_FORMAT,
        "artifact_kind": ARTIFACT_KIND,
        "profile": "phi3",
        "compile_payload_sha256": payload_digest,
        "source_revisions": dict(compile_payload["git"]),
        "bundle_identity": f"{BUNDLE_FORMAT}:{payload_digest}",
        "model_type": "phi3",
        "architecture": "Phi3ForCausalLM",
        "dtype": "float32",
        "model": asdict(config.model),
        "screen": {
            "width": config.screen[0],
            "height": config.screen[1],
            "scale": config.model.scale,
            "detail": config.model.detail,
            "hud": config.model.hud,
        },
        "wad": {"name": config.wad, "sha256": sha256_file(wad_path)},
        "map": config.map,
        "region": asdict(config.region),
        "pose": {**asdict(config.run.pose), "scene_origin": list(origin)},
        "vocab_size": int(vocab_size),
        # Two distinct identities: the fingerprint hashes the row vocabulary's
        # semantic slot/type structure plus its row count
        # (tokenizer/identity.py); the sha256 hashes the ordered tokenizer
        # word strings themselves (tokenizer/standard.py).
        "row_vocab_fingerprint": row_vocab_fingerprint,
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256,
        "formatter_data_sha256": aggregate_hash(
            files, ["doom_vocab.json", "doom_tables.json"]
        ),
        "frame_decoder_data_sha256": aggregate_hash(
            files, ["doom_vocab.json", "doom_palette.json"]
        ),
        "bos_token_id": int(bos_row),
        "eos_token_id": int(eos_row),
        "pad_token_id": int(eos_row),
        "prompt": {
            "path": prompt_path.relative_to(bundle).as_posix(),
            "sha256": sha256_bytes(prompt_bytes),
            "row_ids_sha256": sha256_bytes(prompt_ids),
            "n_rows": len(prompt_rows),
        },
        "generation": {"max_new_tokens": int(config.run.max_new_tokens)},
        "schedule": report.schedule_provenance.to_dict(),
        "compile": {"n_layers": int(report.n_layers)},
        "files": files,
        "validation": {"complete": False, "format_version": LAYOUT_VERSION},
    }


def validate_bundle_manifest(
    bundle_dir: str | Path,
    *,
    expected_payload: dict[str, Any] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate Doom-owned identity without loading model weights.

    A normal probe verifies the layout version, every declared file's size,
    and the hash of every hash-bearing file. Safetensor shards are the only
    hash-exempt files: their sizes and the index's declared name->shard map
    are checked, but shard contents are not hashed — a deliberate cost
    trade-off that keeps completeness probes cheap.
    """
    bundle = Path(bundle_dir)
    path = bundle / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Doom bundle has no {MANIFEST_NAME}: {bundle}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != BUNDLE_FORMAT
        or manifest.get("artifact_kind") != ARTIFACT_KIND
    ):
        raise ValueError("directory is not a supported Doom Phi-3 bundle")
    if manifest.get("profile") != "phi3":
        raise ValueError("Doom production bundle does not declare the Phi-3 profile")
    if manifest.get("validation", {}).get("format_version") != LAYOUT_VERSION:
        raise ValueError(
            f"Doom bundle does not declare layout version {LAYOUT_VERSION}"
        )
    if not allow_incomplete and not manifest.get("validation", {}).get("complete"):
        raise ValueError("Doom bundle manifest is not complete")
    if expected_payload is not None and manifest.get(
        "compile_payload_sha256"
    ) != compile_payload_sha256(expected_payload):
        raise ValueError(
            "Doom bundle compile payload does not match the requested build"
        )

    required = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "doom_vocab.json",
        "doom_tables.json",
        "doom_palette.json",
        "README.md",
        "infer.py",
        "examples/e1m1_prompt.txt",
        "tools/pretty_text.py",
        "tools/txt_to_png.py",
        "tools/README.md",
    }
    declared = manifest.get("files", {})
    missing = sorted(
        name
        for name in required
        if name not in declared or not (bundle / name).is_file()
    )
    if missing:
        raise ValueError(f"Doom bundle is incomplete; missing: {', '.join(missing)}")
    for name, facts in declared.items():
        file_path = bundle / name
        if not file_path.is_file():
            raise ValueError(f"Doom bundle declares a missing file: {name}")
        if file_path.stat().st_size != facts.get("size"):
            raise ValueError(f"Doom bundle file size mismatch: {name}")
        if name.endswith(".safetensors"):
            # Shards are the only hash-exempt entries: size-checked, contents
            # not hashed (deliberate cost trade-off; see module docstring).
            # Verify a digest if one is present anyway.
            if "sha256" in facts and sha256_file(file_path) != facts["sha256"]:
                raise ValueError(f"Doom bundle file hash mismatch: {name}")
            continue
        # Every other entry MUST carry a digest: a manifest that omits one
        # would otherwise silently downgrade validation to size-only —
        # fail closed instead.
        if "sha256" not in facts:
            raise ValueError(f"Doom bundle manifest lacks a digest for: {name}")
        if sha256_file(file_path) != facts["sha256"]:
            raise ValueError(f"Doom bundle file hash mismatch: {name}")
    index = json.loads((bundle / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        raise ValueError("Doom bundle safetensors index references no shards")
    undeclared = sorted(shards - set(declared))
    if undeclared:
        raise ValueError(
            f"Doom bundle index references undeclared shards: {', '.join(undeclared)}"
        )
    config = json.loads((bundle / "config.json").read_text())
    if config.get("model_type") != "phi3" or config.get("architectures") != [
        "Phi3ForCausalLM"
    ]:
        raise ValueError("Doom bundle does not have stock Phi-3 identity")
    if "auto_map" in config:
        raise ValueError("Doom stock model config must not contain auto_map")
    tokenizer_config = json.loads((bundle / "tokenizer_config.json").read_text())
    if "auto_map" in tokenizer_config:
        raise ValueError("Doom stock tokenizer config must not contain auto_map")
    forbidden = [
        path.name
        for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py")
        for path in bundle.glob(pattern)
    ]
    if forbidden:
        raise ValueError(f"Doom stock bundle contains auto-loaded code: {forbidden}")
    return manifest


def is_complete_hf_bundle(
    bundle_dir: str | Path, *, expected_payload: dict[str, Any] | None = None
) -> bool:
    try:
        validate_bundle_manifest(bundle_dir, expected_payload=expected_payload)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False
    return True
