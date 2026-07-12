"""Direct stock-Phi-3 Doom bundle construction and validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import RenderConfig, resolve_wad_path, validate_compile_payload

_MANIFEST = "doom_bundle_manifest.json"
_FORMAT = "torchwright_doom.phi3.v1"
_ARTIFACT_KIND = "hf_phi3_bundle"


@dataclass(frozen=True)
class BundleReport:
    destination: str
    n_layers: int
    vocab_size: int
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_payload_sha256(payload: dict[str, Any]) -> str:
    return _sha_bytes(_canonical_json(payload))


def _remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


@contextmanager
def _outer_bundle_transaction(destination: str | Path) -> Iterator[Path]:
    """Stage a complete bundle beside its destination and publish with rollback."""
    final = Path(destination).absolute()
    final.parent.mkdir(parents=True, exist_ok=True)
    root = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.doom-staging-", dir=final.parent)
    )
    bundle = root / "bundle"
    backup = final.parent / f".{final.name}.previous-{root.name}"
    try:
        yield bundle
        if not bundle.is_dir():
            raise RuntimeError("Doom bundle transaction produced no staged directory")
        if final.exists():
            os.replace(final, backup)
        try:
            os.replace(bundle, final)
        except BaseException:
            if backup.exists():
                os.replace(backup, final)
            raise
        _remove(backup)
    except BaseException:
        if backup.exists() and not final.exists():
            os.replace(backup, final)
        raise
    finally:
        _remove(root)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _stamp_model_config(
    bundle: Path,
    *,
    config: RenderConfig,
    vocab_fingerprint: str,
    eos_row: int,
) -> None:
    path = bundle / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pad_token_id"] = int(eos_row)
    # Phi3's GenerationMixin resets its cache at this field even for default
    # RoPE, where no short/long-RoPE switch exists.  The compiled model has one
    # position regime spanning the full configured context.
    payload["original_max_position_embeddings"] = config.model.max_seq_len
    payload["doom_screen_config"] = {
        "width": config.screen[0],
        "height": config.screen[1],
        "scale": config.model.scale,
        "detail": config.model.detail,
        "hud": config.model.hud,
    }
    payload["doom_vocab_fingerprint"] = vocab_fingerprint
    _write_json(path, payload)


def _write_model_card(bundle: Path, config: RenderConfig) -> None:
    text = f"""---
library_name: transformers
pipeline_tag: text-generation
---

# TorchWright Doom — {config.map}

This is a stock Hugging Face `Phi3ForCausalLM` that renders DOOM through
ordinary autoregressive inference. The model and the data-only fast tokenizer
load through ordinary Transformers auto classes without remote code.

The bundled `examples/e1m1_prompt.txt` is the executable prompt. Run
`examples/infer.py` to produce canonical emitted row ids and raw tokenizer text.
`examples/pretty_text.py` formats that text for reading, while
`examples/txt_to_png.py` independently decodes its cursor/pixel protocol into a
PNG. Neither post-processing program participates in inference or performs
geometry, visibility, lighting, texture selection, or sorting.

Screen: {config.screen[0]}×{config.screen[1]}; map: {config.map}; fp32 weights;
eager attention is the validated implementation.
"""
    (bundle / "README.md").write_text(text, encoding="utf-8")


def _file_manifest(bundle: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != _MANIFEST:
            relative = path.relative_to(bundle).as_posix()
            facts: dict[str, Any] = {"size": path.stat().st_size}
            # The safetensors index binds every tensor key to a shard. Avoid a
            # second tens-of-GB read solely to hash the already validated shard
            # bytes; hash all Doom-owned data, configs, examples, and the index.
            if path.suffix != ".safetensors":
                facts["sha256"] = _sha_file(path)
            files[relative] = facts
    return files


def _aggregate_hash(files: dict[str, dict[str, Any]], names: list[str]) -> str:
    return _sha_bytes(_canonical_json({name: files[name]["sha256"] for name in names}))


def _candidate_manifest(
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
    files = _file_manifest(bundle)
    prompt_bytes = prompt_path.read_bytes()
    prompt_ids = _canonical_json(prompt_rows)
    payload_digest = compile_payload_sha256(compile_payload)
    return {
        "format": _FORMAT,
        "artifact_kind": _ARTIFACT_KIND,
        "profile": "phi3",
        "compile_payload_sha256": payload_digest,
        "source_revisions": dict(compile_payload["git"]),
        "bundle_identity": f"{_FORMAT}:{payload_digest}",
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
        "wad": {"name": config.wad, "sha256": _sha_file(wad_path)},
        "map": config.map,
        "region": asdict(config.region),
        "pose": {**asdict(config.run.pose), "scene_origin": list(origin)},
        "vocab_size": int(vocab_size),
        "row_vocab_fingerprint": row_vocab_fingerprint,
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256,
        "formatter_data_sha256": _aggregate_hash(
            files, ["doom_vocab.json", "doom_tables.json"]
        ),
        "frame_decoder_data_sha256": _aggregate_hash(
            files, ["doom_vocab.json", "doom_palette.json"]
        ),
        "bos_token_id": int(bos_row),
        "eos_token_id": int(eos_row),
        "pad_token_id": int(eos_row),
        "prompt": {
            "path": prompt_path.relative_to(bundle).as_posix(),
            "sha256": _sha_bytes(prompt_bytes),
            "row_ids_sha256": _sha_bytes(prompt_ids),
            "n_rows": len(prompt_rows),
        },
        "generation": {"max_new_tokens": int(config.run.max_new_tokens)},
        "schedule": report.schedule_provenance.to_dict(),
        "compile": {"n_layers": int(report.n_layers)},
        "files": files,
        "validation": {"complete": False, "format_version": 1},
    }


def validate_bundle_manifest(
    bundle_dir: str | Path,
    *,
    expected_payload: dict[str, Any] | None = None,
    allow_incomplete: bool = False,
    verify_all_hashes: bool = False,
) -> dict[str, Any]:
    """Validate Doom-owned identity without loading model weights."""
    bundle = Path(bundle_dir)
    path = bundle / _MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"Doom bundle has no {_MANIFEST}: {bundle}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != _FORMAT
        or manifest.get("artifact_kind") != _ARTIFACT_KIND
    ):
        raise ValueError("directory is not a supported Doom Phi-3 bundle")
    if manifest.get("profile") != "phi3":
        raise ValueError("Doom production bundle does not declare the Phi-3 profile")
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
        "examples/e1m1_prompt.txt",
        "examples/infer.py",
        "examples/pretty_text.py",
        "examples/txt_to_png.py",
        "examples/README.md",
    }
    declared = manifest.get("files", {})
    missing = sorted(
        name
        for name in required
        if name not in declared or not (bundle / name).is_file()
    )
    if missing:
        raise ValueError(f"Doom bundle is incomplete; missing: {', '.join(missing)}")
    index = json.loads((bundle / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards or any(not (bundle / shard).is_file() for shard in shards):
        raise ValueError("Doom bundle safetensors index references missing shards")
    if verify_all_hashes:
        for name, facts in declared.items():
            if "sha256" in facts and _sha_file(bundle / name) != facts["sha256"]:
                raise ValueError(f"Doom bundle file hash mismatch: {name}")
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


def _validate_complete_staged_bundle(
    bundle: Path, config: RenderConfig, prompt_rows: list[int]
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from ..formatter import DoomFormatter

    manifest = validate_bundle_manifest(
        bundle, allow_incomplete=True, verify_all_hashes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(bundle)
    if len(tokenizer) != manifest["vocab_size"]:
        raise ValueError("tokenizer width differs from Doom manifest")
    if tokenizer.bos_token_id != manifest["bos_token_id"]:
        raise ValueError("tokenizer BOS id differs from Doom manifest")
    if tokenizer.eos_token_id != manifest["eos_token_id"]:
        raise ValueError("tokenizer EOS id differs from Doom manifest")
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise ValueError("tokenizer padding does not reuse DONE/EOS")
    prompt_text = (bundle / manifest["prompt"]["path"]).read_text()
    actual_prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if actual_prompt != prompt_rows:
        raise ValueError("bundled prompt text does not reproduce build-time rows")
    formatter = DoomFormatter.from_bundle(bundle, allow_incomplete=True)
    raw_prompt = tokenizer.decode(
        prompt_rows, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not isinstance(raw_prompt, str):
        raise TypeError("tokenizer returned a batched decode for one prompt row list")
    if (
        formatter.rows_from_raw_text(
            formatter.parse_pretty_text(formatter.format_text(raw_prompt))
        )
        != prompt_rows
    ):
        raise ValueError("DoomFormatter changed bundled prompt row identity")

    model = AutoModelForCausalLM.from_pretrained(
        bundle, attn_implementation="eager", dtype=torch.float32
    ).eval()
    if type(model).__name__ != "Phi3ForCausalLM":
        raise ValueError(f"expected Phi3ForCausalLM, loaded {type(model).__name__}")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Doom bundle contains a non-fp32 model parameter")
    if any(name.endswith(".bias") for name, _ in model.named_parameters()):
        raise ValueError("Doom Phi-3 bundle unexpectedly contains projection biases")
    expected = {
        "hidden_size": config.model.d,
        "head_dim": config.model.d_head,
        "max_position_embeddings": config.model.max_seq_len,
        "original_max_position_embeddings": config.model.max_seq_len,
        "vocab_size": manifest["vocab_size"],
        "bos_token_id": manifest["bos_token_id"],
        "eos_token_id": manifest["eos_token_id"],
        "pad_token_id": manifest["pad_token_id"],
        "num_hidden_layers": manifest["compile"]["n_layers"],
        "rms_norm_eps": 1e-5,
    }
    for name, value in expected.items():
        if getattr(model.config, name) != value:
            raise ValueError(
                f"stock Phi-3 config {name}={getattr(model.config, name)!r} "
                f"does not match expected {value!r}"
            )
    if model.config.hidden_act != "silu":
        raise ValueError("Doom production model is not SwiGLU/SiLU")
    hidden_cap = config.model.d_hidden or config.model.d
    if not (0 < model.config.intermediate_size <= hidden_cap):
        raise ValueError("stock Phi-3 intermediate width exceeds the Doom compile cap")
    partial = model.config.rope_parameters.get("partial_rotary_factor")
    expected_partial = (config.model.d_rot or config.model.d_head) / config.model.d_head
    if partial != expected_partial:
        raise ValueError("stock Phi-3 partial-RoPE setting differs from the Doom graph")
    cache = DynamicCache()
    first = torch.tensor([[manifest["bos_token_id"]]], dtype=torch.long)
    with torch.no_grad():
        result = model(
            input_ids=first,
            past_key_values=cache,
            cache_position=torch.tensor([0]),
            use_cache=True,
        )
        second = model(
            input_ids=torch.tensor([[manifest["eos_token_id"]]], dtype=torch.long),
            past_key_values=result.past_key_values,
            cache_position=torch.tensor([1]),
            use_cache=True,
        )
    if (
        result.logits.shape[-1] != manifest["vocab_size"]
        or second.logits.shape[-1] != manifest["vocab_size"]
    ):
        raise ValueError("stock Phi-3 smoke test produced the wrong logit width")
    del model


def compile_phi3_bundle(
    config: RenderConfig,
    *,
    wad_path: str | Path,
    destination: str | Path,
    compile_payload: dict[str, Any],
    verbose: bool = False,
) -> BundleReport:
    """Compile and atomically publish one complete production Doom bundle."""
    if config.map.upper() != "E1M1":
        raise ValueError(
            "the production HF bundle contract currently supports E1M1 only"
        )
    wad_path = Path(wad_path)
    validate_compile_payload(compile_payload, config, wad_path)

    from ..constants import SCREEN_HEIGHT, SCREEN_WIDTH

    if (SCREEN_WIDTH, SCREEN_HEIGHT) != config.screen:
        raise RuntimeError(
            f"screen dims {SCREEN_WIDTH}x{SCREEN_HEIGHT} were imported before "
            f"this config's {config.screen[0]}x{config.screen[1]} was applied; "
            "call apply_screen_env(config) before importing graph/tokenizer modules"
        )

    from torchwright.compiler import CompileProfile, compile_hf_bundle

    from ..bundle_examples import write_bundle_examples
    from ..tokenizer.freeze import write_frozen_data
    from ..tokenizer.identity import vocab_fingerprint
    from ..tokenizer.standard import (
        build_standard_tokenizer,
        canonical_words,
        doom_special_tokens,
        ordered_words_sha256,
    )
    from .compiled_model import build_graph
    from .wad_scene import load_render_scene, pose_from_world, prefill_rows_for

    next_token, _rope, embedding, asset_banks = build_graph(
        d_head=config.model.d_head,
        max_positions=config.model.max_seq_len,
        d_rot=config.model.d_rot,
        asset_config=config.asset_config(),
        wad_path=wad_path,
    )
    compiler_vocab = list(embedding.tokenizer.vocab)
    special = doom_special_tokens(compiler_vocab)
    words = canonical_words(config.asset_config())
    tokenizer = build_standard_tokenizer(config.asset_config(), words=words)

    scene = load_render_scene(config, base_dir=wad_path.parent)
    pose = pose_from_world(scene)
    prompt_rows = prefill_rows_for(scene, pose)
    prompt_text = " ".join(words[row] for row in prompt_rows) + "\n"

    with _outer_bundle_transaction(destination) as stage:
        report = compile_hf_bundle(
            next_token,
            embedding,
            stage,
            d=config.model.d,
            d_head=config.model.d_head,
            max_seq_len=config.model.max_seq_len,
            max_layers=config.model.max_layers,
            optimize=config.model.optimize,
            d_hidden=config.model.d_hidden,
            trim_heads=config.model.trim_heads,
            rms_norm_const_exp=63,
            architecture=CompileProfile.PHI3,
            bias=False,
            bos_token=special.bos_compiler_string,
            eos_token=special.eos_compiler_string,
            verbose=verbose,
            add_bos_token=False,
            write_tokenizer=False,
        )
        tokenizer.save_pretrained(stage)
        fingerprint = vocab_fingerprint()
        write_frozen_data(
            stage,
            asset_config=config.asset_config(),
            palette=asset_banks.playpal,
            origin=scene.origin,
        )
        _stamp_model_config(
            stage,
            config=config,
            vocab_fingerprint=fingerprint,
            eos_row=special.eos_row,
        )
        written_examples = write_bundle_examples(stage, prompt_text=prompt_text)
        _write_model_card(stage, config)
        prompt_path = next(
            path for path in written_examples if path.name == "e1m1_prompt.txt"
        )
        manifest = _candidate_manifest(
            bundle=stage,
            config=config,
            wad_path=wad_path,
            compile_payload=compile_payload,
            report=report,
            prompt_rows=prompt_rows,
            prompt_path=prompt_path,
            vocab_size=len(words),
            row_vocab_fingerprint=fingerprint,
            tokenizer_vocab_sha256=ordered_words_sha256(words),
            bos_row=special.bos_row,
            eos_row=special.eos_row,
            origin=scene.origin,
        )
        _write_json(stage / _MANIFEST, manifest)
        _validate_complete_staged_bundle(stage, config, prompt_rows)
        manifest["validation"]["complete"] = True
        _write_json(stage / _MANIFEST, manifest)

    return BundleReport(
        destination=str(destination),
        n_layers=int(report.n_layers),
        vocab_size=len(words),
        manifest=manifest,
    )


def compile_phi3_bundle_from_config(
    config: RenderConfig,
    *,
    destination: str | Path,
    compile_payload: dict[str, Any],
    base_dir: str | Path | None = None,
    verbose: bool = False,
) -> BundleReport:
    wad_path = resolve_wad_path(config, base_dir=base_dir)
    return compile_phi3_bundle(
        config,
        wad_path=wad_path,
        destination=destination,
        compile_payload=compile_payload,
        verbose=verbose,
    )
