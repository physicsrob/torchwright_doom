from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from torchwright_doom.inference.hf_bundle import (
    _outer_bundle_transaction,
    compile_payload_sha256,
    is_complete_hf_bundle,
    validate_bundle_manifest,
)
from torchwright_doom.config import RenderConfig
from torchwright_doom.identity import compile_payload_domain


def _fake_bundle(path: Path, payload: dict, *, complete: bool = True) -> None:
    path.mkdir(parents=True)
    required = {
        "config.json": json.dumps(
            {"model_type": "phi3", "architectures": ["Phi3ForCausalLM"]}
        ),
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"model.embed_tokens.weight": "model-00001.safetensors"}}
        ),
        "model-00001.safetensors": "fake",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "doom_vocab.json": "{}",
        "doom_tables.json": "{}",
        "doom_palette.json": "{}",
        "README.md": "model card",
        "infer.py": "# portable inference program\n",
        "examples/e1m1_prompt.txt": "begin\n",
        "tools/pretty_text.py": "# pretty\n",
        "tools/txt_to_png.py": "# png\n",
        "tools/README.md": "# tools\n",
    }
    for name, value in required.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    files: dict[str, dict] = {}
    for name, value in required.items():
        facts: dict = {"size": len(value.encode())}
        # Layout v2 verifies every hash-bearing file in a normal probe;
        # safetensor shards are the only hash-exempt (size-checked) files.
        if not name.endswith(".safetensors"):
            facts["sha256"] = hashlib.sha256(value.encode()).hexdigest()
        files[name] = facts
    manifest = {
        "format": "torchwright_doom.phi3.v1",
        "artifact_kind": "hf_phi3_bundle",
        "profile": "phi3",
        "compile_payload_sha256": compile_payload_sha256(payload),
        "files": files,
        "validation": {"complete": complete, "format_version": 2},
    }
    (path / "doom_bundle_manifest.json").write_text(json.dumps(manifest))


def _edit_manifest(bundle: Path, mutate) -> None:
    manifest_path = bundle / "doom_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest))


def test_lightweight_complete_bundle_probe(tmp_path: Path) -> None:
    payload = {"artifact": {"kind": "hf_phi3_bundle"}}
    bundle = tmp_path / "bundle"
    _fake_bundle(bundle, payload)
    assert is_complete_hf_bundle(bundle, expected_payload=payload)
    assert not is_complete_hf_bundle(bundle, expected_payload={"different": True})


def test_public_manifest_validation_rejects_incomplete(tmp_path: Path) -> None:
    payload = {"artifact": {"kind": "hf_phi3_bundle"}}
    bundle = tmp_path / "bundle"
    _fake_bundle(bundle, payload, complete=False)
    with pytest.raises(ValueError, match="not complete"):
        validate_bundle_manifest(bundle)
    validate_bundle_manifest(bundle, allow_incomplete=True)


def test_missing_or_old_layout_version_fails_closed(tmp_path: Path) -> None:
    """Pre-v2 cache entries (and manifests with no version at all) are
    rejected even when no expected compile payload is passed."""
    payload = {"artifact": {"kind": "hf_phi3_bundle"}}
    bundle = tmp_path / "bundle"
    _fake_bundle(bundle, payload)
    validate_bundle_manifest(bundle)

    _edit_manifest(bundle, lambda m: m["validation"].update(format_version=1))
    with pytest.raises(ValueError, match="layout version"):
        validate_bundle_manifest(bundle)
    assert not is_complete_hf_bundle(bundle)

    _edit_manifest(bundle, lambda m: m["validation"].pop("format_version"))
    with pytest.raises(ValueError, match="layout version"):
        validate_bundle_manifest(bundle, allow_incomplete=True)


def test_declared_size_mismatch_fails(tmp_path: Path) -> None:
    payload = {"artifact": {"kind": "hf_phi3_bundle"}}
    bundle = tmp_path / "bundle"
    _fake_bundle(bundle, payload)
    (bundle / "doom_tables.json").write_text("{}  ")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_bundle_manifest(bundle)
    assert not is_complete_hf_bundle(bundle)


def test_truncated_shard_fails(tmp_path: Path) -> None:
    payload = {"artifact": {"kind": "hf_phi3_bundle"}}
    bundle = tmp_path / "bundle"
    _fake_bundle(bundle, payload)
    (bundle / "model-00001.safetensors").write_text("fa")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_bundle_manifest(bundle)


def test_altered_hash_bearing_file_fails(tmp_path: Path) -> None:
    """A same-size content change to a hash-bearing file (here a shipped
    tool) fails the normal completeness probe — no verify flag needed."""
    payload = {"artifact": {"kind": "hf_phi3_bundle"}}
    bundle = tmp_path / "bundle"
    _fake_bundle(bundle, payload)
    original = (bundle / "tools/pretty_text.py").read_text()
    (bundle / "tools/pretty_text.py").write_text("# ALTERED" + original[9:])
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bundle_manifest(bundle)


def test_indexed_but_undeclared_shard_fails(tmp_path: Path) -> None:
    payload = {"artifact": {"kind": "hf_phi3_bundle"}}
    bundle = tmp_path / "bundle"
    _fake_bundle(bundle, payload)
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001.safetensors",
            "lm_head.weight": "model-00002.safetensors",
        }
    }
    (bundle / "model-00002.safetensors").write_text("fake")
    index_text = json.dumps(index)
    (bundle / "model.safetensors.index.json").write_text(index_text)
    _edit_manifest(
        bundle,
        lambda m: m["files"].__setitem__(
            "model.safetensors.index.json",
            {
                "size": len(index_text.encode()),
                "sha256": hashlib.sha256(index_text.encode()).hexdigest(),
            },
        ),
    )
    with pytest.raises(ValueError, match="undeclared shard"):
        validate_bundle_manifest(bundle)


def test_legacy_examples_layout_is_not_accepted(tmp_path: Path) -> None:
    """A bundle carrying the retired examples/infer.py instead of a root
    infer.py is incomplete."""
    payload = {"artifact": {"kind": "hf_phi3_bundle"}}
    bundle = tmp_path / "bundle"
    _fake_bundle(bundle, payload)
    (bundle / "infer.py").rename(bundle / "examples" / "infer.py")
    with pytest.raises(ValueError, match="missing"):
        validate_bundle_manifest(bundle)
    assert not is_complete_hf_bundle(bundle)


def test_outer_transaction_preserves_previous_destination_on_failure(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "identity").write_text("old")
    with pytest.raises(RuntimeError, match="injected"):
        with _outer_bundle_transaction(destination) as stage:
            stage.mkdir()
            (stage / "identity").write_text("new")
            raise RuntimeError("injected")
    assert (destination / "identity").read_text() == "old"


def test_outer_transaction_replaces_complete_destination(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "stale").write_text("old")
    with _outer_bundle_transaction(destination) as stage:
        stage.mkdir()
        (stage / "identity").write_text("new")
    assert not (destination / "stale").exists()
    assert (destination / "identity").read_text() == "new"


def test_direct_builder_passes_exact_graph_embedding_and_phi3_contract(
    tmp_path: Path, monkeypatch
) -> None:
    import torchwright.compiler

    from torchwright_doom import constants
    from torchwright_doom import embedding as embedding_mod
    from torchwright_doom.asset_banks import PLAYPAL
    from torchwright_doom.inference import compiled_model, hf_bundle
    from torchwright_doom.prompt import scene as prompt_scene
    from torchwright_doom.tokenizer.rows import row_index
    from torchwright_doom.vocab import BOS, DONE

    config = RenderConfig()
    monkeypatch.setattr(constants, "SCREEN_WIDTH", config.screen[0])
    monkeypatch.setattr(constants, "SCREEN_HEIGHT", config.screen[1])
    wad = tmp_path / "doom1.wad"
    wad.write_bytes(b"test-wad")
    payload = {
        **compile_payload_domain(config, wad),
        "git": {"torchwright": "tw", "torchwright_doom": "doom"},
    }
    compiler_vocab = [
        embedding_mod._row_label(ttype, values)
        for ttype, values in embedding_mod.TOKEN_VOCAB.row_to_token
    ]
    exact_embedding = SimpleNamespace(tokenizer=SimpleNamespace(vocab=compiler_vocab))
    graph_output = object()
    banks = SimpleNamespace(playpal=PLAYPAL)
    monkeypatch.setattr(
        compiled_model,
        "build_graph",
        lambda **kwargs: (graph_output, object(), exact_embedding, banks),
    )
    scene = SimpleNamespace(origin=(0.0, 0.0))
    monkeypatch.setattr(
        prompt_scene, "load_render_scene", lambda *args, **kwargs: scene
    )
    monkeypatch.setattr(prompt_scene, "pose_from_world", lambda scene: object())
    prompt_rows = [row_index(BOS, {}), row_index(DONE, {})]
    monkeypatch.setattr(
        prompt_scene, "prefill_rows_for", lambda scene, pose: prompt_rows
    )
    monkeypatch.setattr(
        hf_bundle, "_validate_complete_staged_bundle", lambda *args: None
    )

    captured: dict[str, Any] = {}

    def fake_compile(output, embedding, destination, **kwargs):
        captured.update(output=output, embedding=embedding, kwargs=kwargs)
        destination = Path(destination)
        destination.mkdir(parents=True)
        (destination / "config.json").write_text(
            json.dumps({"model_type": "phi3", "architectures": ["Phi3ForCausalLM"]})
        )
        (destination / "model-00001.safetensors").write_text("fake")
        (destination / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "model-00001.safetensors"}})
        )
        provenance = SimpleNamespace(
            to_dict=lambda: {"selected_origin": "heuristic", "delivery": "fresh"}
        )
        return SimpleNamespace(n_layers=3, schedule_provenance=provenance)

    monkeypatch.setattr(torchwright.compiler, "compile_hf_bundle", fake_compile)
    destination = tmp_path / "published"
    report = hf_bundle.compile_phi3_bundle(
        config,
        wad_path=wad,
        destination=destination,
        compile_payload=payload,
    )
    assert captured["output"] is graph_output
    assert captured["embedding"] is exact_embedding
    assert captured["kwargs"]["architecture"].value == "phi3"
    assert captured["kwargs"]["bias"] is False
    assert captured["kwargs"]["write_tokenizer"] is False
    assert captured["kwargs"]["rms_norm_const_exp"] == 63
    assert report.n_layers == 3
    model_config = json.loads((destination / "config.json").read_text())
    assert model_config["original_max_position_embeddings"] == config.model.max_seq_len
    assert json.loads((destination / "doom_bundle_manifest.json").read_text())[
        "validation"
    ]["complete"]
