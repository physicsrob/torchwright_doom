from __future__ import annotations

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
from torchwright_doom.inference.config import (
    RenderConfig,
    compile_payload_domain,
)


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
        "examples/e1m1_prompt.txt": "begin\n",
        "examples/infer.py": "",
        "examples/pretty_text.py": "",
        "examples/txt_to_png.py": "",
        "examples/README.md": "",
    }
    for name, value in required.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    manifest = {
        "format": "torchwright_doom.phi3.v1",
        "artifact_kind": "hf_phi3_bundle",
        "profile": "phi3",
        "compile_payload_sha256": compile_payload_sha256(payload),
        "files": {
            name: {"sha256": "not-checked", "size": len(value)}
            for name, value in required.items()
        },
        "validation": {"complete": complete, "format_version": 1},
    }
    (path / "doom_bundle_manifest.json").write_text(json.dumps(manifest))


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
    from torchwright_doom.inference import compiled_model, hf_bundle, wad_scene
    from torchwright_doom.inference.tokens_bridge import row_index
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
    monkeypatch.setattr(wad_scene, "load_render_scene", lambda *args, **kwargs: scene)
    monkeypatch.setattr(wad_scene, "pose_from_world", lambda scene: object())
    prompt_rows = [row_index(BOS, {}), row_index(DONE, {})]
    monkeypatch.setattr(wad_scene, "prefill_rows_for", lambda scene, pose: prompt_rows)
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
