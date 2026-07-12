from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

import torchwright_doom.inference_example as inference_example
from torchwright_doom.asset_banks import PLAYPAL
from torchwright_doom.asset_config import DEFAULT_ASSET_CONFIG
from torchwright_doom.bundle_examples import write_bundle_examples
from torchwright_doom.frame_decoder_kernel import _pixels, _write_png
from torchwright_doom.inference.decode import decode_rows_to_pixels
from torchwright_doom.inference.tokens_bridge import row_index
from torchwright_doom.tokenizer.freeze import build_vocab_blob
from torchwright_doom.vocab import (
    PIXEL,
    SET_CURSOR_DIRECTION_Y,
    SET_CURSOR_X,
    SET_CURSOR_Y,
)


def _imports(path: Path) -> set[str]:
    imports = set()
    for line in path.read_text().splitlines():
        if line.startswith("import "):
            imports.add(line.split()[1].split(".", 1)[0])
        elif line.startswith("from "):
            imports.add(line.split()[1].split(".", 1)[0])
    return imports


def test_bundle_examples_have_the_required_dependency_boundaries(
    tmp_path: Path,
) -> None:
    write_bundle_examples(tmp_path, prompt_text="begin")
    examples = tmp_path / "examples"
    infer_imports = _imports(examples / "infer.py")
    assert "torchwright" not in infer_imports
    assert "torchwright_doom" not in infer_imports
    assert infer_imports <= set(sys.stdlib_module_names) | {
        "__future__",
        "torch",
        "transformers",
    }
    for name in ("pretty_text.py", "txt_to_png.py"):
        assert _imports(examples / name) <= set(sys.stdlib_module_names) | {
            "__future__"
        }


def test_bundle_infer_is_the_exact_production_source(tmp_path: Path) -> None:
    write_bundle_examples(tmp_path, prompt_text="begin")
    source = Path(inference_example.__file__)
    assert (tmp_path / "examples/infer.py").read_bytes() == source.read_bytes()


def test_portable_infer_accepts_text_and_writes_ids_plus_raw_text(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "bundle"
    output = tmp_path / "out"
    model_dir.mkdir()
    canonical = b"canonical\n"
    manifest = {
        "validation": {"complete": True},
        "bundle_identity": "bundle:test",
        "compile_payload_sha256": "payload",
        "row_vocab_fingerprint": "vocab",
        "prompt": {
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "row_ids_sha256": "canonical-row-hash",
        },
        "generation": {"max_new_tokens": 5},
    }
    (model_dir / "doom_bundle_manifest.json").write_text(json.dumps(manifest))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("foo bar\n")

    class Inputs(dict):
        def __init__(self, rows: list[int]):
            self.input_ids = torch.tensor([rows], dtype=torch.long)
            super().__init__(input_ids=self.input_ids)

        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 4
        pad_token_id = 4

        def __call__(self, text, *, return_tensors=None, add_special_tokens=False):
            rows = [1, 2] if text.strip() == "foo bar" else [3, 4]
            return Inputs(rows) if return_tensors else {"input_ids": rows}

        def decode(self, rows, **kwargs):
            assert rows == [3, 4]
            return "three done"

    class Model:
        config = type(
            "Config",
            (),
            {
                "max_position_embeddings": 16,
                "original_max_position_embeddings": 16,
                "_attn_implementation": "eager",
            },
        )()

        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self.weight

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["use_cache"] is True
            assert kwargs["max_new_tokens"] == 5
            streamer = kwargs["streamer"]
            streamer.put(kwargs["input_ids"])
            streamer.put(torch.tensor([3]))
            streamer.put(torch.tensor([4]))
            streamer.end()
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    captured = {}

    def fake_model_load(path, **kwargs):
        captured.update(kwargs)
        return Model()

    monkeypatch.setattr(
        inference_example.AutoTokenizer, "from_pretrained", lambda path: Tokenizer()
    )
    monkeypatch.setattr(
        inference_example.AutoModelForCausalLM, "from_pretrained", fake_model_load
    )
    assert (
        inference_example.main(
            [
                "--model",
                str(model_dir),
                "--prompt",
                str(prompt),
                "--output",
                str(output),
                "--device",
                "cuda",
            ]
        )
        == 0
    )
    assert captured["disable_mmap"] is True
    assert captured["device_map"] == "cuda"
    payload = json.loads((output / "output.ids.json").read_text())
    assert payload["prompt"]["row_ids"] == [1, 2]
    assert payload["prompt"]["matches_bundled_prompt"] is False
    assert payload["attention_implementation"] == "eager"
    assert payload["emitted_row_ids"] == [3, 4]
    assert payload["generation"] == {
        "mode": "transformers_generate",
        "max_new_tokens": 5,
        "termination_reason": "terminal",
    }
    assert (output / "output.txt").read_text() == "three done\n"


def test_portable_infer_runs_a_real_saved_phi3_bundle(tmp_path: Path) -> None:
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import Phi3Config, Phi3ForCausalLM, PreTrainedTokenizerFast

    model_dir = tmp_path / "tiny-bundle"
    output = tmp_path / "real-out"
    model_dir.mkdir()
    vocab = {"begin": 0, "scene": 1, "done": 2}
    raw = Tokenizer(WordLevel(vocab=vocab, unk_token=None))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="begin",
        eos_token="done",
        pad_token="done",
        unk_token=None,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.save_pretrained(model_dir)
    model = Phi3ForCausalLM(
        Phi3Config(
            vocab_size=len(vocab),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            original_max_position_embeddings=16,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=2,
        )
    )
    model.save_pretrained(model_dir)

    prompt = model_dir / "prompt.txt"
    prompt_bytes = b"begin scene\n"
    prompt.write_bytes(prompt_bytes)
    prompt_rows = [0, 1]
    manifest = {
        "validation": {"complete": True},
        "bundle_identity": "bundle:real-tiny",
        "compile_payload_sha256": "payload",
        "row_vocab_fingerprint": "vocab",
        "prompt": {
            "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "row_ids_sha256": hashlib.sha256(
                json.dumps(prompt_rows, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "generation": {"max_new_tokens": 2},
    }
    (model_dir / "doom_bundle_manifest.json").write_text(json.dumps(manifest))

    assert (
        inference_example.main(
            [
                "--model",
                str(model_dir),
                "--prompt",
                str(prompt),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = json.loads((output / "output.ids.json").read_text())
    assert payload["bundle"] == "bundle:real-tiny"
    assert payload["prompt"]["matches_bundled_prompt"] is True
    assert payload["prompt"]["row_ids"] == prompt_rows
    assert 1 <= len(payload["emitted_row_ids"]) <= 2
    assert (output / "output.txt").read_text().strip()


def test_standalone_frame_decoder_matches_host_cursor_protocol(tmp_path: Path) -> None:
    rows = [
        row_index(SET_CURSOR_DIRECTION_Y, {}),
        row_index(SET_CURSOR_X, {"x": 2}),
        row_index(SET_CURSOR_Y, {"y": 3}),
        row_index(PIXEL, {"color": 5, "w": 1}),
        row_index(PIXEL, {"color": 6, "w": 1}),
    ]
    vocab = build_vocab_blob(
        DEFAULT_ASSET_CONFIG.wall_names, DEFAULT_ASSET_CONFIG.flat_names
    )
    standalone = _pixels(rows, vocab, [list(rgb) for rgb in PLAYPAL])
    assert standalone == decode_rows_to_pixels(rows)
    png = tmp_path / "frame.png"
    _write_png(png, 8, 8, standalone)
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
