"""Bundle layout gate: root ``infer.py`` + ``tools/`` byte-identical to their
sources, dependency boundaries intact, and the copied inference program's
placement-derived defaults working from a real bundle root."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest
import torch

from torchwright_doom.asset_banks import PLAYPAL
from torchwright_doom.asset_config import DEFAULT_ASSET_CONFIG
from torchwright_doom.bundle.layout import write_bundle_layout
from torchwright_doom.inference.decode import decode_rows_to_pixels
from torchwright_doom.inference.tokens_bridge import row_index
from torchwright_doom.portable.txt_to_png import _pixels, _write_png
from torchwright_doom.tokenizer.freeze import build_vocab_blob
from torchwright_doom.vocab import (
    PIXEL,
    SET_CURSOR_DIRECTION_Y,
    SET_CURSOR_X,
    SET_CURSOR_Y,
)

ROOT = Path(__file__).resolve().parents[2]
INFER_SOURCE = ROOT / "torchwright_doom" / "infer.py"
PORTABLE_DIR = ROOT / "torchwright_doom" / "portable"


@lru_cache(maxsize=1)
def _infer_module() -> ModuleType:
    """Load the standalone inference program from its file path — the same
    standalone shape a bundle consumer runs. The package never imports it
    (`import torchwright_doom.infer` is forbidden by the runtime policy)."""
    spec = importlib.util.spec_from_file_location("doom_bundle_infer", INFER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _imports(path: Path) -> set[str]:
    imports = set()
    for line in path.read_text().splitlines():
        if line.startswith("import "):
            imports.add(line.split()[1].split(".", 1)[0])
        elif line.startswith("from "):
            imports.add(line.split()[1].split(".", 1)[0])
    return imports


def test_bundle_layout_has_the_required_dependency_boundaries(
    tmp_path: Path,
) -> None:
    write_bundle_layout(tmp_path, prompt_text="begin")
    infer_imports = _imports(tmp_path / "infer.py")
    assert "torchwright" not in infer_imports
    assert "torchwright_doom" not in infer_imports
    assert infer_imports <= set(sys.stdlib_module_names) | {
        "__future__",
        "torch",
        "transformers",
    }
    for name in ("tools/pretty_text.py", "tools/txt_to_png.py"):
        assert _imports(tmp_path / name) <= set(sys.stdlib_module_names) | {
            "__future__"
        }


def test_bundle_layout_is_byte_identical_to_sources(tmp_path: Path) -> None:
    written = write_bundle_layout(tmp_path, prompt_text="begin")
    expected = {
        tmp_path / "infer.py": INFER_SOURCE,
        tmp_path / "tools/pretty_text.py": PORTABLE_DIR / "pretty_text.py",
        tmp_path / "tools/txt_to_png.py": PORTABLE_DIR / "txt_to_png.py",
    }
    for target, source in expected.items():
        assert target.read_bytes() == source.read_bytes(), target
    assert (tmp_path / "examples/e1m1_prompt.txt").read_text() == "begin\n"
    assert (tmp_path / "tools/README.md").read_text().startswith("# Reproduce")
    assert set(written) == {
        *expected,
        tmp_path / "examples/e1m1_prompt.txt",
        tmp_path / "tools/README.md",
    }
    # The old layout must not reappear.
    assert not (tmp_path / "examples" / "infer.py").exists()


def test_portable_infer_accepts_text_and_writes_ids_plus_raw_text(
    tmp_path: Path, monkeypatch
) -> None:
    infer = _infer_module()
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
        infer.AutoTokenizer, "from_pretrained", lambda path: Tokenizer()
    )
    monkeypatch.setattr(infer.AutoModelForCausalLM, "from_pretrained", fake_model_load)
    assert (
        infer.main(
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


def _tiny_phi3_bundle(model_dir: Path, *, original_max: int = 16) -> list[int]:
    """Save a real tiny stock Phi-3 + tokenizer + bundle layout; returns the
    bundled prompt's row ids."""
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import Phi3Config, Phi3ForCausalLM, PreTrainedTokenizerFast

    model_dir.mkdir(parents=True, exist_ok=True)
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
            original_max_position_embeddings=original_max,
            bos_token_id=0,
            eos_token_id=2,
            pad_token_id=2,
        )
    )
    model.save_pretrained(model_dir)

    write_bundle_layout(model_dir, prompt_text="begin scene")
    prompt_bytes = (model_dir / "examples/e1m1_prompt.txt").read_bytes()
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
    return prompt_rows


def test_copied_bundle_root_infer_resolves_its_own_defaults(tmp_path: Path) -> None:
    """Run the exact copied ``<bundle>/infer.py`` as a subprocess with
    ``--model`` and ``--prompt`` omitted: the program must resolve its own
    bundle directory and the bundled default prompt (Decision 1)."""
    model_dir = tmp_path / "tiny-bundle"
    output = tmp_path / "real-out"
    prompt_rows = _tiny_phi3_bundle(model_dir)

    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(model_dir / "infer.py"),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
        check=True,
        cwd=elsewhere,
    )
    payload = json.loads((output / "output.ids.json").read_text())
    assert payload["bundle"] == "bundle:real-tiny"
    assert payload["prompt"]["matches_bundled_prompt"] is True
    assert payload["prompt"]["row_ids"] == prompt_rows
    assert 1 <= len(payload["emitted_row_ids"]) <= 2
    assert (output / "output.txt").read_text().strip()


def test_portable_infer_rejects_inconsistent_position_capacity(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "tiny-bundle-bad-positions"
    _tiny_phi3_bundle(model_dir, original_max=8)
    with pytest.raises(RuntimeError, match="position"):
        _infer_module().main(
            [
                "--model",
                str(model_dir),
                "--output",
                str(tmp_path / "out"),
                "--device",
                "cpu",
            ]
        )


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
