from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from torchwright_doom.asset_banks import PLAYPAL
from torchwright_doom.interpret.formatter import DoomFormatter
from torchwright_doom.tokenizer.rows import row_index
from torchwright_doom.tokenizer.freeze import write_frozen_data
from torchwright_doom.tokenizer.identity import screen_config, vocab_fingerprint
from torchwright_doom.tokenizer.standard import canonical_words, ordered_words_sha256
from torchwright_doom.vocab import PLAYER_X_MARK, VALUE


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, complete: bool = True) -> Path:
    paths = write_frozen_data(tmp_path, palette=PLAYPAL, origin=(1024.0, -4096.0))
    words = canonical_words()
    manifest = {
        "format": "torchwright_doom.phi3.v1",
        "artifact_kind": "hf_phi3_bundle",
        "vocab_size": len(words),
        "row_vocab_fingerprint": vocab_fingerprint(),
        "tokenizer_vocab_sha256": ordered_words_sha256(words),
        "screen": {
            "width": screen_config()["width"],
            "height": screen_config()["height"],
        },
        "files": {path.name: {"sha256": _sha(path)} for path in paths},
        "validation": {"complete": complete, "format_version": 1},
    }
    (tmp_path / "doom_bundle_manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_formatter_round_trips_canonical_text(tmp_path) -> None:
    bundle = _bundle(tmp_path)
    formatter = DoomFormatter.from_bundle(bundle)
    rows = [
        row_index(PLAYER_X_MARK, {}),
        row_index(VALUE, {"v": 0.0}),
    ]
    raw = formatter.raw_text_from_rows(rows)
    pretty = formatter.format_text(raw)
    assert "viewx(" in pretty
    assert formatter.parse_pretty_text(pretty) == raw
    assert formatter.rows_from_raw_text(formatter.parse_pretty_text(pretty)) == rows


def test_public_formatter_rejects_incomplete_bundle(tmp_path) -> None:
    bundle = _bundle(tmp_path, complete=False)
    try:
        DoomFormatter.from_bundle(bundle)
    except ValueError as exc:
        assert "not complete" in str(exc)
    else:
        raise AssertionError("public formatter accepted an incomplete bundle")
    DoomFormatter.from_bundle(bundle, allow_incomplete=True)


def test_formatter_kernel_is_stdlib_only() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "torchwright_doom"
        / "portable"
        / "pretty_text.py"
    ).read_text()
    imports = set()
    for line in source.splitlines():
        if line.startswith("import "):
            imports.add(line.split()[1].split(".", 1)[0])
        elif line.startswith("from "):
            imports.add(line.split()[1].split(".", 1)[0])
    assert imports <= set(sys.stdlib_module_names) | {"__future__"}
