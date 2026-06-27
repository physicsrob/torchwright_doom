"""``DoomTokenizer`` — a stock HuggingFace slow tokenizer over the DOOM stream.

The core is a pure ``WordLevel`` map: each ``W_EMBED`` row id has one
context-free label (``embedding._row_label``), and ``_convert_token_to_id`` /
``_convert_id_to_token`` are that 1:1 bijection (no arithmetic). The readability
and the one contextual operation (de-quantizing a ``VALUE`` by its preceding
marker, plus header-break layout) live in ``_tokenize`` /
``convert_tokens_to_string`` — the pre-tokenizer / decoder boundary where HF
tokenizers conventionally do deterministic, marker-driven string work. The
``surface`` module is that string layer; this class only bridges it to HF's API
and the row-id bijection in ``tokens_bridge``.

Subject to the import-time-vocab caveat (``embedding`` builds the screen-sized
vocab at import): construct under the same screen config the artifact was
compiled at. ``save_vocabulary`` records that config + a vocab fingerprint so
``from_pretrained`` fails loud on a mismatch rather than silently re-keying ids.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizer

from ..asset_config import DEFAULT_ASSET_CONFIG, AssetConfig
from ..constants import COLUMN_COUNT, PIXEL_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH
from ..embedding import TOKEN_VOCAB, _row_label
from ..inference.tokens_bridge import row_to_token, token_to_row
from ..tokens import FloatSlot, IntSlot, Token
from ..vocab import BOS, DONE, VOCAB_TYPES
from . import surface

VOCAB_FILES_NAMES = {"vocab_file": "doom_vocab.json"}

# Special tokens map onto existing rows — never appended ids (so nothing shifts):
_EOS = _row_label(DONE, {})  # "done" — the generation stop (cli.py)
_BOS = _row_label(BOS, {})  # "bos" — the position-0 anchor (build_prompt prepends it)


def screen_config() -> dict[str, int]:
    """The import-time screen dimensions the vocab was built against."""
    return {
        "width": int(SCREEN_WIDTH),
        "height": int(SCREEN_HEIGHT),
        "column_count": int(COLUMN_COUNT),
        "pixel_width": int(PIXEL_WIDTH),
    }


def vocab_fingerprint() -> str:
    """A stable hash of the vocab's identity: every type's name and slot grid,
    plus the total row count. Two builds with the same fingerprint enumerate
    the same id<->token table; a drift (slot widened, type added, screen
    rescaled) changes it."""
    signature: list[Any] = []
    for ttype in VOCAB_TYPES:
        slots: list[Any] = []
        for name, slot in ttype.slots.items():
            if isinstance(slot, IntSlot):
                slots.append([name, "int", slot.lo, slot.hi])
            elif isinstance(slot, FloatSlot):
                slots.append([name, "float", slot.lo, slot.hi, slot.levels])
        signature.append([ttype.name, slots])
    payload = json.dumps([signature, TOKEN_VOCAB.n_rows], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class DoomTokenizer(PreTrainedTokenizer):
    """Slow ``PreTrainedTokenizer`` wrapping the readable surface."""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab_file: str | None = None,
        *,
        asset_config: AssetConfig | None = None,
        eos_token: str = _EOS,
        bos_token: str = _BOS,
        pad_token: str = _EOS,
        **kwargs,
    ):
        # Build the id<->label bijection BEFORE super().__init__ — it processes
        # the special tokens and consults get_vocab(), which needs these.
        self._id_to_label: list[str] = [
            _row_label(ttype, values) for ttype, values in TOKEN_VOCAB.row_to_token
        ]
        self._label_to_id: dict[str, int] = {
            label: row for row, label in enumerate(self._id_to_label)
        }
        # The WordLevel map is a bijection only if every row has a UNIQUE label.
        # The pretty per-id labels (aliases / decoded words / value-in-name tags)
        # make injectivity load-bearing, so guard it loud here rather than let a
        # collision silently collapse two rows onto one id.
        assert len(self._label_to_id) == len(self._id_to_label), (
            "token labels are not injective — two rows share a label; see "
            "tokenizer.display.token_label"
        )
        config = asset_config or DEFAULT_ASSET_CONFIG
        self._asset_config = config
        # Surface knobs for the stock view: scene-relative coords, BAM angles,
        # readable WAD texture names. None changes the id stream.
        self._knobs: dict[str, Any] = dict(
            wall_names=config.wall_names,
            flat_names=config.flat_names,
        )

        if vocab_file is not None:
            _validate_vocab_file(vocab_file)

        # bos is the position-0 anchor but is NOT auto-prepended: build_prompt
        # already prepends it, so build_inputs_with_special_tokens stays identity
        # (the PreTrainedTokenizer default). No unk token (the stream is always
        # in-vocab). pad == eos.
        super().__init__(
            eos_token=eos_token,
            bos_token=bos_token,
            pad_token=pad_token,
            unk_token=None,
            **kwargs,
        )

    # --- WordLevel id<->label bijection -----------------------------------

    @property
    def vocab_size(self) -> int:
        return TOKEN_VOCAB.n_rows

    def get_vocab(self) -> dict[str, int]:
        return {**self._label_to_id, **self.added_tokens_encoder}

    def _convert_token_to_id(self, token: str) -> int:
        return self._label_to_id[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_label[index]

    # --- contextual string layer (surface) --------------------------------

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Readable text -> context-free row labels (the WordLevel pieces). Uses
        the grammar's :func:`surface.encode`, which ignores whitespace — so a
        formatted (header-laid-out) figure tokenizes the same as the flat
        stream."""
        labels = []
        for ttype, values in surface.encode(text, **self._knobs):
            row = token_to_row(Token(ttype, dict(values)))
            labels.append(self._id_to_label[row])
        return labels

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Row labels -> the **flat** vanilla surface text (de-quantized, single
        spaces). The grammar's :func:`surface.decode`; the blog figure adds the
        whitespace layout separately via :func:`surface.format`."""
        toks = [row_to_token(self._label_to_id[label]) for label in tokens]
        return surface.decode(toks, **self._knobs)

    # --- serialization -----------------------------------------------------

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> tuple[str, ...]:
        """Emit the **self-contained, torch-free** tokenizer bundle: the standalone
        ``tokenization_doom.DoomTokenizer`` plus the two frozen JSON artifacts,
        loadable with only ``transformers`` (+ ``trust_remote_code``). Delegates to
        :func:`freeze.export_bundle`, which freezes the pretty vocab + carrier
        tables and routes ``custom_object_save`` through the standalone class.

        This is deliberately *not* the ``PreTrainedTokenizer`` default (which would
        copy *this* file — and so torch — into the bundle). The in-package
        :meth:`save_vocabulary` still writes the tiny identity card for the
        in-package self-reload validation path. The base-class ``legacy_format`` /
        ``push_to_hub`` knobs don't apply to the frozen bundle and are ignored."""
        from . import freeze

        return tuple(
            freeze.export_bundle(str(save_directory), asset_config=self._asset_config)
        )

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        """Write the small ``doom_vocab.json`` identity card — screen config,
        fingerprint, and expected row count — *not* the 133k-line label table
        (the labels are rebuilt deterministically from the vocab)."""
        os.makedirs(save_directory, exist_ok=True)
        name = (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES[
            "vocab_file"
        ]
        path = os.path.join(save_directory, name)
        Path(path).write_text(
            json.dumps(
                {
                    "screen": screen_config(),
                    "n_rows_expected": TOKEN_VOCAB.n_rows,
                    "fingerprint": vocab_fingerprint(),
                },
                indent=2,
            )
        )
        return (path,)


def _validate_vocab_file(vocab_file: str) -> None:
    """Assert the saved identity card matches the current import-time vocab,
    so a screen / vocab mismatch fails loud instead of silently re-keying ids."""
    data = json.loads(Path(vocab_file).read_text())
    expected_rows = data.get("n_rows_expected")
    if expected_rows != TOKEN_VOCAB.n_rows:
        raise ValueError(
            f"doom_vocab.json expects {expected_rows} rows but the imported "
            f"vocab has {TOKEN_VOCAB.n_rows} — wrong screen config? Saved "
            f"screen={data.get('screen')}, current={screen_config()}."
        )
    current = vocab_fingerprint()
    if data.get("fingerprint") != current:
        raise ValueError(
            "doom_vocab.json fingerprint mismatch: the saved vocabulary is not "
            "the one this process built (a type/slot/screen drift). "
            f"saved={data.get('fingerprint')!r} current={current!r}."
        )
