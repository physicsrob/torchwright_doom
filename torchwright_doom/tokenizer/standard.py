"""Stock, data-only Hugging Face tokenizer for Doom row ids.

The tokenizer is intentionally context free: every model row has one compact
semantic word and whitespace is the only token boundary.  Contextual carrier
folding and presentation layout belong to
:class:`torchwright_doom.interpret.formatter.DoomFormatter`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..asset_config import DEFAULT_ASSET_CONFIG, AssetConfig
from ..embedding import TOKEN_VOCAB
from .rows import row_index
from ..vocab import BOS, DONE
from .display import token_word


@dataclass(frozen=True)
class DoomSpecialTokens:
    bos_row: int
    eos_row: int
    bos_compiler_string: str
    eos_compiler_string: str


def doom_special_tokens(compiler_vocab: Sequence[str]) -> DoomSpecialTokens:
    """Resolve Doom's BEGIN/DONE rows against the compiler vocabulary."""
    vocab = list(compiler_vocab)
    bos_row = row_index(BOS, {})
    eos_row = row_index(DONE, {})
    if not (0 <= bos_row < len(vocab) and 0 <= eos_row < len(vocab)):
        raise ValueError(
            f"special-token rows {(bos_row, eos_row)} outside vocabulary of "
            f"width {len(vocab)}"
        )
    if bos_row == eos_row:
        raise ValueError("Doom BOS and DONE rows must be distinct")
    bos = vocab[bos_row]
    eos = vocab[eos_row]
    if vocab.count(bos) != 1 or vocab.index(bos) != bos_row:
        raise ValueError(f"compiler BOS string {bos!r} is not unique at row {bos_row}")
    if vocab.count(eos) != 1 or vocab.index(eos) != eos_row:
        raise ValueError(f"compiler EOS string {eos!r} is not unique at row {eos_row}")
    return DoomSpecialTokens(bos_row, eos_row, bos, eos)


def canonical_words(asset_config: AssetConfig | None = None) -> list[str]:
    """Return the ordered, asset-aware canonical word for every model row."""
    config = asset_config or DEFAULT_ASSET_CONFIG
    words = [
        token_word(
            ttype,
            values,
            wall_names=config.wall_names,
            flat_names=config.flat_names,
        )
        for ttype, values in TOKEN_VOCAB.row_to_token
    ]
    if any(any(ch.isspace() for ch in word) for word in words):
        raise ValueError("canonical Doom tokenizer words must not contain whitespace")
    if len(set(words)) != len(words):
        seen: dict[str, int] = {}
        for row, word in enumerate(words):
            if word in seen:
                raise ValueError(
                    f"canonical tokenizer word collision: rows {seen[word]} and "
                    f"{row} both use {word!r}"
                )
            seen[word] = row
    if len(words) != TOKEN_VOCAB.n_rows:
        raise AssertionError("canonical tokenizer width differs from model vocabulary")
    return words


def ordered_words_sha256(words: Sequence[str]) -> str:
    """Stable identity of the ordered tokenizer vocabulary."""
    encoded = json.dumps(list(words), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_standard_tokenizer(
    asset_config: AssetConfig | None = None,
    *,
    words: Sequence[str] | None = None,
):
    """Build the stock fast WordLevel tokenizer without adding any row ids."""
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    ordered = list(words) if words is not None else canonical_words(asset_config)
    vocab = {word: row for row, word in enumerate(ordered)}
    if len(vocab) != len(ordered):
        raise ValueError("standard tokenizer vocabulary is not injective")

    bos_row = row_index(BOS, {})
    eos_row = row_index(DONE, {})
    raw = Tokenizer(WordLevel(vocab=vocab, unk_token=None))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token=ordered[bos_row],
        eos_token=ordered[eos_row],
        pad_token=ordered[eos_row],
        unk_token=None,
        clean_up_tokenization_spaces=False,
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.init_kwargs["add_bos_token"] = False
    if len(tokenizer) != len(ordered) or tokenizer.vocab_size != len(ordered):
        raise ValueError(
            "special-token registration changed the Doom tokenizer vocabulary width"
        )
    if tokenizer.bos_token_id != bos_row or tokenizer.eos_token_id != eos_row:
        raise ValueError("standard tokenizer special-token ids disagree with Doom rows")
    if tokenizer.pad_token_id != eos_row:
        raise ValueError("standard tokenizer padding must reuse the DONE/EOS row")
    return tokenizer


def save_standard_tokenizer(
    destination: str | Path, asset_config: AssetConfig | None = None
) -> list[str]:
    tokenizer = build_standard_tokenizer(asset_config)
    return list(tokenizer.save_pretrained(str(destination)))
