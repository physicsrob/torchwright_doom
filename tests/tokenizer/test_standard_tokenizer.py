from __future__ import annotations

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from transformers import AutoTokenizer

from torchwright_doom.embedding import TOKEN_VOCAB
from torchwright_doom.inference.tokens_bridge import row_index
from torchwright_doom.tokenizer import display
from torchwright_doom.tokenizer.standard import (
    build_standard_tokenizer,
    canonical_words,
)
from torchwright_doom.vocab import BOS, DONE


def test_token_word_is_compact_label_from_same_fields() -> None:
    for ttype, values in TOKEN_VOCAB.row_to_token:
        label = display.token_label(ttype, values)
        word = display.token_word(ttype, values)
        assert word == label.replace(", ", ",")
        assert not any(ch.isspace() for ch in word)


def test_standard_tokenizer_is_exhaustive_bijection(tmp_path) -> None:
    words = canonical_words()
    tokenizer = build_standard_tokenizer(words=words)
    assert len(words) == TOKEN_VOCAB.n_rows
    assert tokenizer.get_vocab() == {word: row for row, word in enumerate(words)}
    assert tokenizer.convert_tokens_to_ids(words) == list(range(len(words)))
    assert tokenizer.bos_token_id == row_index(BOS, {})
    assert tokenizer.eos_token_id == row_index(DONE, {})
    assert tokenizer.pad_token_id == tokenizer.eos_token_id

    tokenizer.save_pretrained(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "tokenizer.json",
        "tokenizer_config.json",
    }
    loaded = AutoTokenizer.from_pretrained(tmp_path)
    assert len(loaded) == len(words)
    assert loaded.get_vocab() == tokenizer.get_vocab()
    sample = [0, len(words) // 2, row_index(DONE, {})]
    text = loaded.decode(
        sample, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert loaded(text, add_special_tokens=False)["input_ids"] == sample


def test_unknown_word_fails_instead_of_mapping_to_a_row() -> None:
    tokenizer = build_standard_tokenizer()
    with pytest.raises(Exception, match="UNK"):
        tokenizer("definitely-not-a-doom-row", add_special_tokens=False)
