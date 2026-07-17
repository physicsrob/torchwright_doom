"""Codec parity gate.

``tokenizer/codec.py`` (the project-side contract) and the portable
prettifier kernel's embedded copy must agree on encode/decode over the frozen
vocabulary — including unknown words, negative rows, the upper-bound row,
empty input, and whitespace/newline behavior. The codec's encode must also be
byte-identical to the inline ``" ".join(words[row] ...)`` it replaced, so the
canonical prompt text for the committed configs cannot change.
"""

from __future__ import annotations

import pytest

from torchwright_doom.model.asset_config import DEFAULT_ASSET_CONFIG
from torchwright_doom.portable.pretty_text import DoomTextFormatter
from torchwright_doom.tokenizer.codec import raw_text_from_rows, rows_from_raw_text
from torchwright_doom.tokenizer.freeze import build_tables, build_vocab_blob

from ..prefill_fixture import TINY_BSP_SCENE, row_index


@pytest.fixture(scope="module")
def frozen():
    vocab_blob = build_vocab_blob(
        DEFAULT_ASSET_CONFIG.wall_names, DEFAULT_ASSET_CONFIG.flat_names
    )
    kernel = DoomTextFormatter(vocab_blob, build_tables())
    words = list(vocab_blob["words"])
    # A real token stream (the shared tiny BSP scene), plus the vocabulary
    # boundary rows.
    rows = [row_index(t, dict(v)) for t, v in TINY_BSP_SCENE]
    rows += [0, len(words) - 1]
    return words, kernel, rows


def test_encode_parity_and_inline_join_identity(frozen) -> None:
    words, kernel, rows = frozen
    text = raw_text_from_rows(words, rows)
    assert text == kernel.raw_text_from_rows(rows)
    # Byte-identity with the inline join the codec replaced (run validation
    # and prompt construction previously used this exact expression).
    assert text == " ".join(words[row] for row in rows)
    assert not text.endswith("\n")


def test_decode_parity_and_round_trip(frozen) -> None:
    words, kernel, rows = frozen
    text = raw_text_from_rows(words, rows)
    assert rows_from_raw_text(words, text) == rows
    assert kernel.rows_from_raw_text(text) == rows
    # Whitespace and newline tolerance is identical: the artifact boundary
    # appends one newline, and any whitespace split decodes the same.
    for variant in (text + "\n", text.replace(" ", "\n"), "  " + text + "  \n\n"):
        assert rows_from_raw_text(words, variant) == rows
        assert kernel.rows_from_raw_text(variant) == rows


def test_empty_input_parity(frozen) -> None:
    words, kernel, _ = frozen
    assert raw_text_from_rows(words, []) == ""
    assert kernel.raw_text_from_rows([]) == ""
    assert rows_from_raw_text(words, "") == []
    assert kernel.rows_from_raw_text("") == []
    assert rows_from_raw_text(words, " \n ") == []
    assert kernel.rows_from_raw_text(" \n ") == []


def test_unknown_word_parity(frozen) -> None:
    words, kernel, _ = frozen
    with pytest.raises(ValueError, match="unknown canonical Doom word"):
        rows_from_raw_text(words, "definitely-not-a-doom-word")
    with pytest.raises(ValueError, match="unknown canonical Doom word"):
        kernel.rows_from_raw_text("definitely-not-a-doom-word")


def test_out_of_range_row_parity(frozen) -> None:
    """Negative rows are rejected, never wrapped (Python list semantics would
    silently alias row -1 to the last word); the upper bound is exclusive."""
    words, kernel, _ = frozen
    for bad in (-1, len(words)):
        with pytest.raises(ValueError, match="outside frozen vocabulary"):
            raw_text_from_rows(words, [bad])
        with pytest.raises(ValueError, match="outside frozen vocabulary"):
            kernel.raw_text_from_rows([bad])


def test_codec_matches_the_stock_tokenizer_interchange(frozen) -> None:
    """The codec's encode is exactly the stock WordLevel tokenizer's decode
    (and its decode the tokenizer's encode) — the contract they both speak."""
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from torchwright_doom.tokenizer.standard import build_standard_tokenizer

    words, _, rows = frozen
    tokenizer = build_standard_tokenizer(words=words)
    text = raw_text_from_rows(words, rows)
    assert (
        tokenizer.decode(
            rows, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        == text
    )
    assert tokenizer(text, add_special_tokens=False)["input_ids"] == rows
