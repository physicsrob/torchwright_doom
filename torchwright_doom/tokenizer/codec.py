"""Raw-word codec: frozen vocabulary words <-> model row ids.

Contract code, not presentation: the standard tokenizer's interchange text is
space-joined canonical words, and these two pure functions are the exact
inverse pair over an ordered words list. No project imports, no graph imports
— deliberately cheap to import anywhere.

The portable prettifier (``portable/pretty_text.py``) keeps an embedded copy
of the same rules because a downloaded bundle cannot import project code;
``tests/tokenizer/test_codec.py`` pins the two implementations together.
"""

from __future__ import annotations

from typing import Sequence


def raw_text_from_rows(words: Sequence[str], rows: Sequence[int]) -> str:
    """Space-joined canonical words with **no** trailing newline — callers
    that write files append exactly one ``"\\n"`` at the artifact boundary.

    A valid row satisfies ``0 <= row < len(words)``; negative rows are
    rejected rather than inheriting Python list wraparound.
    """
    out = []
    for row in rows:
        try:
            index = int(row)
        except (TypeError, ValueError):
            raise ValueError("Doom row outside frozen vocabulary") from None
        if not 0 <= index < len(words):
            raise ValueError("Doom row outside frozen vocabulary")
        out.append(words[index])
    return " ".join(out)


def rows_from_raw_text(words: Sequence[str], text: str) -> list[int]:
    """Whitespace-split word -> row lookup over the ordered words list."""
    word_to_row = {word: row for row, word in enumerate(words)}
    rows = []
    for word in text.split():
        try:
            rows.append(word_to_row[word])
        except KeyError:
            raise ValueError(f"unknown canonical Doom word: {word!r}") from None
    return rows
