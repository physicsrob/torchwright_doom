"""Surface invariants that hold at any screen config (run in-process).

* **Bijection coverage** — the id<->token round trip ``token_to_row(
  row_to_token(r)) == r`` (exhaustive for small types, sampled for large).
* **Grammar coverage** — every ``TokenType`` in ``VOCAB_TYPES`` is reachable
  by the surface: a representative token of each type (markers paired with
  their carrier) round-trips render->parse back to the same rows. A new type
  with no handling fails here, not silently.
* **Marker-table sanity** — the shared ``MARKER_RANGE`` / ``ANGLE_MARKERS``
  tables reference only real vocab types and don't overlap.
"""

from __future__ import annotations

from typing import Any

from torchwright_doom.asset_config import DEFAULT_ASSET_CONFIG
from torchwright_doom.embedding import TOKEN_VOCAB
from torchwright_doom.inference.tokens_bridge import row_to_token, token_to_row
from torchwright_doom.marker_ranges import ANGLE_MARKERS, MARKER_RANGE
from torchwright_doom.tokens import Token
from torchwright_doom.tokenizer import surface
from torchwright_doom.vocab import ANGLE_VALUE, VALUE, VOCAB_TYPES

_NAMES: dict[str, Any] = dict(
    wall_names=DEFAULT_ASSET_CONFIG.wall_names,
    flat_names=DEFAULT_ASSET_CONFIG.flat_names,
)
_CARRIERS = (VALUE, ANGLE_VALUE)


def _sample_rows(start: int, end: int, cap: int = 5000, n: int = 64) -> list[int]:
    size = end - start
    if size <= cap:
        return list(range(start, end))
    step = max(1, size // n)
    rows = list(range(start, end, step))
    rows.append(end - 1)  # always include the last row
    return sorted(set(rows))


def test_bijection_round_trip() -> None:
    for ttype in VOCAB_TYPES:
        start, end = TOKEN_VOCAB.type_to_row_range[ttype]
        for row in _sample_rows(start, end):
            assert token_to_row(row_to_token(row)) == row, (ttype.name, row)


def _representative_stream(ttype):
    """A minimal token stream exercising ``ttype`` — a mid-range instance, with
    its carrier appended when it is a marker."""
    start, end = TOKEN_VOCAB.type_to_row_range[ttype]
    marker_tok = row_to_token((start + end) // 2)
    stream = [marker_tok]
    covered = {ttype}
    if ttype in MARKER_RANGE:
        stream.append(Token(VALUE, {"v": 0.0}))
        covered.add(VALUE)
    elif ttype in ANGLE_MARKERS:
        stream.append(Token(ANGLE_VALUE, {"angle": 0}))
        covered.add(ANGLE_VALUE)
    return stream, covered


def test_grammar_coverage() -> None:
    covered = set()
    for ttype in VOCAB_TYPES:
        if ttype in _CARRIERS:
            continue  # reached only via a marker, below
        stream, hit = _representative_stream(ttype)
        rows = [token_to_row(t) for t in stream]
        text = surface.render(stream, **_NAMES)
        back = surface.parse(text, **_NAMES)
        rows2 = [token_to_row(Token(t, dict(v))) for t, v in back]
        assert rows2 == rows, f"{ttype.name}: render/parse changed the row stream"
        covered |= hit

    missing = [t.name for t in VOCAB_TYPES if t not in covered]
    assert not missing, f"types unreachable by the surface: {missing}"


def test_marker_tables_reference_real_types() -> None:
    vocab = set(VOCAB_TYPES)
    assert set(MARKER_RANGE) <= vocab
    assert set(ANGLE_MARKERS) <= vocab
    # A marker carries exactly one kind of carrier.
    assert not (set(MARKER_RANGE) & set(ANGLE_MARKERS))
    # The carriers are never themselves markers.
    assert not ({VALUE, ANGLE_VALUE} & (set(MARKER_RANGE) | set(ANGLE_MARKERS)))
