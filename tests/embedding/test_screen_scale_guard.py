"""Screen-scale guard: no bare default-dimension literals.

The production config renders at 320x200 (``configs/e1m1.yaml``
``model.scale: 1`` via ``apply_screen_env``); 60x50 remains the
bare-import test default in ``constants.py``. Screen size stays a
configuration choice only if every screen-derived
width/range references ``SCREEN_WIDTH`` / ``SCREEN_HEIGHT`` symbolically
and never a bare ``60`` / ``50`` literal. This test fails if a bare
``60`` or ``50`` NUMBER literal appears in ``vocab.py`` or
``value_ranges.py`` (``constants.py`` is where the dimensions are
legitimately *defined*).
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import torchwright_doom.model.value_ranges as value_ranges
import torchwright_doom.model.vocab as vocab


def _bare_screen_literals(path: str) -> list[tuple[int, str]]:
    src = Path(path).read_text()
    hits: list[tuple[int, str]] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NUMBER and tok.string in ("60", "50"):
            hits.append((tok.start[0], tok.string))
    return hits


def test_no_bare_screen_dim_literals() -> None:
    for module in (vocab, value_ranges):
        assert module.__file__ is not None
        hits = _bare_screen_literals(module.__file__)
        assert not hits, (
            f"{module.__name__} has bare 60/50 literals at {hits}; use "
            f"SCREEN_WIDTH/SCREEN_HEIGHT so screen size remains configurable"
        )
