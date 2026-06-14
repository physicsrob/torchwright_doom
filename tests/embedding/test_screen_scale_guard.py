"""Anti-toy-scale guard: no bare screen-dimension literals.

The production config renders at 160x100 (``configs/e1m1.yaml``
``model.scale: 2`` via ``apply_screen_env``); the 60x50 fixture
scale remains the bare-import default in ``constants.py``. The scale
stays a config swap (not a re-port) only if every screen-derived
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

import torchwright_doom.value_ranges as value_ranges
import torchwright_doom.vocab as vocab


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
            f"SCREEN_WIDTH/SCREEN_HEIGHT so the 160x100 retarget stays a "
            f"constant swap"
        )
