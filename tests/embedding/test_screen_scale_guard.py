"""Plan A: anti-toy-scale guard for the deferred 160x100 retarget.

The port runs at the sandbox fixture scale 60x50; the retarget to the
real 160x100 is a deferred one-line change in ``constants.py``. That
stays a constant swap only if every screen-derived width/range
references ``SCREEN_WIDTH`` / ``SCREEN_HEIGHT`` symbolically and never a
bare ``60`` / ``50`` literal. This test fails if a bare ``60`` or ``50``
NUMBER literal appears in ``vocab.py`` or ``value_ranges.py``
(``constants.py`` is where the dimensions are legitimately *defined*).
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
        hits = _bare_screen_literals(module.__file__)
        assert not hits, (
            f"{module.__name__} has bare 60/50 literals at {hits}; use "
            f"SCREEN_WIDTH/SCREEN_HEIGHT so the 160x100 retarget stays a "
            f"constant swap"
        )
