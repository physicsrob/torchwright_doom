"""pydoom — the plain-Python DOOM the compiled graph descends from.

Vendored from ``doom_sandbox`` (its ``reference`` pixel renderer +
``reference_drafter`` AR-token state machine), retargeted onto
``torchwright_doom``'s own token / asset / lighting / geometry modules so it
emits native ``TokenType``s with no sandbox dependency.

- :mod:`renderer` — the pure-Python pixel-pass renderer (``--compare`` / PNG and
  the whole-frame routing test read it).
- :mod:`drafter` — the speculative-decoding draft model (``make run`` proposes
  tokens with it; the compiled graph verifies them).

Both descend from the same hand-port of DOOM's C source as the compiled graph;
see ``plan_remove_doom_sandbox.md`` for the glossary and the lineage caveat.
"""

from __future__ import annotations

from ._scene import GameState, Pixel, Scene, TextureImage
from .drafter import ARDrafter, expected_ar_tokens
from .renderer import expected_pixel_color_options, expected_pixel_pass

__all__ = [
    "ARDrafter",
    "GameState",
    "Pixel",
    "Scene",
    "TextureImage",
    "expected_ar_tokens",
    "expected_pixel_color_options",
    "expected_pixel_pass",
]
