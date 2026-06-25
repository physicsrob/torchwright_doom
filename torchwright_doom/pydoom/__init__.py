"""pydoom — the plain-Python DOOM the compiled graph descends from.

Vendored from the former ``doom_sandbox`` submodule (its ``reference`` pixel
renderer + ``reference_drafter`` AR-token state machine), retargeted onto
``torchwright_doom``'s own token / asset / lighting / geometry modules so it
emits native ``TokenType``s with no dependency on the former submodule.

- :mod:`renderer` — the pure-Python pixel-pass renderer (``--compare`` / PNG and
  the whole-frame routing test read it).
- :mod:`drafter` — the reference AR-token state machine; its public entry
  point :func:`~torchwright_doom.pydoom.drafter.expected_ar_tokens` produces
  the canonical render-token sequence the graph gates compare against.

Both descend from the same hand-port of DOOM's C source as the compiled graph.
"""

from __future__ import annotations

from ._scene import GameState, Pixel, Scene, TextureImage
from .drafter import expected_ar_tokens
from .renderer import expected_pixel_color_options, expected_pixel_pass

__all__ = [
    "GameState",
    "Pixel",
    "Scene",
    "TextureImage",
    "expected_ar_tokens",
    "expected_pixel_color_options",
    "expected_pixel_pass",
]
