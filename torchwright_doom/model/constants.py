"""Shared scale constants for the vocab + value-range layers.

This module runs once at compile time: it builds part of the computation
graph that torchwright lowers into the transformer's weights. Nothing here
executes during inference — at render time, only the compiled transformer
runs. Coined terms: see GLOSSARY.md. Screen size is fixed at compile time —
each config compiles its own weights.

This module sits *below* :mod:`vocab` and :mod:`value_ranges` in the import
graph: both import from here, neither is imported by here. That breaks the
cycle that would otherwise form once ``vocab.py`` imports
``value_derived_columns`` from ``value_ranges.py`` while ``value_ranges.py``
needs ``SCREEN_WIDTH`` for its resolution-scaled ranges (R5 wall scale, R7
drawseg width). Keep this module dependency-free.

Porting scale: the production config renders at 320×200
(``configs/e1m1.yaml``, ``model.scale: 1``); the 80×50 consumer checkpoint is
``configs/e1m1_lowres.yaml`` (``model.scale: 4``). ``apply_screen_env``
(root ``config.py``) exports the screen dims via the env vars read
below before graph modules import. The 60×50 defaults here are the
bare-import fallback — the reference renderer (pydoom) fixture scale,
which is what tests that import the graph without a config see (it gives
exact token-by-token prompt parity with the pydoom fixtures). Every
screen-derived width must reference these names, never a literal 60/50
(guarded by a test), so the scale stays a config swap.
"""

from __future__ import annotations

import os

_DEFAULT_SCREEN_WIDTH = 60
_DEFAULT_SCREEN_HEIGHT = 50
_SUPPORTED_RENDER_SCALES = {1, 2, 4}


def _screen_dim_from_env(name: str, default: int, *, minimum: int = 2) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _screen_dims_from_scale() -> tuple[int, int] | None:
    raw = os.environ.get("TORCHWRIGHT_DOOM_RENDER_SCALE")
    if raw is None or raw == "":
        return None
    try:
        scale = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"TORCHWRIGHT_DOOM_RENDER_SCALE must be an integer, got {raw!r}"
        ) from exc
    if scale not in _SUPPORTED_RENDER_SCALES:
        allowed = ", ".join(str(v) for v in sorted(_SUPPORTED_RENDER_SCALES))
        raise ValueError(
            f"TORCHWRIGHT_DOOM_RENDER_SCALE must be one of {{{allowed}}}; "
            f"got {scale}"
        )
    return 320 // scale, 200 // scale


_scaled_dims = _screen_dims_from_scale()
if _scaled_dims is None:
    _default_width, _default_height = _DEFAULT_SCREEN_WIDTH, _DEFAULT_SCREEN_HEIGHT
else:
    _default_width, _default_height = _scaled_dims

SCREEN_WIDTH = _screen_dim_from_env(
    "TORCHWRIGHT_DOOM_SCREEN_WIDTH", _default_width, minimum=2
)
SCREEN_HEIGHT = _screen_dim_from_env(
    "TORCHWRIGHT_DOOM_SCREEN_HEIGHT", _default_height, minimum=2
)

# Status-bar / viewport split. DOOM's default screen (screenblocks 10) renders
# the 3D view into the top 168 of 200 rows and fills the bottom 32 with the
# status bar; the projection horizon sits at the view centre (84), not the
# screen centre. We gate that behind TORCHWRIGHT_DOOM_HUD (set from the YAML
# model.hud field by apply_screen_env). OFF (default) => BAR_HEIGHT 0 =>
# VIEW_HEIGHT == SCREEN_HEIGHT, i.e. today's full-screen view: every VIEW_HEIGHT
# site below collapses to its old SCREEN_HEIGHT value, so a HUD-off build is
# bit-identical to before this split. The bar's own height scales with the
# screen (DOOM ST_HEIGHT 32 of a 200-row screen): 32 at scale 1, 16 at scale 2.
_FULL_SCREEN_HEIGHT = 200
_FULL_BAR_HEIGHT = 32
_HUD = os.environ.get("TORCHWRIGHT_DOOM_HUD", "0")
if _HUD not in ("0", "1"):
    raise ValueError(f"TORCHWRIGHT_DOOM_HUD must be '0' or '1', got {_HUD!r}")
HUD_ENABLED = _HUD == "1"
BAR_HEIGHT = (
    (_FULL_BAR_HEIGHT * SCREEN_HEIGHT) // _FULL_SCREEN_HEIGHT if HUD_ENABLED else 0
)
# The 3D view occupies the top VIEW_HEIGHT rows; the bar rows
# [HUD_TOP, SCREEN_HEIGHT) are painted by the status-bar phase when HUD is
# enabled (statusbar_renderer). The host buffer and every
# vocab encoding range stay at the full SCREEN_HEIGHT (the bar rows are still
# addressable, e.g. by setCursorY for bar pixels); only the 3D clip extent and
# the projection horizon use VIEW_HEIGHT.
VIEW_HEIGHT = SCREEN_HEIGHT - BAR_HEIGHT
HUD_TOP = VIEW_HEIGHT
# Projection horizon — the 3D view's vertical centre.
CENTER_Y = VIEW_HEIGHT / 2.0

# Upper bound (with headroom) on the number of status-bar draw-list items — the
# V_DrawPatch calls the HUD spine walks. The hardcoded E1M1 bar uses ~32; this
# fixed bound sizes the HUD_ITEM index slot without the vocab importing the WAD.
MAX_HUD_ITEMS = 64

# Pixel paint width (DOOM detail mode): low-detail paints 2 screen columns per
# rendered column, high-detail 1. Read directly from the env (like the screen
# dims) so it is fixed before the graph modules import; defaults to high so a
# bare import / unset config renders at today's full horizontal detail.
_DETAIL = os.environ.get("TORCHWRIGHT_DOOM_DETAIL", "high")
if _DETAIL not in ("low", "high"):
    raise ValueError(
        f"TORCHWRIGHT_DOOM_DETAIL must be 'low' or 'high', got {_DETAIL!r}"
    )
PIXEL_WIDTH = 2 if _DETAIL == "low" else 1
# COLUMN_COUNT sizes every per-column structure (rays, the column one-hots, the
# column-index radix, ...). SCREEN_WIDTH stays the screen-coordinate range and
# host buffer width; the projection focal also stays SCREEN_WIDTH-based. They
# are equal except in low-detail, where the view still renders
# COLUMN_COUNT = SCREEN_WIDTH // 2 columns, each painted 2 px wide.
COLUMN_COUNT = SCREEN_WIDTH // PIXEL_WIDTH
