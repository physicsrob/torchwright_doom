"""Shared scale constants for the vocab + value-range layers.

This module sits *below* :mod:`vocab` and :mod:`value_ranges` in the import
graph: both import from here, neither is imported by here. That breaks the
cycle that would otherwise form once ``vocab.py`` imports
``value_derived_columns`` from ``value_ranges.py`` while ``value_ranges.py``
needs ``SCREEN_WIDTH`` for its resolution-scaled ranges (R5 wall scale, R7
drawseg width). Keep this module dependency-free.

Porting scale: the spec09 port runs the whole way at the sandbox fixture
scale of 60×50, which gives exact token-by-token prompt parity with the
sandbox fixtures. The retarget to the real 160×100 is a deferred,
project-level one-line change *here* once the port is in good shape — every
screen-derived width must reference these names, never a literal 60/50
(guarded by a test), so the bump stays a constant swap.
"""

from __future__ import annotations

SCREEN_WIDTH = 60
SCREEN_HEIGHT = 50
