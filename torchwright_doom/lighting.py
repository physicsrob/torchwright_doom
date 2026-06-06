"""Runtime helpers for DOOM wall COLORMAP application.

Real-side port of ``doom_sandbox/implementation/forward/lighting.py``: the
in-graph ``256x32`` ``COLORMAP[row][palette_index]`` indirection (r_data.c
colormaps, r_main.c, r_draw.c). The COLORMAP-row *selection* math
(``doom_wall_colormap_row`` / scalelight / ``NUMCOLORMAPS``) is the already-ported
:mod:`.doom_lighting`; this is the in-graph *application* of a chosen row.

``Vec`` -> ``Node`` and the sandbox ``...api`` imports become ``.std``. The 32
per-row PWLs are built from ``pwl_def`` closures, so this tuple holds no graph
node at import (``global_node_id`` stays ``0``).
"""

from __future__ import annotations

from torchwright.graph import Node

from .std import concat, pick_by_index, pwl_def
from .doom_lighting import NUMCOLORMAPS
from .asset_banks import COLORMAP_ROWS


def _palette_index(value: float) -> int:
    return max(0, min(255, int(round(float(value)))))


def _colormap_row_pwl(row: int):
    values = COLORMAP_ROWS[row]
    return pwl_def(
        lambda palette_index, values=values: float(
            values[_palette_index(palette_index)]
        ),
        breakpoints=256,
        input_range=(0.0, 255.0),
        name=f"colormap_row_{row}",
    )


_COLORMAP_ROW_PWLS = tuple(_colormap_row_pwl(row) for row in range(NUMCOLORMAPS))


# DOOM: COLORMAP[row][palette_index] lookup (r_draw.c: dc_colormap[...] per column).
def apply_colormap_row(raw_palette_index: Node, colormap_row: Node) -> Node:
    """Return the lit palette index for a texel palette index and COLORMAP row.

    ``raw_palette_index`` is an unlit palette index in ``[0, 255]`` (the texel
    fetch output); ``colormap_row`` is the per-texel light level in
    ``[0, NUMCOLORMAPS)``. Builds the 32 candidate lit values (one per COLORMAP
    row, each a 256-breakpoint PWL over the raw index) and picks the chosen
    row's value — a *runtime* table (the candidates depend on the raw index), so
    ``pick_by_index`` lowers to ``dynamic_extract``.
    """
    candidates_by_row = concat(
        *(row_pwl(raw_palette_index) for row_pwl in _COLORMAP_ROW_PWLS)
    )
    return pick_by_index(colormap_row, candidates_by_row, NUMCOLORMAPS)
