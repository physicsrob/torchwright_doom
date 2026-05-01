"""Legacy ray-cast helpers preserved as a transformer-rollout oracle.

The legacy renderer (`render.py`) was deleted because its horizontal
projection was linear-angle (fisheye-distorted on flat screens) — see
``__init__.py``.  The compiled DOOM transformer was, however, *tuned
against* that legacy math, so ``tests/doom/_rollout/reference.py``
still uses these helpers as the oracle for in-progress transformer
work.  When the transformer is reworked to match the new
DOOM-faithful renderer, this module can go away.

These helpers are not exported from the package; import directly:

    from torchwright_doom.reference_renderer._legacy_oracle import (
        project_wall, _ray_angle_for_column,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torchwright_doom.reference_renderer.geometry import intersect_ray_segment
from torchwright_doom.reference_renderer.types import RenderConfig, Segment


def _ray_angle_for_column(col: int, player_angle: int, config: RenderConfig) -> int:
    """Linear-angle column→ray mapping (legacy)."""
    col_offset = col - config.screen_width // 2
    return (player_angle + col_offset * config.fov_columns // config.screen_width) % 256


@dataclass
class WallProjection:
    """Result of projecting one wall under the legacy linear-angle pipeline."""

    seg: Segment
    vis_lo: int
    vis_hi: int


def project_wall(
    player_x: float,
    player_y: float,
    player_angle: int,
    seg: Segment,
    config: RenderConfig,
) -> Optional[WallProjection]:
    """Linear-angle per-column ray cast against one seg."""
    lo = None
    hi = None
    for col in range(config.screen_width):
        ray_angle = _ray_angle_for_column(col, player_angle, config)
        ray_cos = config.trig_table[ray_angle, 0]
        ray_sin = config.trig_table[ray_angle, 1]
        if intersect_ray_segment(player_x, player_y, ray_cos, ray_sin, seg) is not None:
            if lo is None:
                lo = col
            hi = col
    if lo is None or hi is None:
        return None
    return WallProjection(seg=seg, vis_lo=lo, vis_hi=hi)
