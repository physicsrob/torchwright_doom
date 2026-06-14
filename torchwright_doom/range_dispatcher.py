"""Read-only mixed branch owner for SCREEN_RANGE transitions (Phase H).

In the per-token ``forward()`` flow this is the
``render_main.build_branch_outputs`` owner for the ``screen_range`` branch (the
host-visible clip/visplane merge token): that builder constructs a
``RangeDispatcher`` and wires its ``after_screen_range`` head to the
``"screen_range"`` dispatch branch.

Ported from ``doom_sandbox/implementation/forward/range_dispatcher.py``;
``type_switch`` exists real-side (``std``), so the two-arm wall/visplane merge
ports directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import annotated

from .render_ops import one_minus, or_
from .std import type_switch
from .wall_column_renderer import WallColumnRenderer

if TYPE_CHECKING:
    from .seg_projection import SegProjection


@dataclass(frozen=True)
class RangeDispatcher:
    """Owns the single SCREEN_RANGE branch that merges wall and visplane paths."""

    projection: "SegProjection"

    @annotated("dispatch")
    def after_screen_range(self, fallback_out):
        projection = self.projection
        phase_mask = or_(
            projection.core.inp.screen_range_after_clip_update,
            projection.core.inp.screen_range_after_plane_mark,
        )
        return type_switch(
            (
                projection.core.inp.screen_range_after_clip_update,
                WallColumnRenderer(projection).after_completed_clip_update(),
            ),
            (
                projection.core.inp.screen_range_after_plane_mark,
                WallColumnRenderer(projection).after_completed_plane_mark(),
            ),
            (one_minus(phase_mask), fallback_out),
        )
