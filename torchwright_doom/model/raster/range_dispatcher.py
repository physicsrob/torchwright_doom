"""Read-only mixed branch owner for SCREEN_RANGE transitions (Phase H —
port-milestone tag; GLOSSARY: Phase letters).

This module runs once at compile time: it builds part of the computation
graph that torchwright lowers into the transformer's weights. Nothing here
executes during inference — at render time, only the compiled transformer
runs. Coined terms: see GLOSSARY.md.

In the per-token ``forward()`` flow this is the
``render_main.build_branch_outputs`` owner for the ``screen_range`` branch (the
clip/visplane merge token — the merge is computed in-graph; "host-visible"
means only that the token appears in the output stream): that builder constructs a
``RangeDispatcher`` and wires its ``after_screen_range`` head to the
``"screen_range"`` dispatch branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import annotated

from ..render_ops import one_minus, or_
from ..std import type_switch
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
