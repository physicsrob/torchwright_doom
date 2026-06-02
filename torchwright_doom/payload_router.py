"""Route shared numeric payload carrier tokens (Plan F / F5; bbox arm restored
in Plan G).

``VALUE`` / ``ANGLE_VALUE`` are carrier types: their meaning comes from the
marker that precedes them, so this router is the handoff point that delegates a
carrier row to the protocol owner of its marker sequence.

Ported from ``doom_sandbox/implementation/forward/payload_router.py``. Phase F
ran a reduced router with no bbox owner (the ``is_bbox_angle`` arm and the VALUE
fallback both went to ``no_op_out``). Plan G restores the real coupling: the
``BBoxPruner`` (published by ``BspTraversal``) is the VALUE fallback for the bbox
corner rows and owns the ``is_bbox_angle`` ANGLE_VALUE arm.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .bbox_pruning import BBoxPruner
from .seg_projection import SegProjection
from .seg_scanner import SegScanner
from .std import type_switch
from .wall_range_builder import WallRangeBuilder


@dataclass(frozen=True)
class PayloadRouter:
    """Branch builders for generic numeric carrier rows."""

    projection: SegProjection
    bbox: BBoxPruner

    def after_value(self, no_op_out: Node) -> Node:
        """Advance VALUE carrier rows by their marker-defined meaning.

        Drawseg-scalar VALUE rows are owned by ``WallRangeBuilder``; everything
        else falls through to ``BBoxPruner.after_value`` (the bbox corner rows,
        else ``no_op_out``)."""
        return WallRangeBuilder(self.projection).after_drawseg_value(
            self.bbox.after_value(no_op_out)
        )

    def after_angle_value(self, no_op_out: Node) -> Node:
        """Advance ANGLE_VALUE carrier rows by their marker-defined meaning."""
        inp = self.projection.core.inp
        return type_switch(
            (inp.is_scene_angle_payload, no_op_out),
            (
                self.projection.seg.phase.is_projection_angle,
                SegScanner(self.projection).after_projection_angle_value(),
            ),
            (
                inp.angle_after_drawseg_u_phase,
                WallRangeBuilder(self.projection).after_drawseg_u_angle_value(),
            ),
            (
                self.bbox.phase.is_bbox_angle,
                self.bbox.after_bbox_angle_value(),
            ),
        )
