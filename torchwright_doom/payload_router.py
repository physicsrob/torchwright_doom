"""Route shared numeric payload carrier tokens (Plan F / F5 — reduced).

``VALUE`` / ``ANGLE_VALUE`` are carrier types: their meaning comes from the
marker that precedes them, so this router is the handoff point that delegates a
carrier row to the protocol owner of its marker sequence.

Ported from ``doom_sandbox/implementation/forward/payload_router.py``, **reduced
for Phase F**: the ``BBoxPruner`` coupling is removed. The sandbox router takes
a ``bbox`` field, threads ``bbox.after_value(no_op_out)`` as the VALUE fallback,
and routes ``is_bbox_angle -> bbox.after_bbox_angle_value()``. Phase F has no
bbox owner (R_CheckBBox is Phase G), so the VALUE fallback is ``no_op_out``
directly and the bbox ANGLE arm routes to ``no_op_out``. Phase G restores the
real pruner.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .seg_projection import SegProjection
from .seg_scanner import SegScanner
from .std import type_switch
from .wall_range_builder import WallRangeBuilder


@dataclass(frozen=True)
class PayloadRouter:
    """Branch builders for generic numeric carrier rows (reduced: no bbox)."""

    projection: SegProjection

    def after_value(self, no_op_out: Node) -> Node:
        """Advance VALUE carrier rows by their marker-defined meaning.

        The sandbox fallback ``bbox.after_value(no_op_out)`` becomes ``no_op_out``
        (no bbox owner in F); bbox VALUE rows are teacher-forced-and-skipped, so
        they fall through ``after_drawseg_value`` to this fallback.
        """
        return WallRangeBuilder(self.projection).after_drawseg_value(no_op_out)

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
            # DEFERRED (Phase G): is_bbox_angle -> BBoxPruner.after_bbox_angle_value().
            (inp.is_bbox_angle_payload, no_op_out),
        )
