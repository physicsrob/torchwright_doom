"""Route shared numeric payload carrier tokens.

``VALUE`` / ``ANGLE_VALUE`` are carrier types: their meaning comes from the
marker that precedes them, so this router is the handoff point that delegates a
carrier row to the protocol owner of its marker sequence.

In the per-token ``forward()`` flow this is one of the three protocol owners
built in ``render_main.publish_runtime_protocols`` (held on
``RuntimeProtocols.payload_router``): the ``value`` and ``angle`` dispatch
branches route their carriers through ``after_value`` / ``after_angle_value``
here.

The ``BBoxPruner`` (published by ``BspTraversal``) is the VALUE fallback for the
bbox corner rows and owns the ``is_bbox_angle`` ANGLE_VALUE arm.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node
from torchwright.graph import annotated

from ..traversal.bbox_pruning import BBoxPruner
from .seg_projection import SegProjection
from .seg_scanner import SegScanner
from ..std import type_switch
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

    @annotated("dispatch")
    def after_angle_value(self, no_op_out: Node) -> Node:
        """Advance ANGLE_VALUE carrier rows by their marker-defined meaning.

        Annotated ``dispatch`` (it is the carrier router). The routed arms it
        builds are decorated with their own owner codes (SegScanner -> ``proj``,
        WallRangeBuilder/VisplaneMarker -> ``stor``/``pmrk``, BBoxPruner ->
        ``bsp``); those nest as ``dispatch/<code>`` so the owner code is present
        in the path while the ``type_switch`` reduction glue itself reads
        ``dispatch``. (Order-preserving: only a context wraps the unchanged body.)
        """
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
