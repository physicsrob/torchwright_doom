"""R_CheckBBox visibility pruning — Plan E scope boundary (deferred stub).

The real ``BBoxPruner`` (``doom_sandbox/implementation/forward/bbox_pruning.py``,
493 lines) projects each BSP node's back-child bounding box to a screen X-range
(via the atan2 octant angle math) and scans it against ``SolidIntervals`` to
either descend the back child or prune the subtree with ``TRAVERSE_RETURN``.

Plan E defers it for two reasons: (1) its occlusion source is empty until the
wall pass exists (see ``solid_intervals.py``), so it is functionally inert here;
(2) it needs the deadband-sensitive projection-angle numerics that
``render_ops.py`` deliberately excludes. The traversal oracle caps comparison
before the first bbox token, so every ``after_*`` below is uncompared and emits
``NO_OP``. The real bbox owner lands with the projection/visibility phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .emit import emit_token_head as make_token_head
from .past import PastHandle, PastHandleScope
from .protocol_tokens import ProtocolTokenView
from .scene_index import SceneIndex
from .solid_intervals import SolidIntervals
from .vocab import NO_OP


@dataclass(frozen=True)
class BBoxPruner:
    """Deferred R_CheckBBox owner; all branches emit NO_OP this phase."""

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        input_vec: Node,
        inp: ProtocolTokenView,
        scene: SceneIndex,
        solids: SolidIntervals,
        input_angle_or_zero: PastHandle,
        between_side: Node,
        between_child: Node,
    ) -> "BBoxPruner":
        return cls()

    def after_between(self) -> Node:
        return make_token_head(NO_OP)

    def after_boxpos(self) -> Node:
        return make_token_head(NO_OP)

    def after_corner_x_mark_a(self) -> Node:
        return make_token_head(NO_OP)

    def after_corner_y_mark_a(self) -> Node:
        return make_token_head(NO_OP)

    def after_corner_x_mark_b(self) -> Node:
        return make_token_head(NO_OP)

    def after_corner_y_mark_b(self) -> Node:
        return make_token_head(NO_OP)

    def after_world_angle_mark_a(self) -> Node:
        return make_token_head(NO_OP)

    def after_theta_mark_a(self) -> Node:
        return make_token_head(NO_OP)

    def after_world_angle_mark_b(self) -> Node:
        return make_token_head(NO_OP)

    def after_theta_mark_b(self) -> Node:
        return make_token_head(NO_OP)

    def after_scan(self) -> Node:
        return make_token_head(NO_OP)
