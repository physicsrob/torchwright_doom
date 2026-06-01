"""Solid-interval occlusion state — Plan E scope boundary (deferred stub).

In the full renderer ``SolidIntervals`` accumulates the horizontal occlusion
window from the wall pass (``FIND_RUN`` / ``R_STORE_WALL_RANGE`` / ``EMIT_X2``)
and is scanned by ``R_CheckBBox`` to prune off-screen back subtrees. The
BSP-traversal phase (Plan E) emits no drawsegs, so the interval set is
permanently empty and the occlusion scan is a structural no-op — porting the
real interval machinery before the wall pass buys nothing. This stub is
constructed so ``forward()`` and ``BspTraversal.publish`` keep the real call
shape; the real ``solid_intervals.py`` lands with the wall/projection phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from .past import PastHandleScope
from .protocol_tokens import ProtocolTokenView
from .scene_index import SceneIndex


@dataclass(frozen=True)
class SolidIntervals:
    """Deferred occlusion-window state (empty in the traversal phase)."""

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        scene: SceneIndex,
    ) -> "SolidIntervals":
        return cls()
