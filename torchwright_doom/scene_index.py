"""Assemble the render-time scene index for the prescribed prefill stream.

Each forward call sees one token position. The prescribed prefill stream uses
structural headers (`NODE`, `SS`, `SEG`) followed by marker/value tokens. This
module wires together the helpers that interpret the current token, publish
header contexts, and expose query objects used by render code.

Read order:

1. `scene_tokens`: token interpretation.
2. `scene_headers`: current NODE/SS/SEG header contexts.
3. `scene_facts`: queryable player/node/subsector/seg facts.
4. `attention_handles`: generic GraphPast lookup plumbing.

This is the static-scene side. From here the reading path continues onto the
dynamic/dispatch side: `protocol_tokens` / `protocol_registry` ->
`render_main.forward` (see CLAUDE.md "Rough layout" for the full reading path).

Ported from ``doom_sandbox/implementation/forward/scene_index.py``. Changes
from the sandbox source: the import block (``Vec`` -> ``Node``;
``Past`` -> ``GraphPast``; ``one_hot`` / ``N_PLANES_MAX`` from the real-side
shim / vocab) and the ``SceneIndex.build`` ``pos`` annotation. The classmethod
body, the four ``_publish_*_context`` recency helpers, and the published-field
shape (``view``/``nodes``/``subsectors``/``segs``/``assets``/``planes``) are a
line-for-line port. ``AssetIndex()`` is constructed with zero ``past.publish``;
its lookup internals are the lookup track's deliverable (see ``assets.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node, PosEncoding
from torchwright.graph import annotated

from .assets import AssetIndex
from .past import GraphPast
from .scene_facts import (
    NodeIndex,
    PlaneIndex,
    PlayerView,
    SegIndex,
    SubsectorIndex,
)
from .scene_headers import HeaderContext
from .scene_tokens import SceneTokenView
from .std import one_hot
from .vocab import N_PLANES_MAX


@dataclass(frozen=True)
class SceneIndex:
    """Queryable scene facts recovered from the prefill stream.

    Render code should treat this as the scene database: `view` exposes player
    pose, `nodes` exposes BSP node facts, `subsectors` exposes first-seg
    membership, and `segs` exposes seg geometry/ownership.
    """

    view: PlayerView
    nodes: NodeIndex
    subsectors: SubsectorIndex
    segs: SegIndex
    assets: AssetIndex
    planes: PlaneIndex

    @classmethod
    @annotated("scene")
    def build(
        cls,
        input_vec: Node,
        past: GraphPast,
        pos: PosEncoding,
        assets: AssetIndex | None = None,
    ) -> "SceneIndex":
        """Publish this position's scene-index channels and return queries.

        Construction is eager: every field group publishes its backing channels
        before render code consumes the returned `SceneIndex`. ``pos`` is
        accepted for ``forward()``-signature parity but is unused in the body
        (the previous-token read goes through ``past``, which already holds the
        positional encoding).
        """
        # VALUE/ANGLE_VALUE tokens are interpreted by looking at the marker
        # token immediately before them.
        prev_input_type = past.attend_to_offset(past.input_type(), delta_pos=-1)
        token = SceneTokenView(input_vec, prev_input_type)
        # Header tokens establish the current context for following facts.
        node_context = _publish_node_context(past, token)
        subsector_context = _publish_subsector_context(past, token)
        seg_context = _publish_seg_context(past, token)
        plane_context = _publish_plane_context(past, token)
        return cls(
            view=PlayerView.publish(past, token),
            nodes=NodeIndex.publish(past, token, node_context),
            subsectors=SubsectorIndex.publish(past, token, subsector_context),
            segs=SegIndex.publish(
                past,
                token,
                subsector_context,
                seg_context,
            ),
            assets=assets or AssetIndex(),
            planes=PlaneIndex.publish(past, token, plane_context),
        )


def _publish_node_context(past: GraphPast, token: SceneTokenView) -> HeaderContext:
    """Publish the current NODE header context."""
    return HeaderContext.publish(
        past,
        active_name="node_header_active",
        id_name="node_header_j",
        key_name="node_header_key",
        is_header=token.is_node,
        id_value=token.node_j,
        key_value=token.id_lifted_key,
    )


def _publish_subsector_context(past: GraphPast, token: SceneTokenView) -> HeaderContext:
    """Publish the current SS/subsector header context."""
    return HeaderContext.publish(
        past,
        active_name="ss_header_active",
        id_name="ss_header_s",
        key_name="ss_header_key",
        is_header=token.is_subsector,
        id_value=token.subsector_s,
        key_value=token.id_lifted_key,
    )


def _publish_seg_context(past: GraphPast, token: SceneTokenView) -> HeaderContext:
    """Publish the current SEG header context."""
    return HeaderContext.publish(
        past,
        active_name="seg_header_active",
        id_name="seg_header_i",
        key_name="seg_header_key",
        is_header=token.is_seg,
        id_value=token.seg_i,
        key_value=token.id_lifted_key,
    )


def _publish_plane_context(past: GraphPast, token: SceneTokenView) -> HeaderContext:
    """Publish the current PLANE_DEF header context."""
    return HeaderContext.publish(
        past,
        active_name="plane_header_active",
        id_name="plane_header_p",
        key_name="plane_header_key",
        is_header=token.is_plane_def,
        id_value=token.plane_def_p,
        key_value=one_hot(token.plane_def_p, N_PLANES_MAX),
    )
