"""BSP traversal protocol for the wall-pass forward pass (Plan E / E3).

This module owns the autoregressive token transitions that correspond to
DOOM's `R_RenderBSPNode` tree walk:

    BEGIN -> SET_CURSOR_DIRECTION_Y -> THINK_SIDE(0)
    THINK_SIDE(node) -> SIDE_RECORD(node, side(node))
    SIDE_RECORD(node) -> THINK_SIDE(node + 1) or TRAVERSE_ENTER(root)
    TRAVERSE_ENTER(node, depth) -> first/front child
    TRAVERSE_BETWEEN(node, depth) -> second/back child
    TRAVERSE_RETURN(entity, depth) -> BETWEEN, another RETURN, or DONE

The lower-level dynamic stack record is still in `traversal_edges.py`; this
file decides which edges are taken and what each traversal token emits next.

Ported from ``doom_sandbox/implementation/forward/bsp_traversal.py``. Changes
from the sandbox source: the import block (``Vec`` -> ``Node``; ``Past`` ->
``GraphPast``; std helpers / ``make_token`` -> ``make_token_head`` / the side ops
from the real-side shim); ``ZERO`` is created inline as ``constant(0.0)`` (no
import-time nodes). Plan G restores the R_CheckBBox coupling: ``publish`` builds
the :class:`~.bbox_pruning.BBoxPruner` over the now-populated occlusion state and
``after_between`` / the ``after_bbox_*`` delegators route the bbox sub-protocol's
branches to it. The ``SideTable``, ``_think_side_compute`` cross product, the
ENTER/SIDE/RETURN emitters, and the child dispatch are a line-for-line port.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .attention_handles import LiftedKeyValueHandle
from .bbox_pruning import BBoxPruner
from .past import PastHandle, PastHandleScope
from .protocol_tokens import ProtocolTokenView
from .render_ops import IS_SUBSECTOR, SIDE_POSITIVE, add_const, mul_side, sub
from .scene_index import SceneIndex
from .solid_intervals import SolidIntervals
from .std import (
    ScalarEmit,
    bool_to_01,
    constant,
    indicator_to_bool,
    make_token_head,
    select,
)
from .traversal_edges import TraversalEdges
from .vocab import (
    N_NODES_MAX,
    SIDE_RECORD,
    THINK_SIDE,
    TRAVERSE_ENTER,
    VISIT_SUBSECTOR,
)


@dataclass(frozen=True)
class SideTable:
    """Published table of `side(node, player)` values.

    This table is the AR-friendly form of DOOM's `R_PointOnSide(viewx, viewy,
    node)`: one precomputed 0/1 side bit per BSP node for this player pose.
    THINK_SIDE computes the side bit with the full geometry chain, then emits a
    SIDE_RECORD token. On the following position the side value is available as
    an input slot at depth 0, and this table republishes it keyed by node id.
    """

    past: PastHandleScope
    handle: LiftedKeyValueHandle

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
    ) -> "SideTable":
        """Publish the SIDE_RECORD-backed side lookup channels."""
        return cls(
            past=past,
            handle=LiftedKeyValueHandle.publish(
                past,
                "side",
                inp.is_side_record,
                inp.id_lifted_key,
                indicator_to_bool(inp.side_record_side),
            ),
        )

    def pick(self, node: Node) -> Node:
        """Look up the precomputed side bit for `node`."""
        return self.handle.pick(self.past, node)


@dataclass(frozen=True)
class BspTraversal:
    """Current-position context for the BSP tree-walk protocol."""

    past: PastHandleScope
    inp: ProtocolTokenView
    scene: SceneIndex
    edges: TraversalEdges
    enter_child: Node
    between_child: Node
    bbox: BBoxPruner

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        input_vec: Node,
        inp: ProtocolTokenView,
        scene: SceneIndex,
        solids: SolidIntervals,
        input_angle_or_zero: PastHandle,
    ) -> "BspTraversal":
        """Publish traversal channels and recover child choices for this row."""
        side_table = SideTable.publish(past, inp)
        enter_child = _node_first_child(scene, side_table, inp.enter_node)
        between_child = _node_second_child(scene, side_table, inp.between_node)
        enter_child_lifted = _node_first_child_lifted(
            scene, side_table, inp.enter_node
        )
        between_child_lifted = _node_second_child_lifted(
            scene, side_table, inp.between_node
        )
        edges = TraversalEdges.publish(
            past,
            inp,
            enter_child,
            between_child,
            enter_child_lifted,
            between_child_lifted,
        )
        between_side = side_table.pick(inp.between_node)
        bbox = BBoxPruner.publish(
            past,
            input_vec,
            inp,
            scene,
            solids,
            input_angle_or_zero,
            between_side,
            between_child,
        )
        return cls(
            past=past,
            inp=inp,
            scene=scene,
            edges=edges,
            enter_child=enter_child,
            between_child=between_child,
            bbox=bbox,
        )

    def after_set_cursor_direction_y(self) -> Node:
        """Branch for the SET_CURSOR_DIRECTION_Y row (which BEGIN emits). Starts the
        side-bit precompute prefix: THINK_SIDE(0) when the map has BSP nodes, else
        enter subsector 0 directly."""
        return select(
            self.scene.nodes.has_any,
            make_token_head(THINK_SIDE, node=constant(0.0)),
            make_token_head(VISIT_SUBSECTOR, s=constant(0.0), depth=constant(0.0)),
        )

    def after_think_side(self) -> Node:
        """Emit `SIDE_RECORD(node, side(node))` for the current THINK_SIDE row."""
        side = bool_to_01(_think_side_compute(self.scene, self.inp.think_node))
        return make_token_head(SIDE_RECORD, node=self.inp.think_node, side=side)

    def after_side_record(self) -> Node:
        """Advance side precompute, or enter the BSP root after existing nodes."""
        next_think_node = add_const(self.inp.side_record_node, 1.0)
        has_next_node = self.scene.nodes.exists(next_think_node)
        next_side = make_token_head(THINK_SIDE, node=next_think_node)
        enter_root = make_token_head(
            TRAVERSE_ENTER, node=self.scene.nodes.root, depth=constant(0.0)
        )
        return select(has_next_node, next_side, enter_root)

    def after_enter(self) -> Node:
        """Descend from TRAVERSE_ENTER to the first/front child.

        DOOM: R_RenderBSPNode (r_bsp.c:573) — immediate recursion into bsp->children[side].
        """
        return node_child_out(self.enter_child, self.inp.enter_depth)

    def after_between(self) -> Node:
        """Run R_CheckBBox-style pruning before the second/back child."""
        return self.bbox.after_between()

    def after_return(self) -> Node:
        """Pop the dynamic traversal stack after a child/subsector returns."""
        return self.edges.after_return(
            self.past,
            self.inp.return_entity,
            self.inp.return_depth,
        )

    # The bbox sub-protocol (R_CheckBBox: boxpos region, the two extreme corners,
    # their world/theta angles, the occlusion scan) is owned by ``BBoxPruner``;
    # these delegators route each bbox branch's next-token to it.
    def after_bbox_boxpos(self) -> Node:
        return self.bbox.after_boxpos()

    def after_bbox_corner_x_mark_a(self) -> ScalarEmit:
        return self.bbox.after_corner_x_mark_a()

    def after_bbox_corner_y_mark_a(self) -> ScalarEmit:
        return self.bbox.after_corner_y_mark_a()

    def after_bbox_corner_x_mark_b(self) -> ScalarEmit:
        return self.bbox.after_corner_x_mark_b()

    def after_bbox_corner_y_mark_b(self) -> ScalarEmit:
        return self.bbox.after_corner_y_mark_b()

    def after_bbox_world_angle_mark_a(self) -> ScalarEmit:
        return self.bbox.after_world_angle_mark_a()

    def after_bbox_theta_mark_a(self) -> ScalarEmit:
        return self.bbox.after_theta_mark_a()

    def after_bbox_world_angle_mark_b(self) -> ScalarEmit:
        return self.bbox.after_world_angle_mark_b()

    def after_bbox_theta_mark_b(self) -> ScalarEmit:
        return self.bbox.after_theta_mark_b()

    def after_bbox_scan(self) -> Node:
        return self.bbox.after_scan()


def _think_side_compute(scene: SceneIndex, node: Node) -> Node:
    """Compute `side_p` for the current THINK_SIDE position's node.

    DOOM's `R_PointOnSide` stores a BSP partition as `(x, y, dx, dy)` and uses
    the sign of `(dy * (viewx - x)) - (dx * (viewy - y))` to choose which child
    is front for the current viewpoint. The sandbox computes the same general
    cross-product form, without DOOM's integer/XOR fast paths.

    The result is emitted through SIDE_RECORD rather than directly published
    here. Re-embedding that token makes `input.side` depth 0 on the next row,
    keeping traversal-time side lookup shallow.
    """
    px = scene.nodes.px(node)
    py = scene.nodes.py(node)
    dx = scene.nodes.dx(node)
    dy = scene.nodes.dy(node)
    rel_x = sub(scene.view.x, px)
    rel_y = sub(scene.view.y, py)
    side_raw = sub(mul_side(dy, rel_x), mul_side(dx, rel_y))
    return SIDE_POSITIVE(side_raw)


# DOOM: NF_SUBSECTOR flag test in R_RenderBSPNode (r_bsp.c:558-565) — node vs subsector dispatch
def _child_dispatch(child_u: Node, tree_depth: Node) -> Node:
    """Emit the correct traversal token for a node-or-subsector child id."""
    is_ss = IS_SUBSECTOR(child_u)
    ss = add_const(child_u, -float(N_NODES_MAX))
    return select(
        is_ss,
        make_token_head(VISIT_SUBSECTOR, s=ss, depth=tree_depth),
        make_token_head(TRAVERSE_ENTER, node=child_u, depth=tree_depth),
    )


def node_child_out(child: Node, tree_depth: Node) -> Node:
    # DOOM: R_RenderBSPNode (r_bsp.c) — recursive descent to a child at increased tree depth
    child_tree_depth = add_const(tree_depth, 1.0)
    return _child_dispatch(child, child_tree_depth)


# DOOM: R_RenderBSPNode (r_bsp.c:573) — front child selection via bsp->children[side]
def _node_first_child(scene: SceneIndex, side_table: SideTable, node: Node) -> Node:
    """Return the DOOM front child for this node and player side bit."""
    side = side_table.pick(node)
    front_child = scene.nodes.front_child(node)
    back_child = scene.nodes.back_child(node)
    return select(side, front_child, back_child)


# DOOM: R_RenderBSPNode (r_bsp.c:576) — back child (side^1) candidate before R_CheckBBox
def _node_second_child(scene: SceneIndex, side_table: SideTable, node: Node) -> Node:
    """Return the DOOM back child candidate before R_CheckBBox pruning."""
    side = side_table.pick(node)
    front_child = scene.nodes.front_child(node)
    back_child = scene.nodes.back_child(node)
    return select(side, back_child, front_child)


def _node_first_child_lifted(
    scene: SceneIndex, side_table: SideTable, node: Node
) -> Node:
    """The first child's width-3 lifted key, mirroring `_node_first_child`.

    The scalar `_node_first_child` chooses the child id by player side; this
    chooses the same child's `[id, -id^2, 1]` producer key so the traversal
    edge can publish it without re-deriving the square from a computed id.
    """
    side = side_table.pick(node)
    front_child = scene.nodes.front_child_lifted(node)
    back_child = scene.nodes.back_child_lifted(node)
    return select(side, front_child, back_child)


def _node_second_child_lifted(
    scene: SceneIndex, side_table: SideTable, node: Node
) -> Node:
    """The second child's width-3 lifted key, mirroring `_node_second_child`."""
    side = side_table.pick(node)
    front_child = scene.nodes.front_child_lifted(node)
    back_child = scene.nodes.back_child_lifted(node)
    return select(side, back_child, front_child)
