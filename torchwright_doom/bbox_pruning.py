"""R_CheckBBox-style pruning for back-side BSP traversal.

Ported from ``doom_sandbox/implementation/forward/bbox_pruning.py``. Owns the
``TRAVERSE_BETWEEN`` bbox sub-protocol: classify the player against the back
child's bounding box into one of DOOM's nine regions (``R_CheckBBox`` ``boxx`` /
``boxy``), project the two extreme corners to screen columns through the same
atan2 octant + view-relative-theta chain the seg projection uses, then scan that
screen span against the now-populated :class:`SolidIntervals`. A bbox entirely
beyond the last visible column or fully covered by solid walls prunes its
subtree (``TRAVERSE_RETURN``); otherwise traversal descends into the back child.

Changes from the sandbox source:

* ``Vec`` -> ``Node``; ``...api`` helpers come from the real-side ``std`` /
  ``render_ops`` shim; the token declarations / ``ValueRange`` / ``make_value``
  from ``vocab`` / ``value_ranges``.
* ``make_token`` -> ``make_token_head``: the renderer's dispatch folds over emit
  *heads* and stamps one shared derived tail after selecting the winning branch,
  so every owner ``after_*`` emitter returns a head (the convention the
  ``bsp_traversal`` / ``seg_scanner`` owners established).
* The sandbox kept a near-duplicate octant builder here
  (``_signed_world_angle_bbox``, clamp 3072) alongside ``ops.signed_world_angle``
  (clamp 2048). The real ``render_ops.signed_world_angle`` is already unified on
  the wider 3072 clamp, so this port calls it directly and the duplicate (plus
  its ray-matrix / ``Q2``/``Q3`` constants) is dropped.

The dataclasses, the publish/pick structure, and the ``after_*`` emit logic are
otherwise a line-for-line port. The ``_BOX*_LINEAR`` matrices below are plain
constant data passed to ``linear`` (not graph nodes), so the module stays free
of import-time graph nodes (``global_node_id == 0`` after import).
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .attention_handles import RecentMarkerHandle
from .past import PastHandle, PastHandleScope
from .protocol_tokens import ProtocolTokenView
from .render_ops import (
    COORD_GT_ZERO,
    add_const,
    gt_screen,
    max_screen,
    min_screen,
    one_minus,
    sub,
    wrap_signed_angle,
)
from .scene_index import SceneIndex
from .solid_intervals import SolidIntervals
from .std import (
    AngleInputEmit,
    ScalarEmit,
    angle_inputs,
    angle_scalar,
    bool_or,
    bool_to_01,
    concat,
    extract_derived,
    linear,
    make_token_head,
    select,
    split,
    value_scalar,
)
from .value_ranges import ValueRange
from .vocab import (
    BBOX_BOXPOS,
    BBOX_CORNER_X_MARK_A,
    BBOX_CORNER_X_MARK_B,
    BBOX_CORNER_Y_MARK_A,
    BBOX_CORNER_Y_MARK_B,
    BBOX_SCAN,
    BBOX_THETA_MARK_A,
    BBOX_THETA_MARK_B,
    BBOX_WORLD_ANGLE_MARK_A,
    BBOX_WORLD_ANGLE_MARK_B,
    TRAVERSE_RETURN,
)

# boxx/boxy each fold two sign bits into a 0/1/2 region; boxpos = 4·boxy + boxx
# is the flat index into DOOM's nine-region checkcoord table.
_BOXX_LINEAR = [[1.0], [1.0]]
_BOXY_LINEAR = [[1.0], [1.0]]
_BOXPOS_LINEAR = [[4.0], [1.0]]


@dataclass(frozen=True)
class BBoxProjectionPhase:
    angle_after_world_a: Node
    angle_after_theta_a: Node
    angle_after_world_b: Node
    angle_after_theta_b: Node
    is_bbox_angle: Node

    @classmethod
    def from_input(cls, inp: ProtocolTokenView) -> "BBoxProjectionPhase":
        return cls(
            angle_after_world_a=inp.angle_after_bbox_world_a,
            angle_after_theta_a=inp.angle_after_bbox_theta_a,
            angle_after_world_b=inp.angle_after_bbox_world_b,
            angle_after_theta_b=inp.angle_after_bbox_theta_b,
            is_bbox_angle=inp.is_bbox_angle_payload,
        )


@dataclass(frozen=True)
class BBoxContext:
    node: Node
    depth: Node
    child: Node
    top: Node
    bottom: Node
    left: Node
    right: Node

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        scene: SceneIndex,
        side: Node,
        back_child: Node,
    ) -> "BBoxContext":
        row = RecentMarkerHandle.publish(past, "bbox_between", inp.is_between)
        top_value = select(
            side,
            scene.nodes.bbox_top_back(inp.between_node),
            scene.nodes.bbox_top_front(inp.between_node),
        )
        bottom_value = select(
            side,
            scene.nodes.bbox_bottom_back(inp.between_node),
            scene.nodes.bbox_bottom_front(inp.between_node),
        )
        left_value = select(
            side,
            scene.nodes.bbox_left_back(inp.between_node),
            scene.nodes.bbox_left_front(inp.between_node),
        )
        right_value = select(
            side,
            scene.nodes.bbox_right_back(inp.between_node),
            scene.nodes.bbox_right_front(inp.between_node),
        )
        state_h = past.publish(
            "bbox_context_state",
            concat(
                inp.between_node,
                inp.between_depth,
                back_child,
                top_value,
                bottom_value,
                left_value,
                right_value,
            ),
        )
        node, depth, child, top, bottom, left, right = split(
            row.pick(past, state_h),
            [1] * 7,
        )
        return cls(
            node=node,
            depth=depth,
            child=child,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
        )


@dataclass(frozen=True)
class BBoxRange:
    first: Node
    last: Node
    has_width: Node

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        phase: BBoxProjectionPhase,
        x_from_angle: Node,
    ) -> "BBoxRange":
        x_a_row = RecentMarkerHandle.publish(
            past,
            "bbox_theta_a_x",
            phase.angle_after_theta_a,
        )
        x_a_value = past.publish("bbox_theta_a_x", x_from_angle)
        x_a = x_a_row.pick(past, x_a_value)

        first = min_screen(x_a, x_from_angle)
        last = max_screen(x_a, x_from_angle)
        has_width = gt_screen(last, first)
        range_row = RecentMarkerHandle.publish(
            past,
            "bbox_range",
            phase.angle_after_theta_b,
        )
        range_h = past.publish("bbox_range_value", concat(first, last))
        range_first, range_last = split(range_row.pick(past, range_h), [1, 1])
        return cls(
            first=range_first,
            last=range_last,
            has_width=has_width,
        )


@dataclass(frozen=True)
class BBoxPruner:
    past: PastHandleScope
    inp: ProtocolTokenView
    scene: SceneIndex
    solids: SolidIntervals
    input_angle_or_zero: PastHandle
    input_bbox_coord_or_zero: PastHandle
    input_boxpos_or_zero: PastHandle
    boxpos_row: RecentMarkerHandle
    corner_x_row: RecentMarkerHandle
    corner_y_row: RecentMarkerHandle
    world_angle_row: RecentMarkerHandle
    context: BBoxContext
    phase: BBoxProjectionPhase
    bbox_range: BBoxRange

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        input_vec: Node,
        inp: ProtocolTokenView,
        scene: SceneIndex,
        solids: SolidIntervals,
        input_angle_or_zero: PastHandle,
        side: Node,
        back_child: Node,
    ) -> "BBoxPruner":
        phase = BBoxProjectionPhase.from_input(inp)
        input_bbox_coord_or_zero = past.publish(
            "bbox_coord_or_zero",
            inp.value_v0,
        )
        input_boxpos_or_zero = past.publish(
            "bbox_input_boxpos_or_zero",
            inp.boxpos_or_zero,
        )
        # Bbox projection is a marker/VALUE/ANGLE_VALUE protocol. These
        # semantic markers keep the latest logical payload available without
        # depending on fixed distances between protocol rows.
        # Boxpos is copied through the corner markers so both bbox endpoints can
        # recover the same checkcoord row.
        boxpos_row = RecentMarkerHandle.publish(
            past,
            "bbox_recent_boxpos",
            bool_or(
                inp.is_bbox_boxpos,
                inp.is_bbox_corner_x_a_mark,
                inp.is_bbox_corner_y_a_mark,
                inp.is_bbox_corner_x_b_mark,
                inp.is_bbox_corner_y_b_mark,
            ),
        )
        # Corner coordinate payloads are VALUE rows immediately after bbox.x*
        # and bbox.y* markers.
        corner_x_row = RecentMarkerHandle.publish(
            past,
            "bbox_recent_corner_x",
            bool_or(
                inp.is_value_after_bbox_corner_x_a,
                inp.is_value_after_bbox_corner_x_b,
            ),
        )
        corner_y_row = RecentMarkerHandle.publish(
            past,
            "bbox_recent_corner_y",
            bool_or(
                inp.is_value_after_bbox_corner_y_a,
                inp.is_value_after_bbox_corner_y_b,
            ),
        )
        # World-angle payloads are ANGLE_VALUE rows immediately after bbox.angle*
        # markers; theta markers consume them after re-embedding.
        world_angle_row = RecentMarkerHandle.publish(
            past,
            "bbox_recent_world_angle",
            bool_or(
                phase.angle_after_world_a,
                phase.angle_after_world_b,
            ),
        )
        context = BBoxContext.publish(past, inp, scene, side, back_child)
        bbox_range = BBoxRange.publish(
            past,
            phase,
            extract_derived(input_vec, "vatx"),
        )
        return cls(
            past=past,
            inp=inp,
            scene=scene,
            solids=solids,
            input_angle_or_zero=input_angle_or_zero,
            input_bbox_coord_or_zero=input_bbox_coord_or_zero,
            input_boxpos_or_zero=input_boxpos_or_zero,
            boxpos_row=boxpos_row,
            corner_x_row=corner_x_row,
            corner_y_row=corner_y_row,
            world_angle_row=world_angle_row,
            context=context,
            phase=phase,
            bbox_range=bbox_range,
        )

    def after_between(self) -> Node:
        return make_token_head(BBOX_BOXPOS, boxpos=self._boxpos())

    def after_boxpos(self) -> Node:
        boxpos = self.inp.bbox_boxpos
        return select(
            self.inp.boxpos_fails_open,
            self._descend_child(),
            make_token_head(BBOX_CORNER_X_MARK_A, boxpos=boxpos),
        )

    def after_corner_x_mark_a(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R0,
            self._corner_x(self.inp.boxpos_check_a_x_right),
        )

    def after_corner_y_mark_a(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R0,
            self._corner_y(self.inp.boxpos_check_a_y_bottom),
        )

    def after_world_angle_mark_a(self) -> AngleInputEmit:
        return self._world_angle_mark_out()

    def after_theta_mark_a(self) -> ScalarEmit:
        return self._theta_mark_out()

    def after_corner_x_mark_b(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R0,
            self._corner_x(self.inp.boxpos_check_b_x_right),
        )

    def after_corner_y_mark_b(self) -> ScalarEmit:
        return value_scalar(
            ValueRange.R0,
            self._corner_y(self.inp.boxpos_check_b_y_bottom),
        )

    def after_world_angle_mark_b(self) -> AngleInputEmit:
        return self._world_angle_mark_out()

    def after_theta_mark_b(self) -> ScalarEmit:
        return self._theta_mark_out()

    def after_value(self, no_op_out: Node) -> Node:
        boxpos = self._previous_boxpos()
        after_x_a = make_token_head(BBOX_CORNER_Y_MARK_A, boxpos=boxpos)
        after_y_a = make_token_head(BBOX_WORLD_ANGLE_MARK_A)
        after_x_b = make_token_head(BBOX_CORNER_Y_MARK_B, boxpos=boxpos)
        after_y_b = make_token_head(BBOX_WORLD_ANGLE_MARK_B)
        return select(
            self.inp.is_value_after_bbox_corner_y_b,
            after_y_b,
            select(
                self.inp.is_value_after_bbox_corner_x_b,
                after_x_b,
                select(
                    self.inp.is_value_after_bbox_corner_y_a,
                    after_y_a,
                    select(
                        self.inp.is_value_after_bbox_corner_x_a,
                        after_x_a,
                        no_op_out,
                    ),
                ),
            ),
        )

    def after_bbox_angle_value(self) -> Node:
        after_world_a = make_token_head(BBOX_THETA_MARK_A)
        after_theta_a = make_token_head(
            BBOX_CORNER_X_MARK_B,
            boxpos=self._recent_boxpos(),
        )
        after_world_b = make_token_head(BBOX_THETA_MARK_B)
        after_theta_b = select(
            self.bbox_range.has_width,
            make_token_head(BBOX_SCAN, x=self.bbox_range.first),
            self._prune_return(),
        )
        out = select(
            self.phase.angle_after_theta_b,
            after_theta_b,
            select(
                self.phase.angle_after_world_b,
                after_world_b,
                select(
                    self.phase.angle_after_theta_a,
                    after_theta_a,
                    after_world_a,
                ),
            ),
        )
        return out

    # Scan bbox screen-range against solidsegs occlusion (DOOM: R_CheckBBox lines 475-487)
    def after_scan(self) -> Node:
        x = self.inp.bbox_scan_x
        beyond = gt_screen(x, self.bbox_range.last)
        covered, covering_end = self.solids.covered_and_end(
            x,
            self.inp.bbox_scan_x_square,
        )
        next_x = add_const(covering_end, 1.0)
        return select(
            beyond,
            self._prune_return(),
            select(
                covered,
                make_token_head(BBOX_SCAN, x=next_x),
                self._descend_child(),
            ),
        )

    def _descend_child(self) -> Node:
        from .bsp_traversal import node_child_out

        return node_child_out(self.context.child, self.context.depth)

    def _prune_return(self) -> Node:
        return make_token_head(
            TRAVERSE_RETURN,
            entity_u=self.context.node,
            depth=self.context.depth,
        )

    # DOOM: R_CheckBBox boxx/boxy region computation (r_bsp.c lines 404-418)
    def _boxpos(self) -> Node:
        boxx_gt_left = COORD_GT_ZERO(sub(self.scene.view.x, self.context.left))
        boxx_ge_right = one_minus(
            COORD_GT_ZERO(sub(self.context.right, self.scene.view.x))
        )
        boxx = linear(
            concat(bool_to_01(boxx_gt_left), bool_to_01(boxx_ge_right)),
            _BOXX_LINEAR,
        )

        boxy_below_top = COORD_GT_ZERO(sub(self.context.top, self.scene.view.y))
        boxy_le_bottom = one_minus(
            COORD_GT_ZERO(sub(self.scene.view.y, self.context.bottom))
        )
        boxy = linear(
            concat(bool_to_01(boxy_below_top), bool_to_01(boxy_le_bottom)),
            _BOXY_LINEAR,
        )
        return linear(concat(boxy, boxx), _BOXPOS_LINEAR)

    # DOOM: R_CheckBBox checkcoord[] corner indexing (r_bsp.c lines 422-425)
    def _corner_x(self, use_right: Node) -> Node:
        return select(use_right, self.context.right, self.context.left)

    # DOOM: R_CheckBBox checkcoord[] corner indexing (r_bsp.c lines 422-425)
    def _corner_y(self, use_bottom: Node) -> Node:
        return select(use_bottom, self.context.bottom, self.context.top)

    def _previous_boxpos(self) -> Node:
        return self._recent_boxpos()

    def _recent_boxpos(self) -> Node:
        return self.boxpos_row.pick(self.past, self.input_boxpos_or_zero)

    def _world_angle_mark_out(self) -> AngleInputEmit:
        vx = self.corner_x_row.pick(self.past, self.input_bbox_coord_or_zero)
        vy = self.corner_y_row.pick(self.past, self.input_bbox_coord_or_zero)
        dx = sub(vx, self.scene.view.x)
        dy = sub(vy, self.scene.view.y)
        return angle_inputs(dx, dy)

    def _theta_mark_out(self) -> ScalarEmit:
        world_angle = self.world_angle_row.pick(self.past, self.input_angle_or_zero)
        theta = wrap_signed_angle(sub(world_angle, self.scene.view.angle))
        return angle_scalar(theta)
