"""Read-only branch owner for subsector seg scanning and endpoint projection
(Plan F / F4).

Ported from ``doom_sandbox/implementation/forward/seg_scanner.py``. Owns the
``R_Subsector`` / ``R_AddLine`` scan-cycle transitions: start a subsector's seg
loop, backface-cull each seg, run the per-endpoint atan2 / theta-wrap chain
through the marker→ANGLE_VALUE sequence, then the solidsegs clip loop that
queries :class:`SolidIntervals` and emits visible screen ranges via
``EMIT_X2`` → ``R_STORE_WALL_RANGE``.

Changes from the sandbox source: ``Vec`` -> ``Node``; the sandbox-``api``
``make_token`` becomes the real ``make_token_head`` — the renderer's dispatch
folds over emit *heads* and stamps one shared derived tail after selecting the
winning branch, so every owner ``after_*`` returns a head (the convention the
Plan-E ``bsp_traversal`` owner established). Ops come from the ``render_ops``
shim; token types from ``vocab``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import Node

from .render_ops import (
    MUL_CROSS,
    add_const,
    and_,
    is_negative_cross,
    min_screen,
    one_minus,
    same_int,
    signed_world_angle,
    sub,
    wrap_signed_angle,
)
from .seg_cycle import dispatch_projection_angle_phase
from .std import make_token_head, select
from .vocab import (
    ADVANCE_SEG,
    ANGLE_VALUE,
    EMIT_X2,
    FIND_RUN,
    N_NODES_MAX,
    PROCESS_SEG,
    R_STORE_WALL_RANGE,
    THETA_MARK_A,
    THETA_MARK_B,
    TRAVERSE_RETURN,
    WORLD_ANGLE_MARK_A,
    WORLD_ANGLE_MARK_B,
)

if TYPE_CHECKING:
    from .seg_projection import SegProjection


@dataclass(frozen=True)
class SegScanner:
    """Owns the R_Subsector/R_AddLine scan cycle branch transitions."""

    projection: "SegProjection"

    def after_visit_subsector(self) -> Node:
        """Start the R_Subsector-style scan, or return if the leaf is empty."""
        projection = self.projection
        scene = projection.core.scene
        inp = projection.core.inp
        first_seg_id = scene.subsectors.first_seg(inp.visit_ss)
        has_first_seg = scene.subsectors.has_first_seg(inp.visit_ss)
        subsector_entity = add_const(inp.visit_ss, float(N_NODES_MAX))
        return select(
            has_first_seg,
            make_token_head(PROCESS_SEG, i=first_seg_id),
            make_token_head(
                TRAVERSE_RETURN,
                entity_u=subsector_entity,
                depth=inp.visit_depth,
            ),
        )

    def after_process_seg(self) -> Node:
        """Backface-cull the current seg before starting endpoint projection."""
        projection = self.projection
        inp = projection.core.inp
        scene = projection.core.scene
        return select(
            and_(
                self.current_seg_faces_player(),
                one_minus(scene.segs.empty_line(inp.process_i)),
            ),
            make_token_head(WORLD_ANGLE_MARK_A),
            self.advance_after_current_process_seg(),
        )

    def after_world_angle_mark_a(self) -> Node:
        """Emit ANGLE_VALUE(world_a) for endpoint A."""
        return self.world_angle_mark_out(is_b_side=False)

    def after_theta_mark_a(self) -> Node:
        """Emit ANGLE_VALUE(theta_a) from the previous world angle."""
        return self.theta_mark_out()

    def after_world_angle_mark_b(self) -> Node:
        """Emit ANGLE_VALUE(world_b) for endpoint B."""
        return self.world_angle_mark_out(is_b_side=True)

    def after_theta_mark_b(self) -> Node:
        """Emit ANGLE_VALUE(theta_b) from the previous world angle."""
        return self.theta_mark_out()

    def after_projection_angle_value(self) -> Node:
        projection = self.projection
        return dispatch_projection_angle_phase(
            projection.seg.phase,
            after_world_a_out=make_token_head(THETA_MARK_A),
            after_theta_a_out=make_token_head(WORLD_ANGLE_MARK_B),
            after_world_b_out=make_token_head(THETA_MARK_B),
            after_theta_b_out=self.after_theta_b_out(),
        )

    # DOOM: R_ClipSolidWallSegment / solidsegs scan (r_bsp.c line 98-185)
    def after_find_run(self) -> Node:
        """Find or skip to the next visible run for the current projected seg."""
        projection = self.projection
        seg = projection.seg
        drawseg = projection.drawseg
        skip_covered = make_token_head(
            FIND_RUN,
            x=seg.visible_run.skip_to,
        )
        x = drawseg.store_x1
        next_start = seg.solids.next_start_after(x)
        run_end = min_screen(seg.columns.last, add_const(next_start, -1.0))
        emit_run = make_token_head(EMIT_X2, x=run_end)
        advance = make_token_head(ADVANCE_SEG, i=seg.cycle.seg_id)
        return select(
            seg.visible_run.beyond_last,
            advance,
            select(seg.visible_run.covered_at_x, skip_covered, emit_run),
        )

    def after_advance_seg(self) -> Node:
        """Advance to the next seg, or return to BSP traversal."""
        projection = self.projection
        inp = projection.core.inp
        cycle = projection.seg.cycle
        return self.advance_after_seg(
            inp.advance_seg_i,
            cycle.subsector_id,
            cycle.tree_depth,
        )

    # DOOM: R_StoreWallRange (r_segs.c line 376 call from clipping)
    def after_emit_x2(self) -> Node:
        """Hand the horizontally visible fragment to R_StoreWallRange."""
        return make_token_head(
            R_STORE_WALL_RANGE,
            i=self.projection.seg.cycle.seg_id,
        )

    # DOOM: R_PointOnSegSide (r_main.c:215-273), cross-product backface cull
    def current_seg_faces_player(self) -> Node:
        """Return the implementation's R_AddLine backface-cull predicate."""
        cross_z = self.current_seg_cross_z()
        return is_negative_cross(cross_z)

    # DOOM: 2D cross product (r_main.c:R_PointOnSegSide, r_bsp.c line 157)
    def current_seg_cross_z(self) -> Node:
        """Compute the current seg's 2D cross product against the player."""
        projection = self.projection
        scene = projection.core.scene
        inp = projection.core.inp
        ax, ay = scene.segs.endpoint_a(inp.process_i)
        bx, by = scene.segs.endpoint_b(inp.process_i)
        seg_dx = sub(bx, ax)
        seg_dy = sub(by, ay)
        rel_py = sub(scene.view.y, ay)
        rel_px = sub(scene.view.x, ax)
        return sub(MUL_CROSS(seg_dx, rel_py), MUL_CROSS(seg_dy, rel_px))

    def advance_after_current_process_seg(self) -> Node:
        """Skip the current PROCESS_SEG and continue the subsector scan."""
        projection = self.projection
        inp = projection.core.inp
        cycle = projection.seg.cycle
        return self.advance_after_seg(
            inp.process_i,
            cycle.subsector_id,
            cycle.tree_depth,
        )

    def after_theta_b_out(self) -> Node:
        """Start visible-run scanning for a projected seg, or skip it."""
        projection = self.projection
        seg = projection.seg
        find_run_out = make_token_head(FIND_RUN, x=seg.columns.first)
        skip_seg_out = make_token_head(ADVANCE_SEG, i=seg.cycle.seg_id)
        return select(
            seg.columns.is_visible,
            find_run_out,
            skip_seg_out,
        )

    # DOOM: R_AddLine endpoint angle computation (r_bsp.c line 271-272, R_PointToAngle calls)
    def world_angle_mark_out(self, is_b_side: bool) -> Node:
        """Run one endpoint's atan2 chain and emit ANGLE_VALUE(world_*)."""
        projection = self.projection
        scene = projection.core.scene
        seg_id = projection.seg.cycle.seg_id
        if is_b_side:
            vx, vy = scene.segs.endpoint_b(seg_id)
        else:
            vx, vy = scene.segs.endpoint_a(seg_id)
        dx = sub(vx, scene.view.x)
        dy = sub(vy, scene.view.y)
        world_angle = signed_world_angle(dx, dy)
        return make_token_head(ANGLE_VALUE, angle=world_angle)

    # DOOM: player-relative angle conversion (r_bsp.c:R_AddLine lines 284-285)
    def theta_mark_out(self) -> Node:
        """Wrap the previous ANGLE_VALUE world angle into player-relative theta."""
        projection = self.projection
        world_angle = projection.rows.pick_world_angle()
        theta = wrap_signed_angle(sub(world_angle, projection.core.scene.view.angle))
        return make_token_head(ANGLE_VALUE, angle=theta)

    def advance_after_seg(
        self,
        seg_id: Node,
        subsector_id: Node,
        tree_depth: Node,
    ) -> Node:
        """Continue this subsector's seg scan, or return when it is exhausted."""
        scene = self.projection.core.scene
        next_seg_id = add_const(seg_id, 1.0)
        next_seg_exists = scene.segs.exists(next_seg_id)
        next_seg_subsector_id = scene.segs.subsector(next_seg_id)
        has_next = and_(
            next_seg_exists,
            same_int(next_seg_subsector_id, subsector_id),
        )
        subsector_entity = add_const(subsector_id, float(N_NODES_MAX))
        return select(
            has_next,
            make_token_head(PROCESS_SEG, i=next_seg_id),
            make_token_head(
                TRAVERSE_RETURN,
                entity_u=subsector_entity,
                depth=tree_depth,
            ),
        )
