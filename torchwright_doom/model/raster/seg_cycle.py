"""Projection seg-cycle state records for the SegProjection ``seg`` subcontext.

Each record is a frozen dataclass with a ``publish()`` classmethod that publishes the
cross-position channels DOOM's ``R_Subsector`` / ``R_AddLine`` inner loop reads
back at the current row, then recovers the value(s) the branch owner needs.

Changes from the original: ``Vec`` -> ``Node``; the original ``api`` /
``.ops`` imports map to the real ``std`` / ``past`` / ``render_ops`` shims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import Node
from torchwright.graph import annotated

from ..attention_handles import RecentMarkerHandle, lifted_id_query
from ..past import PastHandleScope
from ..render_constants import MATCH_GAIN_LONG
from ..render_ops import (
    ABS_SMALL_INT,
    HAS_PIXEL_WIDTH,
    add_const,
    and_,
    gt_screen,
    max_screen,
    min_screen,
    one_minus,
    sub,
)
from ..std import concat, gate, select, split

if TYPE_CHECKING:
    from ..protocol.protocol_tokens import ProtocolTokenView
    from ..traversal.solid_intervals import SolidIntervals


@dataclass(frozen=True)
class ProjectionPhase:
    """Semantic phase of the current ANGLE_VALUE position."""

    angle_after_world_a: Node
    angle_after_theta_a: Node
    angle_after_world_b: Node
    angle_after_theta_b: Node
    is_projection_angle: Node

    @classmethod
    def from_input(cls, inp: "ProtocolTokenView") -> "ProjectionPhase":
        """Recover projection meaning from the marker before this carrier row."""
        return cls(
            angle_after_world_a=inp.angle_after_world_a,
            angle_after_theta_a=inp.angle_after_theta_a,
            angle_after_world_b=inp.angle_after_world_b,
            angle_after_theta_b=inp.angle_after_theta_b,
            is_projection_angle=inp.is_projection_angle_payload,
        )


@dataclass(frozen=True)
class SubsectorContext:
    """Most recent R_Subsector slots that scope the active seg loop."""

    subsector_id: Node
    tree_depth: Node

    @classmethod
    @annotated("proj")
    def publish(
        cls,
        past: PastHandleScope,
        inp: "ProtocolTokenView",
    ) -> "SubsectorContext":
        subsector_row = RecentMarkerHandle.publish(
            past,
            "current_subsector",
            inp.is_visit_subsector,
        )
        state_value = past.publish(
            "current_subsector_state",
            concat(inp.visit_ss, inp.visit_depth),
        )
        subsector_id, tree_depth = split(
            subsector_row.pick(past, state_value),
            [1, 1],
        )
        return cls(
            subsector_id=subsector_id,
            tree_depth=tree_depth,
        )


@dataclass(frozen=True)
class ProcessSegCycle:
    """Most recent PROCESS_SEG slots carried through the projection cycle."""

    seg_id: Node
    subsector_id: Node
    tree_depth: Node

    @classmethod
    @annotated("proj")
    def publish(
        cls,
        past: PastHandleScope,
        inp: "ProtocolTokenView",
        subsector: SubsectorContext,
    ) -> "ProcessSegCycle":
        """Publish PROCESS_SEG slots and recover the active seg-scan context."""
        process_seg_row = RecentMarkerHandle.publish(past, "pseg", inp.is_process_seg)
        seg_id_value = past.publish("pseg_i", inp.process_i)
        return cls(
            seg_id=process_seg_row.pick(past, seg_id_value),
            subsector_id=subsector.subsector_id,
            tree_depth=subsector.tree_depth,
        )


# DOOM: R_FindPlane (r_plane.c) — floor/ceiling visplane recovery per subsector, called from R_Subsector
@dataclass(frozen=True)
class PlaneIdLookup:
    """Per-position floor/ceiling plane ids for the active subsector."""

    floor_id: Node
    ceiling_id: Node

    @classmethod
    @annotated("proj/R_FindPlane")
    def publish(
        cls,
        past: PastHandleScope,
        inp: "ProtocolTokenView",
        cycle: ProcessSegCycle,
    ) -> "PlaneIdLookup":
        floor_key = past.publish(
            "ss_floor_plane_key",
            gate(inp.is_ss_floor_plane, inp.id_lifted_key),
        )
        floor_value = past.publish("ss_floor_plane_value", inp.ss_floor_plane_p)
        ceiling_key = past.publish(
            "ss_ceiling_plane_key",
            gate(inp.is_ss_ceiling_plane, inp.id_lifted_key),
        )
        ceiling_value = past.publish(
            "ss_ceiling_plane_value",
            inp.ss_ceiling_plane_p,
        )
        ss_query = lifted_id_query(cycle.subsector_id)
        return cls(
            floor_id=past.pick_most_recent(
                ss_query,
                floor_key,
                floor_value,
                match_gain=MATCH_GAIN_LONG,
            ),
            ceiling_id=past.pick_most_recent(
                ss_query,
                ceiling_key,
                ceiling_value,
                match_gain=MATCH_GAIN_LONG,
            ),
        )


@dataclass(frozen=True)
class ProjectedEndpointColumns:
    """Stored endpoint columns for the currently projected seg."""

    first: Node
    last: Node
    is_visible: Node

    @classmethod
    @annotated("proj")
    def publish(
        cls,
        past: PastHandleScope,
        phase: ProjectionPhase,
        x_from_angle: Node,
        theta_le_neg_fov_half: Node,
        theta_ge_pos_fov_half: Node,
    ) -> "ProjectedEndpointColumns":
        """Publish and recover endpoint span values within one PROCESS_SEG cycle."""
        theta_a_x1_active = phase.angle_after_theta_a
        x1_row = RecentMarkerHandle.publish(past, "theta_a_x1", theta_a_x1_active)
        theta_a_state = past.publish(
            "theta_a_projection_state",
            concat(x_from_angle, theta_le_neg_fov_half),
        )
        x1, theta_a_le_neg_c = split(x1_row.pick(past, theta_a_state), [1, 1])

        x_diff = ABS_SMALL_INT(sub(x_from_angle, x1))
        crosses_pixel_column = HAS_PIXEL_WIDTH(x_diff)
        theta_b_ge_pos_c = theta_ge_pos_fov_half
        fov_culled = and_(theta_a_le_neg_c, theta_b_ge_pos_c)
        is_visible = and_(crosses_pixel_column, one_minus(fov_culled))

        first = min_screen(x1, x_from_angle)
        last = max_screen(x1, x_from_angle)
        span_row = RecentMarkerHandle.publish(
            past,
            "projected_seg_span",
            phase.angle_after_theta_b,
        )
        span_value = past.publish("projected_seg_span_value", concat(first, last))
        first_value, last_value = split(span_row.pick(past, span_value), [1, 1])

        return cls(
            first=first_value,
            last=last_value,
            is_visible=is_visible,
        )


@dataclass(frozen=True)
class VisibleRun:
    """The next visible run discovered by a FIND_RUN position."""

    beyond_last: Node
    covered_at_x: Node
    skip_to: Node

    @classmethod
    @annotated("proj")
    def publish(
        cls,
        past: PastHandleScope,
        inp: "ProtocolTokenView",
        columns: ProjectedEndpointColumns,
        solids: "SolidIntervals",
    ) -> "VisibleRun":
        x = inp.find_run_x
        beyond_last = gt_screen(x, columns.last)
        covered_at_x, covering_end = solids.covered_and_end(
            x,
            inp.find_run_x_square,
        )
        skip_to = add_const(covering_end, 1.0)
        return cls(
            beyond_last=beyond_last,
            covered_at_x=covered_at_x,
            skip_to=skip_to,
        )


@annotated("proj")
def dispatch_projection_angle_phase(
    phase: ProjectionPhase,
    *,
    after_world_a_out: Node,
    after_theta_a_out: Node,
    after_world_b_out: Node,
    after_theta_b_out: Node,
) -> Node:
    """Route a projection ANGLE_VALUE by the preceding projection marker."""
    return select(
        phase.angle_after_theta_b,
        after_theta_b_out,
        select(
            phase.angle_after_world_b,
            after_world_b_out,
            select(
                phase.angle_after_theta_a,
                after_theta_a_out,
                after_world_a_out,
            ),
        ),
    )
