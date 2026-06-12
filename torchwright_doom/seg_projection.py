"""Project visited subsector segs into emitted screen-space columns.

The BSP traversal reaches geometry through subsectors; from there this module
owns the local seg-scanning protocol::

    VISIT_SUBSECTOR -> PROCESS_SEG
    PROCESS_SEG -> WORLD_ANGLE_MARK_A              or advance/return
    WORLD_ANGLE_MARK_A -> ANGLE_VALUE(world_a) -> THETA_MARK_A -> ...
    ANGLE_VALUE(theta_b) -> EMIT_X2               or advance/return
    EMIT_X2 -> R_STORE_WALL_RANGE -> ... drawseg scalars ... -> DRAWSEG_U_PHASE

Ported from ``doom_sandbox/implementation/forward/seg_projection.py``.
``SegProjection.publish`` builds the whole per-position projection context as a
sequence of numbered publish phases (the ``# Phase N —`` comments in its body):
the input side channels and seg-scan recovery (phases 1-6), the per-column clip
memory and recent-drawseg state (phase 7), then the wall-column state, runtime
visplane occupancy, seg facts, wall-span draft, and flat pass (phases 8-13). The
``wall`` / ``planes`` / ``flats`` subcontexts on the returned record carry those
later subsystems.

Changes from the sandbox source: ``Vec`` -> ``Node``; sandbox-``api`` / ``.ops``
imports map to the real ``std`` / ``render_ops`` shims; the module-level
``constant`` sentinel is built inside ``publish()`` (no import-time nodes).
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .attention_handles import RecentMarkerHandle
from .past import PastHandle, PastHandleScope
from .protocol_tokens import ProtocolTokenView, screen_column_one_hot
from .render_ops import SCREEN_X_CLAMP, or_
from .scene_index import SceneIndex
from .seg_cycle import (
    PlaneIdLookup,
    ProcessSegCycle,
    ProjectedEndpointColumns,
    ProjectionPhase,
    SubsectorContext,
    VisibleRun,
)
from .solid_intervals import SolidIntervals
from .std import (
    concat,
    constant,
    extract_derived,
    gate,
    indicator_to_bool,
    one_hot,
    select,
)
from .std import sum as vec_sum
from torchwright.ops.arithmetic_ops import clamp
from .render_constants import MATCH_GAIN_LONG
from .flat_state import FlatPassState
from .visplane_state import RuntimeVisplaneState
from .wall_column_state import (
    ClipMemory,
    WallColumnState,
    WallSpanRuntimeDraft,
    WallSpanRuntimeState,
)
from .wall_range_state import RecentDrawsegState, SegLevelFacts

U_TAN_BY_COLUMN_DERIVED_NAME = "u_tan_by_column"


# --- Published projection context (reduced: core / inputs / rows / seg / drawseg) ---


@dataclass(frozen=True)
class CoreContext:
    """Shared per-position inputs for projection branch owners."""

    past: PastHandleScope
    inp: ProtocolTokenView
    scene: SceneIndex
    pos: Node


@dataclass(frozen=True)
class InputChannels:
    """Published current-token side channels consumed by later protocol rows."""

    drawseg_scale_vec: Node
    angle_or_zero: PastHandle
    drawseg_scale_or_zero: PastHandle
    scale_den_inv_or_zero: PastHandle
    scalestep_width_inv_or_zero: PastHandle
    drawseg_scalestep_or_zero: PastHandle
    x_or_zero: PastHandle
    x_key_or_zero: PastHandle
    xtova_cos_or_zero: PastHandle
    u_tan_by_column_or_zero: PastHandle


@dataclass(frozen=True)
class ProjectionRows:
    """Recent marker rows for projection values re-embedded through tokens."""

    past: PastHandleScope
    world_angle_row: RecentMarkerHandle
    find_run_row: RecentMarkerHandle
    emit_x2_row: RecentMarkerHandle
    drawseg_u_angle_row: RecentMarkerHandle
    inputs: InputChannels

    def pick_world_angle(self) -> Node:
        return self.world_angle_row.pick(self.past, self.inputs.angle_or_zero)

    def pick_range_screen_key(self, *, is_stop: bool) -> Node:
        row = self.emit_x2_row if is_stop else self.find_run_row
        return row.pick(self.past, self.inputs.x_key_or_zero)

    def pick_range_xtova_cos(self, *, is_stop: bool) -> Node:
        row = self.emit_x2_row if is_stop else self.find_run_row
        return row.pick(self.past, self.inputs.xtova_cos_or_zero)

    def pick_u_tan_by_column(self) -> Node:
        return self.drawseg_u_angle_row.pick(
            self.past,
            self.inputs.u_tan_by_column_or_zero,
        )


@dataclass(frozen=True)
class SegScanContext:
    """Active subsector/seg-scan state."""

    phase: ProjectionPhase
    cycle: ProcessSegCycle
    columns: ProjectedEndpointColumns
    visible_run: VisibleRun
    solids: SolidIntervals


@dataclass(frozen=True)
class DrawsegScope:
    """Cached drawseg facts plus attention picks for scale sidecars."""

    past: PastHandleScope
    recent_drawseg: RecentDrawsegState
    drawseg_scale_den_row: RecentMarkerHandle
    drawseg_scalestep_den_row: RecentMarkerHandle
    drawseg_scale2_row: RecentMarkerHandle
    drawseg_scalestep_row: RecentMarkerHandle
    inputs: InputChannels

    @property
    def store_i(self) -> Node:
        return self.recent_drawseg.store_i

    @property
    def store_x1(self) -> Node:
        return self.recent_drawseg.store_x1

    @property
    def stop_x(self) -> Node:
        return self.recent_drawseg.stop_x

    @property
    def scale1(self) -> Node:
        return self.recent_drawseg.scale1

    def pick_scale_den_inverse(self) -> Node:
        return self.drawseg_scale_den_row.pick(
            self.past,
            self.inputs.scale_den_inv_or_zero,
        )

    def pick_scalestep_den_inverse(self) -> Node:
        return self.drawseg_scalestep_den_row.pick(
            self.past,
            self.inputs.scalestep_width_inv_or_zero,
        )

    def pick_scale2(self) -> Node:
        return self.drawseg_scale2_row.pick(
            self.past,
            self.inputs.drawseg_scale_or_zero,
        )

    def pick_scalestep(self) -> Node:
        return self.drawseg_scalestep_row.pick(
            self.past,
            self.inputs.drawseg_scalestep_or_zero,
        )


@dataclass(frozen=True)
class WallRuntimeContext:
    """Published wall-column and wall-span runtime state (Phase H)."""

    clip: ClipMemory
    seg_facts: SegLevelFacts
    wall_column: WallColumnState
    wall_span_runtime: WallSpanRuntimeState


@dataclass(frozen=True)
class VisplaneRuntimeContext:
    """Published runtime-visplane and R_CheckPlane handoff state (Phase H)."""

    plane_mark_kind_or_zero: PastHandle
    check_result_key_pub: PastHandle
    check_result_vp_pub: PastHandle
    runtime_visplanes: RuntimeVisplaneState

    def assigned_vp_for_kind(self, past: PastHandleScope, kind: Node) -> Node:
        """Runtime visplane assigned to a floor/ceiling kind by the most recent
        R_CHECK_PLANE_RESULT, recovered from the check-result handoff handles."""
        return past.pick_most_recent(
            one_hot(kind, 2),
            self.check_result_key_pub,
            self.check_result_vp_pub,
            match_gain=MATCH_GAIN_LONG,
        )


@dataclass(frozen=True)
class FlatRuntimeContext:
    """Published flat-pass state (Phase J)."""

    flat_pass: FlatPassState


class WallColumnRenderScalars:
    """Assemble the per-column wall-render scalars at the SCREEN_Y_VALUE row.

    At the SCREEN_Y_VALUE row that follows the column scale VALUE sidecar, gather
    the three scalars a wall column needs -- dc_iscale, the native u coordinate,
    and the colormap (light) row -- and publish them together as the
    ``wallcol_render_state`` channel consumed by the wall-span draft. Both
    column-scale sidecars are published in Phase 1 and recovered here via
    ``attend_to_offset(-1)``; the u coordinate comes from WALL_COL_U at offset -2.
    """

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        scene: SceneIndex,
        *,
        column_scale_inv_or_zero: PastHandle,
        column_scale_diminish_or_zero: PastHandle,
        recent_drawseg_store_i: Node,
    ) -> PastHandle:
        zero_value = constant(0.0)
        dc_iscale_at_publish_point = past.attend_to_offset(
            column_scale_inv_or_zero,
            delta_pos=-1,
        )
        wallcol_dc_iscale_value = select(
            inp.screen_y_after_wall_column_scale,
            dc_iscale_at_publish_point,
            zero_value,
        )
        scale_diminish_at_publish_point = past.attend_to_offset(
            column_scale_diminish_or_zero,
            delta_pos=-1,
        )
        colormap_row_index = clamp(
            vec_sum(
                scene.segs.light_static(recent_drawseg_store_i),
                scale_diminish_at_publish_point,
            ),
            0.0,
            31.0,
        )
        wallcol_cmap_row_value = select(
            inp.screen_y_after_wall_column_scale,
            colormap_row_index,
            zero_value,
        )
        # WALL_COL_U(u_idx) is emitted right after SET_CURSOR_X; at the
        # SCREEN_Y_VALUE row after the scale sidecar the slot is reachable at -2.
        input_wall_col_u_idx_or_zero = past.publish(
            "input_wall_col_u_idx_or_zero",
            inp.wall_col_u_idx,
        )
        u_idx_at_wall_col_u = past.attend_to_offset(
            input_wall_col_u_idx_or_zero,
            delta_pos=-2,
        )
        return past.publish(
            "wallcol_render_state",
            concat(
                wallcol_dc_iscale_value,
                u_idx_at_wall_col_u,
                wallcol_cmap_row_value,
            ),
        )


@dataclass(frozen=True)
class SegProjection:
    """Published per-position render context for the subsector/R_AddLine protocol.

    The ``wall`` and ``planes`` subcontexts carry the wall-column rasterizer and
    runtime visplane occupancy; the ``flats`` subcontext carries the flat
    span/visplane pixel pass (``FlatPassState``).
    """

    core: CoreContext
    inputs: InputChannels
    rows: ProjectionRows
    seg: SegScanContext
    drawseg: DrawsegScope
    wall: WallRuntimeContext
    planes: VisplaneRuntimeContext
    flats: FlatRuntimeContext

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        input_vec: Node,
        inp: ProtocolTokenView,
        scene: SceneIndex,
        solids: SolidIntervals,
        input_angle_or_zero: PastHandle,
        pos: Node,
    ) -> "SegProjection":
        """Publish projection channels and recover the current seg cycle."""
        zero = constant(0.0)
        phase = ProjectionPhase.from_input(inp)

        # Phase 1 — input side channels: per-position handles that later phases
        # recover by recent-marker match or fixed positional offset.
        input_drawseg_scale_vec = inp.value_v5
        input_drawseg_scale_or_zero = past.publish(
            "seg_drawseg_scale_or_zero",
            input_drawseg_scale_vec,
        )
        # Two per-column scale sidecars (consumed by the wall-column state at
        # phase 11 via attend_to_offset(-1)).
        input_column_scale_diminish_or_zero = past.publish(
            "seg_column_scale_diminish_or_zero",
            inp.value_wall_scale_diminish5,
        )
        input_column_scale_inv_or_zero = past.publish(
            "seg_column_scale_inv_or_zero",
            inp.value_inv5,
        )
        input_scale_den_inv_or_zero = past.publish(
            "seg_scale_den_inv_or_zero",
            inp.value_inv6,
        )
        input_scalestep_width_inv_or_zero = past.publish(
            "seg_scalestep_width_inv_or_zero",
            inp.value_inv7,
        )
        input_drawseg_scalestep_or_zero = past.publish(
            "seg_drawseg_scalestep_or_zero",
            inp.value_v8,
        )
        input_u_tan_by_column_or_zero = past.publish(
            "seg_input_u_tan_by_column_or_zero",
            extract_derived(input_vec, U_TAN_BY_COLUMN_DERIVED_NAME),
        )
        input_i_state_or_zero = past.publish(
            "seg_input_i_state_or_zero",
            concat(inp.seg_i_or_zero, inp.id_lifted_key),
        )
        input_x_or_zero = past.publish("seg_input_x_or_zero", inp.seg_x_or_zero)
        input_x_key_or_zero = past.publish(
            "seg_input_x_key_or_zero",
            screen_column_one_hot(input_vec),
        )
        # Phase 2 — cursor_x recovery: SET_CURSOR_X marker -> screen-key pick.
        cursor_x_row = RecentMarkerHandle.publish(
            past,
            "cursor_x",
            inp.is_set_cursor_x,
        )
        # ClipMemory now keys on the column SCALAR (lifted scalar-id equality),
        # not the screen-x one-hot, so recover/publish the cursor column scalar.
        cursor_x_scalar = cursor_x_row.pick(past, input_x_or_zero)
        cursor_x_scalar_pub = past.publish("cursor_x_scalar_value", cursor_x_scalar)
        # Late input sidecar (emitted after cursor_x in protocol order).
        input_xtova_cos_or_zero = past.publish(
            "seg_input_xtova_cos_or_zero",
            vec_sum(
                select(inp.is_find_run, extract_derived(input_vec, "xtova_cos"), zero),
                select(inp.is_emit_x2, inp.emit_x2_xtova_cos, zero),
            ),
        )
        # Phase 3 — recent marker rows: semantic-recency keys for values consumed
        # after re-embedding.
        world_angle_row = RecentMarkerHandle.publish(
            past,
            "projection_recent_world_angle",
            or_(
                phase.angle_after_world_a,
                phase.angle_after_world_b,
            ),
        )
        find_run_row = RecentMarkerHandle.publish(
            past,
            "projection_recent_find_run",
            inp.is_find_run,
        )
        emit_x2_row = RecentMarkerHandle.publish(
            past,
            "projection_recent_emit_x2",
            inp.is_emit_x2,
        )
        store_range_row = RecentMarkerHandle.publish(
            past,
            "projection_recent_store_range",
            inp.is_store_wall_range,
        )
        # clip_update row — consumed by Phase 7 ClipMemory (wall-column occlusion).
        clip_update_row = RecentMarkerHandle.publish(
            past,
            "projection_recent_clip_update",
            inp.is_clip_update,
        )
        drawseg_scale_den_row = RecentMarkerHandle.publish(
            past,
            "drawseg_recent_scale_den",
            or_(
                inp.is_value_after_drawseg_scale1_den,
                inp.is_value_after_drawseg_scale2_den,
            ),
        )
        drawseg_scalestep_den_row = RecentMarkerHandle.publish(
            past,
            "drawseg_recent_scalestep_den",
            inp.is_value_after_drawseg_scalestep_den,
        )
        drawseg_scale1_row = RecentMarkerHandle.publish(
            past,
            "drawseg_recent_scale1",
            inp.is_value_after_drawseg_scale1,
        )
        drawseg_scale2_row = RecentMarkerHandle.publish(
            past,
            "drawseg_recent_scale2",
            inp.is_value_after_drawseg_scale2,
        )
        drawseg_scalestep_row = RecentMarkerHandle.publish(
            past,
            "drawseg_recent_scalestep",
            inp.is_value_after_drawseg_scalestep,
        )
        drawseg_u_angle_row = RecentMarkerHandle.publish(
            past,
            "drawseg_recent_u_angle",
            inp.angle_after_drawseg_u_phase,
        )
        # Phase 4 — seg-cycle recovery: subsector -> process-seg -> plane ids.
        subsector_context = SubsectorContext.publish(past, inp)
        cycle = ProcessSegCycle.publish(past, inp, subsector_context)
        plane_ids = PlaneIdLookup.publish(past, inp, cycle)

        # Phase 5 — plane-mark side channels (consumed by the visplane owner).
        plane_mark_kind_or_zero = past.publish(
            "plane_mark_kind_or_zero",
            select(inp.is_plane_mark, inp.plane_mark_kind, zero),
        )
        plane_mark_p_or_zero = past.publish(
            "plane_mark_p_or_zero",
            select(inp.is_plane_mark, inp.plane_mark_p, zero),
        )
        plane_mark_vp_or_zero = past.publish(
            "plane_mark_vp_or_zero",
            select(inp.is_plane_mark, inp.plane_mark_vp, zero),
        )

        # Phase 6 — projected endpoint columns + visible run.
        x_from_angle = extract_derived(input_vec, "vatx")
        columns = ProjectedEndpointColumns.publish(
            past,
            phase,
            x_from_angle,
            indicator_to_bool(extract_derived(input_vec, "theta_le_neg_fov_half")),
            indicator_to_bool(extract_derived(input_vec, "theta_ge_pos_fov_half")),
        )
        visible_run = VisibleRun.publish(past, inp, columns, solids)

        # Phase 7 — per-column clip memory + recent drawseg state.
        clip = ClipMemory.publish(
            past,
            inp,
            cursor_x_scalar,
            clip_update_row,
            cursor_x_scalar_pub,
        )
        find_run_x = SCREEN_X_CLAMP(find_run_row.pick(past, input_x_or_zero))
        recent_drawseg = RecentDrawsegState.read_from_recent_rows(
            past,
            store_range_row=store_range_row,
            emit_x2_row=emit_x2_row,
            drawseg_scale1_row=drawseg_scale1_row,
            input_i_state_or_zero=input_i_state_or_zero,
            store_x1=find_run_x,
            input_x_or_zero=input_x_or_zero,
            input_drawseg_scale_or_zero=input_drawseg_scale_or_zero,
        )
        # Phase 8 — wall-column state (consumes clip, recent drawseg, plane ids).
        wall_column = WallColumnState.publish(
            past,
            inp,
            scene,
            clip,
            input_x_or_zero,
            input_x_key_or_zero,
            input_drawseg_scale_or_zero,
            recent_drawseg.store_i,
            plane_ids,
        )
        # Phase 9 — check-result marker handoff (consumed by VisplaneMarker).
        check_result_key_pub = past.publish(
            "r_check_plane_result_key",
            gate(
                inp.is_r_check_plane_result,
                one_hot(inp.r_check_plane_result_kind, 2),
            ),
        )
        check_result_vp_pub = past.publish(
            "r_check_plane_result_vp",
            select(inp.is_r_check_plane_result, inp.r_check_plane_result_vp, zero),
        )
        # Phase 10 — runtime visplanes (consumes wall_column + plane-mark channels).
        runtime_visplanes = RuntimeVisplaneState.publish(
            past,
            inp,
            wall_column,
            plane_mark_p_or_zero,
            plane_mark_vp_or_zero,
        )
        # Phase 11 — seg facts + wall-column render scalars. At SEG_KPART rows the
        # lifted seg key comes from the most recent R_STORE_WALL_RANGE row.
        seg_facts = SegLevelFacts.publish(
            past,
            inp,
            scene,
            seg_key_at_kpart_row=recent_drawseg.store_key,
        )
        wallcol_render_state = WallColumnRenderScalars.publish(
            past,
            inp,
            scene,
            column_scale_inv_or_zero=input_column_scale_inv_or_zero,
            column_scale_diminish_or_zero=input_column_scale_diminish_or_zero,
            recent_drawseg_store_i=recent_drawseg.store_i,
        )
        # Phase 12 — wall-span draft.
        wall_span_draft = WallSpanRuntimeDraft.publish(
            past,
            inp,
            pos,
            scene,
            seg_facts,
            wall_column,
            recent_drawseg.store_i,
            wallcol_render_state,
        )
        # Phase 13 — two independent late publishes: the flat pass and
        # the wall-span K-row finish. ``finish()`` gates the K-row y1 state at the
        # SCREEN_Y_VALUE row and does not read flat-pass state, so the order is
        # free; ``FlatPassState`` reads the runtime visplanes (Phase 10).
        flat_pass = FlatPassState.publish(
            past,
            inp,
            scene,
            runtime_visplanes,
            pos,
        )
        wall_span_runtime = wall_span_draft.finish(past, inp)

        inputs = InputChannels(
            drawseg_scale_vec=input_drawseg_scale_vec,
            angle_or_zero=input_angle_or_zero,
            drawseg_scale_or_zero=input_drawseg_scale_or_zero,
            scale_den_inv_or_zero=input_scale_den_inv_or_zero,
            scalestep_width_inv_or_zero=input_scalestep_width_inv_or_zero,
            drawseg_scalestep_or_zero=input_drawseg_scalestep_or_zero,
            x_or_zero=input_x_or_zero,
            x_key_or_zero=input_x_key_or_zero,
            xtova_cos_or_zero=input_xtova_cos_or_zero,
            u_tan_by_column_or_zero=input_u_tan_by_column_or_zero,
        )

        return cls(
            core=CoreContext(past=past, inp=inp, scene=scene, pos=pos),
            inputs=inputs,
            rows=ProjectionRows(
                past=past,
                world_angle_row=world_angle_row,
                find_run_row=find_run_row,
                emit_x2_row=emit_x2_row,
                drawseg_u_angle_row=drawseg_u_angle_row,
                inputs=inputs,
            ),
            seg=SegScanContext(
                phase=phase,
                cycle=cycle,
                columns=columns,
                visible_run=visible_run,
                solids=solids,
            ),
            drawseg=DrawsegScope(
                past=past,
                recent_drawseg=recent_drawseg,
                drawseg_scale_den_row=drawseg_scale_den_row,
                drawseg_scalestep_den_row=drawseg_scalestep_den_row,
                drawseg_scale2_row=drawseg_scale2_row,
                drawseg_scalestep_row=drawseg_scalestep_row,
                inputs=inputs,
            ),
            wall=WallRuntimeContext(
                clip=clip,
                seg_facts=seg_facts,
                wall_column=wall_column,
                wall_span_runtime=wall_span_runtime,
            ),
            planes=VisplaneRuntimeContext(
                plane_mark_kind_or_zero=plane_mark_kind_or_zero,
                check_result_key_pub=check_result_key_pub,
                check_result_vp_pub=check_result_vp_pub,
                runtime_visplanes=runtime_visplanes,
            ),
            flats=FlatRuntimeContext(flat_pass=flat_pass),
        )
