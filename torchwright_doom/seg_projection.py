"""Project visited subsector segs into emitted screen-space columns
(Plan F / F4 — the **reduced** build).

The BSP traversal reaches geometry through subsectors; from there this module
owns the local seg-scanning protocol::

    VISIT_SUBSECTOR -> PROCESS_SEG
    PROCESS_SEG -> WORLD_ANGLE_MARK_A              or advance/return
    WORLD_ANGLE_MARK_A -> ANGLE_VALUE(world_a) -> THETA_MARK_A -> ...
    ANGLE_VALUE(theta_b) -> EMIT_X2               or advance/return
    EMIT_X2 -> R_STORE_WALL_RANGE -> ... drawseg scalars ... -> DRAWSEG_U_PHASE

Ported from ``doom_sandbox/implementation/forward/seg_projection.py``, **reduced
to the Phase-F write side**: it builds the per-position context the seg scan
and drawseg-scalar chain read (Phases 1-7) and **omits** the wall-column /
visplane / flat subsystems (Phases 8-13 — Phase H). Concretely it drops the
imports of ``wall_column_state`` / ``visplane_state`` / ``flat_state`` and the
``wall`` / ``planes`` / ``flats`` subcontexts, and skips ``ClipMemory``
(Phase 7's wall-column occlusion arrays — nothing in the reduced build reads
``clip``; it lives in the deferred ``wall_column_state``). The seam is a
publish-ordering cut: Phases 1-6 are built whole, Phase 7 keeps only
``recent_drawseg`` + the clamped ``find_run_x``.

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
from .std import concat, constant, extract_derived, indicator_to_bool, select
from .std import sum as vec_sum
from .wall_range_state import RecentDrawsegState

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
class SegProjection:
    """Published per-position render context for the subsector/R_AddLine protocol.

    Reduced to the five Phase-F subcontexts -- ``core``, ``inputs``, ``rows``,
    ``seg``, ``drawseg``. The ``wall`` / ``planes`` / ``flats`` subcontexts are
    Phase H and are not built here; branch owners that would consume them are
    NO_OP-stubbed in the reduced dispatch.
    """

    core: CoreContext
    inputs: InputChannels
    rows: ProjectionRows
    seg: SegScanContext
    drawseg: DrawsegScope

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
        # Two per-column scale sidecars (Phase 11 / Phase-H consumers via
        # attend_to_offset(-1)); published here to keep their channel names live
        # for when the wall-column renderer lands. Harmless in the reduced build.
        past.publish(
            "seg_column_scale_diminish_or_zero",
            inp.value_wall_scale_diminish5,
        )
        past.publish(
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
        cursor_x_key = cursor_x_row.pick(past, input_x_key_or_zero)
        past.publish("cursor_x_key_value", cursor_x_key)
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
        # clip_update row (Phase 7 ClipMemory consumer is deferred); published to
        # keep the recency channel coherent across the teacher-forced span.
        RecentMarkerHandle.publish(
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
        PlaneIdLookup.publish(past, inp, cycle)

        # Phase 5 — plane-mark side channels (consumed by the deferred visplane
        # owner; published to keep the channels coherent).
        past.publish(
            "plane_mark_kind_or_zero",
            select(inp.is_plane_mark, inp.plane_mark_kind, zero),
        )
        past.publish(
            "plane_mark_p_or_zero",
            select(inp.is_plane_mark, inp.plane_mark_p, zero),
        )
        past.publish(
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

        # Phase 7 (reduced) — find_run start column + recent drawseg state.
        # ClipMemory (per-column occlusion arrays) is Phase H and omitted.
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
        )
