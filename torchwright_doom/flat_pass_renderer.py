"""Read-only branch owner for R_DrawPlanes / R_MakeSpans transitions.

Real-side port of ``doom_sandbox/implementation/forward/flat_pass_renderer.py``:
the flat-pass control spine that runs *after* the BSP walk completes
(``DRAW_PLANES_BEGIN``), walking planes -> visplanes -> R_MakeSpans columns ->
SPAN_ROWs -> SET_CURSOR_Y (which flips ``flat_span_seen`` so the shared
pixel/cursor branches route the flat arm), terminating at ``DONE`` when the
plane successor returns the sentinel.

Flat pass control flow (the single map of the four-file token state machine
that draws floors/ceilings). The flat pass is split across four files: this
file is the control spine, ``visplane_marker.py`` records each wall seg's
floor/ceiling region DURING the BSP/wall walk (the rows this pass later reads
back), ``flat_state.py`` publishes the per-row span/cursor state those reads
consume, and ``pixel_dispatcher.py`` owns the shared pixel/cursor branches both
the wall pass and this flat pass route through. Token transitions below; the
owner of each arrow is named in brackets. (Reconstructed from the ``after_*``
methods across these files — read those, not this comment, if they disagree.)

    DRAW_PLANES_BEGIN -> SET_CURSOR_DIRECTION_X            [FlatPassRenderer]
    SET_CURSOR_DIRECTION_X -> FLAT_NEXT_PLANE(p = -1)      [FlatPassRenderer]
    FLAT_NEXT_PLANE -> DONE  (no further used plane)       [FlatPassRenderer]
                    -> FLAT_NEXT_VP(next plane, vp = -1)
    FLAT_NEXT_VP -> FLAT_NEXT_PLANE  (no further vp)       [FlatPassRenderer]
                 -> FLAT_VISPLANE_BEGIN(plane, vp)
    FLAT_VISPLANE_BEGIN -> FLAT_NEXT_VP  (sky: skip span)  [FlatPassRenderer]
                        -> MAKE_SPANS_COL(x = minx)
    MAKE_SPANS_COL -> SPAN_CLOSE_SLOT(slot 0 or 1)         [FlatPassRenderer]
                   -> (no close: advance / next vp)
    SPAN_CLOSE_SLOT -> SPAN_ROW(y)  (slot has rows)        [FlatPassRenderer]
                    -> SPAN_CLOSE_SLOT(other slot) / advance
    SPAN_ROW -> SET_CURSOR_Y(y)                            [FlatPassRenderer]
    SET_CURSOR_Y -> SET_CURSOR_X(span x1)  (flat arm)      [PixelDispatcher]
    SET_CURSOR_X -> PIXEL(first flat texel)  (flat arm)    [PixelDispatcher]
    PIXEL -> PIXEL(next flat texel)  (still in span)       [PixelDispatcher]
          -> back to SPAN_ROW / SPAN_CLOSE_SLOT / next vp / MAKE_SPANS_COL
             (span row finished -> after_completed_flat_span_row below)

The SET_CURSOR_Y / SET_CURSOR_X / PIXEL arrows are SHARED with the wall pass:
``PixelDispatcher`` forks each on the runtime boolean ``flat_span_seen`` (true
only once a SPAN_ROW has fired this frame), so before the flat pass starts they
degenerate to the wall arm. The visplane regions this pass walks were recorded
earlier by ``VisplaneMarker`` (one SCREEN_RANGE per marked wall seg) while the
BSP/wall walk was still running.

Changes from the sandbox: ``make_token`` -> ``make_token_head``; ``Vec`` ->
``Node``; the module-level sentinel ``constant``\\ s relocated inside the methods
(no import-time graph nodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import annotated

from .render_ops import add_const, and_, gt_screen, one_minus, same_int
from .std import constant, make_token_head, select
from .vocab import (
    DONE,
    FLAT_NEXT_PLANE,
    FLAT_NEXT_VP,
    FLAT_VISPLANE_BEGIN,
    MAKE_SPANS_COL,
    N_PLANE_SENTINEL,
    N_VP_SENTINEL,
    SET_CURSOR_DIRECTION_X,
    SET_CURSOR_Y,
    SPAN_CLOSE_SLOT,
    SPAN_ROW,
)

if TYPE_CHECKING:
    from torchwright.graph.node import Node

    from .seg_projection import SegProjection


@dataclass(frozen=True)
class FlatPassRenderer:
    """Owns flat-pass branch transitions after wall traversal completes."""

    projection: "SegProjection"

    @annotated("plan/R_DrawPlanes")
    def after_draw_planes_begin(self) -> "Node":
        return make_token_head(SET_CURSOR_DIRECTION_X)

    @annotated("plan/R_DrawPlanes")
    def after_set_cursor_direction_x(self) -> "Node":
        return make_token_head(FLAT_NEXT_PLANE, p=constant(-1.0))

    @annotated("plan/R_DrawPlanes")
    def after_flat_next_plane(self) -> "Node":
        projection = self.projection
        next_plane = projection.planes.runtime_visplanes.next_plane_after(
            projection.core.past,
            projection.core.inp.flat_next_plane_p,
        )
        return select(
            same_int(next_plane, constant(float(N_PLANE_SENTINEL))),
            make_token_head(DONE),
            make_token_head(FLAT_NEXT_VP, p=next_plane, vp=constant(-1.0)),
        )

    @annotated("plan/R_DrawPlanes")
    def after_flat_next_vp(self) -> "Node":
        projection = self.projection
        next_vp = projection.planes.runtime_visplanes.next_vp_after(
            projection.core.past,
            projection.core.inp.flat_next_vp_p,
            projection.core.inp.flat_next_vp_vp,
        )
        return select(
            same_int(next_vp, constant(float(N_VP_SENTINEL))),
            make_token_head(FLAT_NEXT_PLANE, p=projection.core.inp.flat_next_vp_p),
            make_token_head(
                FLAT_VISPLANE_BEGIN,
                p=projection.core.inp.flat_next_vp_p,
                vp=next_vp,
            ),
        )

    # DOOM: R_DrawPlanes sky special-case (r_plane.c:396-420) vs. regular flat
    # path — sky visplanes are skipped (no span pass); a column-drawn sky is
    # a separate future phase.
    @annotated("plan/R_DrawPlanes")
    def after_flat_visplane_begin(self) -> "Node":
        projection = self.projection
        flat = projection.flats.flat_pass.flat_visplane_values(projection.core.past)
        flat_id = projection.core.scene.planes.flat_id(
            projection.core.inp.flat_visplane_p
        )
        is_sky = projection.core.scene.assets.flats.is_sky(flat_id)
        return select(
            is_sky,
            make_token_head(
                FLAT_NEXT_VP,
                p=projection.core.inp.flat_visplane_p,
                vp=projection.core.inp.flat_visplane_vp,
            ),
            make_token_head(MAKE_SPANS_COL, x=flat.minx),
        )

    @annotated("plan/R_MakeSpans")
    def after_make_spans_col(self) -> "Node":
        projection = self.projection
        return select(
            projection.flats.flat_pass.make_slot0_valid,
            make_token_head(SPAN_CLOSE_SLOT, slot=constant(0.0)),
            select(
                projection.flats.flat_pass.make_slot1_valid,
                make_token_head(SPAN_CLOSE_SLOT, slot=constant(1.0)),
                self.after_make_spans_without_close(),
            ),
        )

    @annotated("plan/R_MakeSpans")
    def after_span_close_slot(self) -> "Node":
        projection = self.projection
        slot1 = same_int(projection.core.inp.span_close_slot, constant(1.0))
        make_spans = projection.flats.flat_pass.make_spans_values(projection.core.past)
        selected_valid = select(
            slot1,
            make_spans.slot1_valid,
            make_spans.slot0_valid,
        )
        selected_y1 = select(
            slot1,
            make_spans.slot1_y1,
            make_spans.slot0_y1,
        )
        return select(
            selected_valid,
            make_token_head(SPAN_ROW, y=selected_y1),
            select(
                and_(
                    one_minus(slot1),
                    make_spans.slot1_valid,
                ),
                make_token_head(SPAN_CLOSE_SLOT, slot=constant(1.0)),
                self.after_make_spans_without_close(),
            ),
        )

    @annotated("plan/R_MakeSpans")
    def after_span_row(self) -> "Node":
        return make_token_head(SET_CURSOR_Y, y=self.projection.core.inp.span_row_y)

    # --- Continuation helpers, not dispatch branches (no entry in render_main's
    # branch table). after_make_spans_without_close is called from the two
    # branches above (after_make_spans_col, after_span_close_slot) when a column
    # opens no close slots; after_completed_flat_span_row is a hidden re-entry
    # reached only from PixelDispatcher.after_flat_pixel_color when a flat span's
    # pixel run finishes ---
    @annotated("plan/R_MakeSpans")
    def after_make_spans_without_close(self) -> "Node":
        projection = self.projection
        flat = projection.flats.flat_pass.flat_visplane_values(projection.core.past)
        make_spans = projection.flats.flat_pass.make_spans_values(projection.core.past)
        return select(
            make_spans.is_sentinel,
            make_token_head(FLAT_NEXT_VP, p=flat.p, vp=flat.vp),
            make_token_head(MAKE_SPANS_COL, x=add_const(make_spans.x, 1.0)),
        )

    @annotated("plan/R_MakeSpans")
    def after_completed_flat_span_row(self) -> "Node":
        projection = self.projection
        flat_span = projection.flats.flat_pass.flat_span_values(projection.core.past)
        closure = projection.flats.flat_pass.closure_values(projection.core.past)
        flat = projection.flats.flat_pass.flat_visplane_values(projection.core.past)
        next_row = add_const(flat_span.y, 1.0)
        slot0_finished = same_int(closure.slot, constant(0.0))
        return select(
            gt_screen(flat_span.y_end, flat_span.y),
            make_token_head(SPAN_ROW, y=next_row),
            select(
                and_(slot0_finished, closure.slot1_valid),
                make_token_head(SPAN_CLOSE_SLOT, slot=constant(1.0)),
                select(
                    closure.is_sentinel,
                    make_token_head(FLAT_NEXT_VP, p=flat.p, vp=flat.vp),
                    make_token_head(MAKE_SPANS_COL, x=add_const(closure.make_x, 1.0)),
                ),
            ),
        )
