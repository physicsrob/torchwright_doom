"""Read-only branch owner for R_DrawPlanes / R_MakeSpans transitions (Plan J2).

Real-side port of ``doom_sandbox/implementation/forward/flat_pass_renderer.py``:
the flat-pass control spine that runs *after* the BSP walk completes
(``DRAW_PLANES_BEGIN``), walking planes -> visplanes -> R_MakeSpans columns ->
SPAN_ROWs -> SET_CURSOR_Y (which flips ``flat_span_seen`` so the shared
pixel/cursor branches route the flat arm), terminating at ``DONE`` when the
plane successor returns the sentinel.

Changes from the sandbox: ``make_token`` -> ``make_token_head``; ``Vec`` ->
``Node``; the module-level sentinel ``constant``\\ s relocated inside the methods
(no import-time graph nodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    from .seg_projection import SegProjection


@dataclass(frozen=True)
class FlatPassRenderer:
    """Owns flat-pass branch transitions after wall traversal completes."""

    projection: "SegProjection"

    def after_draw_planes_begin(self) -> "Node":
        return make_token_head(SET_CURSOR_DIRECTION_X)

    def after_set_cursor_direction_x(self) -> "Node":
        return make_token_head(FLAT_NEXT_PLANE, p=constant(-1.0))

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
    # path — sky visplanes are skipped (no span pass) in J; a column-drawn sky is
    # a separate future phase.
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

    def after_span_row(self) -> "Node":
        return make_token_head(
            SET_CURSOR_Y, y=self.projection.core.inp.span_row_y
        )

    def after_make_spans_without_close(self) -> "Node":
        projection = self.projection
        flat = projection.flats.flat_pass.flat_visplane_values(projection.core.past)
        make_spans = projection.flats.flat_pass.make_spans_values(projection.core.past)
        return select(
            make_spans.is_sentinel,
            make_token_head(FLAT_NEXT_VP, p=flat.p, vp=flat.vp),
            make_token_head(MAKE_SPANS_COL, x=add_const(make_spans.x, 1.0)),
        )

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
                    make_token_head(
                        MAKE_SPANS_COL, x=add_const(closure.make_x, 1.0)
                    ),
                ),
            ),
        )
