"""Wall-column runtime state for the SegProjection ``wall`` subcontext (Phase H).

Sentinel/constant nodes are built inside the publish methods, not at module
scope — see GLOSSARY.md 'the import-time-node rule'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


from torchwright.graph import Node
from torchwright.graph import annotated

from .assets import _H_IDX_OH_WIDTH
from .attention_handles import RecentMarkerHandle
from .constants import CENTER_Y, VIEW_HEIGHT
from .past import PastHandle, PastHandleScope
from .render_constants import MATCH_GAIN_CLIP, OPEN_CLIP_CEILING, PART_NONE
from .render_ops import (
    CEIL_Y,
    CEIL_Y_WIDE,
    CLIP_Y_CLAMP,
    FLOOR_Y_WIDE,
    SCALE_CLAMP,
    SCREEN_X_CLAMP,
    SPAN_Y_CLAMP,
    add_const,
    and_,
    column_from_screen_x,
    gt_height,
    gt_screen,
    gt_y_ceil_boundary,
    gt_y_floor_boundary,
    le_span_y,
    max_screen,
    min_screen,
    mul_height_scale,
    one_minus,
    or_,
    radix_col_key as _radix_col_key,
    same_int,
    sub,
)
from .std import (
    compare,
    concat,
    constant,
    gate,
    indicator_to_bool,
    linear,
    one_hot,
    pick_by_one_hot,
    reduce_sum,
    select,
    split,
    sum as node_sum,
)

if TYPE_CHECKING:
    from .protocol_tokens import ProtocolTokenView
    from .scene_index import SceneIndex
    from .seg_cycle import PlaneIdLookup
    from .wall_range_state import SegLevelFacts


_PART_IS_MID_LINEAR = [[1.0], [0.0], [0.0]]
_PART_IS_UPPER_LINEAR = [[0.0], [1.0], [0.0]]

# Radix split for the ClipMemory column key: the screen column c becomes a
# (bucket = c // B, digit = c % B) pair of one-hots so exact column equality
# needs only B + N_BUCKETS cols (16 fixture / 26 real) instead of SCREEN_WIDTH+1.
# Unlike a width-3 lifted key, the one-hot dot is a sum of non-negative one-hot
# products — NO large-magnitude cancellation — so the gained matched dot is exact
# (2 * MATCH_GAIN_CLIP, computed without subtracting ~1e9 terms) and the
# 8-per-position recency tiebreak survives under any fp32 matmul accumulation
# order (the lifted key's match_gain*c^2 cancellation lost it on A100).
# The clip-memory column key is the shared render_ops radix scheme
# (render_ops.radix_col_key, aliased in the import block above).
# Column scalar published on rows that are NOT a clip update. Any value that can
# never equal a real column (0..SCREEN_WIDTH) works: a query recovers it and the
# same_int presence test reads ABSENT, so the column falls to the default clip.
_ABSENT_COLUMN = -1.0


def _part_is_mid(part_oh: Node) -> Node:
    return indicator_to_bool(linear(part_oh, _PART_IS_MID_LINEAR))


def _part_is_upper(part_oh: Node) -> Node:
    return indicator_to_bool(linear(part_oh, _PART_IS_UPPER_LINEAR))


@dataclass(frozen=True)
class ClipMemory:
    """Current column's vertical clip, defaulting to the open clip when unset.

    DOOM: ceilingclip[]/floorclip[] (r_plane.c) — per-column occlusion arrays
    read each R_RenderSegLoop column; walls mark both as fully opaque.

    ``ceiling``/``floor`` are the resolved values (recovered clip when
    ``present``, the open clip otherwise). The raw pick outputs and the
    presence flag are exposed so depth-critical consumers can compute
    their recovered-clip and open-clip variants in parallel and pick once
    on ``present`` at the end, instead of chaining behind the resolved
    selects (paint-cascade flatten). ``recovered_*`` are only meaningful
    where ``present`` is +1.
    """

    ceiling: Node
    floor: Node
    recovered_ceiling: Node
    recovered_floor: Node
    present: Node

    @classmethod
    @annotated("paint")
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        current_x_scalar: Node,
        clip_update_row: RecentMarkerHandle,
        cursor_x_scalar_pub: PastHandle,
    ) -> "ClipMemory":
        """Recover the current column's (ceiling, floor) clip, defaulting to the
        open clip ``(-1, VIEW_HEIGHT)`` when the column has no prior update.

        The per-column key is a radix (bucket, digit) pair of one-hots (was a
        width-(SCREEN_WIDTH+1) one-hot; width N_BUCKETS+B = 16/26). The dot has
        no orthogonal "no match" within a bucket, so the default fallback is
        recovered explicitly: each clip row carries its own column scalar, the
        recovered scalar is compared to the query column, and a mismatch (no real
        update for this column) selects the default open clip. The recency
        tiebreak in ``pick_most_recent`` makes the most recent update to a column
        win, and also prevents a symmetric blend of columns ``x-1`` / ``x+1``
        from averaging to ``x`` (distinct positions break the tie).
        """
        clip_ceiling_initial = constant(OPEN_CLIP_CEILING)
        clip_floor_initial = constant(float(VIEW_HEIGHT))
        absent_column = constant(_ABSENT_COLUMN)

        query_col = SCREEN_X_CLAMP(current_x_scalar)
        query = _radix_col_key(query_col)

        clip_x_col = clip_update_row.pick(past, cursor_x_scalar_pub)
        range_active = inp.screen_range_after_clip_update

        # Key gated to zero on non-clip-update rows: validity is carried by the
        # column scalar in the value (recovered + same_int below), not a sentinel
        # slot. A gated-zero key scores 0, below any real column's >= 1.
        clip_key = past.publish(
            "clip_range_key",
            gate(range_active, _radix_col_key(clip_x_col)),
        )
        clip_value = past.publish(
            "clip_range_value",
            concat(
                select(range_active, inp.screen_range_y1, clip_ceiling_initial),
                select(range_active, inp.screen_range_y2, clip_floor_initial),
                select(range_active, clip_x_col, absent_column),
            ),
        )
        recovered_ceiling, recovered_floor, recovered_col = split(
            past.pick_most_recent(
                query,
                clip_key,
                clip_value,
                match_gain=MATCH_GAIN_CLIP,
            ),
            [1, 1, 1],
        )
        present = same_int(recovered_col, query_col)
        return cls(
            ceiling=select(present, recovered_ceiling, clip_ceiling_initial),
            floor=select(present, recovered_floor, clip_floor_initial),
            recovered_ceiling=recovered_ceiling,
            recovered_floor=recovered_floor,
            present=present,
        )


@dataclass(frozen=True)
class WallColumnSpanValues:
    """Span visibility and y-bounds for the three wall tiers (mid/upper/lower).

    DOOM: R_RenderSegLoop wall-tier checks (r_segs.c) — per-tier visibility
    gates and y-ranges (yl, yh, pixhigh, pixlow).
    """

    middle_ok: Node
    upper_ok: Node
    lower_ok: Node
    middle_y1: Node
    middle_y2: Node
    upper_y1: Node
    upper_y2: Node
    lower_y1: Node
    lower_y2: Node


@dataclass(frozen=True)
class WallColumnPlaneRanges:
    ceiling_y1: Node
    ceiling_y2: Node
    floor_y1: Node
    floor_y2: Node


@dataclass(frozen=True)
class WallColumnState:
    """Values staged after a wall-column scale/ceiling pair.

    DOOM: R_RenderSegLoop (r_segs.c) per-column state — x, scale, clip bounds,
    ceiling/floor plane marks, and the three wall textures (mid/upper/lower);
    columns drawn by R_DrawColumn (r_draw.c).
    """

    row: RecentMarkerHandle
    current_ceiling_emit: Node
    current_floor_emit: Node
    current_x: Node
    current_ceiling_plane_id: Node
    current_floor_plane_id: Node
    x: PastHandle
    x_key: PastHandle
    span_state: PastHandle
    clip_range: PastHandle
    clip_changed: PastHandle
    floor_plane_id: PastHandle
    floor_emit: PastHandle
    plane_ranges: PastHandle

    @classmethod
    @annotated("paint/R_RenderSegLoop")
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        scene: SceneIndex,
        clip: ClipMemory,
        input_x_or_zero: PastHandle,
        input_x_key_or_zero: PastHandle,
        input_drawseg_scale_or_zero: PastHandle,
        range_seg_i: Node,
        plane_ids: PlaneIdLookup,
    ) -> "WallColumnState":
        """Stage per-column wall state at the SCREEN_Y_VALUE row after a scale pair.

        Phases below, in order:
        - recover this column's x / scale / staged new-ceiling from prior rows;
        - world-relative tier heights (top/bottom/high/low minus the view z);
        - middle-tier span yl/yh (front ceiling..floor), clamped by the clip arrays;
        - portal flag and the wall-clip markceiling/markfloor gates;
        - upper-tier y bounds;
        - lower-tier y bounds, then the portal-floor and new clip (ceiling/floor);
        - clip-changed decision (whether to emit a clip-update record);
        - plane-mark gates (one-sided/closed walls mark planes too);
        - ceiling plane-mark bounds + emit gate;
        - floor plane-mark bounds + emit gate;
        - span visibility flags + the clamped published y bounds;
        - assemble and publish the state handles.
        """
        center_y = constant(float(CENTER_Y))
        # The view bottom (DOOM viewheight): the closed-column ceiling clip and
        # the plane-mark "top below the view" cutoff. VIEW_HEIGHT == SCREEN_HEIGHT
        # when the status bar is off.
        screen_height = constant(float(VIEW_HEIGHT))
        screen_height_m1 = constant(float(VIEW_HEIGHT - 1))
        clip_ceiling_initial = constant(OPEN_CLIP_CEILING)
        one = constant(1.0)

        # --- Phase: recover this column's x / scale / staged new-ceiling ---
        row = RecentMarkerHandle.publish(
            past,
            "wall_column",
            inp.screen_y_after_wall_column_scale,
        )
        seg_i = range_seg_i
        # WallColumnState publishes at the SCREEN_Y_VALUE row that follows
        # SET_CURSOR_X -> WALL_COL_U -> VALUE(scale R5); the SET_CURSOR_X row is
        # at delta_pos=-3. Its x carries the SCREEN coordinate (col * PIXEL_WIDTH),
        # so recover the column (// PIXEL_WIDTH) BEFORE clamping to [0, COLUMN_COUNT):
        # this x becomes wall_column.x / current_x, which the column advance
        # increments and re-emits through screen_x_from_column, so it must be a
        # column. Identity in high-detail; mirrors seg_projection.cursor_x_scalar.
        # (On off-path speculative draft rows the picked value can be a FIND_RUN
        # column sentinel rather than a screen-x; // PIXEL_WIDTH halves it but it
        # stays in-range and off-path.)
        x = SCREEN_X_CLAMP(
            column_from_screen_x(past.attend_to_offset(input_x_or_zero, delta_pos=-3))
        )
        x_key = past.attend_to_offset(input_x_key_or_zero, delta_pos=-3)
        scale = SCALE_CLAMP(
            past.attend_to_offset(
                input_drawseg_scale_or_zero,
                delta_pos=-1,
            )
        )
        # staged_new_ceiling is the SCREEN_Y_VALUE emitted by
        # WallColumnRenderer.wall_column_new_ceiling_from_value; the worldtop/yl
        # recompute below is for the span/plane-mark math, not a second copy of
        # the clip bound.
        staged_new_ceiling = CLIP_Y_CLAMP(inp.screen_y)

        # --- Phase: world-relative tier heights (front/back, ceiling/floor) ---
        worldtop = sub(scene.segs.front_ceiling(seg_i), scene.view.z)
        worldbottom = sub(scene.segs.front_floor(seg_i), scene.view.z)
        worldhigh = sub(scene.segs.back_ceiling(seg_i), scene.view.z)
        worldlow = sub(scene.segs.back_floor(seg_i), scene.view.z)

        # --- Phase: middle-tier span yl/yh (front ceiling..floor), clip-clamped ---
        top_y_raw = sub(center_y, mul_height_scale(worldtop, scale))
        bot_y_raw = sub(center_y, mul_height_scale(worldbottom, scale))
        yl_unclipped = CEIL_Y(top_y_raw)
        # FLOOR_Y_WIDE handles the off-screen-above case for yh: when the
        # front floor is above the viewer's z (worldbottom positive),
        # bot_y_raw is very negative and yh should reflect that to make
        # le_span_y(yl, yh) correctly mark the middle span empty.
        yh_unclipped = FLOOR_Y_WIDE(bot_y_raw)
        # Two-variant clip clamp (paint-cascade flatten): the resolved
        # clip.ceiling/floor sit one select behind the clip pick, and the
        # sequential clamp (+-1 -> gt -> select) chained after them put the
        # span bounds four sublayers past the pick. Instead compute the
        # clamp against the recovered clip and against the open default in
        # parallel — each needs values available at the pick itself — and
        # resolve once on clip.present. gt_screen keeps the integer-aware
        # half-integer thresholds (the +0.5/-0.5 _ceilish/_floorish
        # comparators stay out, as before).
        rec_ceiling_min = add_const(clip.recovered_ceiling, 1.0)
        rec_floor_max = add_const(clip.recovered_floor, -1.0)
        open_ceiling_min = constant(float(OPEN_CLIP_CEILING) + 1.0)
        yl = select(
            clip.present,
            max_screen(yl_unclipped, rec_ceiling_min),
            max_screen(yl_unclipped, open_ceiling_min),
        )
        yh = select(
            clip.present,
            min_screen(yh_unclipped, rec_floor_max),
            min_screen(yh_unclipped, screen_height_m1),
        )

        # --- Phase: portal flag and wall-clip markceiling/markfloor gates ---
        portal = scene.segs.is_portal(seg_i)
        solid_or_closed = one_minus(portal)

        front_ceiling = scene.segs.front_ceiling(seg_i)
        back_ceiling = scene.segs.back_ceiling(seg_i)
        markceiling = and_(
            scene.segs.two_sided(seg_i),
            one_minus(same_int(back_ceiling, front_ceiling)),
        )
        markceiling = and_(markceiling, gt_height(front_ceiling, scene.view.z))

        front_floor = scene.segs.front_floor(seg_i)
        back_floor = scene.segs.back_floor(seg_i)
        markfloor = and_(
            scene.segs.two_sided(seg_i),
            one_minus(same_int(back_floor, front_floor)),
        )
        markfloor = and_(markfloor, gt_height(scene.view.z, front_floor))

        # --- Phase: upper-tier y bounds (front top down to back top) ---
        high_y_raw = sub(center_y, mul_height_scale(worldhigh, scale))
        # FLOOR_Y_WIDE keeps negative values when high_y_raw is below 0 (back
        # ceiling above viewer's horizon). The narrower FLOOR_Y would clamp
        # to 0 and lose the "upper region above screen -> invisible" signal
        # needed by `le_span_y(yl, upper_mid)`.
        upper_mid_unclipped = FLOOR_Y_WIDE(high_y_raw)
        # Same two-variant clip clamp as yl/yh above.
        upper_mid = select(
            clip.present,
            min_screen(upper_mid_unclipped, rec_floor_max),
            min_screen(upper_mid_unclipped, screen_height_m1),
        )
        upper_y1 = yl
        upper_y2 = upper_mid
        # --- Phase: lower-tier y bounds, then portal floor and new clip arrays ---
        lower_geom = gt_height(worldlow, worldbottom)
        low_y_raw = sub(center_y, mul_height_scale(worldlow, scale))
        lower_min = add_const(staged_new_ceiling, 1.0)
        # CEIL_Y_WIDE keeps positive values when low_y_raw is above
        # SCREEN_HEIGHT-1 (back floor below viewer's horizon -> lower
        # region below screen). The narrower CEIL_Y would clamp to
        # SCREEN_HEIGHT-1 and lose the "lower region below screen
        # -> invisible" signal needed by `le_span_y(lower_mid, yh)`.
        low_y_ceil = CEIL_Y_WIDE(low_y_raw)
        lower_mid = select(
            gt_screen(low_y_ceil, lower_min),
            low_y_ceil,
            lower_min,
        )
        lower_y1 = lower_mid
        lower_y2 = yh
        # Two-variant span-ok flag: le(lower_mid, yh) with
        # yh = min(yh_unclipped, eff_floor-1) expands exactly to
        # le(lower_mid, yh_unclipped) AND le(lower_mid, eff_floor-1), so
        # each clip variant is one conjunction over compares available at
        # the clip pick itself, resolved once on clip.present — instead of
        # chaining behind the resolved yh select.
        lower_ok_shared = le_span_y(lower_mid, yh_unclipped)
        lower_span_ok_value = select(
            clip.present,
            and_(lower_ok_shared, le_span_y(lower_mid, rec_floor_max)),
            and_(lower_ok_shared, le_span_y(lower_mid, screen_height_m1)),
        )
        lower_texture = scene.segs.lower_texture(seg_i)
        lower_textured = and_(lower_geom, lower_texture)
        lower_visible = and_(lower_textured, lower_span_ok_value)
        floor_if_lower_textured = select(
            lower_visible,
            lower_y1,
            add_const(yh, 1.0),
        )
        floor_if_lower_geom = select(
            scene.segs.lower_texture(seg_i),
            floor_if_lower_textured,
            select(markfloor, add_const(yh, 1.0), clip.floor),
        )
        portal_floor = select(
            lower_geom,
            floor_if_lower_geom,
            select(markfloor, add_const(yh, 1.0), clip.floor),
        )
        new_ceiling = staged_new_ceiling
        new_floor = select(
            solid_or_closed,
            clip_ceiling_initial,
            CLIP_Y_CLAMP(portal_floor),
        )
        # --- Phase: clip-changed decision (whether to emit a clip-update) ---
        solid_clip_same = and_(
            same_int(clip.ceiling, screen_height),
            same_int(clip.floor, clip_ceiling_initial),
        )
        # The reference emits a clip-update record for every portal column,
        # even when the updated values equal the prior clip arrays.
        clip_changed = select(
            solid_or_closed,
            one_minus(solid_clip_same),
            one,
        )

        # --- Phase: plane-mark gates (one-sided/closed walls mark planes too) ---
        # Plane-mark gates, mirroring `_render_wall_columns` in
        # the reference. The wall-clip `markceiling`/`markfloor` above
        # gate clip-array tightening for portal tiers; plane-mark gating
        # additionally fires for one-sided and closed walls.
        same_ceil = same_int(back_ceiling, front_ceiling)
        same_floor = same_int(back_floor, front_floor)
        markceiling_plane_base = one_minus(and_(portal, same_ceil))
        markfloor_plane_base = one_minus(and_(portal, same_floor))
        markceiling_plane = and_(
            markceiling_plane_base,
            gt_height(front_ceiling, scene.view.z),
        )
        markfloor_plane = and_(
            markfloor_plane_base,
            gt_height(scene.view.z, front_floor),
        )

        # --- Phase: ceiling plane-mark bounds + emit gate ---
        # Bounds (per reference._append_plane_marks):
        # Ceiling: top = ceilingclip+1; bottom = yl-1, clamped to floorclip-1.
        ceiling_top_raw = add_const(clip.ceiling, 1.0)
        ceiling_bottom_raw = add_const(yl, -1.0)
        floorclip_minus_one = add_const(clip.floor, -1.0)
        ceiling_bottom_ge_floor = gt_y_ceil_boundary(ceiling_bottom_raw, clip.floor)
        ceiling_bottom_clamped = select(
            ceiling_bottom_ge_floor,
            floorclip_minus_one,
            ceiling_bottom_raw,
        )
        ceiling_y1_published = SPAN_Y_CLAMP(ceiling_top_raw)
        ceiling_y2_published = SPAN_Y_CLAMP(ceiling_bottom_clamped)
        # Emit gate uses unclamped values: top<=99 AND bottom>=0 AND top<=bottom.
        ceiling_top_le_99 = one_minus(
            gt_y_floor_boundary(ceiling_top_raw, screen_height_m1)
        )
        ceiling_bottom_ge_0 = gt_y_floor_boundary(
            ceiling_bottom_clamped, clip_ceiling_initial
        )
        ceiling_top_le_bottom = one_minus(
            gt_screen(ceiling_y1_published, ceiling_y2_published)
        )
        ceiling_emit_value = and_(
            and_(markceiling_plane, ceiling_top_le_99),
            and_(ceiling_bottom_ge_0, ceiling_top_le_bottom),
        )

        # --- Phase: floor plane-mark bounds + emit gate ---
        # Floor: top = yh+1, clamped to ceilingclip+1; bottom = floorclip-1.
        floor_top_raw = add_const(yh, 1.0)
        ceilingclip_plus_one = add_const(clip.ceiling, 1.0)
        floor_top_le_ceiling = one_minus(
            gt_y_floor_boundary(floor_top_raw, clip.ceiling)
        )
        floor_top_clamped = select(
            floor_top_le_ceiling,
            ceilingclip_plus_one,
            floor_top_raw,
        )
        floor_bottom_raw = floorclip_minus_one
        floor_y1_published = SPAN_Y_CLAMP(floor_top_clamped)
        floor_y2_published = SPAN_Y_CLAMP(floor_bottom_raw)
        floor_top_le_99 = one_minus(
            gt_y_floor_boundary(floor_top_clamped, screen_height_m1)
        )
        floor_bottom_ge_0 = gt_y_floor_boundary(floor_bottom_raw, clip_ceiling_initial)
        floor_top_le_bottom = one_minus(
            gt_screen(floor_y1_published, floor_y2_published)
        )
        floor_emit_value = and_(
            and_(markfloor_plane, floor_top_le_99),
            and_(floor_bottom_ge_0, floor_top_le_bottom),
        )

        # --- Phase: span visibility flags + clamped published y bounds ---
        # (lower_span_ok_value is defined next to lower_mid above, in the
        # same two-variant form as everything here.)
        #
        # Flags: le(max(a, c), min(b, d)) expands exactly to the four
        # pairwise le's (both max arguments below both min arguments), so
        # each clip variant of a span-ok flag is one conjunction over
        # compares available at the clip pick, resolved once on
        # clip.present. All values are integers, so the half-integer
        # le thresholds are untouched.
        mid_ok_shared = le_span_y(yl_unclipped, yh_unclipped)
        middle_span_ok_value = select(
            clip.present,
            and_(
                and_(mid_ok_shared, le_span_y(yl_unclipped, rec_floor_max)),
                and_(
                    le_span_y(rec_ceiling_min, yh_unclipped),
                    le_span_y(rec_ceiling_min, rec_floor_max),
                ),
            ),
            and_(mid_ok_shared, le_span_y(open_ceiling_min, yh_unclipped)),
        )
        up_ok_shared = le_span_y(yl_unclipped, upper_mid_unclipped)
        upper_span_ok_value = select(
            clip.present,
            and_(
                and_(up_ok_shared, le_span_y(yl_unclipped, rec_floor_max)),
                and_(
                    le_span_y(rec_ceiling_min, upper_mid_unclipped),
                    le_span_y(rec_ceiling_min, rec_floor_max),
                ),
            ),
            and_(up_ok_shared, le_span_y(open_ceiling_min, upper_mid_unclipped)),
        )
        # Bounds: SPAN_Y_CLAMP is monotone, so it commutes with max/min —
        # clamp the two candidates first, then one gt+select per bound,
        # resolved on clip.present folded into the select cond. The open
        # variant needs no max/min at all: the open ceiling_min equals the
        # clamp's lower bound and the open floor_max equals its upper
        # bound, so the clamp subsumes them.
        cl_yl_u = SPAN_Y_CLAMP(yl_unclipped)
        cl_yh_u = SPAN_Y_CLAMP(yh_unclipped)
        cl_um_u = SPAN_Y_CLAMP(upper_mid_unclipped)
        cl_rec_c1 = SPAN_Y_CLAMP(rec_ceiling_min)
        cl_rec_f1 = SPAN_Y_CLAMP(rec_floor_max)
        middle_y1 = select(
            and_(clip.present, gt_screen(cl_rec_c1, cl_yl_u)),
            cl_rec_c1,
            cl_yl_u,
        )
        middle_y2 = select(
            and_(clip.present, gt_screen(cl_yh_u, cl_rec_f1)),
            cl_rec_f1,
            cl_yh_u,
        )
        upper_y1_published = middle_y1  # upper_y1 is yl; same clamped bound
        upper_y2_published = select(
            and_(clip.present, gt_screen(cl_um_u, cl_rec_f1)),
            cl_rec_f1,
            cl_um_u,
        )
        lower_y1_published = SPAN_Y_CLAMP(lower_y1)
        lower_y2_published = middle_y2  # lower_y2 is yh; same clamped bound
        new_floor_published = CLIP_Y_CLAMP(new_floor)

        # --- Phase: assemble and publish the state handles ---
        # Values needed by the two-step PLANE_MARK y emission path.
        return cls(
            row=row,
            current_ceiling_emit=ceiling_emit_value,
            current_floor_emit=floor_emit_value,
            current_x=x,
            current_ceiling_plane_id=plane_ids.ceiling_id,
            current_floor_plane_id=plane_ids.floor_id,
            x=past.publish("wall_column_x", x),
            x_key=past.publish("wall_column_x_key", x_key),
            span_state=past.publish(
                "wall_column_span_state",
                concat(
                    middle_span_ok_value,
                    upper_span_ok_value,
                    lower_span_ok_value,
                    middle_y1,
                    middle_y2,
                    upper_y1_published,
                    upper_y2_published,
                    lower_y1_published,
                    lower_y2_published,
                ),
            ),
            clip_range=past.publish(
                "wall_column_clip_range",
                concat(new_ceiling, new_floor_published),
            ),
            clip_changed=past.publish("wall_column_clip_changed", clip_changed),
            floor_plane_id=past.publish(
                "wall_column_floor_plane_id", plane_ids.floor_id
            ),
            floor_emit=past.publish("wall_column_floor_emit", floor_emit_value),
            plane_ranges=past.publish(
                "wall_column_plane_ranges",
                concat(
                    ceiling_y1_published,
                    ceiling_y2_published,
                    floor_y1_published,
                    floor_y2_published,
                ),
            ),
        )

    @annotated("paint")
    def pick(self, past: PastHandleScope, value: PastHandle) -> Node:
        return self.row.pick(past, value)

    @annotated("paint")
    def span_values(self, past: PastHandleScope) -> WallColumnSpanValues:
        values = split(self.pick(past, self.span_state), [1] * 9)
        return WallColumnSpanValues(*values)

    @annotated("paint")
    def clip_range_values(self, past: PastHandleScope) -> tuple[Node, Node]:
        y1, y2 = split(self.pick(past, self.clip_range), [1, 1])
        return y1, y2

    @annotated("paint")
    def plane_range_values(self, past: PastHandleScope) -> WallColumnPlaneRanges:
        values = split(self.pick(past, self.plane_ranges), [1, 1, 1, 1])
        return WallColumnPlaneRanges(*values)


@dataclass(frozen=True)
class SpanStartValues:
    """Per-wall-span init: y bounds, scaling, texture mid, height index, u, colormap.

    DOOM: R_RenderSegLoop / R_DrawColumn setup (r_segs.c, r_draw.c) —
    dc_texturemid, dc_iscale, per-tier y_start/height, texture height for modulo,
    pre-scaled v, lighting colormap row, and texture id.
    """

    y_start: Node
    height: Node
    dc_iscale: Node
    dc_texturemid: Node
    h_idx_oh: Node
    u_native: Node
    cmap_row: Node
    tex_id: Node
    ordinal: Node
    has_next: Node
    next_y: Node
    next_ordinal: Node


@dataclass(frozen=True)
class SpanV0Values:
    """Texture v at the top of a span before per-pixel stepping.

    DOOM: R_DrawColumn (r_draw.c) — v0_at_top is frac = dc_texturemid +
    (dc_yl - centery) * dc_iscale, computed once per span; pos is the screen
    coordinate, not texture.
    """

    pos: Node
    v0_at_top: Node


@dataclass(frozen=True)
class WallSpanRuntimeState:
    """Published wall-span handles and current column K-part y starts."""

    span_start_row: RecentMarkerHandle
    span_v0_row: RecentMarkerHandle
    span_start_state_pub: PastHandle
    span_v0_state_pub: PastHandle
    wallcol_k_y1_pub: PastHandle

    @annotated("paint")
    def span_start_values(self, past: PastHandleScope) -> SpanStartValues:
        values = split(
            self.span_start_row.pick(past, self.span_start_state_pub),
            [
                1,
                1,
                1,
                1,
                _H_IDX_OH_WIDTH,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        )
        return SpanStartValues(*values)

    @annotated("paint")
    def span_v0_values(self, past: PastHandleScope) -> SpanV0Values:
        return SpanV0Values(
            *split(self.span_v0_row.pick(past, self.span_v0_state_pub), [1, 1])
        )

    @annotated("paint")
    def wallcol_k_y1_values(
        self,
        past: PastHandleScope,
        wall_column: WallColumnState,
    ) -> tuple[Node, Node, Node]:
        return tuple(split(wall_column.pick(past, self.wallcol_k_y1_pub), [1, 1, 1]))


@dataclass(frozen=True)
class WallSpanRuntimeDraft:
    """Wall-span runtime drafted at the WALL_SPAN_META row.

    ``finish()`` gates the per-tier K-row y1 state at the SCREEN_Y_VALUE row and
    publishes ``wallcol_k_y1``; it does not read flat-pass state.
    """

    span_start_row: RecentMarkerHandle
    span_v0_row: RecentMarkerHandle
    span_start_state_pub: PastHandle
    span_v0_state_pub: PastHandle
    k0_y1_value: Node
    k1_y1_value: Node
    k2_y1_value: Node

    @classmethod
    @annotated("paint/R_RenderSegLoop")
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        pos: Node,
        scene: SceneIndex,
        seg_facts: SegLevelFacts,
        wall_column: WallColumnState,
        recent_drawseg_i: Node,
        wallcol_render_state: PastHandle,
    ) -> "WallSpanRuntimeDraft":
        part_sentinel = constant(PART_NONE)
        span_ordinal_0 = constant(0.0)
        span_ordinal_1 = constant(1.0)
        span_ordinal_2 = constant(2.0)

        # WALL_SPAN_META is the internal span marker. It precedes the
        # host-visible SET_CURSOR_Y(y), keeping cursor tokens simple while
        # span identity still crosses an embedding boundary.
        span_start_row = RecentMarkerHandle.publish(
            past,
            "span_start",
            inp.is_wall_span_meta,
        )
        span_v0_row = RecentMarkerHandle.publish(
            past,
            "span_v0",
            inp.is_value_after_set_cursor_y,
        )

        seg_i_active = recent_drawseg_i
        k_part_0 = seg_facts.K_part_0(seg_i_active)
        k_part_1 = seg_facts.K_part_1(seg_i_active)
        k_part_2 = seg_facts.K_part_2(seg_i_active)
        wall_span = wall_column.span_values(past)

        part_oh_0 = one_hot(k_part_0, 3)
        part_oh_1 = one_hot(k_part_1, 3)
        part_oh_2 = one_hot(k_part_2, 3)

        def y_start_for_part(part_oh_local: Node) -> Node:
            part_is_mid_local = _part_is_mid(part_oh_local)
            part_is_upper_local = _part_is_upper(part_oh_local)
            return select(
                part_is_mid_local,
                wall_span.middle_y1,
                select(part_is_upper_local, wall_span.upper_y1, wall_span.lower_y1),
            )

        k0_y1_value = y_start_for_part(part_oh_0)
        k1_y1_value = y_start_for_part(part_oh_1)
        k2_y1_value = y_start_for_part(part_oh_2)

        # --- Flat candidate-visibility masks (paint-cascade flatten; see
        # paint_cascade_plan.md, execution record) ---
        # A candidate k_j is visible iff it exists (k_part_j != PART_NONE)
        # and its part's span is ok. The old per-part has_* conjunction is
        # structural here: vocab._K_PART_TABLES only lists parts whose
        # has_* bit is set in ``pat``, so a non-sentinel k_part_j always
        # names a part that exists on this seg.
        #
        # m_j is a 0/1 slot mask over (mid, upper, lower), all-zero when
        # the candidate is absent. Everything below that depends on the
        # span-state read resolves through ONE gated slot-sum + compare —
        # the read-to-publish depth this replaces was the compiled floor's
        # binding chain (nested selects + two-layer or_).
        #
        # Degenerate rows: this state is built eagerly at every position;
        # on non-span rows k_part_j / the ordinal are junk, so the one-hots
        # can be fractional or all-zero. That is broadcast_select's
        # documented junk-mask contract (bounded blend), and the published
        # state is only read back at rows selected by the span_start_row
        # marker, which junk rows never carry.
        exists_1 = one_minus(same_int(k_part_1, part_sentinel))
        exists_2 = one_minus(same_int(k_part_2, part_sentinel))
        m_1 = gate(exists_1, part_oh_1)
        m_2 = gate(exists_2, part_oh_2)
        cursor_y = inp.wall_span_meta_y
        ordinal_at_span = inp.wall_span_meta_ordinal
        is_k0 = same_int(ordinal_at_span, span_ordinal_0)
        is_k1 = same_int(ordinal_at_span, span_ordinal_1)
        selected_part = select(
            is_k0,
            k_part_0,
            select(is_k1, k_part_1, k_part_2),
        )
        part_oh = one_hot(selected_part, 3)
        part_is_mid = _part_is_mid(part_oh)
        part_is_upper = _part_is_upper(part_oh)
        # Selected-part y bounds / height as one-hot picks instead of the
        # nested two-select ladder — the ladder was the binding chain
        # between the span-state read and this publish. On a real span row
        # ``part_oh`` is a clean one-hot (the ordinal is real there and the
        # next-ordinal chain only advances to existing parts); on junk rows
        # it is fractional/all-zero and the pick blends toward zero —
        # bounded, published, never read back (span_start_row marker). The
        # per-part height table entries are linear in the read, so they
        # fuse into the pick rather than trailing it.
        span_y_start_value = pick_by_one_hot(
            part_oh,
            concat(wall_span.middle_y1, wall_span.upper_y1, wall_span.lower_y1),
        )
        span_y_end_value = pick_by_one_hot(
            part_oh,
            concat(wall_span.middle_y2, wall_span.upper_y2, wall_span.lower_y2),
        )
        span_height_value = pick_by_one_hot(
            part_oh,
            concat(
                add_const(sub(wall_span.middle_y2, wall_span.middle_y1), 1.0),
                add_const(sub(wall_span.upper_y2, wall_span.upper_y1), 1.0),
                add_const(sub(wall_span.lower_y2, wall_span.lower_y1), 1.0),
            ),
        )

        # --- Flat next-span logic over the masks above ---
        # Candidates drawn after this span: k1 and k2 at ordinal 0, k2 at
        # ordinal 1, none at ordinal 2. ``marked_next`` marks their part
        # slots; the picked sum over marked slots of the ±1 ok flags is
        # 2·(#visible) − (#marked), so "at least one next candidate is
        # visible" is (picked sum + #marked) >= 2 — one compare with
        # threshold 1, margin 1 on both sides (softmax-recovery noise on
        # the ok flags is ~1e-3, far inside).
        oks = concat(wall_span.middle_ok, wall_span.upper_ok, wall_span.lower_ok)
        marked_k1 = gate(is_k0, m_1)
        marked_next = node_sum(marked_k1, gate(or_(is_k0, is_k1), m_2))
        span_has_next_value = compare(
            node_sum(pick_by_one_hot(marked_next, oks), reduce_sum(marked_next)),
            1.0,
        )
        # The next span is k1 exactly when this is ordinal 0 AND k1 is
        # visible (same single-compare shape, k1's slot only); otherwise k2
        # — which matches the old nested selects on every ordinal.
        choose_k1 = compare(
            node_sum(pick_by_one_hot(marked_k1, oks), reduce_sum(marked_k1)),
            1.0,
        )
        span_next_y_value = select(choose_k1, k1_y1_value, k2_y1_value)
        span_next_ordinal_value = select(choose_k1, span_ordinal_1, span_ordinal_2)

        span_dc_iscale_value, span_u_native_value, span_cmap_row_value = split(
            wall_column.pick(past, wallcol_render_state),
            [1, 1, 1],
        )

        dc_tmid_mid = seg_facts.dc_tmid_mid(seg_i_active)
        dc_tmid_upper = seg_facts.dc_tmid_upper(seg_i_active)
        dc_tmid_lower = seg_facts.dc_tmid_lower(seg_i_active)
        span_dc_texturemid_value = select(
            part_is_mid,
            dc_tmid_mid,
            select(part_is_upper, dc_tmid_upper, dc_tmid_lower),
        )
        h_idx_oh_mid = seg_facts.h_idx_oh_mid(seg_i_active)
        h_idx_oh_upper = seg_facts.h_idx_oh_upper(seg_i_active)
        h_idx_oh_lower = seg_facts.h_idx_oh_lower(seg_i_active)
        span_h_idx_oh_value = select(
            part_is_mid,
            h_idx_oh_mid,
            select(part_is_upper, h_idx_oh_upper, h_idx_oh_lower),
        )
        mid_tex_id = scene.segs.mid_tex_id(seg_i_active)
        upper_tex_id = scene.segs.upper_tex_id(seg_i_active)
        lower_tex_id = scene.segs.lower_tex_id(seg_i_active)
        span_tex_id_value = select(
            part_is_mid,
            mid_tex_id,
            select(part_is_upper, upper_tex_id, lower_tex_id),
        )

        span_start_state_pub = past.publish(
            "span_start_state",
            concat(
                cursor_y,
                span_height_value,
                span_dc_iscale_value,
                span_dc_texturemid_value,
                span_h_idx_oh_value,
                span_u_native_value,
                span_cmap_row_value,
                span_tex_id_value,
                ordinal_at_span,
                span_has_next_value,
                span_next_y_value,
                span_next_ordinal_value,
            ),
        )
        span_v0_state_pub = past.publish(
            "span_v0_state",
            concat(pos, inp.value_v3),
        )

        return cls(
            span_start_row=span_start_row,
            span_v0_row=span_v0_row,
            span_start_state_pub=span_start_state_pub,
            span_v0_state_pub=span_v0_state_pub,
            k0_y1_value=k0_y1_value,
            k1_y1_value=k1_y1_value,
            k2_y1_value=k2_y1_value,
        )

    @annotated("paint/R_RenderSegLoop")
    def finish(
        self,
        past: PastHandleScope,
        inp: ProtocolTokenView,
    ) -> WallSpanRuntimeState:
        zero = constant(0.0)
        wallcol_k0_y1_value = select(
            inp.screen_y_after_wall_column_scale,
            self.k0_y1_value,
            zero,
        )
        wallcol_k1_y1_value = select(
            inp.screen_y_after_wall_column_scale,
            self.k1_y1_value,
            zero,
        )
        wallcol_k2_y1_value = select(
            inp.screen_y_after_wall_column_scale,
            self.k2_y1_value,
            zero,
        )
        wallcol_k_y1_pub = past.publish(
            "wallcol_k_y1",
            concat(wallcol_k0_y1_value, wallcol_k1_y1_value, wallcol_k2_y1_value),
        )

        return WallSpanRuntimeState(
            span_start_row=self.span_start_row,
            span_v0_row=self.span_v0_row,
            span_start_state_pub=self.span_start_state_pub,
            span_v0_state_pub=self.span_v0_state_pub,
            wallcol_k_y1_pub=wallcol_k_y1_pub,
        )
