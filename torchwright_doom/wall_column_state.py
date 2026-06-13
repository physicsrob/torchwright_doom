"""Wall-column runtime state for the SegProjection ``wall`` subcontext (Phase H).

Ported from ``doom_sandbox/implementation/forward/wall_column_state.py``. The
sandbox keeps a family of module-level ``constant(...)`` nodes (``_CENTER_Y``,
the clip sentinels, the span ordinals, ``ONE``/``ZERO``/``FALSE``); on the real
side a ``constant`` is a graph ``Node`` with a global auto-incrementing id, so
building one at import time aliases it under the test harness's node-id reset.
Every such node is therefore relocated **inside** the publish functions
(``global_node_id`` must be 0 after import). The plain-list ``linear`` matrices
stay at module level (they are raw arrays, not nodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


from torchwright.graph import Node

from .assets import _H_IDX_OH_WIDTH
from .attention_handles import RecentMarkerHandle
from .constants import CENTER_Y, SCREEN_HEIGHT
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
    gt_height,
    gt_screen,
    gt_y_ceil_boundary,
    gt_y_floor_boundary,
    le_span_y,
    mul_height_scale,
    one_minus,
    or_,
    radix_col_key as _radix_col_key,
    same_int,
    sub,
)
from .std import (
    concat,
    constant,
    gate,
    indicator_to_bool,
    linear,
    one_hot,
    select,
    split,
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
    """

    ceiling: Node
    floor: Node

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        current_x_scalar: Node,
        clip_update_row: RecentMarkerHandle,
        cursor_x_scalar_pub: PastHandle,
    ) -> "ClipMemory":
        """Recover the current column's (ceiling, floor) clip, defaulting to the
        open clip ``(-1, SCREEN_HEIGHT)`` when the column has no prior update.

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
        clip_floor_initial = constant(float(SCREEN_HEIGHT))
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
        center_y = constant(float(CENTER_Y))
        screen_height = constant(float(SCREEN_HEIGHT))
        screen_height_m1 = constant(float(SCREEN_HEIGHT - 1))
        clip_ceiling_initial = constant(OPEN_CLIP_CEILING)
        one = constant(1.0)

        row = RecentMarkerHandle.publish(
            past,
            "wall_column",
            inp.screen_y_after_wall_column_scale,
        )
        seg_i = range_seg_i
        # Clamp to legal screen-x: input_x_or_zero can carry sentinel x=160
        # from FIND_RUN (whose slot is IntSlot(0, SCREEN_WIDTH+1)).
        # Speculative draft-decode rows can dispatch through column-scoped
        # paths with current_x picked from such a sentinel; clamp so downstream
        # make_token calls with screen-x slots stay in range.
        # WallColumnState publishes at the SCREEN_Y_VALUE row that follows
        # SET_CURSOR_X -> WALL_COL_U -> VALUE(scale R5); the SET_CURSOR_X row is
        # at delta_pos=-3.
        x = SCREEN_X_CLAMP(past.attend_to_offset(input_x_or_zero, delta_pos=-3))
        x_key = past.attend_to_offset(input_x_key_or_zero, delta_pos=-3)
        scale = SCALE_CLAMP(
            past.attend_to_offset(
                input_drawseg_scale_or_zero,
                delta_pos=-1,
            )
        )
        staged_new_ceiling = CLIP_Y_CLAMP(inp.screen_y)

        worldtop = sub(scene.segs.front_ceiling(seg_i), scene.view.z)
        worldbottom = sub(scene.segs.front_floor(seg_i), scene.view.z)
        worldhigh = sub(scene.segs.back_ceiling(seg_i), scene.view.z)
        worldlow = sub(scene.segs.back_floor(seg_i), scene.view.z)

        top_y_raw = sub(center_y, mul_height_scale(worldtop, scale))
        bot_y_raw = sub(center_y, mul_height_scale(worldbottom, scale))
        yl_unclipped = CEIL_Y(top_y_raw)
        # FLOOR_Y_WIDE handles the off-screen-above case for yh: when the
        # front floor is above the viewer's z (worldbottom positive),
        # bot_y_raw is very negative and yh should reflect that to make
        # le_span_y(yl, yh) correctly mark the middle span empty.
        yh_unclipped = FLOOR_Y_WIDE(bot_y_raw)
        ceiling_min = add_const(clip.ceiling, 1.0)
        floor_max = add_const(clip.floor, -1.0)
        # With integer yl_unclipped/yh_unclipped, switch to integer-aware
        # gt_screen (threshold 0.5) instead of the y-boundary comparators
        # whose thresholds were tuned to absorb the +0.5/-0.5 _ceilish/_floorish
        # offsets.
        ceiling_unclipped_wins = gt_screen(yl_unclipped, ceiling_min)
        floor_clip_wins = gt_screen(yh_unclipped, floor_max)
        yl = select(ceiling_unclipped_wins, yl_unclipped, ceiling_min)
        yh = select(floor_clip_wins, floor_max, yh_unclipped)

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

        high_y_raw = sub(center_y, mul_height_scale(worldhigh, scale))
        # FLOOR_Y_WIDE keeps negative values when high_y_raw is below 0 (back
        # ceiling above viewer's horizon). The narrower FLOOR_Y would clamp
        # to 0 and lose the "upper region above screen -> invisible" signal
        # needed by `le_span_y(yl, upper_mid)`.
        upper_mid_unclipped = FLOOR_Y_WIDE(high_y_raw)
        upper_mid = select(
            gt_screen(upper_mid_unclipped, floor_max),
            floor_max,
            upper_mid_unclipped,
        )
        upper_y1 = yl
        upper_y2 = upper_mid
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
        lower_texture = scene.segs.lower_texture(seg_i)
        lower_textured = and_(lower_geom, lower_texture)
        lower_visible = and_(lower_textured, le_span_y(lower_y1, lower_y2))
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

        middle_span_ok_value = le_span_y(yl, yh)
        upper_span_ok_value = le_span_y(upper_y1, upper_y2)
        lower_span_ok_value = le_span_y(lower_y1, lower_y2)
        middle_y1 = SPAN_Y_CLAMP(yl)
        middle_y2 = SPAN_Y_CLAMP(yh)
        upper_y1_published = SPAN_Y_CLAMP(upper_y1)
        upper_y2_published = SPAN_Y_CLAMP(upper_y2)
        lower_y1_published = SPAN_Y_CLAMP(lower_y1)
        lower_y2_published = SPAN_Y_CLAMP(lower_y2)
        new_floor_published = CLIP_Y_CLAMP(new_floor)

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

    def pick(self, past: PastHandleScope, value: PastHandle) -> Node:
        return self.row.pick(past, value)

    def span_values(self, past: PastHandleScope) -> WallColumnSpanValues:
        values = split(self.pick(past, self.span_state), [1] * 9)
        return WallColumnSpanValues(*values)

    def clip_range_values(self, past: PastHandleScope) -> tuple[Node, Node]:
        y1, y2 = split(self.pick(past, self.clip_range), [1, 1])
        return y1, y2

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

    def span_v0_values(self, past: PastHandleScope) -> SpanV0Values:
        return SpanV0Values(
            *split(self.span_v0_row.pick(past, self.span_v0_state_pub), [1, 1])
        )

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
        false_ = constant(-1.0)

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
        has_mid = seg_facts.has_mid(seg_i_active)
        has_upper = seg_facts.has_upper(seg_i_active)
        has_lower = seg_facts.has_lower(seg_i_active)
        wall_span = wall_column.span_values(past)
        mid_visible = and_(has_mid, wall_span.middle_ok)
        upper_visible = and_(has_upper, wall_span.upper_ok)
        lower_visible = and_(has_lower, wall_span.lower_ok)

        def y_start_for_part(part_idx: Node) -> Node:
            part_oh_local = one_hot(part_idx, 3)
            part_is_mid_local = _part_is_mid(part_oh_local)
            part_is_upper_local = _part_is_upper(part_oh_local)
            return select(
                part_is_mid_local,
                wall_span.middle_y1,
                select(part_is_upper_local, wall_span.upper_y1, wall_span.lower_y1),
            )

        k0_y1_value = y_start_for_part(k_part_0)
        k1_y1_value = y_start_for_part(k_part_1)
        k2_y1_value = y_start_for_part(k_part_2)

        def part_visible_for(part_idx: Node) -> Node:
            part_oh_local = one_hot(part_idx, 3)
            part_is_mid_local = _part_is_mid(part_oh_local)
            part_is_upper_local = _part_is_upper(part_oh_local)
            visible = select(
                part_is_mid_local,
                mid_visible,
                select(part_is_upper_local, upper_visible, lower_visible),
            )
            exists = one_minus(same_int(part_idx, part_sentinel))
            return and_(exists, visible)

        k1_visible = part_visible_for(k_part_1)
        k2_visible = part_visible_for(k_part_2)
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
        span_y_start_value = select(
            part_is_mid,
            wall_span.middle_y1,
            select(part_is_upper, wall_span.upper_y1, wall_span.lower_y1),
        )
        span_y_end_value = select(
            part_is_mid,
            wall_span.middle_y2,
            select(part_is_upper, wall_span.upper_y2, wall_span.lower_y2),
        )
        span_height_value = add_const(sub(span_y_end_value, span_y_start_value), 1.0)

        more_after_k0 = or_(k1_visible, k2_visible)
        span_has_next_value = select(
            is_k0,
            more_after_k0,
            select(is_k1, k2_visible, false_),
        )
        next_y_after_k0 = select(k1_visible, k1_y1_value, k2_y1_value)
        span_next_y_value = select(is_k0, next_y_after_k0, k2_y1_value)
        next_ordinal_after_k0 = select(k1_visible, span_ordinal_1, span_ordinal_2)
        span_next_ordinal_value = select(
            is_k0,
            next_ordinal_after_k0,
            span_ordinal_2,
        )

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
