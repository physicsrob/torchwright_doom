"""Flat-pass publish state for the SegProjection ``flats`` subcontext.

A *flat* is DOOM's floor/ceiling texture (as opposed to a wall texture); the
flat pass fills floor/ceiling pixels. See ``GLOSSARY.md`` for the other coined
terms used here: ``subcontext`` (the named ``SegProjection`` slice this state
hangs off), ``span`` (a horizontal run of floor/ceiling pixels at one screen
row), and ``visplane`` / ``marker``.

Real-side port of the ``FlatPassState`` that owns two things:

1. **R_MakeSpans span open/close** (``r_plane.c:338-359``). Iterating columns
   ``minx..maxx+1``, it compares the previous column's coverage (``t1/b1``)
   against the current column's (``t2/b2``) and emits close/open row ranges.
   This uses the **raw** ``t1/b1/t2/b2`` in all four sub-steps (rather than
   threading updated ``t1_after``/``b1_after`` between them) and compensates with
   two ``*_non_empty`` guards; it holds because the four row-ranges are disjoint
   (checked by the equivalence test). The packed ``make_spans_state`` names
   ``slot0 = close_top`` / ``slot1 = close_bottom`` positionally.

2. **R_MapPlane affine cursor** (``r_plane.c:144-169``). ``(xfrac0, yfrac0)`` is
   the flat-texture ``(u, v)`` at the span's first pixel; ``(xstep, ystep)`` the
   per-screen-x increment; plus the distance-light colormap row.

The flat-pass control flow that drives the tokens this state feeds is mapped in
``flat_pass_renderer.py``.

Changes from the original: ``Vec`` -> ``Node``; ``...api`` -> ``.std``; the
``.ops`` shims -> ``.render_ops``; module-level ``constant`` / ``floor``
/ ``multiply`` / ``clamp`` nodes relocated inside ``publish()`` (no import-time
graph nodes); plain-list ``linear`` weight matrices stay at module level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.ops.arithmetic_ops import compare

from torchwright.graph import annotated

from .attention_handles import RecentMarkerHandle
from .doom_lighting import MAXLIGHTZ
from .past import PastHandle, PastHandleScope
from .render_constants import MATCH_GAIN_LONG
from .render_ops import (
    FLAT_DIST_INDEX_FLOOR,
    MARKER_PRESENT,
    _abs_coord,
    add_const,
    and_,
    column_from_screen_x,
    le_span_y,
    max_screen,
    min_screen,
    mul_dist_base,
    mul_dist_distscale,
    mul_len_trig,
    mul_ph_yslope,
    or_,
    same_int,
    snap_bool,
    sub,
)
from .std import (
    clamp,
    concat,
    constant,
    gate,
    linear,
    one_hot,
    pick_by_index,
    pick_by_one_hot,
    select,
    split,
)
from .std import sum as vec_sum
from .constants import COLUMN_COUNT, SCREEN_HEIGHT, SCREEN_WIDTH

from torchwright.graph import Node

if TYPE_CHECKING:
    from .protocol_tokens import ProtocolTokenView
    from .scene_index import SceneIndex
    from .visplane_state import RuntimeVisplaneState


# Plain-list weight matrices / raw table data — not graph nodes, fine at module
# level (the no-import-time-nodes rule applies only to ``Node``s).
_NEG_SCREEN_VECTOR = [
    [-1.0 if row == col else 0.0 for col in range(SCREEN_HEIGHT)]
    for row in range(SCREEN_HEIGHT)
]
_GE_Y_MATRIX = [
    [1.0 if y >= (idx - 1) else 0.0 for y in range(SCREEN_HEIGHT)]
    for idx in range(SCREEN_HEIGHT + 2)
]
_NEG1_LINEAR = [[-1.0]]
# Flat texture-step focal: DOOM's centerxfrac = viewwidth/2 = COLUMN_COUNT/2 (80
# in low-detail), the COLUMN-count half-width — NOT the lighting term just below,
# which keeps the full SCREEN_WIDTH (the §3 "evil twin" pair).
_FLAT_HALF_SCREEN_INV_X = [[1.0 / ((COLUMN_COUNT - 1) / 2.0)]]
_FLAT_DIST_DIV16_LINEAR = [[1.0 / 16.0]]
# DOOM: planezlight distance light terms (r_plane.c). Raw data; wrapped in
# constant() inside publish() for the pick_by_index table.
_FLAT_DIST_TERM_DATA = [
    float(-((SCREEN_WIDTH // 2) // (i + 1)) // 2) for i in range(MAXLIGHTZ)
]

# --- flat_span_x1 row-membership chunking ------------------------------------
#
# flat_span_x1 recovers, for a SPAN_ROW at row y, the column where the span
# covering y opened (DOOM spanstart[y]). The recency pick reads a per-opening
# SCREEN_HEIGHT-wide range-membership indicator at row y. That made the head's
# d_qk = (plane 32 + vp 8 + rows 50) + 1 = 91, far over the d_head=32 floor.
#
# A contiguous-range membership is NOT a one-hot equality, so it can't be radixed
# into one bilinear dot (range membership over a (bucket, digit) split needs the
# full bucket x digit grid). Instead split the row axis into K dense chunks of
# CHUNK rows each: chunk k carries the membership indicator restricted to rows
# [k*CHUNK, (k+1)*CHUNK), and the query one-hots row y only inside its own chunk
# (zero elsewhere). The K recency heads run in parallel; a select on y // CHUNK
# keeps the chunk that actually contains y. Each head's d_qk is CHUNK + 1.
#
# The (plane, vp) instance filter is DROPPED (recency alone is sufficient): within
# R_MakeSpans' per-visplane sequential processing, row y's only opening-indicator
# hit is at its spanstart column, and the visplane being closed now is more recent
# than any prior visplane, so the most-recent covering opening IS spanstart[y].
# The exact-math output is therefore identical to the filtered form.
_FLAT_SPAN_CHUNK = 25  # rows per chunk; CHUNK + 1 (recency) <= d_head=32
_N_FLAT_SPAN_CHUNKS = (SCREEN_HEIGHT + _FLAT_SPAN_CHUNK - 1) // _FLAT_SPAN_CHUNK
_FLAT_SPAN_CHUNK_SIZES = [
    min((k + 1) * _FLAT_SPAN_CHUNK, SCREEN_HEIGHT) - k * _FLAT_SPAN_CHUNK
    for k in range(_N_FLAT_SPAN_CHUNKS)
]


# DOOM: R_MakeSpans span-opening iteration (r_plane.c:328-359). Builds a
# screen-height indicator vector marking rows in range [lo, hi].
def _row_range_indicator(lo: Node, hi: Node) -> Node:
    ge_lo = linear(one_hot(add_const(lo, 1.0), SCREEN_HEIGHT + 2), _GE_Y_MATRIX)
    ge_after_hi = linear(
        one_hot(add_const(hi, 2.0), SCREEN_HEIGHT + 2),
        _GE_Y_MATRIX,
    )
    return vec_sum(ge_lo, linear(ge_after_hi, _NEG_SCREEN_VECTOR))


@dataclass(frozen=True)
class FlatVisplaneValues:
    p: Node
    vp: Node
    minx: Node
    maxx: Node


@dataclass(frozen=True)
class MakeSpansValues:
    x: Node
    slot0_valid: Node
    slot0_y1: Node
    slot0_y2: Node
    slot1_valid: Node
    slot1_y1: Node
    slot1_y2: Node
    is_sentinel: Node


@dataclass(frozen=True)
class ClosureValues:
    slot: Node
    y2: Node
    x_close: Node
    slot1_valid: Node
    make_x: Node
    is_sentinel: Node


@dataclass(frozen=True)
class FlatSpanValues:
    y: Node
    y_end: Node
    x1: Node
    x2: Node


@dataclass(frozen=True)
class FlatCursorValues:
    pos: Node
    xfrac0: Node
    yfrac0: Node
    xstep: Node
    ystep: Node
    flat_id: Node
    cmap_row: Node


@dataclass(frozen=True)
class FlatPassState:
    """Flat-pass publish handles and row-scoped values."""

    flat_visplane_row: RecentMarkerHandle
    flat_visplane_state_pub: PastHandle
    make_spans_row: RecentMarkerHandle
    make_spans_state_pub: PastHandle
    make_spans_opening_keys: tuple[PastHandle, ...]
    make_spans_opening_x: PastHandle
    make_slot0_valid: Node
    make_slot1_valid: Node
    closure_row: RecentMarkerHandle
    closure_state_pub: PastHandle
    flat_span_row: RecentMarkerHandle
    flat_span_state_pub: PastHandle
    flat_span_seen: Node
    flat_cursor_x_row: RecentMarkerHandle
    flat_cursor_state_pub: PastHandle

    @classmethod
    @annotated("plan/R_MapPlane")
    def publish(
        cls,
        past: PastHandleScope,
        inp: "ProtocolTokenView",
        scene: "SceneIndex",
        runtime_visplanes: "RuntimeVisplaneState",
        pos: Node,
    ) -> "FlatPassState":
        neg_one_value = constant(-1.0)
        span_close_slot_1 = constant(1.0)
        screen_height_value = constant(float(SCREEN_HEIGHT))
        zero_screen_rows = constant([0.0] * SCREEN_HEIGHT)

        flat_visplane_row = RecentMarkerHandle.publish(
            past,
            "flat_visplane",
            inp.is_flat_visplane_begin,
        )
        flat_minx_value = runtime_visplanes.min_x(
            past,
            inp.flat_visplane_p,
            inp.flat_visplane_vp,
        )
        flat_maxx_value = runtime_visplanes.max_x(
            past,
            inp.flat_visplane_p,
            inp.flat_visplane_vp,
        )
        flat_visplane_state_pub = past.publish(
            "flat_visplane_state",
            concat(
                inp.flat_visplane_p,
                inp.flat_visplane_vp,
                flat_minx_value,
                flat_maxx_value,
            ),
        )

        flat_p, flat_vp, flat_minx, flat_maxx = split(
            flat_visplane_row.pick(past, flat_visplane_state_pub),
            [1, 1, 1, 1],
        )
        make_x = inp.make_spans_x
        make_is_sentinel = same_int(make_x, add_const(flat_maxx, 1.0))
        make_is_minx = same_int(make_x, flat_minx)

        prev_x = add_const(make_x, -1.0)
        cur_col = runtime_visplanes.column_range(past, flat_p, flat_vp, make_x)
        prev_col = runtime_visplanes.column_range(past, flat_p, flat_vp, prev_x)
        cur_top_raw = cur_col.top
        cur_bottom_raw = cur_col.bottom
        prev_top_raw = prev_col.top
        prev_bottom_raw = prev_col.bottom
        # DOOM: R_MakeSpans span closure/opening logic (r_plane.c:338-359).
        # t1/b1 = previous column's top/bottom (sentinel at minx boundary);
        # t2/b2 = current column's top/bottom (sentinel at sentinel boundary).
        t1 = select(make_is_minx, screen_height_value, prev_top_raw)
        b1 = select(make_is_minx, neg_one_value, prev_bottom_raw)
        t2 = select(make_is_sentinel, screen_height_value, cur_top_raw)
        b2 = select(make_is_sentinel, neg_one_value, cur_bottom_raw)

        prev_non_empty = le_span_y(t1, b1)
        cur_non_empty = le_span_y(t2, b2)

        # RAW t1/b1/t2/b2 in all four sub-steps (not threaded t1_after/b1_after);
        # the *_non_empty guards compensate. Holds only because the four
        # row-ranges are disjoint (see equivalence test).
        close_top_lo = t1
        close_top_hi = min_screen(add_const(t2, -1.0), b1)
        close_top_valid = le_span_y(close_top_lo, close_top_hi)
        close_bottom_lo = max_screen(add_const(b2, 1.0), t1)
        close_bottom_hi = b1
        close_bottom_valid = and_(
            le_span_y(close_bottom_lo, close_bottom_hi),
            cur_non_empty,
        )

        open_top_lo = t2
        open_top_hi = min_screen(add_const(t1, -1.0), b2)
        open_top_valid = le_span_y(open_top_lo, open_top_hi)
        open_bottom_lo = max_screen(add_const(b1, 1.0), t2)
        open_bottom_hi = b2
        open_bottom_valid = and_(
            le_span_y(open_bottom_lo, open_bottom_hi),
            prev_non_empty,
        )

        opening_indicator = vec_sum(
            select(
                open_top_valid,
                _row_range_indicator(open_top_lo, open_top_hi),
                zero_screen_rows,
            ),
            select(
                open_bottom_valid,
                _row_range_indicator(open_bottom_lo, open_bottom_hi),
                zero_screen_rows,
            ),
        )
        opening_active = and_(
            inp.is_make_spans_col,
            or_(open_top_valid, open_bottom_valid),
        )
        # Split the SCREEN_HEIGHT-wide opening membership into K dense row-chunks
        # so each recency head fits d_head=32 (see _FLAT_SPAN_CHUNK). Each chunk
        # key is the membership restricted to that chunk's rows, gated to zero on
        # non-opening rows; the (plane, vp) filter is dropped (recency suffices).
        opening_indicator_chunks = split(opening_indicator, _FLAT_SPAN_CHUNK_SIZES)
        make_spans_opening_keys = tuple(
            past.publish(
                f"flat_opening_key_{k}",
                gate(opening_active, chunk),
            )
            for k, chunk in enumerate(opening_indicator_chunks)
        )
        make_spans_opening_x = past.publish("flat_opening_x", make_x)

        make_spans_row = RecentMarkerHandle.publish(
            past,
            "make_spans",
            inp.is_make_spans_col,
        )
        # Packed slot0 = close_top, slot1 = close_bottom positionally; the
        # recovered MakeSpansValues names them slot0/slot1.
        make_spans_state_pub = past.publish(
            "make_spans_state",
            concat(
                make_x,
                close_top_valid,
                close_top_lo,
                close_top_hi,
                close_bottom_valid,
                close_bottom_lo,
                close_bottom_hi,
                make_is_sentinel,
            ),
        )
        make_spans_state = MakeSpansValues(
            *split(make_spans_row.pick(past, make_spans_state_pub), [1] * 8)
        )

        slot_is_one = same_int(inp.span_close_slot, span_close_slot_1)
        picked_slot_y2 = select(
            slot_is_one,
            make_spans_state.slot1_y2,
            make_spans_state.slot0_y2,
        )
        closure_row = RecentMarkerHandle.publish(
            past,
            "flat_closure",
            inp.is_span_close_slot,
        )
        closure_state_pub = past.publish(
            "flat_closure_state",
            concat(
                inp.span_close_slot,
                picked_slot_y2,
                add_const(make_spans_state.x, -1.0),
                make_spans_state.slot1_valid,
                make_spans_state.x,
                make_spans_state.is_sentinel,
            ),
        )

        # One recency head per row-chunk: chunk k recovers x1 if row y is in chunk
        # k (its query one-hots y - k*CHUNK, zero outside the chunk), then a select
        # on y's chunk index keeps the head that actually contains y. The highest
        # chunk threshold y >= k*CHUNK that holds wins, which is exactly y // CHUNK.
        flat_span_x1_per_chunk = [
            past.pick_most_recent(
                one_hot(
                    add_const(inp.span_row_y, -k * _FLAT_SPAN_CHUNK),
                    _FLAT_SPAN_CHUNK_SIZES[k],
                ),
                key,
                make_spans_opening_x,
                match_gain=MATCH_GAIN_LONG,
            )
            for k, key in enumerate(make_spans_opening_keys)
        ]
        flat_span_x1_value = flat_span_x1_per_chunk[0]
        for k in range(1, _N_FLAT_SPAN_CHUNKS):
            flat_span_x1_value = select(
                compare(inp.span_row_y, k * _FLAT_SPAN_CHUNK - 0.5),
                flat_span_x1_per_chunk[k],
                flat_span_x1_value,
            )
        flat_span_row = RecentMarkerHandle.publish(
            past,
            "flat_span_row",
            inp.is_span_row,
        )
        closure_values_at_span = ClosureValues(
            *split(closure_row.pick(past, closure_state_pub), [1] * 6)
        )
        flat_span_state_pub = past.publish(
            "flat_span_state",
            concat(
                inp.span_row_y,
                closure_values_at_span.y2,
                flat_span_x1_value,
                closure_values_at_span.x_close,
            ),
        )
        flat_span_seen = MARKER_PRESENT(flat_span_row.pick(past, flat_span_row.marker))
        flat_cursor_x_active = and_(inp.is_set_cursor_x, flat_span_seen)
        flat_cursor_x_row = RecentMarkerHandle.publish(
            past,
            "flat_cursor_x",
            flat_cursor_x_active,
        )

        flat_yslope_pub = past.publish("flat_yslope", inp.span_row_yslope)
        yslope_picked = flat_span_row.pick(past, flat_yslope_pub)
        active_plane_id = flat_p
        plane_height_value = scene.planes.height(active_plane_id)
        plane_flat_id_value = scene.planes.flat_id(active_plane_id)
        # DOOM: abs(pl->height - viewz) (r_plane.c:427) — camera-to-flat distance.
        planeheight_value = _abs_coord(
            sub(plane_height_value, scene.view.z),
        )
        # DOOM: distance = FixedMul(planeheight, yslope[y]) (r_plane.c:144).
        distance_value = mul_ph_yslope(planeheight_value, yslope_picked)
        length_value = mul_dist_distscale(
            distance_value,
            inp.cursor_x_distscale,
        )
        cursor_x_oh = one_hot(column_from_screen_x(inp.cursor_x), COLUMN_COUNT)
        cos_angle_value = pick_by_one_hot(
            cursor_x_oh,
            scene.view.ray_x_by_screen,
        )
        sin_angle_value = pick_by_one_hot(
            cursor_x_oh,
            scene.view.ray_y_by_screen,
        )
        # DOOM: ds_xfrac = viewx + FixedMul(distance, finecosine[angle]) (r_plane.c:157-158).
        xfrac0_length_cos = mul_len_trig(length_value, cos_angle_value)
        xfrac0_native_value = vec_sum(scene.view.x, xfrac0_length_cos)
        yfrac0_length_sin = mul_len_trig(length_value, sin_angle_value)
        neg_view_y = linear(scene.view.y, _NEG1_LINEAR)
        yfrac0_native_value = sub(neg_view_y, yfrac0_length_sin)
        basexscale_value = linear(
            scene.view.angle_sin,
            _FLAT_HALF_SCREEN_INV_X,
        )
        baseyscale_value = linear(
            scene.view.angle_cos,
            _FLAT_HALF_SCREEN_INV_X,
        )
        # DOOM: ds_xstep/ds_ystep = FixedMul(distance, base{x,y}scale) (r_plane.c:145-146).
        xstep_native_value = mul_dist_base(distance_value, basexscale_value)
        ystep_native_value = mul_dist_base(distance_value, baseyscale_value)
        plane_light_static_value = scene.planes.light_static(active_plane_id)
        flat_distance_div16 = linear(
            distance_value,
            _FLAT_DIST_DIV16_LINEAR,
        )
        # DOOM: index = distance >> LIGHTZSHIFT; planezlight[index] (r_plane.c:164-169).
        flat_distance_index = FLAT_DIST_INDEX_FLOOR(flat_distance_div16)
        flat_distance_term = pick_by_index(
            flat_distance_index,
            constant(_FLAT_DIST_TERM_DATA),
            MAXLIGHTZ,
        )
        flat_colormap_row_value = clamp(
            vec_sum(plane_light_static_value, flat_distance_term),
            0.0,
            31.0,
        )
        flat_cursor_state_pub = past.publish(
            "flat_cursor_state",
            concat(
                pos,
                xfrac0_native_value,
                yfrac0_native_value,
                xstep_native_value,
                ystep_native_value,
                plane_flat_id_value,
                flat_colormap_row_value,
            ),
        )

        return cls(
            flat_visplane_row=flat_visplane_row,
            flat_visplane_state_pub=flat_visplane_state_pub,
            make_spans_row=make_spans_row,
            make_spans_state_pub=make_spans_state_pub,
            make_spans_opening_keys=make_spans_opening_keys,
            make_spans_opening_x=make_spans_opening_x,
            make_slot0_valid=close_top_valid,
            make_slot1_valid=close_bottom_valid,
            closure_row=closure_row,
            closure_state_pub=closure_state_pub,
            flat_span_row=flat_span_row,
            flat_span_state_pub=flat_span_state_pub,
            flat_span_seen=flat_span_seen,
            flat_cursor_x_row=flat_cursor_x_row,
            flat_cursor_state_pub=flat_cursor_state_pub,
        )

    @annotated("plan")
    def flat_visplane_values(self, past: PastHandleScope) -> FlatVisplaneValues:
        return FlatVisplaneValues(
            *split(
                self.flat_visplane_row.pick(past, self.flat_visplane_state_pub),
                [1] * 4,
            )
        )

    @annotated("plan/R_MakeSpans")
    def make_spans_values(self, past: PastHandleScope) -> MakeSpansValues:
        # pick_most_recent recovers the stored ±1 booleans with ~1e-3 softmax
        # noise; the consumers feed them into selects over wide emit *heads*,
        # where the approximate select amplifies a noisy cond by its large
        # offset M (~deviation·M) and flips the emitted token. Snap the boolean
        # fields back to a clean ±1 (the float64→float32 sharp-step discipline);
        # the y/x coordinate fields ride the slot quantization and stay raw.
        x, s0v, s0y1, s0y2, s1v, s1y1, s1y2, sent = split(
            self.make_spans_row.pick(past, self.make_spans_state_pub), [1] * 8
        )
        return MakeSpansValues(
            x=x,
            slot0_valid=snap_bool(s0v),
            slot0_y1=s0y1,
            slot0_y2=s0y2,
            slot1_valid=snap_bool(s1v),
            slot1_y1=s1y1,
            slot1_y2=s1y2,
            is_sentinel=snap_bool(sent),
        )

    @annotated("plan/R_MakeSpans")
    def closure_values(self, past: PastHandleScope) -> ClosureValues:
        slot, y2, x_close, s1v, make_x, sent = split(
            self.closure_row.pick(past, self.closure_state_pub), [1] * 6
        )
        return ClosureValues(
            slot=slot,
            y2=y2,
            x_close=x_close,
            slot1_valid=snap_bool(s1v),
            make_x=make_x,
            is_sentinel=snap_bool(sent),
        )

    @annotated("plan")
    def flat_span_values(self, past: PastHandleScope) -> FlatSpanValues:
        return FlatSpanValues(
            *split(self.flat_span_row.pick(past, self.flat_span_state_pub), [1] * 4)
        )

    @annotated("plan/R_MapPlane")
    def flat_cursor_values(self, past: PastHandleScope) -> FlatCursorValues:
        values = split(
            self.flat_cursor_x_row.pick(past, self.flat_cursor_state_pub),
            [1, 1, 1, 1, 1, 1, 1],
        )
        return FlatCursorValues(*values)
