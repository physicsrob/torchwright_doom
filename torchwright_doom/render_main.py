"""Autoregressive renderer dispatch: the per-token forward pass.

``forward()`` builds the read-side ``SceneIndex`` + ``ProtocolTokenView``,
publishes the runtime protocol owners (BSP traversal, seg projection, and the
payload router), builds each branch's next-token, then dispatches by the
current input token's type.

**The output head (the fan-out fix).** The original dispatch sums one type-gated
*full* ``d_embed`` row per transition; at ~64 transitions that needs a huge
residual and will not compile. The port replaces it (see ``dispatch_next_token``):
branches are built as emit *heads* (``head_width()`` — the constant derived tail
dropped, ≈ 19 cols), transitions that select the same head are grouped with
their predicates OR-ed, the distinct gated heads are summed (see
``dispatch_next_token`` for the ``max_fanout`` depth/width trade-off), and one
shared ``emit_derived_zero`` is concatenated at the end. The full row is
byte-identical to a per-transition ``make_token`` row, so the teacher-forced
oracle is unchanged, and the dispatch width barely grows with the token count.

Changes from the original: ``Vec`` -> ``Node``; ``Past`` -> ``GraphPast``
/ ``PastHandleScope``; ``make_token`` -> ``make_token_head`` + a shared derived
tail; ``ForwardOutput`` -> the next-token ``Node`` returned directly. The
original's prefill-replay ``select`` is not ported (see ``dispatch_next_token``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from torchwright.graph import Node, PosEncoding
from torchwright.graph import annotate, annotated

from .bsp_traversal import BspTraversal
from .emit import emit_derived_zero
from .flat_pass_renderer import FlatPassRenderer
from .past import GraphPast, PastHandleScope
from .payload_router import PayloadRouter
from .pixel_dispatcher import PixelDispatcher
from .psprite_renderer import PspriteRenderer
from .protocol_registry import DISPATCH_TRANSITIONS
from .protocol_tokens import ProtocolTokenView
from .render_ops import _ATAN_ABS_RANGE, signed_world_angle
from .scene_index import SceneIndex
from .seg_projection import SegProjection
from .range_dispatcher import RangeDispatcher
from .seg_scanner import SegScanner
from .visplane_marker import VisplaneMarker
from .wall_column_renderer import WallColumnRenderer
from .solid_intervals import SolidIntervals
from .std import (
    AngleInputEmit,
    ScalarEmit,
    angle_scalar,
    bool_or,
    bool_to_01,
    clamp,
    clamp_to_slot,
    concat,
    make_token_head,
    pick_by_one_hot,
    select,
    type_switch,
)
from .vocab import ANGLE_VALUE, DONE, NO_OP, SET_CURSOR_DIRECTION_Y
from .wall_range_builder import WallRangeBuilder

BranchOutputs = Mapping[str, Node]


def _dispatch_head(token_type) -> Node:
    """A dispatch-owned (registry owner "main") inert/begin/done emit head,
    annotated ``dispatch`` so its emit nodes carry the router's provenance."""
    with annotate("dispatch"):
        from .std import make_token_head as _make_token_head

        return _make_token_head(token_type)


@dataclass(frozen=True)
class RuntimeProtocols:
    """Published protocol owners needed by branch construction: the BSP traversal
    (which also publishes the ``BBoxPruner`` for R_CheckBBox visibility pruning),
    the seg projection, and the payload router that routes numeric-carrier tokens
    by the preceding marker."""

    traversal: BspTraversal
    projection: SegProjection
    payload_router: PayloadRouter


def publish_runtime_protocols(
    input_vec: Node,
    past: PastHandleScope,
    inp: ProtocolTokenView,
    scene: SceneIndex,
    pos: Node,
) -> RuntimeProtocols:
    """Publish runtime protocol channels before branch candidates consume them.

    Order matters: ``input_angle_or_zero`` and ``SolidIntervals`` are published
    before the owners that read them. Both (plus ``input_vec``) are threaded into
    ``BspTraversal.publish`` so its ``BBoxPruner`` can scan the occlusion state,
    and the payload router is built with the bbox arm
    (``is_bbox_angle`` -> ``BBoxPruner.after_bbox_angle_value``)."""
    with annotate("input"):
        input_angle_or_zero = past.publish(
            "input_angle_or_zero",
            ANGLE_VALUE.extract(input_vec, "angle"),
        )
    solids = SolidIntervals.publish(past, inp, scene)
    traversal = BspTraversal.publish(
        past,
        input_vec,
        inp,
        scene,
        solids,
        input_angle_or_zero,
    )
    projection = SegProjection.publish(
        past,
        input_vec,
        inp,
        scene,
        solids,
        input_angle_or_zero,
        pos,
    )
    payload_router = PayloadRouter(projection=projection, bbox=traversal.bbox)
    return RuntimeProtocols(
        traversal=traversal,
        projection=projection,
        payload_router=payload_router,
    )


@annotated("dispatch")
def dispatch_next_token(
    inp: ProtocolTokenView,
    branches: BranchOutputs,
) -> Node:
    """Select the one branch's next-token that matches the current input token.

    This is the transformer's output head. The original's ``type_switch`` sums
    one type-gated *full* ``d_embed`` row **per transition** — at ~64
    transitions that needs a huge residual to compile. Two changes shrink it to
    a handful of narrow columns while staying a single flat sum (no deep
    ``select`` chain — that compiles to dozens of sequential layers and overflows
    ``reference_eval``'s recursion):

    1. **Gate over heads, stamp the tail once.** Each branch is built at
       ``head_width()`` (the constant derived tail dropped — ≈ 19 cols under the
       shared-slot-column embedding); the sum runs over heads and one shared
       :func:`emit_derived_zero` is concatenated afterward. The result is
       byte-identical to the matching branch's full ``make_token`` row, so the
       teacher-forced oracle is unchanged.
    2. **One gated copy per *distinct* head, not per transition.** Every
       transition's predicate is read off ``inp``; transitions that select the
       *same* head node (all the deferred branches share one ``no_op`` head) are
       grouped and their predicates OR-ed, so the sum has one term per distinct
       next-token head (~8 this phase) rather than 64. Because exactly one
       transition predicate is +1, exactly one grouped predicate is +1.

    **Not ported:** the original also wraps dispatch in a prefill-replay
    ``select`` so prefill rows re-emit their input verbatim (a render-trace
    nicety; those emissions are discarded and ``BEGIN`` — the AR seed — is not
    replayed). That replay is **deferred** (which is why this function takes no
    ``input_vec``): it would select the full input row, whose W_EMBED rows span
    a ~100x dynamic range, and a single input ``value_type`` can't satisfy both
    the read-side interval arithmetic and the whole-row replay guard. The AR
    rollout and the oracle are unaffected (prefill emissions are discarded; the
    AR seed comes from the ``begin`` branch).
    """
    # max_fanout bounds how many gated heads are summed per accumulator step:
    # smaller = fewer copies live at once (narrower residual), larger = a shallower
    # reduction tree (fewer layers). Measured on the e1m1 forward (scripts/
    # analyze_forward_cost.py): the reduction is the dominant depth driver —
    # fanout=2 (a serial accumulator) costs ~35 layers (66 total), fanout=8 a
    # shallow tree ~22 (44 total), flat ~10 (32 total). The width cost is small
    # (each head is ~19 cols + its inputs; +~3% peak from 2→8). 8 captures most of
    # the depth win at a residual that still fits a modest d. The reassociation is
    # output-identical (scripts/check_fanout_equivalence.py: 0 next-token diffs).
    head = type_switch(*_distinct_head_pairs(inp, branches), max_fanout=8)
    return concat(head, emit_derived_zero())


@annotated("dispatch")
def _group_by_value(pairs: list[tuple[Node, Node]]) -> list[tuple[Node, Node]]:
    """Collapse ``(value, ±1-predicate)`` pairs sharing the SAME value node into
    one ``(value, predicate)`` entry per distinct value — first-seen order, the
    grouped predicate being the OR of the pairs that share that value.

    The shared core of both dispatch picks: :func:`_distinct_head_pairs` groups the
    emit *heads* a transition selects, and :func:`_collapse_scalar_emits` groups a
    carrier's branch *scalars* (the world-angle pre-pass points all four world
    branches at one shared atan, so their four pairs collapse to one). Because
    exactly one input predicate is +1 per position, the OR is +1 iff any grouped
    pair fires — one-hot is preserved."""
    groups: dict[int, tuple[Node, list[Node]]] = {}
    order: list[int] = []
    for value, pred in pairs:
        key = id(value)
        if key not in groups:
            groups[key] = (value, [])
            order.append(key)
        groups[key][1].append(pred)
    return [
        (value, preds[0] if len(preds) == 1 else bool_or(*preds))
        for value, preds in (groups[key] for key in order)
    ]


@annotated("dispatch")
def _distinct_head_pairs(
    inp: ProtocolTokenView,
    branches: BranchOutputs,
) -> tuple[tuple[Node, Node], ...]:
    """Group transitions by the head node they select, OR-ing the predicates of
    transitions that share a head. Returns one ``(predicate, head)`` pair per
    distinct head, in first-seen order — the input to ``type_switch``."""
    pairs = [
        (branches[t.branch], getattr(inp, t.predicate)) for t in DISPATCH_TRANSITIONS
    ]
    return tuple((pred, head) for head, pred in _group_by_value(pairs))


# DOOM: R_RenderPlayerView (r_main.c) — top-level per-frame render dispatch
def forward(
    input_vec: Node, past: GraphPast, pos: PosEncoding, asset_index=None
) -> Node:
    # Provenance: each subsystem call below re-annotates to its own TOP-LEVEL code
    # (SceneIndex.build -> `scene`, the publish/branch builders -> their owners,
    # dispatch_next_token -> `dispatch`). forward() therefore does NOT wrap the
    # whole body in one label (that would nest every subsystem under it); it only
    # labels its own glue: the current-token decode block (`input`) and the bare
    # wiring node (the position scalar -> `dispatch`).
    #
    # `SceneIndex.build` publishes scene channels through the bare `GraphPast`;
    # the `PastHandleScope` wrap below is intentional and must follow it. It is a
    # distinct local (`scope`) rather than a rebind of `past` so the two types
    # stay separable.
    scene = SceneIndex.build(input_vec, past, pos, assets=asset_index)
    # Current-token decode + memory-fetch handles (the `input` subsystem): the
    # input-type code, the two previous-type reads, and the typed token view.
    with annotate("input"):
        scope = PastHandleScope(past)
        prev_input_type = scope.attend_to_offset(scope.input_type(), delta_pos=-1)
        prev_prev_input_type = scope.attend_to_offset(scope.input_type(), delta_pos=-2)
        inp = ProtocolTokenView(
            input_vec,
            prev_input_type,
            prev_prev_input_type,
        )

    # The projection texel path (Phase J) reads the position as a *scalar* value
    # (the original's ``pos`` is a 1-Vec): pixel_index = pos - span_v0.pos - 1, and the
    # span-v0 / flat-cursor publishes stamp it. Extract the raw integer counter
    # column from the PosEncoding (``SceneIndex.build``'s ``pos`` is unused).
    with annotate("dispatch"):
        pos_scalar = pos.get_position_scalar()
    protocols = publish_runtime_protocols(input_vec, scope, inp, scene, pos_scalar)
    branches = build_branch_outputs(inp, protocols)
    return dispatch_next_token(inp, branches)


def build_branch_outputs(
    inp: ProtocolTokenView, protocols: RuntimeProtocols
) -> dict[str, Node]:
    """Build each dispatch branch's next-token head from its owner.

    The dict's keys equal ``{t.branch for t in DISPATCH_TRANSITIONS}`` (so the
    dispatch is well-formed). Each branch routes to its owner: the traversal /
    begin / done heads; the seg-scan loop (``visit`` … ``emit_x2``) and the
    drawseg-scalar chain (``store_wall_range`` … ``drawseg_u_phase``) to the
    ``SegScanner`` / ``WallRangeBuilder``; the ``value`` / ``angle`` carriers to
    the ``PayloadRouter``; the R_CheckBBox sub-protocol (``between`` + ``bbox_*``)
    to the ``BBoxPruner`` (via the ``BspTraversal`` delegators); and the
    wall-column / visplane / flat / pixel owners. Any registry branch left without
    an explicit head falls back to the shared NO_OP head via the ``setdefault``
    loop.

    Values are emit *heads* (``make_token_head`` / ``after_*``), not full rows;
    the dispatch stamps the shared derived tail after selection. Numeric-carrier
    branches (``VALUE`` / ``ANGLE_VALUE``) instead return a :class:`ScalarEmit`
    (just their 1-wide scalar); the four world-angle branches return an
    :class:`AngleInputEmit` (just their atan2 ``(dx, dy)``). Two collapse passes
    fold these back to ``branch -> head Node`` before returning:
    :func:`_collapse_world_angle_inputs` routes the four deferred atan inputs to
    ONE shared :func:`signed_world_angle` (re-wrapped as an ``ANGLE_VALUE``
    ``ScalarEmit``), then :func:`_collapse_scalar_emits` folds every carrier's
    branches into ONE shared digit-quad head.
    """
    # The inert / begin / done heads are dispatch-owned (registry owner "main").
    with annotate("dispatch"):
        no_op_out = make_token_head(NO_OP)
    traversal = protocols.traversal
    projection = protocols.projection
    payload_router = protocols.payload_router
    seg_scan = SegScanner(projection)
    wall_range = WallRangeBuilder(projection)
    wall_cols = WallColumnRenderer(projection)
    pixels = PixelDispatcher(projection)
    flats = FlatPassRenderer(projection)
    weapon = PspriteRenderer(projection)
    visplanes = VisplaneMarker(projection)
    ranges = RangeDispatcher(projection)
    branches: dict[str, "Node | ScalarEmit | AngleInputEmit"] = {
        # Inert / begin (dispatch-owned, registry owner "main").
        "no_op": no_op_out,
        "done": _dispatch_head(DONE),
        "begin": _dispatch_head(SET_CURSOR_DIRECTION_Y),
        # BSP traversal (BspTraversal).
        "think": traversal.after_think_side(),
        "side_record": traversal.after_side_record(),
        "enter": traversal.after_enter(),
        "between": traversal.after_between(),
        "return_": traversal.after_return(),
        # SET_CURSOR_DIRECTION_Y is emitted both at frame start (BEGIN) and at the
        # weapon phase start (R_DrawPlayerSprites); fork on weapon_seen so the
        # weapon arm walks to its first column instead of starting the BSP walk.
        "set_cursor_direction_y": select(
            projection.flats.weapon.weapon_seen,
            weapon.after_set_cursor_direction_y_weapon(),
            traversal.after_set_cursor_direction_y(),
        ),
        # R_CheckBBox visibility pruning (BBoxPruner via BspTraversal).
        "bbox_boxpos": traversal.after_bbox_boxpos(),
        "bbox_corner_x_a": traversal.after_bbox_corner_x_mark_a(),
        "bbox_corner_y_a": traversal.after_bbox_corner_y_mark_a(),
        "bbox_corner_x_b": traversal.after_bbox_corner_x_mark_b(),
        "bbox_corner_y_b": traversal.after_bbox_corner_y_mark_b(),
        "bbox_world_a": traversal.after_bbox_world_angle_mark_a(),
        "bbox_theta_a": traversal.after_bbox_theta_mark_a(),
        "bbox_world_b": traversal.after_bbox_world_angle_mark_b(),
        "bbox_theta_b": traversal.after_bbox_theta_mark_b(),
        "bbox_scan": traversal.after_bbox_scan(),
        # Payload carriers (PayloadRouter, routed by previous marker).
        "value": payload_router.after_value(no_op_out),
        "angle": payload_router.after_angle_value(no_op_out),
        # Seg scan + endpoint projection (SegScanner).
        "visit": seg_scan.after_visit_subsector(),
        "process": seg_scan.after_process_seg(),
        "find_run": seg_scan.after_find_run(),
        "world_a": seg_scan.after_world_angle_mark_a(),
        "theta_a": seg_scan.after_theta_mark_a(),
        "world_b": seg_scan.after_world_angle_mark_b(),
        "theta_b": seg_scan.after_theta_mark_b(),
        "advance_seg": seg_scan.after_advance_seg(),
        "emit_x2": seg_scan.after_emit_x2(),
        # Drawseg / wall-range setup (WallRangeBuilder).
        "store_wall_range": wall_range.after_store_wall_range(),
        "seg_kpart": wall_range.after_seg_kpart(),
        "seg_dc_tmid_mid": wall_range.after_seg_dc_tmid_mid(),
        "seg_dc_tmid_upper": wall_range.after_seg_dc_tmid_upper(),
        "seg_dc_tmid_lower": wall_range.after_seg_dc_tmid_lower(),
        "drawseg_meta": wall_range.after_drawseg_meta(),
        "drawseg_scale1_den": wall_range.after_drawseg_scale1_den(),
        "drawseg_scale1": wall_range.after_drawseg_scale1(),
        "drawseg_scale2_den": wall_range.after_drawseg_scale2_den(),
        "drawseg_scale2": wall_range.after_drawseg_scale2(),
        "drawseg_scalestep_den": wall_range.after_drawseg_scalestep_den(),
        "drawseg_scalestep": wall_range.after_drawseg_scalestep(),
        "drawseg_bsilheight": wall_range.after_drawseg_bsilheight(),
        "drawseg_tsilheight": wall_range.after_drawseg_tsilheight(),
        "drawseg_u_phase": wall_range.after_drawseg_u_phase(),
        # Visplane check + plane marks (VisplaneMarker).
        "r_check_plane": visplanes.after_r_check_plane(),
        "r_check_plane_result": visplanes.after_r_check_plane_result(),
        "plane_mark": visplanes.after_plane_mark(),
        # Wall columns / spans / clip (WallColumnRenderer).
        "wall_col_u": wall_cols.after_wall_col_u(),
        "wall_span_meta": wall_cols.after_wall_span_meta(),
        "clip_update": wall_cols.after_clip_update(),
        "screen_y": wall_cols.after_screen_y_value(no_op_out),
        # Wall + flat pixels (PixelDispatcher). The three shared
        # pixel/cursor branches fork on flat_span_seen (wall arm / flat arm).
        "wall_column": pixels.after_wall_column(),
        "set_cursor_y": pixels.after_set_cursor_y(),
        "pixel_color": pixels.after_pixel_color(),
        # Flat floor/ceiling span pass (FlatPassRenderer).
        "draw_planes_begin": flats.after_draw_planes_begin(),
        "set_cursor_direction_x": flats.after_set_cursor_direction_x(),
        "flat_next_plane": flats.after_flat_next_plane(),
        "flat_next_vp": flats.after_flat_next_vp(),
        "flat_visplane_begin": flats.after_flat_visplane_begin(),
        "make_spans_col": flats.after_make_spans_col(),
        "span_close_slot": flats.after_span_close_slot(),
        "span_row": flats.after_span_row(),
        # Player weapon pass (PspriteRenderer). The phase-begin head; the rest of
        # the weapon walk rides the shared SET_CURSOR_* / PIXEL branches above,
        # forked on weapon_seen.
        "draw_psprites_begin": weapon.after_draw_psprites_begin(),
        # Host-visible screen-range merge (RangeDispatcher).
        "screen_range": ranges.after_screen_range(no_op_out),
    }
    # Any registry branch not built above falls back to the shared NO_OP head,
    # keeping the dispatch well-formed.
    for transition in DISPATCH_TRANSITIONS:
        branches.setdefault(transition.branch, no_op_out)
    # Route the four world-angle branches' deferred atan inputs to one shared
    # atan (4 atans -> 1), then collapse every numeric carrier to one head.
    collapsed = _collapse_world_angle_inputs(inp, branches)
    return _collapse_scalar_emits(inp, collapsed)


@annotated("dispatch")
def _branch_predicates(inp: ProtocolTokenView) -> dict[str, list[Node]]:
    """Map each dispatch branch to its transition predicate node(s), read off
    ``inp`` exactly as :func:`_distinct_head_pairs` does.

    A branch reached by more than one transition accumulates one predicate per
    transition (the caller ORs them). Shared by both collapse passes
    (:func:`_collapse_world_angle_inputs`, :func:`_collapse_scalar_emits`) so they
    mask on identical predicates."""
    branch_preds: dict[str, list[Node]] = defaultdict(list)
    for transition in DISPATCH_TRANSITIONS:
        branch_preds[transition.branch].append(getattr(inp, transition.predicate))
    return branch_preds


@annotated("dispatch")
def _one_hot_predicate(branch_preds: Mapping[str, list[Node]], name: str) -> Node:
    """The single ±1 dispatch predicate for branch ``name`` — the OR of every
    transition that selects it. Exactly one branch's predicate is +1 per
    position, so the masks built from these are one-hot at the active row."""
    conds = branch_preds[name]
    assert conds, f"branch {name!r} has no dispatch-transition predicate"
    return conds[0] if len(conds) == 1 else bool_or(*conds)


def _collapse_world_angle_inputs(
    inp: ProtocolTokenView,
    branches: Mapping[str, "Node | ScalarEmit | AngleInputEmit"],
) -> dict[str, "Node | ScalarEmit"]:
    """Route the four world-angle branches' deferred atan2 inputs to ONE shared
    :func:`signed_world_angle` — the 4-atans-to-1 collapse (see the module note in
    ``emit.py``). ``signed_world_angle`` is the widest computation in
    ``forward()``; only one of the four world-angle branches is ever the live
    output, so three of the four atans are pure waste.

    Each world-angle branch (``world_a`` / ``world_b`` / ``bbox_world_a`` /
    ``bbox_world_b``) returns its endpoint's ``(dx, dy)`` wrapped in an
    :class:`AngleInputEmit` instead of running its own atan. Exactly one of the
    four branch predicates is +1 per position (mutual exclusivity), so the active
    branch's ``(dx, dy)`` is picked float-exact (``pick_by_one_hot``,
    ``approximate=False``) and a single ``signed_world_angle`` runs over the
    winning pair. The angle is re-wrapped as an ``ANGLE_VALUE`` :class:`ScalarEmit`
    at all four branch keys, so the downstream :func:`_collapse_scalar_emits` folds
    them into the one shared ANGLE_VALUE head exactly as before. Byte-identical at
    the active row; 8 ray-halves (4 sites × tan/cot) → 2.

    The discriminator is ``isinstance(out, AngleInputEmit)`` — **not** the
    ``"projection_angle"`` / ``"bbox_angle"`` payload group, which also tags the
    four *theta* branches. Theta reads the emitted world angle back and applies
    ``wrap_signed_angle`` (not an atan), so it must stay its own ``ScalarEmit``.
    """
    members = [
        (name, out) for name, out in branches.items() if isinstance(out, AngleInputEmit)
    ]
    if not members:
        return dict(branches)

    # Branch-selection glue (mask + per-candidate clamp + the float-exact pick of
    # the active branch's (dx, dy)) is dispatch work.
    with annotate("dispatch"):
        branch_preds = _branch_predicates(inp)
        mask = concat(
            *[
                bool_to_01(_one_hot_predicate(branch_preds, name))
                for name, _e in members
            ]
        )

        # Clamp each candidate (dx, dy) to the atan's ±3072 square BEFORE the pick.
        # The raw `dx = sub(vertex, view)` carries a tracked value_range wider than
        # 3072, which would blow `broadcast_select`'s additive offset past its
        # sanity bound. Byte-identical at the active row: `signed_world_angle`
        # re-clamps to the same ±3072 internally via `_abs_coord`, so the clamp is
        # idempotent there.
        dxs = [
            clamp(emit.dx, -_ATAN_ABS_RANGE, _ATAN_ABS_RANGE) for _n, emit in members
        ]
        dys = [
            clamp(emit.dy, -_ATAN_ABS_RANGE, _ATAN_ABS_RANGE) for _n, emit in members
        ]
        # Two d_fill=1 picks sharing the SAME mask — simpler than one d_fill=2 pick
        # + split (no slot-major interleave), and the second pick's cost is
        # negligible against the 1024-wide atan. Both are float-exact at the active
        # row.
        picked_dx = pick_by_one_hot(mask, concat(*dxs), d_fill=1)
        picked_dy = pick_by_one_hot(mask, concat(*dys), d_fill=1)

    # The deferred world-angle atan2 is projection work (the seg/bbox
    # R_PointToAngle), even though it is structurally invoked here at dispatch
    # time after the 4-atans-to-1 collapse. Annotate it `proj` so its (wide)
    # nodes carry their true provenance rather than the dispatch glue's.
    with annotate("proj"):
        shared = angle_scalar(signed_world_angle(picked_dx, picked_dy))
    return {
        name: (shared if isinstance(out, AngleInputEmit) else out)
        for name, out in branches.items()
    }


@annotated("dispatch")
def _collapse_scalar_emits(
    inp: ProtocolTokenView,
    branches: Mapping[str, "Node | ScalarEmit"],
) -> dict[str, Node]:
    """Collapse every numeric-carrier branch into ONE shared digit-quad head per
    carrier — the residual-width keystone (see ``scripts/COST_NOTES.md``).

    Each numeric-carrier branch (``VALUE`` / ``ANGLE_VALUE``) returns a
    :class:`ScalarEmit` carrying only its 1-wide scalar, not a full
    ~255-wide-quad head. Here, for each carrier, the active branch's own
    dispatch predicate (exactly one is +1 at any position) picks that branch's
    scalar from the carrier's candidates (``pick_by_one_hot`` — a flat
    ``broadcast_select`` that is float-exact at the winning row), and ONE
    :func:`make_token_head` emits it. Pointing all of a carrier's branch keys at
    that single head node means :func:`_distinct_head_pairs` groups them into one
    OR-ed dispatch term. ~24 eager 255-wide quads collapse to 2.

    Byte-identical to building each branch's head eagerly: the picked scalar is
    the winning branch's scalar bit-for-bit (``approximate=False``), and the
    shared head clamps + quantizes it exactly as the per-branch head did — the
    gate that used to mask the *quantized bytes* now picks the *scalar* before
    the single shared quantization, on a row where exactly one predicate fires.
    """
    # branch -> its dispatch predicate(s), read exactly as _distinct_head_pairs.
    branch_preds = _branch_predicates(inp)

    # Group the ScalarEmit branches by carrier, in first-seen branch order.
    by_carrier: dict[int, list[tuple[str, ScalarEmit]]] = {}
    for name, out in branches.items():
        if isinstance(out, ScalarEmit):
            by_carrier.setdefault(id(out.carrier), []).append((name, out))

    shared_head: dict[str, Node] = {}
    for members in by_carrier.values():
        carrier = members[0][1].carrier
        (slot_name,) = carrier.slots  # VALUE -> "v"; ANGLE_VALUE -> "angle"
        # Collapse members that share the SAME scalar node into one pick entry:
        # the world-angle pre-pass points all four world branches at one shared
        # atan, so they would otherwise build four identical table entries. The
        # per-branch VALUE carriers each carry a distinct scalar, so this is a
        # no-op for them. (Mirrors _distinct_head_pairs' head grouping.)
        grouped = _group_by_value(
            [
                (emit.scalar, _one_hot_predicate(branch_preds, name))
                for name, emit in members
            ]
        )
        mask = concat(*[bool_to_01(pred) for _scalar, pred in grouped])
        # Clamp each candidate to the slot range BEFORE the pick so the
        # broadcast_select offset stays the slot's magnitude (an un-clamped
        # angle's value_type is ±millions); byte-identical at the winning row.
        table = concat(
            *[clamp_to_slot(carrier, slot_name, scalar) for scalar, _pred in grouped]
        )
        picked = pick_by_one_hot(mask, table, d_fill=1)
        head = make_token_head(carrier, **{slot_name: picked})
        for name, _emit in members:
            shared_head[name] = head

    # ScalarEmit branches now point at their shared head; everything else (a
    # Node already) passes through unchanged.
    return {name: shared_head.get(name, out) for name, out in branches.items()}
