"""Autoregressive renderer dispatch — reduced to the BSP-traversal spine
(Plan E / E1).

Mirrors ``doom_sandbox/implementation/forward/main.py``: ``forward()`` builds
the read-side ``SceneIndex`` + ``ProtocolTokenView``, publishes the runtime
protocol owners, builds each branch's next-token, then dispatches by token type.

**The output head (the fan-out fix).** The sandbox dispatch sums one type-gated
*full* ``d_embed`` row per transition; at ~64 transitions that needs a huge
residual and will not compile. The port replaces it (see ``dispatch_next_token``):
branches are built as emit *heads* (``head_width()`` — the constant derived tail
dropped, ≈ 19 cols under the shared-slot-column embedding), transitions that
select the same head are grouped with their predicates OR-ed, the ~8 distinct
gated heads are summed (``max_fanout=2`` so the gated copies free incrementally),
and one shared ``emit_derived_zero`` is concatenated at the end. The full row is
byte-identical to the sandbox's, so the teacher-forced oracle is unchanged. With
the shared-slot-column layout the whole forward compiles at d≈1600 (peak ~1432);
heads are ~19 cols each, so the dispatch width barely grows with token count.

**Single-pass scoping (Plan E).** A literal port of ``main.py`` would drag in
the whole renderer (``SegProjection`` and seven owners). This reduced spine
publishes **only** ``BspTraversal`` and builds the **traversal/begin/no_op/done**
heads for real; every deferred branch (the bbox sub-protocol and the projection /
wall / flat / payload owners) shares the one ``no_op`` head. The projection owner
lands in a later phase, at which point its real branches split out of that shared
term. A free-run therefore walks the BSP tree and stops at the first subsector
(``test_forward_ar_rollout``); pixels arrive with projection.

Changes from the sandbox source: ``Vec`` -> ``Node``; ``Past`` -> ``GraphPast``
/ ``PastHandleScope``; ``make_token`` -> ``make_token_head`` + a shared derived
tail; ``ForwardOutput`` -> the next-token ``Node`` returned directly; the reduced
``RuntimeProtocols`` / ``publish_runtime_protocols`` / ``build_branch_outputs``;
and the prefill-replay ``select`` is deferred (see ``dispatch_next_token``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torchwright.graph import Node, PosEncoding

from .bsp_traversal import BspTraversal
from .emit import emit_derived_zero
from .past import GraphPast, PastHandleScope
from .payload_router import PayloadRouter
from .protocol_registry import DISPATCH_TRANSITIONS
from .protocol_tokens import ProtocolTokenView
from .scene_index import SceneIndex
from .seg_projection import SegProjection
from .seg_scanner import SegScanner
from .solid_intervals import SolidIntervals
from .std import bool_or, concat, make_token_head, type_switch
from .vocab import ANGLE_VALUE, DONE, NO_OP, SET_CURSOR_DIRECTION_Y
from .wall_range_builder import WallRangeBuilder

BranchOutputs = Mapping[str, Node]


@dataclass(frozen=True)
class RuntimeProtocols:
    """Published protocol owners needed by branch construction.

    Plan F grows this from traversal-only to traversal + the reduced
    seg-projection owner + the payload router. R_CheckBBox visibility pruning
    (Phase G) and the wall-column / visplane / flat owners (Phase H/J) are still
    deferred, so their branches collapse into the shared NO_OP head."""

    traversal: BspTraversal
    projection: SegProjection
    payload_router: PayloadRouter


def publish_runtime_protocols(
    input_vec: Node,
    past: PastHandleScope,
    inp: ProtocolTokenView,
    scene: SceneIndex,
    pos: PosEncoding,
) -> RuntimeProtocols:
    """Publish runtime protocol channels before branch candidates consume them.

    Order matters: ``input_angle_or_zero`` and ``SolidIntervals`` are published
    before the projection owner that reads them; ``BspTraversal`` stays the
    Plan-E reduced owner (bbox deferred to Phase G), so the payload router is
    built without a bbox arm (Phase F routes ``is_bbox_angle`` to NO_OP)."""
    input_angle_or_zero = past.publish(
        "input_angle_or_zero",
        ANGLE_VALUE.extract(input_vec, "angle"),
    )
    solids = SolidIntervals.publish(past, inp, scene)
    traversal = BspTraversal.publish(past, inp, scene)
    projection = SegProjection.publish(
        past,
        input_vec,
        inp,
        scene,
        solids,
        input_angle_or_zero,
        pos,
    )
    payload_router = PayloadRouter(projection=projection)
    return RuntimeProtocols(
        traversal=traversal,
        projection=projection,
        payload_router=payload_router,
    )


def dispatch_next_token(
    input_vec: Node,
    inp: ProtocolTokenView,
    branches: BranchOutputs,
) -> Node:
    """Select the one branch's next-token that matches the current input token.

    This is the transformer's output head. The sandbox's ``type_switch`` sums
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

    The sandbox also wraps dispatch in a prefill-replay ``select`` so prefill
    rows re-emit their input verbatim (a render-trace nicety; those emissions are
    discarded and ``BEGIN`` — the AR seed — is not replayed). That replay is
    **deferred**: it would select the full ``input_vec``, whose W_EMBED rows span
    a ~100x dynamic range, and a single input ``value_type`` can't satisfy both
    the read-side interval arithmetic and the whole-row replay guard. The AR
    rollout and the oracle are unaffected (prefill emissions are discarded; the
    AR seed comes from the ``begin`` branch).
    """
    # max_fanout=2 sums the gated heads through a running accumulator so at most
    # ~2 gated copies sit on the residual stream at once, instead of all ~8 — the
    # gated copies are full emit heads, so the peak-width saving is what lets the
    # forward compile at a modest residual width.
    head = type_switch(*_distinct_head_pairs(inp, branches), max_fanout=2)
    return concat(head, emit_derived_zero())


def _distinct_head_pairs(
    inp: ProtocolTokenView,
    branches: BranchOutputs,
) -> tuple[tuple[Node, Node], ...]:
    """Group transitions by the head node they select, OR-ing the predicates of
    transitions that share a head. Returns one ``(predicate, head)`` pair per
    distinct head, in first-seen order — the input to ``type_switch``."""
    groups: dict[int, tuple[Node, list[Node]]] = {}
    order: list[int] = []
    for transition in DISPATCH_TRANSITIONS:
        cond = getattr(inp, transition.predicate)
        head = branches[transition.branch]
        key = id(head)
        if key not in groups:
            groups[key] = (head, [])
            order.append(key)
        groups[key][1].append(cond)
    return tuple(
        (conds[0] if len(conds) == 1 else bool_or(*conds), head)
        for head, conds in (groups[key] for key in order)
    )


# DOOM: R_RenderPlayerView (r_main.c) — top-level per-frame render dispatch
def forward(input_vec: Node, past: GraphPast, pos: PosEncoding) -> Node:
    # `SceneIndex.build` publishes scene channels through the bare `GraphPast`;
    # the `PastHandleScope` wrap below is intentional and must follow it. It is a
    # distinct local (`scope`) rather than a rebind of `past` so the two types
    # stay separable.
    scene = SceneIndex.build(input_vec, past, pos)
    scope = PastHandleScope(past)
    prev_input_type = scope.attend_to_offset(scope.input_type(), delta_pos=-1)
    prev_prev_input_type = scope.attend_to_offset(scope.input_type(), delta_pos=-2)
    inp = ProtocolTokenView(
        input_vec,
        prev_input_type,
        prev_prev_input_type,
    )

    protocols = publish_runtime_protocols(input_vec, scope, inp, scene, pos)
    branches = build_branch_outputs(protocols)
    return dispatch_next_token(input_vec, inp, branches)


def build_branch_outputs(protocols: RuntimeProtocols) -> dict[str, Node]:
    """Build the traversal + Phase-F branch heads for real; stub the rest.

    The dict's keys equal ``{t.branch for t in DISPATCH_TRANSITIONS}`` (so the
    dispatch is well-formed). Plan F splits the seg-projection / drawseg branches
    out of the shared ``no_op`` head: the seg-scan loop (``visit`` … ``emit_x2``),
    the drawseg-scalar chain (``store_wall_range`` … ``drawseg_u_phase``), and the
    ``value`` / ``angle`` carriers route to the ``SegScanner`` / ``WallRangeBuilder``
    / ``PayloadRouter`` owners. Still deferred to the *one shared* ``no_op`` head
    via the ``setdefault`` loop: ``R_CheckBBox`` (``between`` + ``bbox_*`` — Phase
    G) and the wall-column / visplane / flat / pixel owners (Phase H/J).

    Values are emit *heads* (``make_token_head`` / ``after_*``), not full rows;
    the dispatch stamps the shared derived tail after selection.
    """
    no_op_out = make_token_head(NO_OP)
    traversal = protocols.traversal
    projection = protocols.projection
    payload_router = protocols.payload_router
    seg_scan = SegScanner(projection)
    wall_range = WallRangeBuilder(projection)
    branches: dict[str, Node] = {
        # Inert / begin.
        "no_op": no_op_out,
        "done": make_token_head(DONE),
        "begin": make_token_head(SET_CURSOR_DIRECTION_Y),
        # BSP traversal (BspTraversal) — the Plan E branches.
        "think": traversal.after_think_side(),
        "side_record": traversal.after_side_record(),
        "enter": traversal.after_enter(),
        "return_": traversal.after_return(),
        "set_cursor_direction_y": traversal.after_set_cursor_direction_y(),
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
    }
    # Every remaining registry branch — the deferred bbox sub-protocol (Phase G)
    # and the wall-column / visplane / flat / pixel owners (Phase H/J) — shares
    # the one NO_OP head.
    for transition in DISPATCH_TRANSITIONS:
        branches.setdefault(transition.branch, no_op_out)
    return branches
