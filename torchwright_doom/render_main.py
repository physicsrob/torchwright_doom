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
from .protocol_registry import DISPATCH_TRANSITIONS
from .protocol_tokens import ProtocolTokenView
from .scene_index import SceneIndex
from .solid_intervals import SolidIntervals
from .std import bool_or, concat, make_token_head, type_switch
from .vocab import ANGLE_VALUE, DONE, NO_OP, SET_CURSOR_DIRECTION_Y

BranchOutputs = Mapping[str, Node]


@dataclass(frozen=True)
class RuntimeProtocols:
    """Published protocol owners needed by branch construction (traversal only
    this phase)."""

    traversal: BspTraversal


def publish_runtime_protocols(
    input_vec: Node,
    past: PastHandleScope,
    inp: ProtocolTokenView,
    scene: SceneIndex,
    pos: PosEncoding,
) -> RuntimeProtocols:
    """Publish runtime protocol channels before branch candidates consume them.

    Reduced to the traversal owner: ``SegProjection`` (and the seven projection
    owners) are deferred, so the projection branches are NO_OP-stubbed below.
    """
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
    return RuntimeProtocols(traversal=traversal)


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
    """Build the traversal/begin/no_op/done branch heads for real; stub the rest.

    The dict's keys equal ``{t.branch for t in DISPATCH_TRANSITIONS}`` (so the
    dispatch is well-formed). Only the branches this phase implements get their
    own head; every deferred branch — ``R_CheckBBox`` (``between`` + the
    ``bbox_*`` sub-protocol) and the projection / wall / flat / payload owners —
    maps to the *one shared* ``no_op`` head via the ``setdefault`` loop. Sharing
    that node is what lets the dispatch collapse ~56 deferred transitions into a
    single gated term (see :func:`dispatch_next_token`). When an owner lands, its
    branches get real heads here and split back out of the shared term.

    Values are emit *heads* (``make_token_head`` / ``after_*``), not full rows;
    the dispatch stamps the shared derived tail after selection.
    """
    no_op_out = make_token_head(NO_OP)
    traversal = protocols.traversal
    branches: dict[str, Node] = {
        # Inert / begin.
        "no_op": no_op_out,
        "done": make_token_head(DONE),
        "begin": make_token_head(SET_CURSOR_DIRECTION_Y),
        # BSP traversal (BspTraversal) — the real Plan E branches.
        "think": traversal.after_think_side(),
        "side_record": traversal.after_side_record(),
        "enter": traversal.after_enter(),
        "return_": traversal.after_return(),
        "set_cursor_direction_y": traversal.after_set_cursor_direction_y(),
    }
    # Every remaining registry branch — the deferred bbox sub-protocol and the
    # projection / wall / flat / payload owners — shares the one NO_OP head.
    for transition in DISPATCH_TRANSITIONS:
        branches.setdefault(transition.branch, no_op_out)
    return branches
