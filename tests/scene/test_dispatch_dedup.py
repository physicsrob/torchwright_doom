"""The dispatch collapses one term per *distinct head*, not per transition.

``render_main.dispatch_next_token`` is only compilable because
``_distinct_head_pairs`` groups the ~64 registry transitions by the head node
they select (the deferred branches all share one ``no_op`` head) and OR-s the
predicates within a group, so ``type_switch`` sums a handful of gated heads
instead of 64. The whole-forward compile gate would fail to allocate if this
regressed; this pins the grouping itself, cheaply and without a compile.
"""

from __future__ import annotations

from typing import cast

from torchwright_doom.protocol_registry import DISPATCH_TRANSITIONS
from torchwright_doom.protocol_tokens import ProtocolTokenView
from torchwright_doom.render_main import _distinct_head_pairs
from torchwright_doom.std import constant


class _FakeInp:
    """Carries one ±1 predicate node per registry predicate name."""


def test_distinct_head_pairs_groups_by_head_node() -> None:
    transitions = list(DISPATCH_TRANSITIONS)
    branch_names = list(dict.fromkeys(t.branch for t in transitions))
    assert len(branch_names) > 3  # sanity: there is something to collapse

    # Mimic build_branch_outputs: all branches share one node except two, which
    # get their own — so the grouping should yield exactly three distinct heads.
    shared = constant(0.0)
    head_a = constant(1.0)
    head_b = constant(2.0)
    branches = {name: shared for name in branch_names}
    branches[branch_names[0]] = head_a
    branches[branch_names[1]] = head_b

    inp = _FakeInp()
    for predicate in {t.predicate for t in transitions}:
        setattr(inp, predicate, constant(1.0))

    pairs = _distinct_head_pairs(cast(ProtocolTokenView, inp), branches)
    heads = [head for _cond, head in pairs]

    # One pair per distinct head node (by identity), in first-seen order.
    assert len({id(h) for h in heads}) == len(pairs)
    assert {id(h) for h in heads} == {id(shared), id(head_a), id(head_b)}
    # It actually collapsed: far fewer pairs than transitions.
    assert len(pairs) == 3 < len(transitions)
