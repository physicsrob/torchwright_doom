"""Plan E mandatory gate: BSP-traversal teacher-forced next-token agreement.

This is the hard gate for the renderer control spine + BSP-traversal owner. It
drives the real ``forward()`` graph teacher-forced on the sandbox's golden token
stream (prefill + autoregressive emission) for a real map/pose, and asserts the
graph emits the *exact same next token* the sandbox emits at every
traversal-owned position the phase implements.

Two comparisons, kept separate (like the Plan D oracle):

1. **Port-correctness (this gate).** Each position's emitted next-token residual
   is evaluated through the graph oracle (``reference_eval`` -> memoised
   ``node.compute``, exact math, no compile), argmaxed against ``W_EMBED.T`` to
   a vocab row, and compared to the row the sandbox emitted. The match is
   *exact* (integer row index) — the emit digit-quad payload argmaxes cleanly.
2. **Compile-fidelity (downstream).** A compiled ``probe_compiled`` of the read
   side lives in ``test_scene_compiled_probe.py``; a free-running compiled AR
   loop is a later phase. This gate is ``reference_eval``-only.

Scope (Plan E): the phase implements the side-test precompute
(``setCursorDirectionY`` -> ``R_PointOnSide`` / ``pointOnSideResult``), the DFS
descent (``bspFront`` = ``TRAVERSE_ENTER``), node-vs-subsector child dispatch,
and the stack-pop (``bspReturn`` = ``TRAVERSE_RETURN``). ``R_CheckBBox`` pruning
(``TRAVERSE_BETWEEN`` -> the bbox sub-protocol) and the projection owner
(``R_Subsector`` / ``R_AddLine`` / ...) are deferred and **NO_OP-stubbed**, so
their positions are skipped — teacher forcing feeds the golden tokens through
them, and the traversal state (side table, DFS edges) is published at those
teacher-forced positions, so later ``TRAVERSE_RETURN`` pops still resolve.

Cross-submodule: ``importorskip``\\ s ``doom_sandbox`` (skipped standalone).
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward
from torchwright_doom.vocab import VOCAB_TYPES

from ..prefill_fixture import row_index, tokens_to_input
from ..sandbox_support import import_sandbox, require_doom_sandbox

# Traversal token types this phase emits a real next-token for (sandbox names).
# TRAVERSE_BETWEEN ("bspCheckBack") delegates to the deferred bbox owner, so it
# is excluded; the projection types (R_Subsector, R_AddLine, ...) are stubbed.
_IMPLEMENTED = {
    "begin",
    "setCursorDirectionY",
    "R_PointOnSide",
    "pointOnSideResult",
    "bspFront",
    "bspReturn",
}
# How many positions past the prefill to teacher-force. Sized to cover the full
# side-test precompute + several subsector descents and stack-pops without
# evaluating the whole (~3.5k-token) rollout.
_AR_SPAN = 240


@pytest.fixture(scope="module")
def traversal_eval():
    """Build the reduced ``forward()`` graph once, teacher-force it on the
    sandbox golden stream, and ``reference_eval`` a single pass — shared by both
    assertions below (the graph + eval is the expensive part)."""
    require_doom_sandbox()

    fixtures = import_sandbox("doom_sandbox.fixtures")
    sb_prefill = import_sandbox("doom_sandbox.implementation.prefill")
    drafter = import_sandbox("doom_sandbox.implementation.reference_drafter")

    name_to_real = {t.name: t for t in VOCAB_TYPES}

    scene = fixtures.load_fixture("e1m1_subset")
    pose = scene.test_poses[0]
    prefill = list(sb_prefill.get_prefill(scene, pose))
    golden = list(drafter.expected_ar_tokens(scene, pose))
    full = prefill + golden
    begin = len(prefill) - 1  # the BEGIN row seeds the AR loop

    n_pos = min(begin + _AR_SPAN, len(full) - 1) + 1
    real_pairs = [(name_to_real[t.type.name], dict(t.values)) for t in full]
    inputs = {"iv": tokens_to_input(real_pairs[:n_pos])}

    d_embed = TOKEN_VOCAB.layout.d_embed
    iv = create_input("iv", d_embed)
    past = GraphPast(input_vec=iv, pos_encoding=create_pos_encoding())
    next_token = forward(iv, past, create_pos_encoding())

    cache = reference_eval(next_token, inputs, n_pos)
    return {
        "emitted": cache[next_token],
        "full": full,
        "real_pairs": real_pairs,
        "begin": begin,
        "n_pos": n_pos,
    }


def test_bsp_traversal_matches_sandbox_next_tokens(traversal_eval) -> None:
    emitted = traversal_eval["emitted"]
    full = traversal_eval["full"]
    real_pairs = traversal_eval["real_pairs"]
    begin = traversal_eval["begin"]
    n_pos = traversal_eval["n_pos"]
    w_embed_t = W_EMBED.t()

    coverage: Counter[str] = Counter()
    mismatches = []
    for i in range(begin, n_pos - 1):
        tname = full[i].type.name
        if tname not in _IMPLEMENTED:
            continue  # deferred (bbox) / projection-owned -> teacher-forced, uncompared
        coverage[tname] += 1
        predicted_row = int(torch.argmax(emitted[i] @ w_embed_t).item())
        expected_row = row_index(*real_pairs[i + 1])
        if predicted_row != expected_row:
            mismatches.append(
                f"pos {i} (in {tname} {dict(full[i].values)}): emitted row "
                f"{predicted_row} != sandbox {full[i + 1].type.name} "
                f"{dict(full[i + 1].values)} (row {expected_row})"
            )

    assert not mismatches, "traversal next-token mismatches:\n" + "\n".join(
        mismatches[:20]
    )

    # The gate is only meaningful if it actually exercised the side test, the
    # descent, and a stack-pop — assert non-trivial coverage of each.
    assert coverage["R_PointOnSide"] >= 8, f"too few side tests compared: {coverage}"
    assert coverage["pointOnSideResult"] >= 8, f"too few side records: {coverage}"
    assert coverage["bspFront"] >= 3, f"too few TRAVERSE_ENTER descents: {coverage}"
    assert (
        coverage["bspReturn"] >= 1
    ), f"no TRAVERSE_RETURN stack-pop exercised: {coverage}"


def test_side_test_bits_match_sandbox(traversal_eval) -> None:
    """Focused check: the R_PointOnSide cross product (the one new numeric) emits
    the exact side bit the sandbox recorded, for every compared node."""
    emitted = traversal_eval["emitted"]
    full = traversal_eval["full"]
    real_pairs = traversal_eval["real_pairs"]
    begin = traversal_eval["begin"]
    n_pos = traversal_eval["n_pos"]
    w_embed_t = W_EMBED.t()

    checked = 0
    for i in range(begin, n_pos - 1):
        if full[i].type.name != "R_PointOnSide":
            continue
        # THINK_SIDE(node) must emit SIDE_RECORD(node, side) with the right side.
        predicted_row = int(torch.argmax(emitted[i] @ w_embed_t).item())
        assert predicted_row == row_index(*real_pairs[i + 1]), (
            f"side bit wrong at node {full[i].values}: "
            f"sandbox emitted {full[i + 1].values}"
        )
        checked += 1
    assert checked >= 8
