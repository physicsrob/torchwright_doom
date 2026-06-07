"""Targeted reproducer: the compiled free-run mispredicts a BSP-traversal node.

Plan K Step 1 found the compiled autoregressive free-run renders the first visible
subsector correctly, then loops forever in the BSP/bbox traversal. This test
isolates the root cause as a single per-position compiled error, with **one wide
teacher-forced forward** (not an AR rollout): feed the *reference* token stream
through the compiled token-id forward and check each position's argmax under the
J2 bars (markers exact, carriers in tolerance, pixels in option set).

The first hard divergence is at rollout token 1395 (stream pos 2451): the
reference emits ``bspCheckBack(node=3, depth=8)`` but the compiled fp32 forward
predicts ``bspCheckBack(node=1, depth=8)``. Across the early divergences the
``depth`` is always exact and only the BSP ``node``/``entity_u`` index is wrong —
the node-resolution attention loses its score gap at recursion depth ~8 and picks
the wrong node. Because the context is teacher-forced (correct), this is a genuine
per-position compiled error (an op over its noise budget), not AR-feedback
amplification.

Fast relative to the free-run: one forward over ~2.5k positions vs thousands of
sequential steps. Localize the offending op next with ``probe_compiled`` /
``probe_attention`` (CLAUDE.md triage), then narrow to an op-level test (D6).
"""

from __future__ import annotations

import pytest

from torchwright_doom.render import compare
from torchwright_doom.render.compiled_model import build_compiled
from torchwright_doom.render.diagnostic import teacher_forced_scan
from torchwright_doom.render.tokens_bridge import sandbox_token_to_row

# Just past the first hard divergence (stream pos 2451). Keeps the single wide
# forward's O(n_pos^2) attention within an L4's memory while still reaching the bug.
_WINDOW = 2470


def _reference_stream():
    fixtures = pytest.importorskip("doom_sandbox.fixtures")
    sb_prefill = pytest.importorskip("doom_sandbox.implementation.prefill")
    drafter = pytest.importorskip("doom_sandbox.implementation.reference_drafter")
    scene = fixtures.load_fixture("e1m1_subset_textured")
    pose = scene.test_poses[0]
    prefill_rows = [sandbox_token_to_row(t) for t in sb_prefill.get_prefill(scene, pose)]
    ar_rows = [sandbox_token_to_row(t) for t in drafter.expected_ar_tokens(scene, pose)]
    return scene, pose, prefill_rows, prefill_rows + ar_rows


def test_compiled_teacher_forced_free_run_is_faithful(device):
    """The compiled forward, fed the correct context, must predict the reference
    next token at every position (under the J2 bars).

    Currently FAILS at rollout token 1395 — bspCheckBack node 3 -> 1 — capturing
    the BSP-traversal node-resolution fp32 divergence (Plan K Step 1)."""
    if device.type != "cuda":
        pytest.skip("compiled full forward needs a GPU")

    scene, pose, prefill_rows, full_rows = _reference_stream()
    begin = len(prefill_rows) - 1
    options = compare.reference_options(scene, pose)

    compiled, _ = build_compiled(device)
    divs = teacher_forced_scan(compiled, full_rows, begin, options, window=_WINDOW)

    assert not divs, (
        f"compiled teacher-forced free-run hard-diverges from the reference at "
        f"{len(divs)} position(s) within the first {_WINDOW}:\n"
        + "\n".join(
            f"  rollout {d.pos - begin} (pos {d.pos}): expected {d.expected_type}"
            f"(row {d.expected_row}) -> predicted {d.predicted_type}"
            f"(row {d.predicted_row})  [{d.kind}: {d.detail}]"
            for d in divs[:10]
        )
    )
