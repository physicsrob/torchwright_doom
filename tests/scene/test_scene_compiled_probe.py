"""Plan E (folded-in): compiled read-side fidelity probe.

Plan D validated the read-side *graph math* (graph oracle vs the reference renderer
(pydoom) / map data)
but never ran the SceneIndex *composition* through an actual compile — only the
Plan A primitives (is_type / extract / emit) had compiled coverage. This test
closes that gap: it compiles a representative ``SceneIndex`` channel set into a
``CompiledHeadless`` transformer and runs ``probe_compiled`` to confirm the
compiled module matches the graph oracle (recursive ``node.compute``) at every
node within tolerance.

This is the *compile-fidelity* half of the two-comparison story (the other half
— port-correctness, graph oracle vs pydoom — is the Plan D ``test_scene_oracle``
gate). A small hand-built prefill keeps the compile tractable; the point is to
exercise the read-side *composition* (the ``pick_most_recent`` header contexts,
the lifted-key lookups, ``mean_where``) through the compiler, not map scale.
"""

from __future__ import annotations

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled, reference_eval
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom import std
from torchwright_doom.bsp_traversal import _think_side_compute
from torchwright_doom.embedding import TOKEN_VOCAB
from torchwright_doom.past import GraphPast
from torchwright_doom.scene_index import SceneIndex
from torchwright_doom.value_ranges import ValueRange
from torchwright_doom.vocab import (
    BEGIN,
    NODE,
    NODE_DX,
    NODE_DY,
    NODE_PX,
    NODE_PY,
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    SEG,
    SEG_AX,
    SS,
)

from ..prefill_fixture import tokens_to_input, value

# Shallow reads (one affine Linear over a FloatSlot, plus the recency/lifted
# picks) — the compile-vs-oracle residual is tiny, far below the DOOM-scale
# ~500 floor. atol=5 gives comfortable margin on the coordinate channels
# (values ~100s) while still being a real fidelity check.
_ATOL = 5.0


def test_scene_index_compiles_and_matches_oracle(device) -> None:
    seq = [
        (PLAYER_X_MARK, {}),
        value(ValueRange.R1, 100.0),
        (PLAYER_Y_MARK, {}),
        value(ValueRange.R1, -30.0),
        (NODE, {"j": 0}),
        (NODE_PX, {}),
        value(ValueRange.R1, 50.0),
        (NODE_PY, {}),
        value(ValueRange.R1, -20.0),
        (NODE, {"j": 1}),
        (NODE_PX, {}),
        value(ValueRange.R1, 200.0),
        (SS, {"s": 0}),
        (SEG, {"i": 0, "is_first_of_ss": 1}),
        (SEG_AX, {}),
        value(ValueRange.R1, 10.0),
        (BEGIN, {}),
    ]
    n_pos = len(seq)
    inputs = {"iv": tokens_to_input(seq)}

    d_embed = TOKEN_VOCAB.layout.d_embed
    pos = create_pos_encoding()
    iv = create_input("iv", d_embed)
    past = GraphPast(input_vec=iv, pos_encoding=pos)
    scene = SceneIndex.build(iv, past, pos)

    # A representative cross-section: player pose, node coords + root recency,
    # node presence (no-match branch), and a seg endpoint — covering mean_where,
    # pick_most_recent, and the lifted/one-hot keyed lookups.
    channels = Concatenate(
        [
            scene.view.x,
            scene.view.y,
            scene.nodes.px(std.constant(0.0)),
            scene.nodes.px(std.constant(1.0)),
            scene.nodes.py(std.constant(0.0)),
            scene.nodes.root,
            scene.nodes.exists(std.constant(0.0)),
            scene.nodes.exists(std.constant(5.0)),
            scene.segs.ax(std.constant(0.0)),
        ]
    )

    # d_head must cover the widest attention key — the presence lookups key on
    # a one-hot of width N_NODES_MAX + 1 = 65, so d_qk = 65.
    compiled = compile_headless(
        channels,
        pos,
        d=2048,
        d_head=128,
        max_layers=60,
        verbose=False,
        device=str(device),
    )

    report = probe_compiled(compiled, channels, inputs, n_pos, atol=_ATOL)
    assert report.first_divergent is None, report.format_short()


def test_side_test_cross_product_compiles_exact(device) -> None:
    """Compiled fidelity of the *new* Plan E numeric: the BSP side-test cross
    product (``R_PointOnSide``).

    ``_think_side_compute`` computes ``sign((dy*(viewx-x)) - (dx*(viewy-y)))``.
    The two ``mul_side`` products are ``multiply_2d`` over a coarse grid
    (``step1=8``, ``step2=37.5``), so the *raw* cross product is a ~6-figure
    value carrying ~step-level absolute grid-quantization error — a probe of the
    intermediate would need an atol scaled to that magnitude and prove little.
    What is load-bearing is the *sign* of that value (the 0/1 side bit), which
    decides front-vs-back child. The two partitions here are chosen so the cross
    product has large margin (|cross| ~ 1000s), so the sign is decided cleanly
    and the compiled bit is *bit-exact* vs the oracle.

    Checked only at the BEGIN position, where the full node table is prefilled.
    At earlier positions the node-coordinate lookups return the no-match default
    (the partition hasn't been emitted yet), so the cross product lands near zero
    — an undefined sign that flips between compiled (GPU PL-approx) and oracle
    (CPU exact math) run-to-run. That is the "FP nondeterminism at tolerance
    boundaries" case, not a fidelity failure: the bit is only meaningful, and
    only consumed by the renderer, once its node's geometry is in the table.

    The two partitions are chosen to land on opposite sides (bits ``[0, 1]``) so
    the check exercises both outcomes rather than passing trivially on a constant.
    """
    seq = [
        (PLAYER_X_MARK, {}),
        value(ValueRange.R1, 100.0),
        (PLAYER_Y_MARK, {}),
        value(ValueRange.R1, -30.0),
        # node 0: cross = (-30)*(100-50) - 40*(-30-(-20)) = -1500 + 400 = -1100 -> bit 0
        (NODE, {"j": 0}),
        (NODE_PX, {}),
        value(ValueRange.R1, 50.0),
        (NODE_PY, {}),
        value(ValueRange.R1, -20.0),
        (NODE_DX, {}),
        value(ValueRange.R2, 40.0),
        (NODE_DY, {}),
        value(ValueRange.R2, -30.0),
        # node 1: cross = 80*(100-60) - 10*(-30-0) = 3200 + 300 = 3500 -> bit 1
        (NODE, {"j": 1}),
        (NODE_PX, {}),
        value(ValueRange.R1, 60.0),
        (NODE_PY, {}),
        value(ValueRange.R1, 0.0),
        (NODE_DX, {}),
        value(ValueRange.R2, 10.0),
        (NODE_DY, {}),
        value(ValueRange.R2, 80.0),
        (BEGIN, {}),
    ]
    n_pos = len(seq)
    inputs = {"iv": tokens_to_input(seq)}

    d_embed = TOKEN_VOCAB.layout.d_embed
    pos = create_pos_encoding()
    iv = create_input("iv", d_embed)
    past = GraphPast(input_vec=iv, pos_encoding=pos)
    scene = SceneIndex.build(iv, past, pos)

    bits = Concatenate(
        [
            std.bool_to_01(_think_side_compute(scene, std.constant(0.0))),
            std.bool_to_01(_think_side_compute(scene, std.constant(1.0))),
        ]
    )

    # No one-hot presence key here (only lifted-key side lookups), so d_head=64
    # comfortably covers the attention key widths.
    compiled = compile_headless(
        bits,
        pos,
        d=2048,
        d_head=64,
        max_layers=60,
        verbose=False,
        device=str(device),
    )

    input_specs = compiled._input_specs
    d_in = max(start + width for _, start, width in input_specs)
    start, width = next((s, w) for nm, s, w in input_specs if nm == "iv")
    full = torch.zeros(n_pos, d_in)
    full[:, start : start + width] = inputs["iv"]

    compiled_bits = compiled(full).cpu()
    oracle_bits = reference_eval(bits, inputs, n_pos)[bits]

    begin = n_pos - 1  # full node table is prefilled at BEGIN
    compiled_at_begin = compiled_bits[begin].round()
    oracle_at_begin = oracle_bits[begin].round()

    # Sanity: the oracle itself lands on the opposite-side bits we designed for,
    # so the fidelity check below isn't passing trivially on a constant.
    assert oracle_at_begin.tolist() == [0.0, 1.0], (
        f"fixture drift: oracle side bits at BEGIN = {oracle_at_begin.tolist()}, "
        "expected [0, 1] (one front, one back partition)"
    )
    assert torch.equal(compiled_at_begin, oracle_at_begin), (
        f"compiled side bits diverge from oracle at BEGIN:\n"
        f"compiled={compiled_at_begin.tolist()}\n"
        f"oracle  ={oracle_at_begin.tolist()}"
    )
