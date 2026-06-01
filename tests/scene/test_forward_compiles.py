"""The whole renderer ``forward()`` compiles to a transformer and stays faithful.

Plan E's oracle (``test_traversal_oracle``) proves the forward's *graph math* is
right via ``reference_eval``; the read-side probe (``test_scene_compiled_probe``)
proves a slice of the read side compiles. This is the gate that the **entire
forward** — read side + the dispatch output head + emit — compiles into a real
``CompiledHeadless`` and matches the graph oracle everywhere.

It is also the regression guard for the dispatch fan-out fix. The literal
sandbox dispatch (``type_switch`` over one *full* ``d_embed`` row per branch)
needs a ~53k-wide residual to compile — it does not. The output head here gates
over emit *heads* (``head_width()`` ≈ 236 cols, derived tail dropped) and sums
**one gated copy per distinct head** (the deferred branches share one NO_OP
head), then stamps the shared derived zero once. That compiles at a modest
residual width in a couple dozen layers; if the dispatch ever regresses to the
wide fan-out, this test fails to allocate.
"""

from __future__ import annotations

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward
from torchwright_doom.value_ranges import ValueRange
from torchwright_doom.vocab import (
    BEGIN,
    NODE,
    NODE_BACK_CHILD,
    NODE_DX,
    NODE_DY,
    NODE_FRONT_CHILD,
    NODE_PX,
    NODE_PY,
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    SEG,
    SEG_AX,
    SET_CURSOR_DIRECTION_Y,
    SS,
)

from ..prefill_fixture import row_index, tokens_to_input, value

# The reduced forward's residual fits at d=5120: ~8 distinct emit heads (236
# each) float together, plus the 820-wide input and the 584-wide shared tail.
# d_head=160 covers the widest attention key (the traversal-edge lookup keys on
# a one-hot of width N_ENTITY_MAX + N_DEPTH_MAX = 145).
_D = 5120
_D_HEAD = 160
# Deep op chains over values in the 10^3-10^6 range (digit-quad payloads); 50 is
# comfortably above the compiled-vs-oracle PL noise here while still a real check.
_ATOL = 50.0


def test_forward_compiles_and_matches_oracle(device) -> None:
    # Smallest scene that exercises the whole spine: one BSP node (both children
    # subsectors) and one subsector/seg, ending at BEGIN — the AR seed position.
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
        (NODE_DX, {}),
        value(ValueRange.R2, 40.0),
        (NODE_DY, {}),
        value(ValueRange.R2, -30.0),
        (NODE_FRONT_CHILD, {"child_u": 64}),
        (NODE_BACK_CHILD, {"child_u": 65}),
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
    next_token = forward(iv, past, pos)

    compiled = compile_headless(
        next_token,
        pos,
        d=_D,
        d_head=_D_HEAD,
        max_layers=200,
        verbose=False,
        device=str(device),
    )

    # Compiled transformer matches the graph oracle at every node.
    report = probe_compiled(compiled, next_token, inputs, n_pos, atol=_ATOL)
    assert report.first_divergent is None, report.format_short()

    # And the seed position emits the right next token: BEGIN -> SET_CURSOR_DIRECTION_Y.
    d_in = max(start + w for _, start, w in compiled._input_specs)
    start, width = next((s, w) for nm, s, w in compiled._input_specs if nm == "iv")
    full = torch.zeros(n_pos, d_in)
    full[:, start : start + width] = inputs["iv"]
    out = compiled(full)
    begin = n_pos - 1
    emitted = int(torch.argmax(out[begin].cpu() @ W_EMBED.t()).item())
    assert emitted == row_index(SET_CURSOR_DIRECTION_Y, {}), (
        f"BEGIN emitted row {emitted}, expected SET_CURSOR_DIRECTION_Y "
        f"({row_index(SET_CURSOR_DIRECTION_Y, {})})"
    )
