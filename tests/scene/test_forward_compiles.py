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
from torchwright_doom.vocab import SET_CURSOR_DIRECTION_Y

from ..prefill_fixture import TINY_BSP_SCENE, pad_iv, row_index, tokens_to_input

# With the shared-slot-column embedding the unforced residual peak is ~1432:
# the 603-wide input + the 584-wide shared derived tail dominate; the ~8 distinct
# emit heads are only ~19 cols each now (8 E8 + the shared slot columns), so the
# dispatch fan-out is no longer the driver. d=2400 leaves comfortable margin;
# d_head=160 covers the widest attention key (the traversal-edge lookup keys on a
# one-hot of width N_ENTITY_MAX + N_DEPTH_MAX = 145; d must be a multiple of it).
_D = 2400
_D_HEAD = 160
# Deep op chains over values in the 10^3-10^6 range (digit-quad payloads); 50 is
# comfortably above the compiled-vs-oracle PL noise here while still a real check.
_ATOL = 50.0


def test_forward_compiles_and_matches_oracle(device) -> None:
    n_pos = len(TINY_BSP_SCENE)
    iv_input = tokens_to_input(TINY_BSP_SCENE)
    inputs = {"iv": iv_input}

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
    out = compiled(pad_iv(compiled, iv_input))
    begin = n_pos - 1
    emitted = int(torch.argmax(out[begin].cpu() @ W_EMBED.t()).item())
    assert emitted == row_index(SET_CURSOR_DIRECTION_Y, {}), (
        f"BEGIN emitted row {emitted}, expected SET_CURSOR_DIRECTION_Y "
        f"({row_index(SET_CURSOR_DIRECTION_Y, {})})"
    )
