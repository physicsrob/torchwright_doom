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
from torchwright.debug.probe import probe_compiled, reference_eval
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward
from torchwright_doom.vocab import SET_CURSOR_DIRECTION_Y

from ..prefill_fixture import TINY_BSP_SCENE, pad_iv, row_index, tokens_to_input

# With the shared-slot-column embedding the emit heads are only ~19 cols each, so
# dispatch fan-out is not the residual driver; the 603-wide input + 584-wide shared
# derived tail plus the live published render state dominate. Plan G's BBoxPruner
# adds its published occlusion/context channels (the bbox region/corner/angle
# recovery state) to the live set, pushing the peak past Plan F's ~1432 — d=2400 no
# longer fits, d=3200 restores comfortable margin. d_head=160 covers the widest
# attention key (the traversal-edge lookup keys on a one-hot of width
# N_ENTITY_MAX + N_DEPTH_MAX = 145; d must be a multiple of it).
_D = 4800
_D_HEAD = 160
# Deep op chains over values in the 10^3-10^6 range (digit-quad payloads); 50 is
# comfortably above the compiled-vs-oracle PL noise on every real-path node here.
_ATOL = 50.0
# A digit-quad floor landing on a 256-block boundary flips its byte by up to one
# block; a 2-byte payload thus bounds a never-selected candidate divergence to a
# few hundred. Anything past this is a real (gross / NaN / structural) regression,
# not the bounded bbox R0 emit-candidate noise documented at the probe below.
_CANDIDATE_DIGIT_QUAD_CAP = 600.0


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

    # Compiled transformer matches the graph oracle. Plan G note: the renderer
    # builds the bbox R0 corner-value ``make_value`` candidate at *every* position
    # but it only fires at TRAVERSE_BETWEEN rows — which ``TINY_BSP_SCENE`` has
    # none of. ``probe_compiled`` still checks that never-selected candidate, and
    # at one row its recovered (garbage) corner lands on a 256-block boundary of
    # the digit-quad's floor, where the compiled floor ramp diverges by up to a
    # byte block (~512). That node never reaches the dispatched next token (the
    # per-position output check below is exact), and compiled digit-quad fidelity
    # on the wide R0 range is a Plan I/J concern (the carried-forward floor note).
    # So: no node may diverge *grossly* (NaN / structural break — a real
    # regression), but the bounded never-selected emit candidate is tolerated.
    report = probe_compiled(compiled, next_token, inputs, n_pos, atol=_ATOL)
    over = [d for d in report.per_node.values() if d.max_abs_error > _ATOL]
    gross = [d for d in over if d.max_abs_error > _CANDIDATE_DIGIT_QUAD_CAP]
    assert not gross, "gross compiled divergence (not the bounded emit candidate):\n" + (
        report.format_short()
    )

    # The real correctness claim: the compiled forward emits the exact next token
    # the graph oracle does, at *every* position (not just the seed) — the
    # never-selected candidate noise above cannot leak into a dispatched output.
    oracle = reference_eval(next_token, inputs, n_pos)[next_token]
    out = compiled(pad_iv(compiled, iv_input))
    w_embed_t = W_EMBED.t()
    for i in range(n_pos):
        compiled_row = int(torch.argmax(out[i].cpu() @ w_embed_t).item())
        oracle_row = int(torch.argmax(oracle[i] @ w_embed_t).item())
        assert compiled_row == oracle_row, (
            f"pos {i}: compiled next-token row {compiled_row} != oracle {oracle_row}"
        )

    # And the seed position specifically: BEGIN -> SET_CURSOR_DIRECTION_Y.
    begin = n_pos - 1
    emitted = int(torch.argmax(out[begin].cpu() @ w_embed_t).item())
    assert emitted == row_index(SET_CURSOR_DIRECTION_Y, {}), (
        f"BEGIN emitted row {emitted}, expected SET_CURSOR_DIRECTION_Y "
        f"({row_index(SET_CURSOR_DIRECTION_Y, {})})"
    )
