"""E6: the compiled renderer free-runs autoregressively, no host computation.

Everything else teacher-forces — the host feeds a known token sequence. This is
the real autoregressive loop: the compiled transformer emits a token, the host
argmaxes it to an id and feeds that id straight back, and the model re-embeds it
*itself* (the in-graph ``Embedding`` node). The host does no geometry, no
arithmetic — just argmax + copy, exactly like decoding any LLM.

Two things are proven:

1. **The feedback loop closes.** Wiring ``forward()``'s ``input_vec`` to a 1-wide
   ``token_ids`` slot through ``build_doom_embedding`` (instead of a pre-embedded
   820-wide input) compiles, and a KV-cached ``.step()`` rollout re-embeds each
   self-emitted token correctly across steps.
2. **It reproduces the intended rollout.** The free-run matches, token for token,
   the exact-math (``reference_eval``) free-run of the same graph — so the
   compiled autoregressive trajectory is the trajectory the graph defines, not an
   artifact of compilation.

This is the one place a *compiled* doom forward is actually executed locally: it
uses the in-process ``compile_headless`` on a tiny scene (fits at ~10 GB). The
real token-I/O artifact's *compile* is validated separately by
``test_forward_compiles`` (``compile_to_onnx``); running that artifact is
memory-infeasible on a 30 GB box (its weights densify to >26 GB — onnxruntime
``bad_alloc``s on load), so artifact-level inference belongs on a larger machine.
The dispatch reduction is ``max_fanout=8`` (render_main), so the free-run compiles
to ~44 layers rather than the ~66 of the old serial fold.

Scope: the rollout walks ``BEGIN -> SET_CURSOR_DIRECTION_Y -> the side-bit
precompute -> TRAVERSE_ENTER -> the descent to a leaf``. At the first subsector
the projection owner would take over; it is deferred this phase (stubbed NO_OP),
so the run stops there — the honest boundary of the traversal-only spine. The
meaningful prefix (a real side test + a real descent, generated autonomously) is
what the loop is validated on.
"""

from __future__ import annotations

import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.past import GraphPast
from torchwright_doom.render.compiled_model import build_graph
from torchwright_doom.render.inference import argmax_rows
from torchwright_doom.render.tokens_bridge import rows_to_input
from torchwright_doom.render_main import forward
from torchwright_doom.vocab import NO_OP

from ..prefill_fixture import TINY_BSP_SCENE, row_index

# Plan E's shared-slot layout shrank the residual peak; Plan F/G's projection +
# BBoxPruner grew it; Plan J's flat-pass span emission grew it again AND raised
# the d_head floor. Radixing the flat keys (see test_forward_compiles) brings the
# whole forward back to H's d=4096 / d_head=32 working point, so this exercises
# the compiled free-run at the real target config. (This compiles the full J
# forward — 85 layers — so the in-process compile_headless run is heavier than the
# pre-J traversal spine; it fits in ~12 GB.)
_D = 4096
_D_HEAD = 32
_MAX_STEPS = 8  # plenty for set_cursor + side precompute + descent on a 1-node scene


def _decode_type(row: int) -> str:
    t, _values = TOKEN_VOCAB.row_to_token[row]
    return t.name


def _exact_math_rollout(prefill_ids: list[int]) -> list[int]:
    """Free-run the graph in exact math (reference_eval), one token at a time."""
    iv = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    pos = create_pos_encoding()
    nt = forward(iv, GraphPast(input_vec=iv, pos_encoding=pos), pos)

    seq = list(prefill_ids)
    emitted: list[int] = []
    for _ in range(_MAX_STEPS):
        rows = torch.stack([W_EMBED[i] for i in seq])  # (len, d_embed)
        cache = reference_eval(nt, {"iv": rows}, len(seq))
        nxt = int(torch.argmax(cache[nt][-1] @ W_EMBED.t()).item())
        emitted.append(nxt)
        seq.append(nxt)
        if nxt == row_index(NO_OP, {}):
            break
    return emitted


def _build_compiled(device, *, d: int, d_head: int, max_layers: int = 200):
    """Compile the token-id forward in-process; returns ``(compiled, output_node)``.

    This test is the one place a compiled doom forward still runs in-process:
    production inference is the cached ONNX artifact, so the in-process compile
    lives here (the graph construction is shared via
    ``render.compiled_model.build_graph``; torchwright's ``compile_headless``
    is its in-process debug/test reference compiler).
    """
    from torchwright.compiler.export import compile_headless

    next_token, pos, _emb, _banks = build_graph()
    compiled = compile_headless(
        next_token,
        pos,
        d=d,
        d_head=d_head,
        max_layers=max_layers,
        verbose=False,
        device=str(device),
    )
    return compiled, next_token


def _compiled_rollout(prefill_ids: list[int], device) -> list[int]:
    """Free-run the compiled transformer: ids in, argmax out, id fed back.

    The gate owns its own AR loop: the shipped rollout harness
    (``render.inference``) speaks only the production owned-``KVCache``
    protocol (and is covered there by ``tests/render/test_spec_decode_logic.py``
    + ``test_windowed_cache.py``), while ``compile_headless`` threads
    grow-per-step KV tuples.  This test's job is the compiled-vs-exact-math
    trajectory, so it drives ``compiled.step`` directly.
    """
    compiled, _ = _build_compiled(device, d=_D, d_head=_D_HEAD, max_layers=200)
    terminal = row_index(NO_OP, {})
    out, past = compiled.step(rows_to_input(prefill_ids), compiled.empty_past(), past_len=0)
    emitted = [argmax_rows(out[-1:])[0]]
    while emitted[-1] != terminal and len(emitted) < _MAX_STEPS:
        pos = len(prefill_ids) + len(emitted) - 1
        out, past = compiled.step(rows_to_input([emitted[-1]]), past, past_len=pos)
        emitted.append(argmax_rows(out[-1:])[0])
    return emitted


def test_compiled_forward_free_runs_bsp_traversal(device) -> None:
    prefill_ids = [row_index(t, s) for t, s in TINY_BSP_SCENE]

    golden = _exact_math_rollout(prefill_ids)
    rollout = _compiled_rollout(prefill_ids, device)

    golden_names = [_decode_type(r) for r in golden]
    rollout_names = [_decode_type(r) for r in rollout]

    # The compiled free-run reproduces the exact-math free-run, token for token.
    assert rollout == golden, (
        "compiled AR rollout diverges from the exact-math free-run:\n"
        f"compiled={rollout_names}\nexact   ={golden_names}"
    )

    # And it did real traversal work autonomously before the projection boundary:
    # the seed transition, a side test, and a descent into the tree.
    assert golden_names[0] == "setCursorDirectionY", golden_names
    assert "R_PointOnSide" in golden_names, golden_names  # THINK_SIDE
    assert "pointOnSideResult" in golden_names, golden_names  # SIDE_RECORD
    assert "bspFront" in golden_names, golden_names  # TRAVERSE_ENTER
