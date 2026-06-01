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

Scope: the rollout walks ``BEGIN -> SET_CURSOR_DIRECTION_Y -> the side-bit
precompute -> TRAVERSE_ENTER -> the descent to a leaf``. At the first subsector
the projection owner would take over; it is deferred this phase (stubbed NO_OP),
so the run stops there — the honest boundary of the traversal-only spine. The
meaningful prefix (a real side test + a real descent, generated autonomously) is
what the loop is validated on.
"""

from __future__ import annotations

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED, build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward
from torchwright_doom.vocab import NO_OP

from ..prefill_fixture import TINY_BSP_SCENE, row_index

_D = 5120
_D_HEAD = 160
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


def _compiled_rollout(prefill_ids: list[int], device) -> list[int]:
    """Free-run the compiled transformer: ids in, argmax out, id fed back."""
    emb = build_doom_embedding("token_ids")
    pos = create_pos_encoding()
    nt = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)
    compiled = compile_headless(
        nt, pos, d=_D, d_head=_D_HEAD, max_layers=200, verbose=False, device=str(device)
    )

    w_embed_t = W_EMBED.t()
    past = compiled.empty_past()
    # Prefill in one shot, then decode one token at a time.
    prefill = torch.tensor([[float(i)] for i in prefill_ids], dtype=torch.float32)
    out, past = compiled.step(prefill, past, past_len=0)
    cur = int(torch.argmax(out[-1].cpu() @ w_embed_t).item())  # emission at BEGIN
    emitted = [cur]
    seq_pos = len(prefill_ids)
    for _ in range(_MAX_STEPS - 1):
        if cur == row_index(NO_OP, {}):
            break
        out, past = compiled.step(
            torch.tensor([[float(cur)]], dtype=torch.float32), past, past_len=seq_pos
        )
        seq_pos += 1
        cur = int(torch.argmax(out[-1].cpu() @ w_embed_t).item())
        emitted.append(cur)
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
