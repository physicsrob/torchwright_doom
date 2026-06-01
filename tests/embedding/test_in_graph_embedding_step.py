"""In-graph embedding + autoregressive ``.step()`` — the OQ-1 de-risk.

The teacher-forced renderer tests feed a *pre-embedded* 820-wide input row per
position (the host computes ``W_EMBED[row]``). A free-running renderer can't do
that: it must emit a token, turn it back into the next input, and step again.
The vanilla-LLM way to close that loop is an **in-graph embedding** — the graph
takes a 1-wide integer ``token_ids`` slot, an ``Embedding`` node gathers
``W_EMBED[id]`` *inside* the model, and the host only argmaxes the output to an
id and feeds it back (the re-embed is a table lookup inside the transformer, so
the host stays dumb).

That path was previously unvalidated end-to-end: no test compiled an
``Embedding``-bearing graph and ran ``.step()`` on it. This test closes that gap
with the smallest meaningful graph — an autoregressive counter:

    THINK_SIDE(node=k)  ->  THINK_SIDE(node=k+1)

Driving it free-running from ``THINK_SIDE(0)`` must walk ``THINK_SIDE(1)``,
``THINK_SIDE(2)``, … — which exercises, every step, the full loop the real
renderer needs: id -> in-graph embed -> read a slot off the embedded row ->
emit the next token -> argmax to an id -> feed the id back -> re-embed.
"""

from __future__ import annotations

import torch

from torchwright.compiler.export import compile_headless
from torchwright.ops.inout_nodes import create_pos_encoding

from torchwright_doom.embedding import W_EMBED, build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_ops import add_const
from torchwright_doom.std import make_token
from torchwright_doom.vocab import THINK_SIDE

from ..prefill_fixture import row_index

_STEPS = 5  # walk THINK_SIDE(0) -> ... -> THINK_SIDE(5); stays within node range


def test_in_graph_embedding_autoregressive_counter(device) -> None:
    # Graph: token id -> in-graph embedding -> read the `node` slot -> emit
    # THINK_SIDE(node + 1). No history needed; this isolates the embed/read/
    # emit/argmax/re-embed loop.
    emb = build_doom_embedding("token_ids")
    next_node = add_const(THINK_SIDE.extract(emb, "node"), 1.0)
    emit = make_token(THINK_SIDE, node=next_node)

    compiled = compile_headless(
        emit,
        create_pos_encoding(),
        d=2048,
        d_head=32,
        max_layers=80,
        verbose=False,
        device=str(device),
    )

    w_embed_t = W_EMBED.t()
    expected = [row_index(THINK_SIDE, {"node": k}) for k in range(_STEPS + 1)]

    # Free-running autoregressive rollout: feed one id, argmax the emitted row
    # back to an id, feed that id next. Exactly the host's job in a real loop.
    past = compiled.empty_past()
    cur_id = expected[0]
    produced = [cur_id]
    for t in range(_STEPS):
        flat = torch.tensor([[float(cur_id)]], dtype=torch.float32)
        out, past = compiled.step(flat, past, past_len=t)
        cur_id = int(torch.argmax(out[0].cpu() @ w_embed_t).item())
        produced.append(cur_id)

    assert produced == expected, (
        "in-graph-embedding AR rollout drifted from the counter sequence:\n"
        f"produced={produced}\nexpected={expected}"
    )


def test_step_matches_full_forward_with_cross_position_handle(device) -> None:
    """The KV-cached ``.step()`` loop must agree with a single full forward when
    the graph attends across positions — the second half of OQ-1.

    The counter above never reads history, so it doesn't exercise the KV cache.
    ``render_main`` reads prior positions via ``GraphPast`` handles (the cheapest
    being ``attend_to_offset``, used for the previous input type). This builds a
    forward that emits ``THINK_SIDE(node = <node slot one position back> + 1)``
    and asserts the incremental step loop emits the *same token at every
    position* as one full forward over the whole sequence — i.e. the KV cache +
    PosEncoding position-shift reproduce the cross-position attention exactly.
    """
    emb = build_doom_embedding("token_ids")
    past = GraphPast(input_vec=emb, pos_encoding=create_pos_encoding())
    node_here = past.publish("node", THINK_SIDE.extract(emb, "node"))
    prev_node = past.attend_to_offset(node_here, delta_pos=-1)
    emit = make_token(THINK_SIDE, node=add_const(prev_node, 1.0))

    compiled = compile_headless(
        emit,
        create_pos_encoding(),
        d=2048,
        d_head=32,
        max_layers=80,
        verbose=False,
        device=str(device),
    )

    # An arbitrary fixed input sequence (teacher-forced — we compare two ways of
    # running the *same* inputs, so the values themselves don't matter).
    ids = [row_index(THINK_SIDE, {"node": v}) for v in (2, 5, 3, 1)]
    w_embed_t = W_EMBED.t()

    full = torch.tensor([[float(i)] for i in ids], dtype=torch.float32)
    out_full = compiled(full)
    full_tokens = [
        int(torch.argmax(out_full[i].cpu() @ w_embed_t).item()) for i in range(len(ids))
    ]

    past_kv = compiled.empty_past()
    step_tokens = []
    for t, i in enumerate(ids):
        out, past_kv = compiled.step(
            torch.tensor([[float(i)]], dtype=torch.float32), past_kv, past_len=t
        )
        step_tokens.append(int(torch.argmax(out[0].cpu() @ w_embed_t).item()))

    assert step_tokens == full_tokens, (
        "KV-cached step loop disagrees with the full forward on a cross-position "
        f"(attend_to_offset) graph:\nstep={step_tokens}\nfull={full_tokens}"
    )
