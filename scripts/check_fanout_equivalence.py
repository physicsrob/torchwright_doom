"""De-risk the depth lever: does raising the dispatch max_fanout change the
emitted next token? type_switch's fold is a sum of gated heads; reassociating it
(serial vs tree vs flat) must be output-identical. This reference_eval's the
token-I/O forward at fanout=2 and fanout=None on TINY_BSP_SCENE and compares the
per-position next-token argmax. Memory-safe (exact math, no compile, n_pos=20).
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import torch

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_pos_encoding
from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED, build_doom_embedding
from torchwright_doom.past import GraphPast, PastHandleScope
from torchwright_doom.scene_index import SceneIndex
from torchwright_doom.protocol_tokens import ProtocolTokenView
from torchwright_doom.render_main import (
    publish_runtime_protocols,
    build_branch_outputs,
    _distinct_head_pairs,
)
from torchwright_doom.emit import emit_derived_zero
from torchwright_doom.std import concat as C, type_switch as TS


def build_forward(emb, pos, fanout):
    gp = GraphPast(input_vec=emb, pos_encoding=pos)
    scene = SceneIndex.build(emb, gp, pos)
    scope = PastHandleScope(gp)
    inp = ProtocolTokenView(
        emb,
        scope.attend_to_offset(scope.input_type(), delta_pos=-1),
        scope.attend_to_offset(scope.input_type(), delta_pos=-2),
    )
    protocols = publish_runtime_protocols(emb, scope, inp, scene, pos)
    branches = build_branch_outputs(inp, protocols)
    head = TS(*_distinct_head_pairs(inp, branches), max_fanout=fanout)
    return C(head, emit_derived_zero())


def main():
    sys.path.insert(0, str(_UMBRELLA / "torchwright_doom" / "tests"))
    from prefill_fixture import TINY_BSP_SCENE, row_index as _row_index

    ids = [_row_index(t, s) for t, s in TINY_BSP_SCENE]
    ids_col = torch.tensor([[float(i)] for i in ids], dtype=torch.float32)
    n_pos = len(ids)
    w_t = W_EMBED.t()

    from torchwright_doom.graph_debug import silenced_graph_asserts

    results = {}
    with silenced_graph_asserts():  # garbage candidates trip range asserts
        for fanout in (2, None):
            pos = create_pos_encoding()
            emb = build_doom_embedding("token_ids")
            nt = build_forward(emb, pos, fanout)
            cache = reference_eval(nt, {"token_ids": ids_col}, n_pos)
            results[fanout] = [
                int(torch.argmax(cache[nt][i] @ w_t).item()) for i in range(n_pos)
            ]

    a, b = results[2], results[None]
    mism = [(i, a[i], b[i]) for i in range(n_pos) if a[i] != b[i]]
    print(f"fanout=2 vs flat: {n_pos} positions, {len(mism)} mismatches")
    for i, x, y in mism:
        print(
            f"  pos {i}: {TOKEN_VOCAB.row_to_token[x][0].name} != {TOKEN_VOCAB.row_to_token[y][0].name}"
        )
    print("RESULT: IDENTICAL" if not mism else "RESULT: DIFFERS")


if __name__ == "__main__":
    main()
