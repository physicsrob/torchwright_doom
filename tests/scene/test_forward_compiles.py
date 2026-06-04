"""The whole renderer ``forward()`` compiles to the real token-I/O artifact.

Migrated from ``compile_headless`` to ``compile_to_onnx``. ``compile_headless`` is
residual-I/O and body-only (no embedding / unembed), and it holds *every* layer's
dense weights resident — so at the doom forward's natural residual width it OOMs
the box (attention alone is ~4·d² per layer × ~44-66 layers). ``compile_to_onnx``
is the **real autoregressive artifact** (token_ids → logits, KV-cached) and it
**streams**: it extracts + sparsifies + frees each layer as it compiles, so the
whole forward compiles at ~one dense layer's worth (~2 GB) and writes a compact
sparse ONNX (~55 MB; ~22% of the dense weight capacity is non-zero).

This gate validates that the entire forward **compiles into a structurally valid
token→token transformer**: the compiler's I1–I4 invariants are enforced during
compilation, so a successful ``compile_to_onnx`` + ``onnx.checker`` == a
structurally correct artifact. The forward's *graph math* is validated separately
by the ``reference_eval`` oracle gates (``test_{projection,bbox,traversal}_oracle``).

It is also the regression guard for the dispatch output head: the literal sandbox
``type_switch`` (one full ``d_embed`` row per branch) needs a ~53k-wide residual;
the head-gated ``max_fanout`` reduction here compiles at a modest ``d``.

NOT validated here: running the compiled model (compiled-value / PL-noise fidelity).
The doom transformer's weights densify to >26 GB, so onnxruntime ``bad_alloc``s
just loading it on a 30 GB box (see ``scripts/probe_onnx_inference.py``) — inference
validation belongs on a larger machine. The in-process free-run on a tiny scene
(``test_forward_ar_rollout``) is the one compiled-behavior check that fits locally.
"""

from __future__ import annotations

import os

import onnx

from torchwright.compiler.export import compile_to_onnx
from torchwright.ops.inout_nodes import create_pos_encoding

from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

# compile_to_onnx streams, so d only sets the cramming point, not peak memory.
# d=6400, d_head=160 (a multiple of 160; covers the widest attention key — the
# traversal-edge one-hot of width N_ENTITY_MAX + N_DEPTH_MAX = 145). At the
# fanout=8 dispatch this lands ~34 layers (was ~44 before the world-angle
# 4-atans-to-1 collapse — the three eliminated 1024-wide atans had forced the
# scheduler to spread the dispatch across more layers under width pressure).
_D = 6400
_D_HEAD = 160


def test_forward_compiles_to_onnx(tmp_path) -> None:
    emb = build_doom_embedding("token_ids")
    pos = create_pos_encoding()
    next_token = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)

    onnx_path = os.path.join(tmp_path, "doom_forward.onnx")
    compile_to_onnx(
        next_token,
        pos,
        embedding=emb,
        output_path=onnx_path,
        d=_D,
        d_head=_D_HEAD,
        max_layers=400,
        verbose=False,
    )

    assert os.path.exists(onnx_path), "compile_to_onnx wrote no ONNX file"
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)  # structurally valid ONNX (I1–I4 already enforced)

    # The token-I/O / KV-cache contract: token_ids + per-layer past in, logits out.
    in_names = {i.name for i in model.graph.input}
    out_names = {o.name for o in model.graph.output}
    assert "token_ids" in in_names, in_names
    assert "past_len" in in_names, in_names
    assert "logits" in out_names, out_names

    # Layer count: the fanout=8 dispatch reduction lands ~34 after the
    # world-angle 4-atans-to-1 collapse (was ~44 before it; ~66 at fanout=2). A
    # ±8 band around the expected count catches a fanout regression (back toward
    # ~66) or a blow-up while tolerating scheduler jitter — and, tightened here,
    # also catches a regression that merely undoes this collapse's depth win
    # (the old 48 ceiling would have let ~44 slip through).
    n_layers = sum(1 for n in in_names if n.startswith("past_K_"))
    assert 26 <= n_layers <= 42, f"unexpected compiled layer count {n_layers}"
