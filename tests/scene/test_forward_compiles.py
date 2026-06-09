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
# d=4096, d_head=32 — Phase H's original working point. Phase J's flat-pass span
# emission adds attention keys that were no_op in H, which both (a) exceeded the
# d_head=32 floor and (b) raised the d_embed residual demand (Risk #8's expected
# width bump). Radixing the flat keys fixes BOTH axes back to H's point:
#   * column_range (was d_qk 101): lifted (plane, vp) instance + radix screen
#     column + sentinel bias (visplane_state.py), d_qk 20;
#   * min_x / max_x / next_vp_after (were 41-42): the plane equality radixed to a
#     (bucket, digit) one-hot pair, d_qk 21-22;
#   * next_plane_after (was a width-(N_PLANES_MAX+1) argmin-above, d_head 35): a
#     3-stage radix successor over the used-plane set, heads d_qk <= 14;
#   * flat_span_x1 (was d_qk 91): the SCREEN_HEIGHT-wide opening membership split
#     into d_head-sized dense row-chunks (flat_state.py), heads d_qk 26.
# Beyond holding d_head=32, narrowing those keys removed their wide
# key-construction nodes from the residual, so the *real* forward_compile fits at
# d=4096 again — bracketed: d=3584 deadlocks ("No progress"), d=4096 schedules.
# (The standalone SCHED_ONLY heuristic is a CP-SAT warm-start seed, far more
# conservative — it deadlocks ~4800 — and is NOT the real d_embed demand.) The
# H-side keys (R_CheckPlane occupied_key, ClipMemory) were already radixed/lifted
# to d_qk <= 32 in Phase H.
_D = 4096
_D_HEAD = 32


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

    # The token-I/O static-cache contract: token_ids + cache_position +
    # per-layer full-S past in, logits out (past_len/Concat is gone — the
    # CUDA-graph-capturable contract, plan_cuda_graph_decode.md).
    in_names = {i.name for i in model.graph.input}
    out_names = {o.name for o in model.graph.output}
    assert "token_ids" in in_names, in_names
    assert "cache_position" in in_names, in_names
    assert "past_len" not in in_names, in_names
    assert "logits" in out_names, out_names

    # The static slot count is baked into past_K_0's first dim.
    past_k0 = next(i for i in model.graph.input if i.name == "past_K_0")
    first_dim = past_k0.type.tensor_type.shape.dim[0]
    assert first_dim.HasField("dim_value") and first_dim.dim_value >= 1, (
        "past_K_0 first dim must be a static cache_stride, got "
        f"{first_dim}"
    )

    # Layer count: Phase J's flat pass lands the forward at 85 layers at d=4096
    # (H was ~45). The jump is the per-position flat-pass compute that was no_op
    # in H — the R_MapPlane cursor PWL chain, R_MakeSpans open/close, and the
    # next_plane_after radix successor's H2 -> H3 data dependency — not the
    # dispatch fold (still max_fanout=8). It's a few layers deeper than at a
    # looser d (81 at d=5120) because the tight residual forces more serialization.
    # Keep the ceiling tight enough to catch a dispatch-fanout regression (a serial
    # fold would add ~13).
    n_layers = sum(1 for n in in_names if n.startswith("past_K_"))
    assert 26 <= n_layers <= 90, f"unexpected compiled layer count {n_layers}"
