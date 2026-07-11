"""The whole renderer compiles to a structurally valid token-I/O artifact.

``compile_to_onnx`` exercises embedding, the complete forward graph, unembedding,
and the KV-cache interface while streaming layer weights to keep memory bounded.
Graph math and compiled behavior are covered separately by the reference oracles
and the short in-process autoregressive rollout.
"""

from __future__ import annotations

import json
import os

import onnx

from torchwright.compiler.export import compile_to_onnx
from torchwright.ops.inout_nodes import create_rope_config

from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

# This lighter-than-production geometry keeps the structural compile gate
# tractable while retaining enough residual and head width for the full graph.
_D = 4608
_D_HEAD = 64
_D_ROT = 32


def test_forward_compiles_to_onnx(tmp_path) -> None:
    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=_D_HEAD, max_positions=65536, d_rot=_D_ROT)
    next_token = forward(emb, GraphPast(input_vec=emb, rope=rope))

    onnx_path = os.path.join(tmp_path, "doom_forward.onnx")
    compile_to_onnx(
        next_token,
        embedding=emb,
        output_path=onnx_path,
        d=_D,
        d_head=_D_HEAD,
        max_layers=400,
        verbose=False,
        # d=4608 is not a power of two, so the pinned-constant identity
        # RMSNorm cannot be emitted — export without the norm (the examples'
        # convention for odd widths).  The production export (d=8192, power
        # of two) keeps the norm with q=63; this gate validates the
        # structural compile, not the norm.
        rms_norm=False,
    )

    assert os.path.exists(onnx_path), "compile_to_onnx wrote no ONNX file"
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)  # structurally valid ONNX (I1–I4 already enforced)

    # The token-I/O static-cache contract: token_ids + cache_position +
    # per-layer past PREFIX VIEWS in, logits out. The past first dimension is
    # the symbolic cache_slots, so one CUDA-graph-capturable artifact serves
    # any attention-window bucket.
    in_names = {i.name for i in model.graph.input}
    out_names = {o.name for o in model.graph.output}
    assert "token_ids" in in_names, in_names
    assert "cache_position" in in_names, in_names
    assert "past_len" not in in_names, in_names
    assert "logits" in out_names, out_names

    # The slot dim is symbolic (stride bucketing); the full stride S lives
    # in the sidecar meta, not the input shape.
    past_k0 = next(i for i in model.graph.input if i.name == "past_K_0")
    first_dim = past_k0.type.tensor_type.shape.dim[0]
    assert first_dim.HasField("dim_param") and first_dim.dim_param == "cache_slots", (
        "past_K_0 first dim must be the symbolic cache_slots, got " f"{first_dim}"
    )
    meta_path = onnx_path.replace(".onnx", ".meta.json")
    with open(meta_path) as f:
        sidecar = json.load(f)
    assert (
        isinstance(sidecar.get("cache_stride"), int) and sidecar["cache_stride"] >= 1
    ), f"sidecar must carry the full cache_stride, got {sidecar.get('cache_stride')!r}"

    # Layer count: Phase J's flat pass lands the forward at 85 layers at d=4096
    # (H was ~45). The jump is the per-position flat-pass compute that was no_op
    # in H — the R_MapPlane cursor PWL chain, R_MakeSpans open/close, and the
    # next_plane_after radix successor's H2 -> H3 data dependency — not the
    # dispatch fold (still max_fanout=8). It's a few layers deeper than at a
    # looser d (81 at d=5120) because the tight residual forces more serialization.
    # Keep the ceiling tight enough to catch a dispatch-fanout regression (a serial
    # fold would add ~13).
    # Upper bound raised from 90 to 100 for RoPE global recency: the
    # ``global_position_from_bos`` readout adds one MLP sublayer (the BOS-weight
    # → position PWL inversion) plus its attention head, landing the count at ~92.
    # Still tight enough to catch a dispatch-fanout regression (a serial fold
    # would add ~13, past 100).
    n_layers = sum(1 for n in in_names if n.startswith("past_K_"))
    assert 26 <= n_layers <= 100, f"unexpected compiled layer count {n_layers}"
