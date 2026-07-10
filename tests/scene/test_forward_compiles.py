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

It is also the regression guard for the dispatch output head: the literal
``type_switch`` (one full ``d_embed`` row per branch) needs a ~53k-wide residual;
the head-gated ``max_fanout`` reduction here compiles at a modest ``d``.

NOT validated here: running the compiled model (compiled-value / PL-noise fidelity).
The doom transformer's weights densify to >26 GB, so ONNX Runtime exhausts a
30 GB box just loading it — inference
validation belongs on a larger machine. The in-process free-run on a tiny scene
(``test_forward_ar_rollout``) is the one compiled-behavior check that fits locally.
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
# Under RoPE the content rides the head grid, so d_head must cover the widest
# content head (~28); d_head=64 / d_rot=32 (NoPE tail = 32) is the lightest working
# point, with d=4096 holding the production 64 head-slots.  (Production export uses
# d_head=128/d_rot=64/d=8192; this gate validates the structural compile at the
# lighter pair.)
# 2026-07-06, univariate collapse (torchwright docs/univariate_collapse_plan.md):
# a collapsed subgraph materializes ALL its boundary members one sublayer above
# the source instead of progressively along the chain, so outputs whose
# consumers sit far downstream stay live longer — higher simultaneous residual
# pressure. The optimize=0 static schedule deadlocked at the old d=4096
# cramming point with the pass on ("No progress: 321 nodes remaining");
# d=4608 schedules with the pass on and off. (The production CP-SAT compile
# at d=8192 / optimize=3 is unaffected: 37 layers, status OPTIMAL, both ways.)
_D = 4608
_D_HEAD = 64
_D_ROT = 32


# The d=4096 atomic-FFN-placement deadlock (torchwright 726f349; xfail'd here
# 2026-07-03) dissolved with the swiglu machine flip: multiply lanes replaced
# the relu-era grid banks, dropping the residual pressure the optimize=0
# static schedule wedged on. The strict xfail XPASSed at the D2 cutover gate
# and was removed per its own instruction.
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
    # per-layer past PREFIX VIEWS in, logits out (past_len/Concat is gone —
    # the CUDA-graph-capturable contract, plan_cuda_graph_decode.md; the
    # past first dim is the SYMBOLIC cache_slots so one graph serves any
    # attention-window bucket, plan_stride_bucketing.md).
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
