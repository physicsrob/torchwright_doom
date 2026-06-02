"""Validate the streaming + sparse token-I/O compile of the doom forward.

compile_to_onnx streams per layer ("one dense layer's worth regardless of depth")
and stores weights as sparse ONNX inits — so it should compile the doom forward at
a small fraction of compile_headless's ~28GB (all dense layers resident). This
also exercises the real token-I/O artifact (token_ids -> logits, KV-cached).

Run with peak-RSS measurement:
    D=6400 /usr/bin/time -v python -m scripts.probe_onnx_compile
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from torchwright.compiler.export import compile_to_onnx
from torchwright.ops.inout_nodes import create_pos_encoding
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

D = int(os.environ.get("D", "6400"))
D_HEAD = int(os.environ.get("D_HEAD", "160"))
out_path = os.environ.get("OUT", "/tmp/doom_forward.onnx")

emb = build_doom_embedding("token_ids")
pos = create_pos_encoding()
fanout_env = os.environ.get("FANOUT")
if fanout_env is None:
    nt = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)
else:
    # Rebuild dispatch with a custom max_fanout (mirrors render_main.forward).
    from torchwright_doom.protocol_tokens import ProtocolTokenView
    from torchwright_doom.scene_index import SceneIndex
    from torchwright_doom.render_main import (
        publish_runtime_protocols, build_branch_outputs, _distinct_head_pairs,
    )
    from torchwright_doom.emit import emit_derived_zero
    from torchwright_doom.std import concat as _C, type_switch as _TS
    from torchwright_doom.past import PastHandleScope
    fanout = None if fanout_env.lower() in ("none", "0", "full") else int(fanout_env)
    gp = GraphPast(input_vec=emb, pos_encoding=pos)
    scene = SceneIndex.build(emb, gp, pos)
    scope = PastHandleScope(gp)
    inp = ProtocolTokenView(
        emb,
        scope.attend_to_offset(scope.input_type(), delta_pos=-1),
        scope.attend_to_offset(scope.input_type(), delta_pos=-2),
    )
    protocols = publish_runtime_protocols(emb, scope, inp, scene, pos)
    branches = build_branch_outputs(protocols)
    nt = _C(_TS(*_distinct_head_pairs(inp, branches), max_fanout=fanout),
            emit_derived_zero())

opt = int(os.environ.get("OPT", "0"))
trim = os.environ.get("TRIM", "1") == "1"
print(f"=== compile_to_onnx token-I/O: d={D} d_head={D_HEAD} fanout={fanout_env} "
      f"optimize={opt} trim_heads={trim} ===")
t0 = time.perf_counter()
compile_to_onnx(nt, pos, embedding=emb, output_path=out_path, d=D, d_head=D_HEAD,
                max_layers=400, verbose=True, trim_heads=trim, optimize=opt)
print(f"compile wall time: {time.perf_counter()-t0:.1f}s")
if os.path.exists(out_path):
    print(f"onnx size: {os.path.getsize(out_path)/1e6:.1f} MB")
