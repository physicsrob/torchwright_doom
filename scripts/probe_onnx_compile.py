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
nt = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)

print(f"=== compile_to_onnx (token-I/O, streaming+sparse): d={D} d_head={D_HEAD} ===")
t0 = time.perf_counter()
trim = os.environ.get("TRIM", "1") == "1"
print(f"[trim_heads={trim}]")
compile_to_onnx(nt, pos, embedding=emb, output_path=out_path, d=D, d_head=D_HEAD,
                max_layers=400, verbose=True, trim_heads=trim)
print(f"compile wall time: {time.perf_counter()-t0:.1f}s")
if os.path.exists(out_path):
    print(f"onnx size: {os.path.getsize(out_path)/1e6:.1f} MB")
