"""Empirically bracket the d_head floor of the doom ``forward()`` graph.

d_head must be >= the widest single attention query/key. This compiles the whole
forward via ``compile_to_onnx`` at a candidate d_head and reports whether it
succeeds: a successful compile proves every attention key fits in that d_head
(so the floor is <= the candidate); a failure (key wider than d_head) brackets it
from below. Run with a few candidates to bracket the floor.

Usage:
    python -m scripts.measure_dhead_floor 64 48 32
"""

from __future__ import annotations

import os
import sys
import tempfile
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

_D = 6400


def try_compile(d_head: int) -> tuple[bool, str]:
    emb = build_doom_embedding("token_ids")
    pos = create_pos_encoding()
    next_token = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)
    with tempfile.TemporaryDirectory() as td:
        try:
            compile_to_onnx(
                next_token,
                pos,
                embedding=emb,
                output_path=os.path.join(td, "f.onnx"),
                d=_D,
                d_head=d_head,
                max_layers=400,
                verbose=False,
            )
            return True, "compiled"
        except Exception as e:  # noqa: BLE001 - reporting the bracket reason
            return False, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    candidates = [int(a) for a in sys.argv[1:]] or [64]
    for dh in candidates:
        ok, msg = try_compile(dh)
        print(f"d_head={dh:4d}: {'PASS' if ok else 'FAIL'}  ({msg[:120]})")
