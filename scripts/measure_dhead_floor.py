"""Empirically bracket the d_head floor of the doom ``forward()`` graph.

d_head must be >= the widest single attention query/key. This compiles the whole
forward via ``compile_to_onnx`` at a candidate d_head and reports:

- ``PASS``         -- compiled; every attention key fits, so the floor <= candidate.
- ``FAIL (width)`` -- the compiler's "needs d_qk=N" assertion: a key is wider than
                      d_head. THIS is the genuine bracket-from-below.
- ``SKIP``         -- the candidate does not divide ``_D`` (the compiler requires
                      d % d_head == 0); an unrelated constraint, NOT a width bracket.
- ``FAIL (other)`` -- any other compile error (OOM, ...); NOT a width bracket.

So bracket the floor with *divisor* candidates only; a bare FAIL that is not
"(width)" tells you nothing about d_head.

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
from torchwright.ops.inout_nodes import create_rope_config

from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

_D = 6400


def try_compile(d_head: int) -> tuple[str, str]:
    # Skip non-divisor candidates up front: the compiler requires d % d_head == 0,
    # which would otherwise raise an AssertionError that looks like a FAIL but is
    # unrelated to key width (a false bracket).
    if _D % d_head != 0:
        return "SKIP", f"d={_D} is not divisible by d_head"
    emb = build_doom_embedding("token_ids")
    # PORT NOTE (RoPE): position is now a rotation inside attention, not a residual
    # node, so the graph is built against a RopeConfig whose d_head MUST equal the
    # compiled d_head (the compile entry points assert it). That couples the swept
    # d_head to graph construction — each candidate rebuilds its own graph here (it
    # already did) with rope d_head tracking the candidate and d_rot = d_head // 2
    # (production ratio: d_head 128 -> d_rot 64).
    #
    # This adds a SECOND floor that did not exist under the old sinusoidal pos. The
    # doom content heads need a NoPE tail (d_head - d_rot) >= 25, and d_rot must be
    # large enough for BOS-position monotonicity at max_positions, so candidates
    # below d_head ~64 now raise at *graph build* here (content-width / NoPE-tail /
    # BOS-monotonicity ValueErrors), before compile is ever reached. Those surface
    # as FAIL (other), NOT FAIL (width): they are the RoPE content-placement floor,
    # a different thing from the old "needs d_qk=N" attention-key-width bracket this
    # script was written to find. d_head must also be even (rotate_half); an odd
    # candidate raises "d_head must be even", also FAIL (other).
    rope = create_rope_config(d_head=d_head, max_positions=65536, d_rot=d_head // 2)
    next_token = forward(emb, GraphPast(input_vec=emb, rope=rope))
    with tempfile.TemporaryDirectory() as td:
        try:
            compile_to_onnx(
                next_token,
                embedding=emb,
                output_path=os.path.join(td, "f.onnx"),
                d=_D,
                d_head=d_head,
                max_layers=400,
                verbose=False,
            )
            return "PASS", "compiled"
        except Exception as e:  # noqa: BLE001 - reporting the bracket reason
            # Only the compiler's "needs d_qk=N" assertion is a genuine width
            # bracket; any other failure (OOM, ...) is unrelated to d_head.
            kind = "FAIL (width)" if "needs d_qk" in str(e) else "FAIL (other)"
            return kind, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    candidates = []
    for arg in sys.argv[1:]:
        try:
            candidates.append(int(arg))
        except ValueError:
            print(f"skipping non-integer argument: {arg!r}", file=sys.stderr)
    for dh in candidates or [64]:
        status, msg = try_compile(dh)
        print(f"d_head={dh:4d}: {status:13s} ({msg[:120]})")
