"""Reproduce the compile-gate scheduler deadlock with a verbose layer trace.

``tests/scene/test_forward_compiles.py::test_forward_compiles_to_onnx``
fails against post-B1 torchwright with ``No progress: 2095 nodes
remaining, 0 free columns`` at d=4096.  This probe compiles the same
graph at the same geometry with ``verbose=True``, so the per-layer
column-occupancy trajectory (occupied before/after, MLP slot usage) and
the deadlock layer are visible.  The gate compiles at ``optimize=0`` —
the static scheduler, not the CP-SAT path — so the trace localizes a
static scheduling/admission change, and re-running it against different
torchwright checkouts (the Modal image mounts the sibling checkout)
brackets which commit moved the trajectory.

Run:

    MODAL_RUN_CPU=8 MODAL_RUN_MEMORY=32768 make modal-run \
        MODULE=scripts.compile_gate_probe CPU_ONLY=1

Deliberately mirrors the test exactly — same d/d_head/d_rot and no
screen-env overrides (this is the gate's graph, not the flagship
config).
"""

from __future__ import annotations

import tempfile
import traceback

from torchwright.compiler.export import compile_to_onnx
from torchwright.ops.inout_nodes import create_rope_config

from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

# tests/scene/test_forward_compiles.py geometry, verbatim.
_D = 4096
_D_HEAD = 64
_D_ROT = 32


def main() -> None:
    # The Modal image mounts the LOCAL sibling torchwright checkout (no git
    # metadata survives the mount) — record which SHA was checked out from
    # the driver side when comparing runs.
    print(f"[probe] geometry: d={_D} d_head={_D_HEAD} d_rot={_D_ROT} optimize=0")

    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=_D_HEAD, max_positions=65536, d_rot=_D_ROT)
    next_token = forward(emb, GraphPast(input_vec=emb, rope=rope))
    print("[probe] graph built, compiling...")

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        try:
            compile_to_onnx(
                next_token,
                embedding=emb,
                output_path=f.name,
                d=_D,
                d_head=_D_HEAD,
                max_layers=400,
                verbose=True,
                rms_norm_const_exp=63,
            )
        except RuntimeError as e:
            print(f"\n[probe] DEADLOCK: {e}")
            traceback.print_exc()
            return
    print("\n[probe] compile SUCCEEDED")


if __name__ == "__main__":
    main()
