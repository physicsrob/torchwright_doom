"""Local structural compile of the HUD-on forward (no render).

Validates that the status-bar spine compiles into a structurally valid
token->token transformer (the compiler's I1-I4 invariants are enforced during
compilation) and reports the layer count, so we catch spine wiring/scheduling
errors before spending a Modal render. Run HUD-on at the lowres scale:

    TORCHWRIGHT_DOOM_HUD=1 TORCHWRIGHT_DOOM_RENDER_SCALE=2 \
    TORCHWRIGHT_DOOM_DETAIL=low .venv/bin/python -m scripts.compile_hud_check
"""

from __future__ import annotations

import os
import tempfile

import onnx

from torchwright.compiler.export import compile_to_onnx
from torchwright.ops.inout_nodes import create_rope_config

from torchwright_doom.constants import HUD_ENABLED, SCREEN_WIDTH, SCREEN_HEIGHT
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward


def main() -> None:
    print(f"HUD_ENABLED={HUD_ENABLED} screen={SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    emb = build_doom_embedding("token_ids")
    # PORT NOTE (RoPE): the old d_head=32 is no longer feasible for the full doom
    # forward — under RoPE the rope d_head MUST equal the compiled d_head, and the
    # content heads need a NoPE tail (d_head - d_rot) >= 25 while d_rot must stay
    # large enough for BOS-position monotonicity at max_positions. d_head=64 /
    # d_rot=32 is the smallest verified-feasible pair (production is 128 / 64).
    rope = create_rope_config(d_head=64, max_positions=65536, d_rot=32)
    next_token = forward(emb, GraphPast(input_vec=emb, rope=rope))

    with tempfile.TemporaryDirectory() as d:
        onnx_path = os.path.join(d, "hud_forward.onnx")
        result = compile_to_onnx(
            next_token,
            embedding=emb,
            output_path=onnx_path,
            d=8192,
            d_head=64,
            max_layers=400,
            verbose=True,
        )
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("STRUCTURALLY VALID ONNX")
        n_layers = getattr(result, "n_layers", None)
        print(
            f"compile result: {result!r}" if n_layers is None else f"layers={n_layers}"
        )


if __name__ == "__main__":
    main()
