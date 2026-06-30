"""Measure the COMPILED value of the bar's setCursorX decode at the screen edge.

When the bar appeared to stick on its last column, the suspected cause was a
value-decode error at the edge (screen-x = SCREEN_WIDTH-1). This probe reads the
actual compiled decode for setCursorX(x) at the edge (159 at lowres) vs interior
values (100, 157, 158) -- and finds it EXACT everywhere (setCursorX is a 1-digit
IntSlot, so there is no floor layer and no round-down). The real bug was a
`gt_screen` threshold units error, not the decode (see FINDINGS_status_bar.md).
Kept as the measurement-not-inference probe. Run HUD-on at the lowres geometry:

    TORCHWRIGHT_DOOM_HUD=1 TORCHWRIGHT_DOOM_RENDER_SCALE=2 TORCHWRIGHT_DOOM_DETAIL=low \
    .venv/bin/python -m scripts.probe_cursor_decode
"""

from __future__ import annotations

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import reference_eval
from torchwright.graph import fresh_graph_session
from torchwright.ops.inout_nodes import create_input

from torchwright_doom.constants import COLUMN_COUNT, PIXEL_WIDTH, SCREEN_WIDTH
from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.protocol_tokens import ProtocolTokenView
from torchwright_doom.std import constant
from torchwright_doom.vocab import SET_CURSOR_X


def _row_for_setcursorx(x: int) -> torch.Tensor:
    start, _end = TOKEN_VOCAB.type_to_row_range[SET_CURSOR_X]
    idx = int(x) - 0  # IntSlot(0, SCREEN_WIDTH); step index = value - lo
    return W_EMBED[start + idx : start + idx + 1].clone()


def main() -> None:
    print(
        f"SCREEN_WIDTH={SCREEN_WIDTH} COLUMN_COUNT={COLUMN_COUNT} "
        f"PIXEL_WIDTH={PIXEL_WIDTH}  (edge value = {SCREEN_WIDTH - 1})"
    )
    xs = [100, 157, 158, 159]
    with fresh_graph_session():
        d_embed = TOKEN_VOCAB.layout.d_embed
        inp = create_input("iv", d_embed)
        # prev-type slots are unused by the cursor pieces; constant dummies keep
        # the graph single-input ("iv" only).
        view = ProtocolTokenView(inp, constant(0.0), constant(0.0))
        out = view.cursor_x  # the raw setCursorX value decode

        rows = torch.cat([_row_for_setcursorx(x) for x in xs], dim=0)
        n = len(xs)
        oracle = reference_eval(out, {"iv": rows}, n)[out]
        compiled = compile_headless(out, d=2048, d_head=32)
        compiled(rows, debug=True)
        comp = compiled.debug_value(out)

        print(f"\n{'x':>5} | {'cursor_x(raw): oracle / compiled':>34}")
        print("-" * 44)
        for i, x in enumerate(xs):
            print(f"{x:>5} | {oracle[i, 0].item():15.3f} /{comp[i, 0].item():15.3f}")


if __name__ == "__main__":
    main()
