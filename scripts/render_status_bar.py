"""Render the baked E1M1 status bar to a PNG for eyeball verification.

Pure CPU asset preview (no model): composites the draw-list with the faithful
``V_DrawPatch`` blit and writes an upscaled PNG. Run locally:

    .venv/bin/python -m scripts.render_status_bar --scale 1 --out /tmp/bar.png
"""

from __future__ import annotations

import argparse

from torchwright_doom.hud_assets import bar_to_rgb, composite_bar
from torchwright_doom.inference.compare import _upscale, _write_png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=1, choices=(1, 2))
    ap.add_argument("--zoom", type=int, default=3, help="integer upscale for the PNG")
    ap.add_argument("--out", default="/tmp/status_bar.png")
    args = ap.parse_args()

    indices = composite_bar(args.scale)
    rgb = bar_to_rgb(indices)
    _write_png(args.out, _upscale(rgb, args.zoom))
    h, w = indices.shape
    unwritten = int((indices < 0).sum())
    print(f"scale={args.scale} bar={w}x{h} unwritten_cells={unwritten} -> {args.out}")


if __name__ == "__main__":
    main()
