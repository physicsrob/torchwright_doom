"""Spike: how does the baked texture count affect the compiled layer count?

Patches ``asset_config``'s texture/flat name lists BEFORE the import cascade
(asset_banks -> assets -> vocab -> embedding -> forward all read it), so the
whole chain — wall banks, lookup tables, vocab tex-id cardinality, W_EMBED —
shrinks consistently. Then runs ONLY the heuristic scheduler
(``schedule_only_capture``: no weight tensors, no GPU) and reports the layer
count + a texture/pixel-bucket attribution.

    python -m scripts.k_texture_cost --keep-wall 5 --keep-flat 3 --seed 0
    python -m scripts.k_texture_cost            # full set
    python -m scripts.k_texture_cost --all-wall # every wall texture in the WAD
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
_UMBRELLA = Path(__file__).resolve().parents[2]
if str(_UMBRELLA) not in sys.path:
    sys.path.insert(0, str(_UMBRELLA))

_TEX_PATS = (
    "tex", "flat", "pixel", "colormap", "playpal", "uv", "texel", "wall_col",
    "lookup", "bank", "asset", "light", "span_row", "_u_", "wallcol",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--keep-wall", type=int, default=None, help="keep this many wall textures (random subset)")
    p.add_argument("--keep-flat", type=int, default=None, help="keep this many flats (random subset)")
    p.add_argument("--all-wall", action="store_true", help="use every wall texture name in the WAD")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d", type=int, default=4096)
    p.add_argument("--d-head", type=int, default=32, dest="d_head")
    args = p.parse_args()

    # --- patch asset_config BEFORE the cascade imports it ---
    import torchwright_doom.asset_config as ac

    rng = random.Random(args.seed)

    if args.all_wall:
        from torchwright_doom.wad_assets import WADReader

        wad = WADReader(str(_UMBRELLA / "torchwright_doom" / "doom1.wad"))
        ac.WALL_TEXTURE_NAMES = tuple(sorted(wad._texture_defs().keys()))
    elif args.keep_wall is not None:
        names = list(ac.WALL_TEXTURE_NAMES)
        rng.shuffle(names)
        ac.WALL_TEXTURE_NAMES = tuple(sorted(names[: args.keep_wall]))
    if args.keep_flat is not None:
        fnames = list(ac.FLAT_NAMES)
        rng.shuffle(fnames)
        ac.FLAT_NAMES = tuple(sorted(fnames[: args.keep_flat]))

    ac.N_WALL_TEXTURES = len(ac.WALL_TEXTURE_NAMES)
    ac.N_FLATS = len(ac.FLAT_NAMES)

    # --- now build the (patched) forward graph ---
    import torchwright.graph.node as _node_module
    from torchwright.ops.inout_nodes import create_pos_encoding
    from torchwright_doom.embedding import TOKEN_VOCAB, build_doom_embedding
    from torchwright_doom.past import GraphPast
    from torchwright_doom.render_main import forward
    from torchwright_doom.asset_banks import WALL_BANKS

    from scripts.analyze_forward_cost import schedule_only_capture

    n_banks = len(WALL_BANKS)
    sizes = sorted({(b.width, b.height) for b in WALL_BANKS if b.bank_id != 0})

    _node_module.global_node_id = 0
    in_node = build_doom_embedding("token_ids")
    pos = create_pos_encoding()
    nt = forward(in_node, GraphPast(input_vec=in_node, pos_encoding=pos),
                 create_pos_encoding())

    res = schedule_only_capture(nt, pos, d=args.d, d_head=args.d_head)
    n_layers, node_to_layer = res[0], res[1]
    id_to_node = res[2]
    peak_width = res[3]

    # texture/pixel-bucket layer span
    tex_layers = set()
    for nid, layer in node_to_layer.items():
        nm = (getattr(id_to_node.get(nid), "name", "") or "").lower()
        if any(pat in nm for pat in _TEX_PATS):
            tex_layers.add(layer)

    print(
        f"[tex-cost] n_wall={ac.N_WALL_TEXTURES:3d} n_flat={ac.N_FLATS:2d} "
        f"n_wall_banks={n_banks} sizes={sizes}  d={args.d}  "
        f"d_embed={TOKEN_VOCAB.layout.d_embed}  "
        f"TOTAL_LAYERS={n_layers}  peak_resid_width={peak_width}  "
        f"texture/pixel-bucket layers touched={len(tex_layers)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
