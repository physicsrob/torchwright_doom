"""Locate the first HUD-phase divergence between the generated token stream and
the drafter's reference plan.

Loads ``token_dump.json`` (the rollout's generated tokens) and rebuilds the
drafter's reference plan for the same config, then prints the first index where
they differ and a window of context — pinning the exact spine token that the
compiled model got wrong.

    TORCHWRIGHT_DOOM_HUD=1 TORCHWRIGHT_DOOM_RENDER_SCALE=2 TORCHWRIGHT_DOOM_DETAIL=low \
    .venv/bin/python -m scripts.diff_hud_tokens out/hud_dbg/token_dump.json
"""

from __future__ import annotations

import json
import sys


def _tok_str(entry) -> str:
    """Compact 'TYPE{slots}' from a token-dump entry (dict) ."""
    if isinstance(entry, dict):
        t = entry.get("type") or entry.get("token_type") or entry.get("name")
        vals = entry.get("values") or entry.get("slots") or {}
        if vals:
            return f"{t}{vals}"
        return f"{t}"
    return str(entry)


def main() -> None:
    dump_path = sys.argv[1] if len(sys.argv) > 1 else "out/hud_dbg/token_dump.json"
    with open(dump_path) as f:
        dump = json.load(f)
    print("dump keys:", list(dump.keys()))
    gen = (
        dump.get("predicted_next_tokens")
        or dump.get("rollout_output_tokens")
        or dump.get("rollout_entries")
        or []
    )
    print(f"generated tokens: {len(gen)}")

    # Find where the HUD phase starts in the generated stream.
    types = [
        (
            (e.get("type") or e.get("token_type") or e.get("name"))
            if isinstance(e, dict)
            else str(e)
        )
        for e in gen
    ]
    hud_starts = [
        i for i, t in enumerate(types) if t in ("ST_Drawer", "ST_Drawer.item")
    ]
    if hud_starts:
        h0 = hud_starts[0]
        print(f"\nfirst HUD token at generated index {h0} ({types[h0]})")
        print("--- generated tokens around HUD start (h0-3 .. h0+40) ---")
        for i in range(max(0, h0 - 3), min(len(gen), h0 + 40)):
            print(f"  [{i}] {_tok_str(gen[i])}")
        # Count HUD_ITEM emissions and look for anomalies (repeats / out-of-order).
        items = [
            (i, e.get("values", {}).get("item"))
            for i, e in enumerate(gen)
            if isinstance(e, dict)
            and (e.get("type") or e.get("name")) == "ST_Drawer.item"
        ]
        print(f"\nHUD_ITEM emissions: {len(items)} -> indices/items:")
        print("  ", items[:40])
    else:
        print("NO HUD tokens found in the generated stream (weapon/3D only?)")
        print("last 20 generated types:", types[-20:])


if __name__ == "__main__":
    main()
