"""Why doesn't table_lookup_2d's range-aware clamp skip fire on doom?

Monkeypatches torchwright's ``_clamp_or_skip`` to log every decision
(op name, the index node's static value range, gate_mult, top, and
whether the clamp was skipped) while building the production forward
graph.  No lowering/scheduling — construction is where the decision
happens.

**Screen-env trap**: run under the production env or you measure the
60x50 hud-off graph:

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \\
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \\
    TORCHWRIGHT_DOOM_HUD=1 python -m scripts.clamp_range_probe
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from torchwright.ops.swiglu import map_select

_orig = map_select._clamp_or_skip
_decisions: list[tuple] = []


def _logging_clamp_or_skip(index, top, gate_mult, name):
    r = index.value_type.value_range
    out = _orig(index, top, gate_mult, name)
    skipped = out[0] is index
    _decisions.append((name, r, gate_mult, top, skipped, index))
    return out


map_select._clamp_or_skip = _logging_clamp_or_skip

from torchwright.ops.inout_nodes import create_rope_config
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward


def _describe(n, depth=0, max_depth=3):
    pad = "    " * depth
    name = getattr(n, "name", "") or type(n).__name__
    try:
        r = n.value_type.value_range
    except Exception as exc:  # noqa: BLE001 - report, don't die mid-dump
        r = f"<range error: {exc!r}>"
    print(f"{pad}{name}  d={n.d_output}  range={r}")
    if depth < max_depth:
        for inp in getattr(n, "inputs", []) or []:
            _describe(inp, depth + 1, max_depth)


def main():
    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=128, max_positions=65536, d_rot=64)
    forward(emb, GraphPast(input_vec=emb, rope=rope))

    kept = [d for d in _decisions if not d[4]]
    skipped = [d for d in _decisions if d[4]]
    print(
        f"\n=== clamp decisions: {len(_decisions)} total, "
        f"{len(skipped)} skipped, {len(kept)} kept ==="
    )
    for name, r, g, top, s, _ in _decisions:
        need_lo = g * r.lo if r.is_finite() else float("nan")
        need_hi = g * r.hi if r.is_finite() else float("nan")
        verdict = "SKIP" if s else "KEEP"
        print(
            f"[{verdict}] {name}: range={r} gate_mult={g} top={top} "
            f"-> g*range=[{need_lo:.3f}, {need_hi:.3f}] vs [0, {top}]"
        )

    print("\n=== input trees of KEPT clamps (first 12) ===")
    seen: set[int] = set()
    shown = 0
    for name, r, g, top, _, index in kept:
        if id(index) in seen or shown >= 12:
            continue
        seen.add(id(index))
        shown += 1
        print(f"\n--- {name} (need g*x in [0, {top}], g={g}) ---")
        _describe(index)


if __name__ == "__main__":
    main()
