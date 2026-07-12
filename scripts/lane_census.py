"""FFN lane census: where the MLP hidden lanes go, by op family and by site.

Pure graph construction (no compile, no weights) — the same walk as
``scripts/widest_nodes.py``, but sized the way compute is actually paid
post-cutover: an FFN node's lane count is its ``gate_proj`` row count
(the torchwright ``b12bac3`` pattern; ``d_output`` measures residual
width, not lanes). Buckets by annotation subtree, by node name, and by
op-family name pattern, plus per-site cross-tabs for the floor / table
/ in_range families.

**Screen-env trap**: the graph modules read screen dims from env at
import; without the production env this measures the 60x50 hud-off
graph (materially different — 112k vs 147k total lanes). For
production-topology numbers run with:

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \\
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \\
    TORCHWRIGHT_DOOM_HUD=1 python -m scripts.lane_census

Run the command above to produce current production-topology numbers.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from torchwright.ops.inout_nodes import create_rope_config
from torchwright_doom.model.embedding import build_doom_embedding
from torchwright_doom.model.past import GraphPast
from torchwright_doom.model.render_main import forward


def _build():
    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=128, max_positions=65536, d_rot=64)
    return forward(emb, GraphPast(input_vec=emb, rope=rope))


def _collect(output_node):
    seen: dict[int, object] = {}
    stack = [output_node]
    while stack:
        n = stack.pop()
        if n.node_id in seen:
            continue
        seen[n.node_id] = n
        for inp in n.inputs:
            if inp.node_id not in seen:
                stack.append(inp)
    return list(seen.values())


# Op families by name pattern; first match wins.
_FAMILIES = [
    ("floor_int", ("floor_int",)),
    ("table_lookup_2d", ("table_lookup_2d",)),
    ("colormap_row", ("colormap_row",)),
    ("in_range", ("in_range",)),
    ("select-family", ("select", "cond_gate", "broadcast_select", "compare", "snap")),
    ("ray_count/scaled", ("ray_count", "ray_scaled")),
    ("reciprocal", ("reciprocal", "recip")),
    ("pwl", ("pwl_",)),
    ("sawtooth", ("sawtooth",)),
    ("visplane_instance", ("visplane_instance",)),
    ("bos_weight", ("bos_weight",)),
]


def _family(n) -> str:
    nm = (n.name or "").lower()
    for label, pats in _FAMILIES:
        if any(p in nm for p in pats):
            return label
    return "OTHER"


def _site_crosstab(ffn, lanes, pattern: str, title: str, with_widths: bool = True):
    print(f"\n=== {title} lanes by annotation prefix (depth 2) ===")
    agg: Counter = Counter()
    widths: dict[str, Counter] = defaultdict(Counter)
    for n in ffn:
        if pattern in (n.name or ""):
            key = "/".join((n.annotation or "<none>").split("/")[:2])
            agg[key] += lanes(n)
            widths[key][lanes(n)] += 1
    for a, w in agg.most_common():
        if with_widths:
            dist = ", ".join(
                f"{c}x{ww}"
                for ww, c in sorted(widths[a].items(), key=lambda kv: -kv[0])[:8]
            )
            print(f"  {w:>8}  {a:<40} [{dist}]")
        else:
            print(f"  {w:>8}  {a}")


def main():
    out = _build()
    nodes = _collect(out)
    print(f"reachable nodes: {len(nodes)}")

    by_type_c: Counter = Counter()
    for n in nodes:
        by_type_c[type(n).__name__] += 1
    print("\n=== node count by type ===")
    for t, c in by_type_c.most_common():
        print(f"  {c:>6}  {t}")

    ffn = [n for n in nodes if type(n).__name__ == "FFN"]
    relu = [n for n in nodes if "ReLU" in type(n).__name__]
    lanes = lambda n: n.gate_proj.shape[0]  # noqa: E731

    total = sum(lanes(n) for n in ffn)
    print(f"\nFFN nodes: {len(ffn)}   total lanes: {total}")
    if relu:
        print(
            f"!! legacy ReLU nodes still present: {len(relu)}, "
            f"width {sum(len(n) for n in relu)}"
        )

    for depth in (1, 2):
        agg_w: Counter = Counter()
        agg_c: Counter = Counter()
        for n in ffn:
            parts = (n.annotation or "<none>").split("/")
            key = "/".join(parts[:depth])
            agg_w[key] += lanes(n)
            agg_c[key] += 1
        print(f"\n=== FFN lanes by annotation prefix (depth {depth}, top 30) ===")
        for a, w in agg_w.most_common(30):
            print(f"  {w:>8} ({100.0*w/total:5.1f}%)  ({agg_c[a]:>5} nodes)  {a}")

    by_name: dict[tuple, int] = defaultdict(int)
    for n in ffn:
        by_name[(n.name or "<unnamed>", lanes(n))] += 1
    rows = sorted(by_name.items(), key=lambda kv: -(kv[0][1] * kv[1]))
    print("\n=== FFN lanes by name (top 60: count x lanes = total) ===")
    for (nm, w), cnt in rows[:60]:
        print(f"  {w*cnt:>8} = {cnt:>4} x {w:<6}  {nm}")

    fam_w: Counter = Counter()
    fam_c: Counter = Counter()
    for n in ffn:
        fam_w[_family(n)] += lanes(n)
        fam_c[_family(n)] += 1
    print("\n=== FFN lanes by op family (exact, whole graph) ===")
    for f, w in fam_w.most_common():
        print(f"  {w:>8} ({100.0*w/total:5.1f}%)  ({fam_c[f]:>5} nodes)  {f}")

    other: Counter = Counter()
    for n in ffn:
        if _family(n) == "OTHER":
            other[n.name or "<unnamed>"] += lanes(n)
    print("\n=== OTHER breakdown by name (top 30) ===")
    for nm, w in other.most_common(30):
        print(f"  {w:>8}  {nm}")

    _site_crosstab(ffn, lanes, "floor_int", "floor_int")
    _site_crosstab(ffn, lanes, "table_lookup_2d", "table_lookup_2d")
    _site_crosstab(ffn, lanes, "in_range", "in_range", with_widths=False)

    cm = sum(lanes(n) for n in ffn if "colormap_row" in (n.name or ""))
    cm_c = sum(1 for n in ffn if "colormap_row" in (n.name or ""))
    print(f"\ncolormap_row total: {cm} lanes across {cm_c} nodes")


if __name__ == "__main__":
    main()
