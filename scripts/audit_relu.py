"""Audit ReLU-unit usage in the compiled doom ``forward()`` graph.

Every approximate op lowers to one or more ``Linear -> ReLU -> Linear`` sublayers
(``torchwright.ops.linear_relu_linear``). The ``ReLU`` node's width
(``len(node)``) is the number of ReLU units that sublayer computes *per
position* — e.g. a ``floor_int`` over 255 integer steps builds a
``floor_int_step`` ReLU of width 510 (two units per step) plus a
``floor_int_saturate`` ReLU of width 255.

**ReLU units are a distinct cost axis from residual width.** ``analyze_forward_cost``
measures the persistent residual stream (live columns across layers); this script
measures the transient MLP-hidden compute (the ReLU activations inside each
sublayer). The two move independently: the route-then-emit-once collapse barely
changed the residual peak but cut ReLU units massively (24 floor_int quads -> 2).

This is a GRAPH property (no compile, memory-safe ~graph construction only). The
script walks ``forward()``, inventories every ReLU node, and aggregates several
ways. The classification ruleset (``_OP_RULES``) is meant to be edited — anything
unmatched surfaces as ``OTHER:<name>`` so nothing hides.

Usage:
    python -m scripts.audit_relu                 # the doom forward() graph
    python -m scripts.audit_relu --top 30        # show more of each table
    python -m scripts.audit_relu --csv out.csv   # dump the raw inventory
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from torchwright.graph.relu import ReLU
from torchwright.ops.inout_nodes import create_rope_config
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

# --- op classification (by the linear_relu_linear `name=` label) ------------
# Ordered, first-match-wins, matched against the ReLU node's name. The labels
# come from the `name=` argument each op passes to linear_relu_linear (the
# ReLU node is named f"{name}_relu"). Edit freely; OTHER:<norm> catches misses.
_OP_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("floor_int", ("floor_int",)),  # digit-quad high byte + any continuous floor
    ("thermometer_floor_div", ("thermometer",)),
    # atan2 octant: signed_world_angle's ray-count staircase (1024-wide). Before
    # "multiply" so ray_* doesn't fall through (no overlap, but keep it explicit).
    (
        "atan/ray (octant)",
        ("ray_count", "ray_scaled", "atan", "octant", "signed_world"),
    ),
    ("multiply", ("mul_", "multiply", "_prod")),  # quarter-square products
    ("broadcast_select", ("broadcast_select",)),  # before "select" (substring)
    ("select", ("select",)),  # two-way + select_cond_gates
    ("cond_gate", ("cond_gate",)),
    ("square", ("square",)),
    ("reciprocal", ("reciprocal", "_norm1", "_norm2", "recip", "_inv")),
    ("table_lookup", ("staircase", "_2d", "_flatten", "lookup", "map_to_table")),
    ("piecewise_linear", ("piecewise", "pwl")),
    ("in_range/compare", ("in_range", "compare")),
    ("clamp", ("clamp",)),
    ("abs", ("abs",)),
    ("wrap_angle", ("wrap",)),
]

# Coarse subsystem family per op (APPROXIMATE — by op identity, not call-site: a
# floor_int may be an emit digit-quad OR a geometry column-floor; this groups by
# the op, which is the cleanly-graph-derivable signal). Fallback: "other".
_FAMILY: dict[str, str] = {
    "floor_int": "emit/quantize",
    "thermometer_floor_div": "emit/quantize",
    "atan/ray (octant)": "geometry",
    "multiply": "geometry",
    "square": "geometry",
    "reciprocal": "geometry",
    "abs": "geometry",
    "wrap_angle": "geometry",
    "table_lookup": "geometry/lookup",
    "broadcast_select": "dispatch/select",
    "select": "dispatch/select",
    "cond_gate": "dispatch/select",
    "in_range/compare": "compare",
    "clamp": "compare",
    "relu(bare)": "compare (abs/step)",
}


def _classify_op(name: str) -> str:
    low = name.lower()
    for label, pats in _OP_RULES:
        if any(p in low for p in pats):
            return label
    if not name or name == "_relu":
        return "relu(bare)"
    return f"OTHER:{_normalize(name)}"


def _family(op: str) -> str:
    return _FAMILY.get(op, "other" if not op.startswith("OTHER:") else "other")


_CHUNK_RANGE = re.compile(r"_\d+_\d+$")
_TRAIL_INT = re.compile(r"_\d+$")


def _normalize(name: str) -> str:
    """Collapse instance-specific suffixes so siblings aggregate together:
    strip the ``_relu`` tag, a trailing chunk range (``_0_512``), then a single
    trailing index (``_3``)."""
    n = name[:-5] if name.endswith("_relu") else name
    n = _CHUNK_RANGE.sub("", n)
    n = _TRAIL_INT.sub("", n)
    return n or "<unnamed>"


@dataclass(frozen=True)
class ReluRecord:
    name: str  # the raw ReLU node name (f"{op}_relu")
    norm: str  # instance-suffixes stripped
    op: str  # canonical op bucket
    width: int  # ReLU units (= len(node))


def collect_relu_inventory(output_node) -> list[ReluRecord]:
    """DFS the graph from ``output_node``; one record per distinct ReLU node."""
    seen: set[int] = set()
    out: list[ReluRecord] = []
    stack = [output_node]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        if isinstance(node, ReLU):
            nm = getattr(node, "name", "") or ""
            out.append(ReluRecord(nm, _normalize(nm), _classify_op(nm), len(node)))
        for inp in getattr(node, "inputs", []) or []:
            if id(inp) not in seen:
                stack.append(inp)
    return out


# --- aggregation + printing -------------------------------------------------
def _agg(records: list[ReluRecord], key) -> dict[str, tuple[int, int]]:
    """key -> (total ReLU units, node count)."""
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        a = acc[key(r)]
        a[0] += r.width
        a[1] += 1
    return {k: (u, c) for k, (u, c) in acc.items()}


def _print_table(
    title: str,
    agg: dict[str, tuple[int, int]],
    total_units: int,
    total_nodes: int,
    top: int,
) -> None:
    rows = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)
    print(f"\n=== {title} ===")
    print(f"  {'units':>8} {'%units':>7} {'nodes':>6} {'%nodes':>7} {'mean':>6}  key")
    shown_u = shown_n = 0
    for k, (u, c) in rows[:top]:
        shown_u += u
        shown_n += c
        print(
            f"  {u:8d} {100*u/total_units:6.1f}% {c:6d} "
            f"{100*c/total_nodes:6.1f}% {u/c:6.0f}  {k}"
        )
    if len(rows) > top:
        ru = total_units - shown_u
        rn = total_nodes - shown_n
        print(
            f"  {ru:8d} {100*ru/total_units:6.1f}% {rn:6d} "
            f"{100*rn/total_nodes:6.1f}% {'':6}  ... ({len(rows)-top} more)"
        )


def _print_width_buckets(
    records: list[ReluRecord], total_units: int, total_nodes: int
) -> None:
    edges = [(1, 1), (2, 4), (5, 16), (17, 64), (65, 256), (257, 1024), (1025, 1 << 30)]
    labels = ["1", "2-4", "5-16", "17-64", "65-256", "257-1024", "1025+"]
    print("\n=== by ReLU width bucket (the shape of the demand) ===")
    print(f"  {'width':>10} {'units':>8} {'%units':>7} {'nodes':>6} {'%nodes':>7}")
    for (lo, hi), lab in zip(edges, labels):
        u = sum(r.width for r in records if lo <= r.width <= hi)
        c = sum(1 for r in records if lo <= r.width <= hi)
        if c:
            print(
                f"  {lab:>10} {u:8d} {100*u/total_units:6.1f}% {c:6d} "
                f"{100*c/total_nodes:6.1f}%"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="rows per table")
    ap.add_argument("--csv", type=str, default="", help="dump raw inventory to CSV")
    args = ap.parse_args()

    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=128, max_positions=65536, d_rot=64)
    nt = forward(emb, GraphPast(input_vec=emb, rope=rope))

    records = collect_relu_inventory(nt)
    total_units = sum(r.width for r in records)
    total_nodes = len(records)
    print(
        f"ReLU inventory for doom forward(): {total_nodes} ReLU nodes, "
        f"{total_units} ReLU units total (per position)."
    )

    _print_table(
        "by op family (approximate subsystem)",
        _agg(records, lambda r: _family(r.op)),
        total_units,
        total_nodes,
        args.top,
    )
    _print_table(
        "by op type", _agg(records, lambda r: r.op), total_units, total_nodes, args.top
    )
    _print_table(
        "by normalized name (finer)",
        _agg(records, lambda r: r.norm),
        total_units,
        total_nodes,
        args.top,
    )
    _print_width_buckets(records, total_units, total_nodes)

    print("\n=== top widest individual ReLU nodes ===")
    print(f"  {'width':>6}  op / name")
    for r in sorted(records, key=lambda r: r.width, reverse=True)[: args.top]:
        print(f"  {r.width:6d}  {r.op:22s} {r.name}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "norm", "op", "width"])
            for r in records:
                w.writerow([r.name, r.norm, r.op, r.width])
        print(f"\nwrote raw inventory ({total_nodes} rows) to {args.csv}")


if __name__ == "__main__":
    main()
