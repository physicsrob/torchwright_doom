"""Score a store_assignment schedule's residual-column occupancy per layer.

Mirrors the CP-SAT ``residual_cumulative`` accounting (cpsat_scheduler.py):
scheduled nodes that use residual columns occupy ``len(node)`` columns over
``[layer, cancel)``; freeable inputs occupy ``[0, cancel)``; capacity is
``d - (1 + reserve_residual)``.  One known bias: free-Adds reuse a dead
addend's columns and start one layer late, but the emitted JSON does not
record ``is_free`` — so this scorer OVER-counts by one layer x width for
each free-Add.  Calibrate the bias by scoring a schedule the solver
certified feasible: its apparent overflow above capacity IS the bias.

Usage (local, from the commit's checkout):

    python -m scripts.cpsat_occupancy <schedule.json> [<schedule2.json> ...]
"""

from __future__ import annotations

import json
import sys

from scripts.cpsat_prod_harness import build_production_model


def main() -> None:
    from torchwright.compiler.forward.cpsat_scheduler import (
        build_graph_model,
        uses_residual,
    )
    from torchwright.compiler.graph_identity import canonical_ids

    pm = build_production_model("configs/e1m1.yaml")
    gm = build_graph_model(pm.output_node, None)
    capacity = pm.d - (1 + pm.n_reserved_residual)
    canon = canonical_ids(pm.output_node)
    width_by_canon = {canon[n.node_id]: len(n) for n in gm.graph.get_all_nodes()}
    resid_canon = {canon[n.node_id] for n in gm.schedulable if uses_residual(n, gm)}
    input_canon = {canon[n.node_id] for n in gm.input_nodes if n is not gm.output_node}

    node_by_canon = {
        canon[n.node_id]: n for n in gm.graph.get_all_nodes() if n.node_id in canon
    }

    def _tag(cid: int) -> str:
        n = node_by_canon[cid]
        ann = getattr(n, "annotation", None) or "<unannotated>"
        return ann

    band_frac = 0.95
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--band="):
            band_frac = float(a.split("=", 1)[1])

    for path in paths:
        d = json.loads(open(path).read())
        n_layers = d["n_layers"]
        layer = {int(k): v for k, v in d["node_to_layer"].items()}
        cancel = {int(k): v for k, v in d["node_to_cancel_layer"].items()}
        occ = [0] * n_layers
        live: list[list[int]] = [[] for _ in range(n_layers)]
        for cid, li in layer.items():
            if cid not in resid_canon:
                continue
            for lay in range(li, min(cancel.get(cid, n_layers), n_layers)):
                occ[lay] += width_by_canon[cid]
                live[lay].append(cid)
        for cid in input_canon:
            for lay in range(min(cancel.get(cid, n_layers), n_layers)):
                occ[lay] += width_by_canon[cid]
                live[lay].append(cid)
        over = [(lay, o - capacity) for lay, o in enumerate(occ) if o > capacity]
        print(
            f"OCCUPANCY {path}: n_layers={n_layers} capacity={capacity} "
            f"peak={max(occ)} ({100 * max(occ) / capacity:.1f}%) "
            f"mean={sum(occ) / len(occ):.0f} ({100 * sum(occ) / len(occ) / capacity:.1f}%) "
            f"layers_over_capacity={len(over)} "
            f"max_overflow={max((v for _, v in over), default=0)}",
            flush=True,
        )
        profile = " ".join(f"{lay}:{o}" for lay, o in enumerate(occ))
        print(f"  profile: {profile}", flush=True)
        if over:
            print(f"  overflow layers: {over}", flush=True)

        # ---- Pressure attribution: who occupies the high-pressure band ----
        band = [lay for lay, o in enumerate(occ) if o >= band_frac * capacity]
        if not band:
            band = [max(range(n_layers), key=lambda lay: occ[lay])]
        by_subsystem: dict[str, int] = {}
        by_node: dict[int, int] = {}
        for lay in band:
            for cid in live[lay]:
                w = width_by_canon[cid]
                by_subsystem[_tag(cid)] = by_subsystem.get(_tag(cid), 0) + w
                by_node[cid] = by_node.get(cid, 0) + w
        total_band = sum(occ[lay] for lay in band)
        print(
            f"  PRESSURE band (occ >= {band_frac:.0%} cap): layers {band}, "
            f"{total_band} column-layers",
            flush=True,
        )
        print("  by subsystem (column-layers in band):", flush=True)
        for tag, cols in sorted(by_subsystem.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {cols:>7}  {100 * cols / total_band:5.1f}%  {tag}", flush=True)
        print("  top nodes (column-layers in band):", flush=True)
        for cid, cols in sorted(by_node.items(), key=lambda kv: -kv[1])[:20]:
            n = node_by_canon[cid]
            print(
                f"    {cols:>7}  w={width_by_canon[cid]:>5}  "
                f"[{layer.get(cid, 0)},{cancel.get(cid, n_layers)})  "
                f"{type(n).__name__}  {_tag(cid)}  "
                f"name={getattr(n, 'name', None)!r}",
                flush=True,
            )


if __name__ == "__main__":
    main()
