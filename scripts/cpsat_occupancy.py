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
    resid_canon = {
        canon[n.node_id] for n in gm.schedulable if uses_residual(n, gm)
    }
    input_canon = {
        canon[n.node_id] for n in gm.input_nodes if n is not gm.output_node
    }

    for path in sys.argv[1:]:
        d = json.loads(open(path).read())
        n_layers = d["n_layers"]
        layer = {int(k): v for k, v in d["node_to_layer"].items()}
        cancel = {int(k): v for k, v in d["node_to_cancel_layer"].items()}
        occ = [0] * n_layers
        for cid, li in layer.items():
            if cid not in resid_canon:
                continue
            for lay in range(li, min(cancel.get(cid, n_layers), n_layers)):
                occ[lay] += width_by_canon[cid]
        for cid in input_canon:
            for lay in range(min(cancel.get(cid, n_layers), n_layers)):
                occ[lay] += width_by_canon[cid]
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


if __name__ == "__main__":
    main()
