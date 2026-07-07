"""Layer-domain statistics of the production CP-SAT model, one commit at a time.

Quantifies the 'the flattens freed the schedule variables' claim: build the
production-exact model (via cpsat_prod_harness.build_production_model), run
the production warm start, compute the tightened per-node layer domains at
the production horizon (hint+1), and report how pinned the model is.

Run locally from a checkout/worktree of the commit under test:

    /path/to/umbrella/.venv/bin/python -m scripts.cpsat_domain_stats

One DOMAIN_STATS line per run; CPU-only, no solver.
"""

from __future__ import annotations

import statistics

from scripts.cpsat_prod_harness import build_production_model


def main() -> None:
    from torchwright.compiler.forward.cpsat_scheduler import (
        _compute_layer_bounds,
        build_graph_model,
    )

    pm = build_production_model("configs/e1m1.yaml")
    pm.run_warm_start()
    gm = build_graph_model(pm.output_node, None)
    horizon = pm.solver_max_layers
    es, ls = _compute_layer_bounds(gm, pm.policy, True, horizon)
    widths = [ls[nid] - es[nid] + 1 for nid in es]
    pinned = sum(1 for w in widths if w == 1)
    print(
        f"DOMAIN_STATS cp={pm.cp_layers} hint={pm.hint_n_layers} "
        f"horizon={horizon} n={len(widths)} pinned={pinned} "
        f"({100 * pinned / len(widths):.1f}%) "
        f"mean_width={statistics.mean(widths):.2f} "
        f"median_width={statistics.median(widths)} "
        f"total_slack={sum(w - 1 for w in widths)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
