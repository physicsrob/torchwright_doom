"""Attribute the CP-SAT lower bound to a constraint family (depth exp 7).

Context (depth_experiments_log.md): at HEAD the DAG floor is 33 but every
solver configuration in ``cpsat_space_experiments`` proves lb=38 — the
lower bound left the critical path, so a per-layer RESOURCE family now
prices ~5 layers above the dependency floor. This script names the family.

Method: decision probes instead of optimization. For each configuration
(all families on; then each cumulative family disabled in turn) build the
model exactly like the harness (same d/d_head/d_hidden geometry, so the
numbers are comparable to its lb=38), add ``n_layers <= K`` as a hard
constraint, and report the solver STATUS at a fixed budget:

  * INFEASIBLE  -> this configuration still forbids K layers;
  * FEASIBLE    -> dropping that family admits K layers (family is the
                   binder, or part of it);
  * UNKNOWN     -> budget exhausted without a proof either way.

K=37 answers "which family prices us above the baseline artifact"; K=33
(run for the flipped families only) answers "does dropping it recover the
full DAG floor".

Run on Modal (64 CPUs to match the production compile container):

    MODAL_RUN_CPU=64 MODAL_RUN_MEMORY=65536 UV_PROJECT=... PYTHONPATH=$PWD \\
    TORCHWRIGHT_DOOM_<production screen env> \\
    make modal-run MODULE=scripts.cpsat_lb_attribution CPU_ONLY=1
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ortools.sat.python import cp_model

from torchwright.compiler.forward.cpsat_scheduler import (
    CONSTRAINT_FAMILIES,
    build_cpsat_model,
)
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.ops.inout_nodes import create_rope_config
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

D = 8192
D_HEAD = 64  # match cpsat_space_experiments so statuses compare to its lb=38
D_HIDDEN = 16384
MAX_LAYERS = 60
BUDGET_S = float(os.environ.get("LB_ATTRIB_BUDGET_S", "150"))

_CUMULATIVE_FAMILIES = (
    "attn_cumulative",
    "mlp_cumulative",
    "residual_cumulative",
)


def _probe(output_node, disabled: frozenset, k: int) -> str:
    built = build_cpsat_model(
        output_node,
        None,
        d=D,
        d_head=D_HEAD,
        d_hidden=D_HIDDEN,
        max_layers=MAX_LAYERS,
        assume_zero_init=True,
        _disabled_families=disabled,
    )
    built.model.Add(built.n_layers_var <= k)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = BUDGET_S
    solver.parameters.num_search_workers = int(os.environ.get("TW_CPSAT_WORKERS", "16"))
    status = solver.Solve(built.model)
    return solver.StatusName(status)


def main() -> None:
    print(f"=== graph build (d={D} d_head={D_HEAD} d_hidden={D_HIDDEN}) ===")
    t0 = time.perf_counter()
    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=D_HEAD, max_positions=65536, d_rot=D_HEAD // 2)
    nt = forward(emb, GraphPast(input_vec=emb, rope=rope))
    graph = GraphAnalyzer(nt)
    output_node = graph.get_output_node()
    print(f"graph: {len(graph.get_all_nodes())} nodes in {time.perf_counter() - t0:.1f}s")

    configs: list[tuple[str, frozenset]] = [("all-on", frozenset())]
    configs += [(f"minus-{fam}", frozenset({fam})) for fam in _CUMULATIVE_FAMILIES]

    flipped: list[tuple[str, frozenset]] = []
    for name, disabled in configs:
        unknown = disabled - CONSTRAINT_FAMILIES
        assert not unknown, unknown
        t0 = time.perf_counter()
        st = _probe(output_node, disabled, k=37)
        print(
            f"--> K=37 [{name}]: {st}  ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        if name != "all-on" and st != "INFEASIBLE":
            flipped.append((name, disabled))

    for name, disabled in flipped:
        t0 = time.perf_counter()
        st = _probe(output_node, disabled, k=33)
        print(
            f"--> K=33 [{name}]: {st}  ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
