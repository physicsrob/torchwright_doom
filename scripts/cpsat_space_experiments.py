"""CP-SAT search-space experiments on the forward() schedule.

Motivated by the d=8192/h=16384 opt2 compile (2026-06-09): the 180s solve
spent 25s in presolve + 17s completing the hint, every improvement came from
local-search/LNS workers (the 24 shared-tree workers and the lower-bound
cohort contributed nothing), and the lower bound never moved off the
critical path.  This harness measures whether shrinking the search space /
reallocating workers buys a better schedule for the same budget.

Builds the DOOM forward() graph and the heuristic warm-start ONCE (the
heuristic is deterministic, so every variant shares an identical hint), then
runs `solve_schedule` under several configurations:

  baseline        production parameters, two seeds (variance gauge)
  no-shared-tree  shared_tree_num_workers=0 (frees 24 workers for LNS/LS)
  lns-heavy       no-shared-tree + drop the lower-bound/shaving subsolvers
  tight-domains   per-node [earliest, latest] layer domains baked into the
                  model (instead of letting presolve derive them)
  combo           tight-domains + lns-heavy
  fixed-routing   flex_routing=False diagnostic: is sublayer-routing freedom
                  where the gains come from?
  combo-600s      best combo at a 600s budget (probe toward the LB floor)

Run on Modal (needs 64 CPUs to match the production compile container):

    MODAL_RUN_CPU=64 MODAL_RUN_MEMORY=32768 MODAL_RUN_TIMEOUT=7200 \\
        make modal-run MODULE=scripts.cpsat_space_experiments CPU_ONLY=1

Env knobs: D (8192), D_HEAD (32), DH (16384), BUDGET_S (180),
VARIANTS (comma list to subset, e.g. VARIANTS=baseline-s0,tight-domains).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
# Match the production e1m1 render geometry (scale=4 -> 80x50): the token
# vocab is screen-dependent and built when the vocab module is imported, so
# these must be set before any torchwright_doom import.
os.environ.setdefault("TORCHWRIGHT_DOOM_RENDER_SCALE", "4")
os.environ.setdefault("TORCHWRIGHT_DOOM_SCREEN_WIDTH", "80")
os.environ.setdefault("TORCHWRIGHT_DOOM_SCREEN_HEIGHT", "50")
os.environ.setdefault("TW_CPSAT_WORKERS", "64")
for _p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from torchwright.compiler.forward.compile import _run_heuristic_warm_start
from torchwright.compiler.forward.cpsat_scheduler import (
    Costs,
    _compute_layer_bounds,
    build_graph_model,
    solve_schedule,
)
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_pos_encoding
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward

D = int(os.environ.get("D", "8192"))
D_HEAD = int(os.environ.get("D_HEAD", "32"))
D_HIDDEN = int(os.environ.get("DH", "16384"))
BUDGET_S = float(os.environ.get("BUDGET_S", "180"))

# The subsolvers that contributed nothing in the measured d8192 run: the
# whole lower-bound / shaving cohort (the LB never moved off the critical
# path in 180s).
_LB_SUBSOLVERS = [
    "lb_tree_search",
    "objective_lb_search",
    "objective_lb_search_max_lp",
    "objective_lb_search_no_lp",
    "objective_shaving_max_lp",
    "objective_shaving_no_lp",
    "max_lp",
    "max_lp_sym",
]

_VARIANTS: list[tuple[str, dict]] = [
    ("baseline-s0", {}),
    ("baseline-s7", {"solver_params": {"random_seed": 7}}),
    ("no-shared-tree", {"solver_params": {"shared_tree_num_workers": 0}}),
    (
        "lns-heavy",
        {
            "solver_params": {
                "shared_tree_num_workers": 0,
                "ignore_subsolvers": _LB_SUBSOLVERS,
            }
        },
    ),
    ("tight-domains", {"tighten_domains": True}),
    (
        "combo",
        {
            "tighten_domains": True,
            "solver_params": {
                "shared_tree_num_workers": 0,
                "ignore_subsolvers": _LB_SUBSOLVERS,
            },
        },
    ),
    # Hint-free: with routing pinned, the heuristic's routing hints can
    # contradict the pinned literals (observed: ortools SIGSEGV mid-search
    # with the full hint + pinned routing).
    ("fixed-routing", {"flex_routing": False, "no_hint": True}),
    (
        "combo-600s",
        {
            "tighten_domains": True,
            "time_budget_s": 600.0,
            "solver_params": {
                "shared_tree_num_workers": 0,
                "ignore_subsolvers": _LB_SUBSOLVERS,
            },
        },
    ),
    # Force the horizon below the best known schedule (58 from the measured
    # opt2 run): domains shrink hard (the serial tail pins), but the 71-layer
    # hint no longer fits, so CP-SAT starts cold.  Tests whether a small
    # space beats a warm start.
    (
        "squeeze-59",
        {
            "tighten_domains": True,
            "max_layers": 59,
            "no_hint": True,
            "solver_params": {"shared_tree_num_workers": 0},
        },
    ),
    # Feasibility at the floor: the critical path (= the proven optimum, 50)
    # is computable in milliseconds, so skip the staircase descent and ask
    # for ANY schedule at that horizon.  Domains collapse hard (every
    # critical-path node pins to one layer); all workers hunt for one
    # solution instead of grinding an objective.  51 = fallback rung.
    (
        "squeeze-50",
        {
            "tighten_domains": True,
            "max_layers": 50,
            "no_hint": True,
            "time_budget_s": 300.0,
        },
    ),
    (
        "squeeze-51",
        {
            "tighten_domains": True,
            "max_layers": 51,
            "no_hint": True,
            "time_budget_s": 300.0,
        },
    ),
    # Width-bound-regime probe (run with --d=4096 --dh=4096): cold solve just
    # under the d4096 heuristic's 85 layers.  Asks whether cold construction
    # works at all when width binds, or only in the slack-width regime.
    (
        "squeeze-80",
        {
            "tighten_domains": True,
            "max_layers": 80,
            "no_hint": True,
            "time_budget_s": 300.0,
        },
    ),
    # Repeatability draws for the floor+1 recipe (variance is timing-driven,
    # so repeated runs are independent draws; seeds just label them).
    (
        "squeeze-51-r2",
        {
            "tighten_domains": True,
            "max_layers": 51,
            "no_hint": True,
            "time_budget_s": 300.0,
            "solver_params": {"random_seed": 7},
        },
    ),
    (
        "squeeze-51-r3",
        {
            "tighten_domains": True,
            "max_layers": 51,
            "no_hint": True,
            "time_budget_s": 300.0,
            "solver_params": {"random_seed": 13},
        },
    ),
    # Plateau-gradient objective: lexicographic earliness term rewards every
    # compacting move, not just ones that drop the max layer (the pure-depth
    # staircase is why 180s outcomes vary 50..64 run-to-run).
    ("earliness", {"costs": Costs(alpha=1, earliness=1)}),
    (
        "earliness-tight",
        {
            "costs": Costs(alpha=1, earliness=1),
            "tighten_domains": True,
            "solver_params": {"shared_tree_num_workers": 0},
        },
    ),
    # Residual-occupancy secondary: minimize column-layers of values sitting
    # in the stream (waste), aimed at the width pinch points that gate layer
    # drops — where plain earliness measurably distracted the search.
    ("waste", {"costs": Costs(alpha=1, waste=1)}),
    (
        "waste-tight",
        {
            "costs": Costs(alpha=1, waste=1),
            "tighten_domains": True,
            "solver_params": {"shared_tree_num_workers": 0},
        },
    ),
]


# Trajectory-capture variants: dense incumbent streams (a secondary
# objective surfaces intra-plateau states; pure depth only snapshots the
# drops themselves).  Used to test which schedule metrics actually lead
# layer drops — validating a gradient direction before baking it into the
# objective.
_TRACE_VARIANTS: list[tuple[str, dict]] = [
    ("trace-plain", {"time_budget_s": 300.0}),
    ("trace-earliness", {"costs": Costs(alpha=1, earliness=1), "time_budget_s": 300.0}),
    ("trace-waste", {"costs": Costs(alpha=1, waste=1), "time_budget_s": 300.0}),
]


def _snapshot_metrics(snap, lens, keep_ids, input_lens, input_keep, horizon):
    """Candidate schedule metrics for one incumbent snapshot."""
    layers, cancels, ic = snap["layers"], snap["cancels"], snap["input_cancels"]
    n_layers = snap["n_layers"]
    top = n_layers - 1
    occ = [0] * (horizon + 1)
    waste = 0
    for nid, L in layers.items():
        c, w = cancels[nid], lens[nid]
        if nid not in keep_ids:
            waste += w * (c - L)
        for layer in range(L, min(c, horizon)):
            occ[layer] += w
    for nid, c in ic.items():
        w = input_lens[nid]
        if nid not in input_keep:
            waste += w * c
        for layer in range(min(c, horizon)):
            occ[layer] += w
    return dict(
        n_layers=n_layers,
        sum_layers=sum(layers.values()),
        waste=waste,
        top_count=sum(1 for L in layers.values() if L == top),
        top_cols=sum(lens[nid] for nid, L in layers.items() if L == top),
        top3_cols=sum(lens[nid] for nid, L in layers.items() if L >= top - 2),
        peak_occ=max(occ),
    )


def _lead_analysis(rows) -> None:
    """Which metrics improved across each plateau that ended in a drop?"""
    metrics = [k for k in rows[0] if k != "n_layers"]
    plateaus: list[tuple[list, bool]] = []
    cur = [rows[0]]
    for r in rows[1:]:
        if r["n_layers"] < cur[-1]["n_layers"]:
            plateaus.append((cur, True))
            cur = [r]
        else:
            cur.append(r)
    plateaus.append((cur, False))
    multi = [p for p, dropped in plateaus if dropped and len(p) >= 2]
    print(
        f"    {len(plateaus)} plateaus, {len(multi)} with >=2 snapshots "
        f"ending in a drop; metric trend across those plateaus:"
    )
    for m in metrics:
        fell = rose = flat = 0
        for plat, dropped in plateaus:
            if not dropped or len(plat) < 2:
                continue
            d = plat[-1][m] - plat[0][m]
            fell, rose, flat = (
                fell + (d < 0),
                rose + (d > 0),
                flat + (d == 0),
            )
        print(
            f"      {m:<11} fell-before-drop {fell:>3}  rose {rose:>3}  flat {flat:>3}"
        )


def _parse_log(log: str) -> dict:
    """Extract presolve time, incumbent trajectory, and final LB."""
    out: dict = {"presolve_s": None, "first_incumbent_s": None, "traj": [], "lb": None}
    m = re.search(r"Starting search at ([\d.]+)s", log)
    if m:
        out["presolve_s"] = float(m.group(1))
    for m in re.finditer(r"^#\d+\s+([\d.]+)s\s+best:(\d+)", log, re.M):
        t, best = float(m.group(1)), int(m.group(2))
        out["traj"].append((t, best))
        if out["first_incumbent_s"] is None:
            out["first_incumbent_s"] = t
    lbs = re.findall(r"next:\[(\d+),", log)
    if lbs:
        out["lb"] = int(lbs[-1])
    return out


def main() -> None:
    # Variant subset + geometry: env for local runs, or --variants= / --d= /
    # --d-head= / --dh= via the modal-run ARGS passthrough (env vars do NOT
    # reach the container).
    global D, D_HEAD, D_HIDDEN
    spec = os.environ.get("VARIANTS", "")
    for arg in sys.argv[1:]:
        if arg.startswith("--variants="):
            spec = arg.split("=", 1)[1]
        elif arg.startswith("--d="):
            D = int(arg.split("=", 1)[1])
        elif arg.startswith("--d-head="):
            D_HEAD = int(arg.split("=", 1)[1])
        elif arg.startswith("--dh="):
            D_HIDDEN = int(arg.split("=", 1)[1])
    only = {v for v in spec.split(",") if v}

    print(
        f"=== graph build (d={D} d_head={D_HEAD} d_hidden={D_HIDDEN}) ===", flush=True
    )
    t0 = time.perf_counter()
    emb = build_doom_embedding("token_ids")
    pos = create_pos_encoding()
    nt = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)
    graph = GraphAnalyzer(nt)
    output_node = graph.get_output_node()
    n_nodes = len(graph.get_all_nodes())
    print(f"graph: {n_nodes} nodes in {time.perf_counter() - t0:.1f}s", flush=True)

    # --- heuristic warm start (once; deterministic, shared by all variants)
    input_nodes = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]
    rmap = ResidualStreamMap(D)
    rmap.allocate(pos)
    rmap.mark_clean(rmap.get_indices(pos))
    for n in input_nodes:
        if n is pos:
            continue
        rmap.allocate(n)
        rmap.mark_clean(rmap.get_indices(n))
    computed = set(input_nodes)
    policy = SchedulingPolicy()

    t0 = time.perf_counter()
    hint_layers, hint_routing, hint_cancel, hint_n = _run_heuristic_warm_start(
        graph=graph,
        d=D,
        d_head=D_HEAD,
        pos_encoding=pos,
        d_hidden=D_HIDDEN,
        residual_map=rmap,
        computed=computed,
        clusters=None,
        admission_budget_fraction=0.4,
        policy=policy,
        output_node=output_node,
        max_layers=200,
    )
    print(
        f"warm start: {hint_n} layers, {len(hint_layers)} hinted nodes "
        f"({time.perf_counter() - t0:.1f}s)  [production d8192 run: 71 layers, "
        f"13197 nodes — mismatch means the harness diverges from prod]",
        flush=True,
    )
    solver_max_layers = min(200, hint_n + 1)

    # --- soundness check: the (feasible) hint must satisfy the tightened
    # domains, otherwise _compute_layer_bounds is WRONG, not just tight.
    gm = build_graph_model(output_node, pos)
    es, ls = _compute_layer_bounds(gm, policy, True, solver_max_layers)
    bad = [
        (nid, L, es.get(nid), ls.get(nid))
        for nid, L in hint_layers.items()
        if nid in es and not (es[nid] <= L <= ls[nid])
    ]
    crit = max(es.values()) + 1
    slack0 = sum(1 for nid in es if es[nid] == ls[nid])
    print(
        f"bounds: critical path >= {crit} layers; {slack0}/{len(es)} nodes "
        f"fully pinned at horizon {solver_max_layers}; "
        f"hint violations: {len(bad)}",
        flush=True,
    )
    if bad:
        print(f"FIRST VIOLATIONS: {bad[:5]}", flush=True)
        print("ABORTING tight-domain variants would be unsound; exiting.", flush=True)
        raise SystemExit(1)

    # Structures for trajectory-metric computation (trace-* variants).
    lens = {n.node_id: len(n) for n in gm.schedulable}
    keep_ids = {n.node_id for n in gm.pinned_nodes} | {
        n.node_id
        for n in gm.schedulable
        if any(isinstance(c, Concatenate) for c in gm.consumers_eff.get(n, set()))
    }
    freeable = [n for n in gm.input_nodes if n is not pos and n is not gm.output_node]
    input_lens = {n.node_id: len(n) for n in freeable}
    input_keep = {
        n.node_id
        for n in freeable
        if any(isinstance(c, Concatenate) for c in gm.consumers_eff.get(n, set()))
    }

    rows = []
    for name, overrides in _VARIANTS + _TRACE_VARIANTS:
        if only and name not in only:
            continue
        if name.startswith("trace-") and not only:
            continue  # capture runs are opt-in via --variants
        kwargs = dict(
            d=D,
            d_head=D_HEAD,
            d_hidden=D_HIDDEN,
            flex_routing=True,
            time_budget_s=BUDGET_S,
            max_layers=solver_max_layers,
            policy=policy,
            assume_zero_init=True,
            hint_layers=hint_layers,
            hint_routing=hint_routing,
            hint_cancel=hint_cancel,
            log_search_progress=True,
        )
        kwargs.update({k: v for k, v in overrides.items()})
        if kwargs.pop("no_hint", False):
            kwargs["hint_layers"] = None
            kwargs["hint_routing"] = None
            kwargs["hint_cancel"] = None
        trace: list[dict] | None = None
        if name.startswith("trace-"):
            trace = []
            kwargs["solution_trace"] = trace
        print(f"\n=== variant {name} ===", flush=True)
        t0 = time.perf_counter()
        assignment, stats = solve_schedule(output_node, pos, **kwargs)
        wall = time.perf_counter() - t0
        info = _parse_log(stats.solver_log)
        # Secondary objectives scale the raw best/bound values; divide back
        # to layer counts (alpha=1: best // scale == n_layers).
        scale = stats.objective_scale
        if scale > 1:
            info["traj"] = [(t, b // scale) for t, b in info["traj"]]
            info["lb"] = info["lb"] // scale if info["lb"] is not None else None
        traj_txt = " ".join(f"{b}@{t:.0f}s" for t, b in info["traj"])
        row = dict(
            name=name,
            status=stats.status_name,
            n_layers=(assignment.n_layers if assignment else -1),
            lb=info["lb"],
            presolve_s=info["presolve_s"],
            first_inc_s=info["first_incumbent_s"],
            wall_s=wall,
        )
        rows.append(row)
        print(
            f"--> {name}: status={stats.status_name} "
            f"n_layers={row['n_layers']} lb={row['lb']} "
            f"presolve={row['presolve_s']}s first_incumbent={row['first_inc_s']}s "
            f"wall={wall:.0f}s\n    trajectory: {traj_txt}",
            flush=True,
        )
        if trace:
            metric_rows = [
                _snapshot_metrics(
                    s, lens, keep_ids, input_lens, input_keep, kwargs["max_layers"]
                )
                for s in trace
            ]
            print(f"    {len(trace)} incumbents captured; snapshots:", flush=True)
            header = list(metric_rows[0])
            print("      " + " ".join(f"{h:>10}" for h in ["t"] + header))
            prev_nl = None
            for idx, (s, mr) in enumerate(zip(trace, metric_rows)):
                # Print layer transitions + every 10th plateau snapshot.
                if mr["n_layers"] != prev_nl or idx % 10 == 0:
                    print(
                        f"      {s['t']:>10.1f} "
                        + " ".join(f"{mr[h]:>10}" for h in header)
                    )
                prev_nl = mr["n_layers"]
            _lead_analysis(metric_rows)

    print("\n=== SUMMARY (hint = {} layers) ===".format(hint_n), flush=True)
    print(
        f"{'variant':<16} {'status':<10} {'layers':>6} {'LB':>4} "
        f"{'presolve':>8} {'first_inc':>9} {'wall':>6}"
    )
    for r in rows:
        print(
            f"{r['name']:<16} {r['status']:<10} {r['n_layers']:>6} "
            f"{str(r['lb']):>4} {str(r['presolve_s']):>8} "
            f"{str(r['first_inc_s']):>9} {r['wall_s']:>6.0f}"
        )


if __name__ == "__main__":
    main()
