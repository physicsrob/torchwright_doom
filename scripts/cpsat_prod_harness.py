"""Production-exact CP-SAT solve harness (depth series, phase 0).

Every earlier probe in this series diverged from the production compile in at
least one model-construction input (measured divergences: d_head 64 vs 128;
``reserve_residual`` 0 vs 2 — the pinned-constant RMSNorm columns; ``bias``
True vs False, so solver d_hidden 16384 vs 16383; ``lower()`` with default
collapse flags vs production's always-on collapse passes; the bare
``forward(emb, GraphPast(...))`` graph vs the WAD/AssetIndex production graph;
scale-4 screen env vs the production 320x200).  Any conclusion drawn from
those probes inherits that risk.  This harness replicates the production
solve's model construction bit-for-bit and PROVES it with a calibration gate:
the ``graph_fingerprint`` it computes must equal the fingerprint the
production compile printed (the fingerprint is a pure function of lowered
topology + geometry + solver knobs, so equality == construction exactness).

Replication contract (mirrors ``torchwright/compiler/forward/compile.py`` at
the ``optimize=3`` production path, and ``inference/compile_cache.py`` for the
graph inputs):

- graph: ``inference.compiled_model.build_graph`` (WAD + AssetIndex +
  ``apply_screen_env`` from the config) — never the bare forward();
- lower: ``collapse_univariate=True, collapse_pl=True,
  collapse_lane_cap=d_hidden // 4`` (compile.py's exact kwargs);
- residual seed: inputs sorted by node_id, ``reserve_node_id_above`` before
  the const-1 mint, ``assume_zero_init`` free-pool mark_clean, THEN the
  pinned-constant RMSNorm reservation (order matters — replicated verbatim);
- knobs: ``policy=SchedulingPolicy()``, ``costs=Costs()``,
  ``flex_routing=True``, ``cancel_slack=2``, ``tighten_domains=True``,
  ``reserve_residual=len(rms_spec.reserved_cols)``,
  ``solver d_hidden = d_hidden - 1`` under ``bias=False``;
- warm start: ``_run_heuristic_warm_start`` with the compile.py scaffolding
  (including ``bias``), horizon ``min(config.max_layers, hint_n + 1)``;
- floor probe: the optimize>=2 cold solve at horizon ``critical_path + 1``
  with budget ``min(150, budget)`` — part of the production flow, so part of
  the replication;
- ``TW_CPSAT_WORKERS=64`` set here (env does not forward through
  ``make modal-run``; ``modal_render.compile_remote`` sets the same).

Subcommands:

  fingerprint   build + lower + fingerprint (+ optional warm-start depth).
                The calibration gate: the printed fingerprint must match the
                production compile's print (HEAD 5d86ed63ded1..., baseline
                93bb98e 89aa36948a5f...).
  solve         replicate the production solve flow (floor probe + hinted
                descent), with seed / budget / solver-params overrides for
                the statistical re-baseline.  Emits the winning schedule in
                ``store_assignment`` JSON format for cache injection.
  probe-k       fixed-K decision probe: ``n_layers <= K`` as a hard
                constraint; SAT hunt (hinted or cold) or refutation run.
                Emits the schedule JSON on a SAT hit.

Run on Modal (64 CPUs to match the production compile container):

    MODAL_RUN_CPU=64 MODAL_RUN_MEMORY=65536 MODAL_RUN_TIMEOUT=7200 \\
    PYTHONPATH=$PWD make modal-run CPU_ONLY=1 \\
        MODULE=scripts.cpsat_prod_harness ARGS="fingerprint --warm-start"

    ... ARGS="solve --budget=300 --seed=3"
    ... ARGS="probe-k --k=33 --budget=3600 --no-hint"

(``PYTHONPATH=$PWD`` when running from a git worktree, so
``add_local_python_source`` ships THIS tree's torchwright_doom.)
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
# Production compile container width (modal_render.compile_remote sets the
# same); env does NOT forward through `make modal-run`, so set it here,
# before any solve.  Override by exporting it locally if you really want to.
os.environ.setdefault("TW_CPSAT_WORKERS", "64")
for _p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from ortools.sat.python import cp_model

from torchwright.compiler.forward.compile import (
    _reserve_rms_norm_columns,
    _run_heuristic_warm_start,
)
from torchwright.compiler.forward.cpsat_scheduler import (
    ATTN,
    MLP,
    Costs,
    ScheduleAssignment,
    build_cpsat_model,
    critical_path_layers,
    solve_schedule,
    _validate_hint,
)
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.compiler.graph_identity import canonical_ids, graph_fingerprint
from torchwright.graph.misc import LiteralValue
from torchwright.graph.node import reserve_node_id_above

# NOTE: torchwright_doom is imported INSIDE build_production_model, after
# apply_screen_env — the token vocab is screen-sized at import time (the
# screen-env trap; see modal_run.py's forwarding comment).

# The lower-bound / shaving cohort that contributed nothing in the measured
# d8192 production run (copied from scripts/cpsat_space_experiments.py).
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


@dataclass
class ProdModel:
    """Everything the production solve derives before the solver runs."""

    config_path: str
    d: int
    d_head: int
    d_hidden: int  # raw (fingerprint / cache-meta value), 16384 in production
    solver_d_hidden: int  # what solve_schedule sees: d_hidden - 1 under bias=False
    bias: bool
    assume_zero_init: bool
    max_layers: int  # config horizon (200)
    n_reserved_residual: int  # RMSNorm pinned columns (2 in production)
    policy: SchedulingPolicy
    costs: Costs
    output_node: Any  # lowered graph output
    graph: GraphAnalyzer
    fingerprint: str
    cp_layers: int  # dependency floor of the lowered graph
    # Warm start (None until run_warm_start()):
    hint_layers: dict | None = None
    hint_routing: dict | None = None
    hint_cancel: dict | None = None
    hint_n_layers: int = 0
    _residual_map: Any = field(default=None, repr=False)
    _computed: Any = field(default=None, repr=False)

    @property
    def solver_max_layers(self) -> int:
        if self.hint_n_layers > 0:
            return min(self.max_layers, self.hint_n_layers + 1)
        return self.max_layers

    def run_warm_start(self) -> None:
        """The compile.py warm start, verbatim (mutates a clone only)."""
        t0 = time.perf_counter()
        (
            self.hint_layers,
            self.hint_routing,
            self.hint_cancel,
            self.hint_n_layers,
        ) = _run_heuristic_warm_start(
            graph=self.graph,
            d=self.d,
            d_head=self.d_head,
            pos_encoding=None,
            d_hidden=self.d_hidden,
            residual_map=self._residual_map,
            computed=self._computed,
            clusters=None,
            admission_budget_fraction=0.4,
            policy=self.policy,
            output_node=self.output_node,
            max_layers=self.max_layers,
            bias=self.bias,
        )
        print(
            f"warm start: {self.hint_n_layers} layers, "
            f"{len(self.hint_layers)} layer hints / "
            f"{len(self.hint_routing)} routing / {len(self.hint_cancel)} cancel "
            f"({time.perf_counter() - t0:.1f}s); solver horizon "
            f"{self.solver_max_layers}",
            flush=True,
        )


def build_production_model(config_path: str, verbose: bool = False) -> ProdModel:
    """Build graph + lower + residual seed + fingerprint, production-exactly."""
    # Screen env BEFORE any torchwright_doom graph import (vocab is built
    # screen-sized at import).
    from torchwright_doom.inference.config import (
        apply_screen_env,
        load_render_config,
        resolve_wad_path,
    )

    config = load_render_config(config_path)
    apply_screen_env(config)
    m = config.model
    print(
        f"config {config_path}: d={m.d} d_head={m.d_head} d_rot={m.d_rot} "
        f"d_hidden={m.d_hidden} bias={m.bias} optimize={m.optimize} "
        f"scale={m.scale} screen={config.screen} detail={m.detail} "
        f"hud={m.hud} max_layers={m.max_layers} max_seq_len={m.max_seq_len} "
        f"assume_zero_init={m.assume_zero_init}",
        flush=True,
    )

    from torchwright_doom.inference.compiled_model import build_graph

    wad_path = resolve_wad_path(config)
    t0 = time.perf_counter()
    next_token, _rope, _emb, _banks = build_graph(
        d_head=m.d_head,
        max_positions=m.max_seq_len,
        d_rot=m.d_rot,
        asset_config=config.asset_config(),
        wad_path=wad_path,
    )
    print(f"graph build: {time.perf_counter() - t0:.1f}s (wad={wad_path})", flush=True)

    d = m.d
    d_hidden = m.d_hidden if m.d_hidden is not None else d

    # Lower exactly as forward_compile (compile.py): both collapse passes on,
    # lane cap d_hidden//4.  lower() owns fusion — no explicit pre-fuse.
    from torchwright.compiler.lower import lower

    t0 = time.perf_counter()
    lowered = lower(
        next_token,
        verbose=verbose,
        collapse_univariate=True,
        collapse_pl=True,
        collapse_lane_cap=d_hidden // 4,
    )
    output_node = lowered.output_node
    print(f"lower: {time.perf_counter() - t0:.1f}s", flush=True)

    graph = GraphAnalyzer(output_node)
    n_nodes = len(graph.get_all_nodes())

    # Residual-stream seed, replicated from forward_compile in ORDER:
    # const-1 mint (after reserve_node_id_above), inputs sorted by node_id,
    # assume_zero_init free-pool mark_clean, THEN the RMSNorm reservation.
    input_nodes = sorted(
        (n for n in graph.get_all_nodes() if graph.is_input_node(n)),
        key=lambda n: n.node_id,
    )
    residual_map = ResidualStreamMap(d)
    reserve_node_id_above(graph.get_all_nodes())
    const_one = LiteralValue(torch.ones(1), name="rope_self_match_const_one")
    residual_map.allocate(const_one)
    residual_map.mark_clean(residual_map.get_indices(const_one))
    for node in input_nodes:
        residual_map.allocate(node)
        residual_map.mark_clean(residual_map.get_indices(node))
    if m.assume_zero_init:
        residual_map.mark_clean(set(residual_map._free))
    computed = set(input_nodes)

    # Pinned-constant RMSNorm: production is rms_norm ON (compile_to_onnx
    # default at the supported power-of-two d) with q=63
    # (compile_to_onnx_path's default), eps 1e-5.
    rms_spec = _reserve_rms_norm_columns(residual_map, d, 1e-5, 63)
    n_reserved = len(rms_spec.reserved_cols)

    policy = SchedulingPolicy()
    fp = graph_fingerprint(
        output_node,
        d=d,
        d_head=m.d_head,
        d_hidden=d_hidden,
        flex_routing=True,
        assume_zero_init=m.assume_zero_init,
        cancel_slack=2,
        policy=policy,
        reserve_residual=n_reserved,
        bias=m.bias,
    )
    cp = critical_path_layers(output_node, None, policy=policy, flex_routing=True)

    solver_d_hidden = d_hidden if m.bias else d_hidden - 1
    print(
        f"lowered graph: {n_nodes} nodes, {len(input_nodes)} inputs; "
        f"critical path (DAG floor) = {cp}; reserve_residual={n_reserved}; "
        f"solver_d_hidden={solver_d_hidden}",
        flush=True,
    )
    print(f"GRAPH_FINGERPRINT {fp}", flush=True)

    return ProdModel(
        config_path=config_path,
        d=d,
        d_head=m.d_head,
        d_hidden=d_hidden,
        solver_d_hidden=solver_d_hidden,
        bias=m.bias,
        assume_zero_init=m.assume_zero_init,
        max_layers=m.max_layers,
        n_reserved_residual=n_reserved,
        policy=policy,
        costs=Costs(),
        output_node=output_node,
        graph=graph,
        fingerprint=fp,
        cp_layers=cp,
        _residual_map=residual_map,
        _computed=computed,
    )


# ---------------------------------------------------------------------------
# Solver-log parsing (copied from scripts/cpsat_space_experiments.py)


def _parse_log(log: str) -> dict:
    """Extract presolve time, incumbent trajectory, final LB, and var count."""
    out: dict = {
        "presolve_s": None,
        "first_incumbent_s": None,
        "traj": [],
        "lb": None,
        "n_vars": None,
    }
    m = re.search(r"Starting search at ([\d.]+)s", log)
    if m:
        out["presolve_s"] = float(m.group(1))
    m = re.search(r"#Variables:\s*([\d']+)", log)
    if m:
        out["n_vars"] = int(m.group(1).replace("'", ""))
    for m in re.finditer(r"^#\d+\s+([\d.]+)s\s+best:(\d+)", log, re.M):
        t, best = float(m.group(1)), int(m.group(2))
        out["traj"].append((t, best))
        if out["first_incumbent_s"] is None:
            out["first_incumbent_s"] = t
    lbs = re.findall(r"next:\[(\d+),", log)
    if lbs:
        out["lb"] = int(lbs[-1])
    return out


def _print_solve_report(name: str, stats, assignment, wall_s: float) -> None:
    info = _parse_log(stats.solver_log)
    scale = getattr(stats, "objective_scale", 1)
    if scale > 1:
        info["traj"] = [(t, b // scale) for t, b in info["traj"]]
        info["lb"] = info["lb"] // scale if info["lb"] is not None else None
    traj = " ".join(f"{b}@{t:.0f}s" for t, b in info["traj"])
    print(
        f"--> {name}: status={stats.status_name} "
        f"n_layers={assignment.n_layers if assignment else -1} "
        f"lb={info['lb']} bound={stats.best_objective_bound} "
        f"n_vars={info['n_vars']} presolve={info['presolve_s']}s "
        f"first_incumbent={info['first_incumbent_s']}s wall={wall_s:.0f}s",
        flush=True,
    )
    if traj:
        print(f"    trajectory: {traj}", flush=True)


def _print_warnings(caught: list[warnings.WarningMessage]) -> None:
    for w in caught:
        print(f"RUNTIME_WARNING [{w.category.__name__}] {w.message}", flush=True)


# ---------------------------------------------------------------------------
# Schedule JSON emission (store_assignment payload format, byte-compatible)


def emit_schedule_json(pm: ProdModel, assignment, meta: dict) -> None:
    """Print the schedule in schedule_cache.store_assignment's exact payload
    format (canonical-id keys), gzip+base64 so it survives a Modal log.

    Decode locally:
        awk '/B64_BEGIN/{f=1;next}/B64_END/{f=0}f' log | base64 -d | \\
            gunzip > <fingerprint>.json
    then `modal volume put torchwright-doom-schedule-cache <fingerprint>.json
    <fingerprint>.json` (store_assignment's one-way min-ratchet applies only
    to in-process writes; a manual put overwrites — check the existing entry
    first).
    """
    canon = canonical_ids(pm.output_node)
    if not set(assignment.node_to_layer) <= set(canon):
        print("SCHEDULE_JSON_SKIPPED: scheduled node unreachable from output")
        return
    payload = {
        "node_to_layer": {canon[k]: v for k, v in assignment.node_to_layer.items()},
        "node_to_cancel_layer": {
            canon[k]: v
            for k, v in assignment.node_to_cancel_layer.items()
            if k in canon
        },
        "node_to_routing": {canon[k]: v for k, v in assignment.node_to_routing.items()},
        "n_layers": assignment.n_layers,
        "meta": meta,
    }
    raw = json.dumps(payload).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    b64 = base64.b64encode(gzip.compress(raw)).decode("ascii")
    print(
        f"SCHEDULE_JSON fingerprint={pm.fingerprint} "
        f"n_layers={assignment.n_layers} sha256={digest} bytes={len(raw)}",
        flush=True,
    )
    print("SCHEDULE_JSON_B64_BEGIN", flush=True)
    for i in range(0, len(b64), 120):
        print(b64[i : i + 120])
    print("SCHEDULE_JSON_B64_END", flush=True)


def _schedule_meta(pm: ProdModel, stats) -> dict:
    """The meta dict compile.py writes with store_assignment, key-for-key."""
    return {
        "status_name": stats.status_name,
        "best_objective_bound": stats.best_objective_bound,
        "is_optimal": stats.is_optimal,
        "d": pm.d,
        "d_head": pm.d_head,
        "d_hidden": pm.d_hidden,
    }


# ---------------------------------------------------------------------------
# Subcommands


def set_hint_from_embedded(pm: ProdModel) -> None:
    """Replace the heuristic warm-start hint with the best-known schedule
    embedded in scripts/_hint_payload.py (ratchet-descend mode).

    The payload is store_assignment-format (canonical-id keys); remap onto
    the current graph exactly as schedule_cache.load_assignment does.  A
    fingerprint mismatch means the embedded schedule belongs to a different
    graph — refuse loudly rather than hint garbage.
    """
    from scripts import _hint_payload

    if _hint_payload.FINGERPRINT != pm.fingerprint:
        raise SystemExit(
            f"--hint-embedded: payload fingerprint "
            f"{_hint_payload.FINGERPRINT[:12]} != graph {pm.fingerprint[:12]}"
        )
    payload = json.loads(gzip.decompress(base64.b64decode(_hint_payload.PAYLOAD_B64)))
    current_by_canon = {c: nid for nid, c in canonical_ids(pm.output_node).items()}
    pm.hint_layers = {
        current_by_canon[int(k)]: v for k, v in payload["node_to_layer"].items()
    }
    pm.hint_cancel = {
        current_by_canon[int(k)]: v for k, v in payload["node_to_cancel_layer"].items()
    }
    pm.hint_routing = {
        current_by_canon[int(k)]: v for k, v in payload["node_to_routing"].items()
    }
    pm.hint_n_layers = payload["n_layers"]
    print(
        f"embedded hint: n_layers={pm.hint_n_layers}, "
        f"{len(pm.hint_layers)} layer / {len(pm.hint_routing)} routing / "
        f"{len(pm.hint_cancel)} cancel; solver horizon {pm.solver_max_layers}",
        flush=True,
    )


def _solver_params(args) -> dict | None:
    params: dict = {}
    if getattr(args, "lns_heavy", False):
        params["shared_tree_num_workers"] = 0
        params["ignore_subsolvers"] = list(_LB_SUBSOLVERS)
    if args.params:
        params.update(json.loads(args.params))
    if args.seed is not None:
        params["random_seed"] = args.seed
    return params or None


def cmd_fingerprint(args) -> None:
    pm = build_production_model(args.config, verbose=args.verbose)
    if args.warm_start:
        pm.run_warm_start()
    print(
        f"CALIBRATION fingerprint={pm.fingerprint[:12]} cp={pm.cp_layers} "
        f"reserve_residual={pm.n_reserved_residual} "
        f"solver_d_hidden={pm.solver_d_hidden} "
        f"hint_n={pm.hint_n_layers if args.warm_start else 'not-run'}",
        flush=True,
    )


def cmd_solve(args) -> None:
    pm = build_production_model(args.config, verbose=args.verbose)
    if args.hint_embedded:
        set_hint_from_embedded(pm)
    else:
        pm.run_warm_start()
    params = _solver_params(args)

    # ---- Seed sweep: one build + warm start, N main solves (statistical
    # re-baseline mode).  The floor probe is skipped (it has its own seed
    # sensitivity; sweep it explicitly via repeated --floor-probe=only runs
    # if that is the question).  Emits the best draw's schedule.
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        best: tuple[Any, Any] | None = None
        results = []
        for seed in seeds:
            seed_params = dict(params or {})
            seed_params["random_seed"] = seed
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assignment, stats = solve_schedule(
                    pm.output_node,
                    None,
                    d=pm.d,
                    d_head=pm.d_head,
                    d_hidden=pm.solver_d_hidden,
                    costs=pm.costs,
                    flex_routing=True,
                    time_budget_s=args.budget,
                    max_layers=pm.solver_max_layers,
                    policy=pm.policy,
                    reserve_residual=pm.n_reserved_residual,
                    assume_zero_init=pm.assume_zero_init,
                    tighten_domains=True,
                    hint_layers=pm.hint_layers or None,
                    hint_routing=pm.hint_routing or None,
                    hint_cancel=pm.hint_cancel or None,
                    log_search_progress=True,
                    solver_params=seed_params,
                )
            _print_warnings(caught)
            _print_solve_report(
                f"seed-{seed}", stats, assignment, time.perf_counter() - t0
            )
            n = assignment.n_layers if assignment else None
            results.append((seed, stats.status_name, n))
            if assignment is not None and (
                best is None or assignment.n_layers < best[0].n_layers
            ):
                best = (assignment, stats)
        print(
            "SWEEP_SUMMARY " + " ".join(f"seed{s}:{st}:{n}" for s, st, n in results),
            flush=True,
        )
        if best is not None and not args.no_emit:
            emit_schedule_json(pm, best[0], _schedule_meta(pm, best[1]))
        return

    assignment = None
    stats = None

    # ---- Floor probe (the optimize>=2 production step): cold solve at
    # horizon cp+1, budget min(150, budget).
    if args.floor_probe != "off" and pm.cp_layers + 1 < pm.solver_max_layers:
        probe_budget = min(150.0, args.budget)
        print(
            f"floor probe: horizon {pm.cp_layers + 1} "
            f"(critical path {pm.cp_layers}), budget {probe_budget:.0f}s",
            flush=True,
        )
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assignment, stats = solve_schedule(
                pm.output_node,
                None,
                d=pm.d,
                d_head=pm.d_head,
                d_hidden=pm.solver_d_hidden,
                costs=pm.costs,
                flex_routing=True,
                time_budget_s=probe_budget,
                max_layers=pm.cp_layers + 1,
                policy=pm.policy,
                reserve_residual=pm.n_reserved_residual,
                assume_zero_init=pm.assume_zero_init,
                tighten_domains=True,
                log_search_progress=True,
                solver_params=params,
            )
        _print_warnings(caught)
        _print_solve_report("floor-probe", stats, assignment, time.perf_counter() - t0)

    if args.floor_probe == "only":
        if assignment is not None:
            emit_schedule_json(pm, assignment, _schedule_meta(pm, stats))
        return

    # ---- Warm-start descent (the production main solve).
    if assignment is None:
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assignment, stats = solve_schedule(
                pm.output_node,
                None,
                d=pm.d,
                d_head=pm.d_head,
                d_hidden=pm.solver_d_hidden,
                costs=pm.costs,
                flex_routing=True,
                time_budget_s=args.budget,
                max_layers=pm.solver_max_layers,
                policy=pm.policy,
                reserve_residual=pm.n_reserved_residual,
                assume_zero_init=pm.assume_zero_init,
                tighten_domains=True,
                hint_layers=pm.hint_layers or None,
                hint_routing=pm.hint_routing or None,
                hint_cancel=pm.hint_cancel or None,
                log_search_progress=True,
                solver_params=params,
            )
        _print_warnings(caught)
        _print_solve_report("main-solve", stats, assignment, time.perf_counter() - t0)

    if assignment is not None and not args.no_emit:
        emit_schedule_json(pm, assignment, _schedule_meta(pm, stats))


def cmd_probe_k(args) -> None:
    pm = build_production_model(args.config, verbose=args.verbose)
    use_hint = args.hint or args.hint_embedded
    if args.hint_embedded:
        set_hint_from_embedded(pm)
    elif use_hint:
        pm.run_warm_start()

    k = args.k
    cancel_slack = None if args.cancel_slack < 0 else args.cancel_slack
    disabled = frozenset(f for f in (args.disable_families or "").split(",") if f)

    # squeeze: the model's horizon IS K (hardest domain pruning; the
    # production floor probe's construction).  wide: horizon stays at the
    # warm-start's, with n_layers <= K as a hard constraint on top — the
    # hint stays in-domain so the solver can repair it downward.
    if args.mode == "squeeze":
        model_max_layers = k
    else:
        if not use_hint:
            raise SystemExit("--mode=wide requires --hint (its point is the hint)")
        model_max_layers = pm.solver_max_layers

    t0 = time.perf_counter()
    built = build_cpsat_model(
        pm.output_node,
        None,
        d=pm.d,
        d_head=pm.d_head,
        d_hidden=pm.solver_d_hidden,
        costs=pm.costs,
        flex_routing=True,
        max_layers=model_max_layers,
        cancel_slack=cancel_slack,
        policy=pm.policy,
        reserve_residual=pm.n_reserved_residual,
        assume_zero_init=pm.assume_zero_init,
        tighten_domains=not args.no_tighten,
        hint_layers=pm.hint_layers if use_hint else None,
        hint_cancel=pm.hint_cancel if use_hint else None,
        _disabled_families=disabled,
    )
    built.model.Add(built.n_layers_var <= k)
    print(
        f"probe-k model: K<={k} mode={args.mode} horizon={model_max_layers} "
        f"cancel_slack={cancel_slack} hint={use_hint} "
        f"disabled={sorted(disabled) or 'none'} "
        f"built in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    # Hint application — solve_schedule's AddHint loop, guards and all.
    if use_hint:
        assert (
            pm.hint_layers is not None
            and pm.hint_routing is not None
            and pm.hint_cancel is not None
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _validate_hint(
                built,
                pm.hint_layers,
                pm.hint_routing,
                pm.hint_cancel,
                max_layers=model_max_layers,
                strict=False,
            )
        _print_warnings(caught)
        applied = 0
        for nid, layer in pm.hint_layers.items():
            if nid in built.layer_var and 0 <= layer < model_max_layers:
                built.model.AddHint(built.layer_var[nid], layer)
                applied += 1
        for nid, route in pm.hint_routing.items():
            if nid in built.is_attn:
                built.model.AddHint(built.is_attn[nid], 1 if route == ATTN else 0)
                applied += 1
        for nid, layer in pm.hint_cancel.items():
            if nid in built.cancel_layer and 0 <= layer <= model_max_layers:
                built.model.AddHint(built.cancel_layer[nid], layer)
                applied += 1
            elif nid in built.input_cancel_layer and 0 <= layer <= model_max_layers:
                built.model.AddHint(built.input_cancel_layer[nid], layer)
                applied += 1
        print(f"hint: {applied} AddHint calls survived the guards", flush=True)

    # Decision strategy — solve_schedule's, verbatim (schedule the deepest
    # critical-path nodes first, earliest layer first).
    if not args.no_decision_strategy:
        nodes_by_cp = sorted(
            built.gm.schedulable,
            key=lambda n: -built.gm.graph.get_critical_path_length(n),
        )
        built.model.AddDecisionStrategy(
            [built.layer_var[n.node_id] for n in nodes_by_cp],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE,
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.budget
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = int(os.environ.get("TW_CPSAT_WORKERS", "16"))
    params = _solver_params(args)
    if params:
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                getattr(solver.parameters, key).extend(value)
            else:
                setattr(solver.parameters, key, value)
    log_buf: list[str] = []
    solver.log_callback = log_buf.append

    t0 = time.perf_counter()
    status = solver.Solve(built.model)
    wall = time.perf_counter() - t0
    status_name = solver.StatusName(status)
    log = "\n".join(log_buf)
    info = _parse_log(log)
    print(
        f"--> probe K<={k}: {status_name} lb={info['lb']} "
        f"n_vars={info['n_vars']} presolve={info['presolve_s']}s "
        f"first_incumbent={info['first_incumbent_s']}s wall={wall:.0f}s",
        flush=True,
    )
    if info["traj"]:
        print(
            "    trajectory: " + " ".join(f"{b}@{t:.0f}s" for t, b in info["traj"]),
            flush=True,
        )

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Read the assignment back exactly as solve_schedule does.
        node_to_layer: dict[int, int] = {}
        node_to_cancel: dict[int, int] = {}
        node_to_routing: dict[int, str] = {}
        for n in built.gm.schedulable:
            node_to_layer[n.node_id] = solver.Value(built.layer_var[n.node_id])
            node_to_cancel[n.node_id] = solver.Value(built.cancel_layer[n.node_id])
            node_to_routing[n.node_id] = (
                ATTN if solver.Value(built.is_attn[n.node_id]) else MLP
            )
        for nid, var in built.input_cancel_layer.items():
            node_to_cancel[nid] = solver.Value(var)
        assignment = ScheduleAssignment(
            node_to_layer=node_to_layer,
            node_to_cancel_layer=node_to_cancel,
            node_to_routing=node_to_routing,
            n_layers=solver.Value(built.n_layers_var),
        )
        print(f"SAT: schedule with n_layers={assignment.n_layers}", flush=True)
        meta = {
            "status_name": status_name,
            "best_objective_bound": float(solver.BestObjectiveBound()),
            "is_optimal": status == cp_model.OPTIMAL,
            "d": pm.d,
            "d_head": pm.d_head,
            "d_hidden": pm.d_hidden,
        }
        emit_schedule_json(pm, assignment, meta)
    elif status_name == "INFEASIBLE":
        print(
            f"UNSAT: no schedule with n_layers <= {k} exists under this "
            f"constraint set (families disabled: {sorted(disabled) or 'none'})",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1m1.yaml")
    ap.add_argument("--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fingerprint", help="calibration: build + fingerprint")
    p.add_argument("--warm-start", action="store_true", help="also report hint depth")
    p.set_defaults(fn=cmd_fingerprint)

    p = sub.add_parser("solve", help="replicate the production solve flow")
    p.add_argument("--budget", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--seeds",
        default=None,
        help="csv seed sweep: one build, N main solves, no floor probe",
    )
    p.add_argument("--params", default=None, help="JSON CpSolver param overrides")
    p.add_argument("--lns-heavy", action="store_true")
    p.add_argument(
        "--floor-probe",
        choices=["auto", "off", "only"],
        default="auto",
        help="auto = production flow; off = skip the cold cp+1 probe",
    )
    p.add_argument(
        "--hint-embedded",
        action="store_true",
        help="hint with scripts/_hint_payload.py's best-known schedule "
        "instead of the heuristic warm start (ratchet-descend)",
    )
    p.add_argument("--no-emit", action="store_true")
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("probe-k", help="fixed-K decision probe (n_layers <= K)")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--mode", choices=["squeeze", "wide"], default="squeeze")
    p.add_argument("--hint", action="store_true")
    p.add_argument(
        "--hint-embedded",
        action="store_true",
        help="hint with scripts/_hint_payload.py's best-known schedule",
    )
    p.add_argument("--budget", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--params", default=None, help="JSON CpSolver param overrides")
    p.add_argument("--lns-heavy", action="store_true")
    p.add_argument(
        "--cancel-slack",
        type=int,
        default=2,
        help="production 2; 0 = zero window; -1 = unrestricted (None)",
    )
    p.add_argument("--disable-families", default="")
    p.add_argument("--no-tighten", action="store_true")
    p.add_argument("--no-decision-strategy", action="store_true")
    p.set_defaults(fn=cmd_probe_k)

    args = ap.parse_args()
    print(
        f"harness: TW_CPSAT_WORKERS={os.environ['TW_CPSAT_WORKERS']} "
        f"argv={sys.argv[1:]}",
        flush=True,
    )
    args.fn(args)


if __name__ == "__main__":
    main()
