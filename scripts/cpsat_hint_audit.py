"""Audit the production warm-start hint against the CP-SAT descent model.

Why: the production e1m1 compile (optimize=2) reliably ends in the silent
heuristic fallback — the descent solve gets a complete no-eager warm-start
hint yet finds ZERO incumbents in its 180s budget (solver log: ``best:inf``
throughout, ``best_solutions: 0``).  The documented signature of an
*infeasible* hint (docs/cpsat_scheduler.md §5, the eager-free discovery) is
exactly this: CP-SAT silently drops a hint that violates the model and
cold-searches into UNKNOWN.  A feasible hint historically produced a first
incumbent in ~70-80s on a larger model (post-J, ~80k vars).

What this does: runs the REAL production compile path (e1m1 config, hud env,
linear fusion, forward_compile's own pre-solve setup and warm start) with
``solve_schedule`` intercepted at the descent call, then — instead of
solving — audits the captured hint against the captured model:

  Phase 1  Domain audit (no solve): every hinted value checked against its
           variable's domain in the built model (catches tighten_domains
           bounds that exclude the hint, AddHint guard drops, routing hints
           conflicting with pinned routings).
  Phase 2  Hard-fix feasibility: pin layer+routing+cancel to the hint as
           hard equalities and solve.  OPTIMAL/FEASIBLE => the hint IS a
           model point and the fallback is a pure search failure.
           INFEASIBLE => the hint violates the model (a bug — localize).
  Phase 3  (only if Phase 2 is INFEASIBLE) Assumption core: re-add the
           equalities under assumption literals and ask CP-SAT for a
           sufficient-for-infeasibility subset, naming the exact nodes.

Run (CPU-only; graph build + warm start take a few minutes):

    ../.venv/bin/python -m scripts.cpsat_hint_audit
    ../.venv/bin/python -m scripts.cpsat_hint_audit --budget 300 --fix-budget 120
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()

_UMBRELLA = Path(__file__).resolve().parents[2]
# Snapshot mode (same contract as baseline_soundness_scan.py): run the audit
# against a checked-out torchwright/torchwright_doom pair instead of HEAD.
#   TW_SNAPSHOT_PATHS=<tw_dir>:<twd_dir> TW_VENV_SITE=<site-packages> \
#       ../.venv/bin/python -S scripts/cpsat_hint_audit.py
# (-S skips .pth files so the HEAD editable-install finder never activates.)
_SNAP = os.environ.get("TW_SNAPSHOT_PATHS")
if _SNAP:
    sys.path[:0] = _SNAP.split(os.pathsep) + [os.environ["TW_VENV_SITE"]]
else:
    for _p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

# The schedule cache would skip the solve entirely — the audit needs the
# real descent call to fire.
os.environ.pop("TW_SCHEDULE_CACHE_DIR", None)


class _AuditDone(Exception):
    """Unwinds forward_compile once the descent call has been audited."""


# NEVER call IntVar.Proto()/.proto here: ortools 9.15 returns corrupted
# memory from it (documented in cpsat_scheduler commit 4876043) — the
# corruption poisons the model and a later Solve() segfaults.  Layer-domain
# checks go through _compute_layer_bounds (the source of the tightened
# domains) instead.


def _node_desc(gm, nid: int) -> str:
    node = next((n for n in gm.graph.get_all_nodes() if n.node_id == nid), None)
    if node is None:
        return f"id={nid} <not in graph>"
    ann = getattr(node, "annotation", "") or ""
    return f"id={nid} {type(node).__name__} name={node.name!r} ann={ann!r}"


def _audit(output_node, pos_encoding, kw, fix_budget_s: float, core_budget_s: float):
    from ortools.sat.python import cp_model

    from torchwright.compiler.forward.cpsat_scheduler import (
        ATTN,
        Costs,
        _compute_layer_bounds,
        build_cpsat_model,
    )
    from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy

    hint_layers = kw.get("hint_layers") or {}
    hint_routing = kw.get("hint_routing") or {}
    hint_cancel = kw.get("hint_cancel") or {}
    max_layers = kw["max_layers"]

    build_kw = dict(
        d=kw["d"],
        d_head=kw["d_head"],
        d_hidden=kw["d_hidden"],
        costs=kw.get("costs", Costs()),
        flex_routing=kw.get("flex_routing", True),
        max_layers=max_layers,
        cancel_slack=kw.get("cancel_slack", 2),
        policy=kw.get("policy"),
        reserve_heads=kw.get("reserve_heads", 0),
        reserve_residual=kw.get("reserve_residual", 0),
        tighten_domains=kw.get("tighten_domains", False),
        # The production model is hint-aware: the captured hints size the
        # per-node cancel windows (hint-aware widening), so the audit must
        # build the same widened model production solves.
        hint_layers=hint_layers or None,
        hint_cancel=hint_cancel or None,
    )
    print("\n=== captured descent call ===", flush=True)
    for k, v in build_kw.items():
        print(f"  {k} = <{len(v)} entries>" if isinstance(v, dict) else f"  {k} = {v}")
    print(
        f"  hints: layers={len(hint_layers)} routing={len(hint_routing)} "
        f"cancel={len(hint_cancel)}"
    )

    t0 = time.perf_counter()
    built = build_cpsat_model(output_node, pos_encoding, **build_kw)
    print(f"  model built in {time.perf_counter() - t0:.1f}s", flush=True)
    if built.cancel_window_delta:
        print(
            f"  cancel windows widened for {len(built.cancel_window_delta)} "
            f"nodes (max +{max(built.cancel_window_delta.values())})",
            flush=True,
        )
    gm = built.gm
    lv, ca, ia, ica = (
        built.layer_var,
        built.cancel_layer,
        built.is_attn,
        built.input_cancel_layer,
    )

    # ---- Phase 1: domain audit ----
    # Layer domains come from _compute_layer_bounds (what build_cpsat_model
    # uses under tighten_domains); cancel/routing var domains are the full
    # [0, max_layers] / {0,1} so only the AddHint guard applies to them.
    print("\n=== phase 1: domain audit ===", flush=True)
    if build_kw["tighten_domains"]:
        lo_b, hi_b = _compute_layer_bounds(
            gm,
            build_kw["policy"] or SchedulingPolicy(),
            build_kw["flex_routing"],
            max_layers,
            # build_cpsat_model's own `d_hidden` is already the usable
            # hidden-slot count; the static routing rule reads it to decide
            # whether a Linear's MLP bypass fits.
            usable_slots=build_kw["d_hidden"],
        )
    else:
        lo_b = hi_b = None
    guard_dropped: list[str] = []
    out_of_domain: list[str] = []
    for nid, L in hint_layers.items():
        if nid not in lv or not (0 <= L < max_layers):
            guard_dropped.append(f"layer  {_node_desc(gm, nid)} hint={L}")
        elif lo_b is not None and not (lo_b[nid] <= L <= hi_b[nid]):
            out_of_domain.append(
                f"layer  {_node_desc(gm, nid)} hint={L} "
                f"domain=[{lo_b[nid]},{hi_b[nid]}]"
            )
    for nid, route in hint_routing.items():
        if nid not in ia:
            guard_dropped.append(f"route  {_node_desc(gm, nid)} hint={route}")
    for nid, L in hint_cancel.items():
        if nid not in ca and nid not in ica:
            guard_dropped.append(f"cancel {_node_desc(gm, nid)} hint={L}")
        elif not (0 <= L <= max_layers):
            guard_dropped.append(f"cancel {_node_desc(gm, nid)} hint={L}")
    print(f"  guard-dropped hints (never reach AddHint): {len(guard_dropped)}")
    # `hint_layers` deliberately includes Concatenate nodes (the warm start
    # records every newly-computed node); the model has no vars for them, so
    # Concatenate drops are expected — only non-Concatenate drops are news.
    from collections import Counter

    kinds = Counter(s.split()[2] for s in guard_dropped)
    print(f"    by node type: {dict(kinds)}")
    interesting = [s for s in guard_dropped if "Concatenate" not in s]
    for s in interesting[:20]:
        print(f"    {s}")
    if len(interesting) > 20:
        print(f"    ... {len(interesting) - 20} more non-Concatenate")
    print(f"  hints outside their variable's domain:     {len(out_of_domain)}")
    for s in out_of_domain[:40]:
        print(f"    {s}")
    if len(out_of_domain) > 40:
        print(f"    ... {len(out_of_domain) - 40} more")

    # ---- Phase 2: hard-fix feasibility ----
    print("\n=== phase 2: hard-fix the hint, solve ===", flush=True)
    model = built.model
    n_fixed = 0
    for nid, L in hint_layers.items():
        if nid in lv:
            model.Add(lv[nid] == L)
            n_fixed += 1
    for nid, route in hint_routing.items():
        if nid in ia:
            model.Add(ia[nid] == (1 if route == ATTN else 0))
            n_fixed += 1
    for nid, L in hint_cancel.items():
        if nid in ca:
            model.Add(ca[nid] == L)
            n_fixed += 1
        elif nid in ica:
            model.Add(ica[nid] == L)
            n_fixed += 1
    # Presolve-only pass first: with every decision variable fixed by an
    # equality, propagation alone usually decides feasibility — and it avoids
    # the parallel search machinery (observed SIGSEGV inside Solve on this
    # model with the equalities added, laptop and Modal alike).
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = fix_budget_s
    solver.parameters.num_search_workers = 1
    solver.parameters.stop_after_presolve = True
    t1 = time.perf_counter()
    print("  presolve-only pass...", flush=True)
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    print(
        f"  presolve-only -> {status_name} ({time.perf_counter() - t1:.1f}s)",
        flush=True,
    )
    if status_name not in ("OPTIMAL", "FEASIBLE", "INFEASIBLE"):
        n_workers = int(os.environ.get("TW_CPSAT_WORKERS", "1"))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = fix_budget_s
        solver.parameters.num_search_workers = n_workers
        t1 = time.perf_counter()
        print(f"  full solve (workers={n_workers})...", flush=True)
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
    print(
        f"  fixed {n_fixed} vars -> {status_name} "
        f"({time.perf_counter() - t1:.1f}s)",
        flush=True,
    )
    if status_name in ("OPTIMAL", "FEASIBLE"):
        print(
            f"  hint IS a model point: n_layers="
            f"{solver.Value(built.n_layers_var)} — the fallback is a pure "
            f"SEARCH failure, not an infeasible hint."
        )
        return

    # ---- Phase 3: constraint-family bisect ----
    # (An assumption-based unsat core was tried first and came back
    # unminimized — all 16k literals — so it localizes nothing.)
    from torchwright.compiler.forward.cpsat_scheduler import CONSTRAINT_FAMILIES
    from torchwright.graph import Concatenate

    print("\n=== phase 3: constraint-family bisect ===", flush=True)

    def _fixed_status(disabled: frozenset, budget: float) -> str:
        b = build_cpsat_model(
            output_node, pos_encoding, _disabled_families=disabled, **build_kw
        )
        m = b.model
        for nid, L in hint_layers.items():
            if nid in b.layer_var:
                m.Add(b.layer_var[nid] == L)
        for nid, route in hint_routing.items():
            if nid in b.is_attn:
                m.Add(b.is_attn[nid] == (1 if route == ATTN else 0))
        for nid, L in hint_cancel.items():
            if nid in b.cancel_layer:
                m.Add(b.cancel_layer[nid] == L)
            elif nid in b.input_cancel_layer:
                m.Add(b.input_cancel_layer[nid] == L)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = budget
        s.parameters.num_search_workers = 1
        s.parameters.stop_after_presolve = True
        st = s.StatusName(s.Solve(m))
        if st not in ("OPTIMAL", "FEASIBLE", "INFEASIBLE"):
            s = cp_model.CpSolver()
            s.parameters.max_time_in_seconds = budget
            s.parameters.num_search_workers = 1
            st = s.StatusName(s.Solve(m))
        return st

    # Baseline: ALL families off.  Only the always-on constraints remain
    # (cancel >= layer+1; keep-forever cancel == max_layers for pinned /
    # Concat-consumed nodes; routing pins).  INFEASIBLE here means the hint
    # conflicts with an always-on constraint, and the per-family rows below
    # are all noise.
    all_fams = frozenset(CONSTRAINT_FAMILIES)
    st0 = _fixed_status(all_fams, core_budget_s / 12)
    print(f"  [ALL families OFF] -> {st0}", flush=True)
    if st0 == "INFEASIBLE":
        # Localize in pure Python: cancel >= layer+1 and keep-forever.
        keep_forever: set[int] = {n.node_id for n in gm.pinned_nodes}
        for n in gm.schedulable:
            if any(isinstance(c, Concatenate) for c in gm.consumers_eff.get(n, set())):
                keep_forever.add(n.node_id)
        bad_keep = [
            (nid, L)
            for nid, L in hint_cancel.items()
            if nid in keep_forever and L != max_layers
        ]
        bad_order = [
            (nid, L)
            for nid, L in hint_cancel.items()
            if nid in hint_layers and L < hint_layers[nid] + 1
        ]
        print(f"  cancel hints on keep-forever nodes (!= max_layers): {len(bad_keep)}")
        for nid, L in bad_keep[:20]:
            print(f"    cancel={L} (model wants {max_layers})  {_node_desc(gm, nid)}")
        print(f"  cancel hints with cancel < layer+1: {len(bad_order)}")
        for nid, L in bad_order[:20]:
            print(f"    cancel={L} layer={hint_layers[nid]}  {_node_desc(gm, nid)}")
    print("  --- enable ONE family (rest off): INFEASIBLE => that family alone")
    print("      rejects the heuristic schedule ---")
    for fam in sorted(CONSTRAINT_FAMILIES):
        st = _fixed_status(all_fams - {fam}, core_budget_s / 12)
        flag = "  <== REJECTS the hint" if st == "INFEASIBLE" else ""
        print(f"    only {fam:<22} -> {st}{flag}", flush=True)
    print("  --- disable ONE family (rest on): feasible => that family is the")
    print("      only rejector ---")
    for fam in sorted(CONSTRAINT_FAMILIES):
        st = _fixed_status(frozenset({fam}), core_budget_s / 12)
        flag = "  <== sole rejector" if st in ("OPTIMAL", "FEASIBLE") else ""
        print(f"    without {fam:<22} -> {st}{flag}", flush=True)

    # ---- Phase 4: cancel-window violations, node by node (pure Python) ----
    # Mirrors the UNIFORM cancel_slack window: for a non-keep-forever
    # schedulable node, cancel <= last_consumer + 1 + K (consumers = the
    # Concat-transparent effective consumers that have a layer var); with no
    # such consumer, cancel <= layer + 1 + K.  Inputs: same with birth 0.
    # Deliberately delta-blind (ignores the hint-aware widening): phases 3-4
    # only run when phase 2 rejected the hint, and the raw overshoot against
    # the uniform window is the number that localizes WHY.  A node listed
    # here with `over_by` <= its widening delta is admitted by the production
    # model — check `built.cancel_window_delta` before reading it as a bug.
    window = build_kw["cancel_slack"]
    print(f"\n=== phase 4: cancel-window (K={window}) violations ===", flush=True)

    def _window_violations(nodes, birth_of):
        out = []
        for n in nodes:
            nid = n.node_id
            if nid not in hint_cancel or n in gm.pinned_nodes:
                continue
            cons = gm.consumers_eff.get(n, set())
            if any(isinstance(c, Concatenate) for c in cons):
                continue  # keep-forever in the model; no window constraint
            cons_layers = [
                hint_layers[c.node_id]
                for c in cons
                if c.node_id in lv and c.node_id in hint_layers
            ]
            base = max(cons_layers) if cons_layers else birth_of(nid)
            ub = base + 1 + window
            if hint_cancel[nid] > ub:
                out.append((hint_cancel[nid] - ub, nid, hint_cancel[nid], base))
        return out

    viol = _window_violations(gm.schedulable, lambda nid: hint_layers.get(nid, 0))
    viol_in = _window_violations(
        [n for n in gm.input_nodes if n is not gm.output_node], lambda nid: 0
    )
    allv = sorted(viol + viol_in, reverse=True)
    print(f"  schedulable violations: {len(viol)}, input violations: {len(viol_in)}")
    if allv:
        worst = allv[0][0]
        print(
            f"  max overshoot: {worst} layers -> window K={window + worst} "
            f"would admit this hint"
        )
        for over, nid, cl, base in allv[:25]:
            print(
                f"    cancel={cl} last_consumer/birth={base} over_by={over}  "
                f"{_node_desc(gm, nid)}"
            )
        if len(allv) > 25:
            print(f"    ... {len(allv) - 25} more")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1m1.yaml")
    ap.add_argument(
        "--mode",
        choices=["audit", "descent"],
        default="audit",
        help=(
            "audit: hint-vs-model audit (phases 1-3). descent: run the REAL "
            "solve_schedule on the captured production call (use with "
            "--no-tighten / --budget to test search-behavior hypotheses)."
        ),
    )
    ap.add_argument(
        "--no-tighten",
        action="store_true",
        help="descent mode: override tighten_domains=False",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=None,
        help="descent mode: override time_budget_s",
    )
    ap.add_argument(
        "--cancel-slack",
        type=int,
        default=None,
        help="descent mode: override the cancel-window K",
    )
    ap.add_argument(
        "--no-fuse",
        action="store_true",
        help=(
            "disable the linear-fusion pass before compile (native-Block "
            "graph, no folds) — discriminates 2b (native blocks) from 2c "
            "(block-aware fusion) as the source of a hint violation"
        ),
    )
    ap.add_argument(
        "--fix-budget",
        type=float,
        default=120.0,
        help="seconds for the phase-2 hard-fix solve",
    )
    ap.add_argument(
        "--core-budget",
        type=float,
        default=600.0,
        help="seconds for the phase-3 assumption-core solve",
    )
    args = ap.parse_args()

    # Screen env BEFORE any vocab/embedding/scene import (the DOOM vocab is
    # screen-sized at import time) — exactly what modal_render.compile_remote
    # does.  Verifying the wrong (hud-off) graph proves nothing.
    from torchwright_doom.inference.config import apply_screen_env, load_render_config

    candidates = [
        Path(args.config),
        _UMBRELLA / "torchwright_doom" / args.config,
        Path("/root/configs") / Path(args.config).name,  # modal-run layout
    ]
    config_path = next((p for p in candidates if p.is_file()), None)
    if config_path is None:
        raise SystemExit(f"config not found; tried {[str(p) for p in candidates]}")
    config = load_render_config(str(config_path))
    apply_screen_env(config)

    import torchwright.compiler.forward.compile as compile_mod
    from torchwright.compiler.forward.cpsat_scheduler import SolveStats

    from torchwright_doom.inference.compiled_model import compile_to_onnx_path
    from torchwright_doom.inference.config import resolve_wad_path

    if args.no_fuse:
        import torchwright.graph.optimize as optimize_mod

        optimize_mod.fuse_consecutive_linears = lambda *a, **k: 0
        print("[audit] fusion DISABLED (--no-fuse)", flush=True)

    captured: dict = {}

    def _intercept(output_node, pos_encoding=None, **kw):
        if kw.get("hint_layers") is None:
            # Floor probe: skip the solve (production's probe returns UNKNOWN
            # at this scale anyway) so the compile proceeds to the descent.
            print("  [audit] floor probe intercepted — skipping", flush=True)
            return None, SolveStats(
                status_name="SKIPPED",
                objective_value=-1,
                best_objective_bound=-1.0,
                wall_time_s=0.0,
                solver_log="",
                total_attn_heads=-1,
                total_mlp_bypass_slots=-1,
                is_optimal=False,
            )
        print("  [audit] descent call intercepted", flush=True)
        captured["output_node"] = output_node
        captured["pos_encoding"] = pos_encoding
        captured["kw"] = kw
        raise _AuditDone()

    compile_mod.solve_schedule = _intercept

    out_path = (
        Path(os.environ.get("TMPDIR", "/tmp")) / "cpsat_hint_audit" / "model.onnx"
    )
    t0 = time.perf_counter()
    try:
        compile_to_onnx_path(
            out_path,
            d=config.model.d,
            d_head=config.model.d_head,
            d_rot=config.model.d_rot,
            d_hidden=config.model.d_hidden,
            max_layers=config.model.max_layers,
            max_seq_len=config.model.max_seq_len,
            cache_stride=config.model.cache_stride,
            trim_heads=config.model.trim_heads,
            optimize=config.model.optimize,
            verbose=True,
            asset_config=config.asset_config(),
            wad_path=resolve_wad_path(config),
        )
        raise SystemExit(
            "compile completed without hitting the descent solve — "
            "did a schedule cache hit skip it?"
        )
    except _AuditDone:
        pass
    print(
        f"[audit] graph + warm start captured in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    if args.mode == "descent":
        from torchwright.compiler.forward.cpsat_scheduler import (
            solve_schedule as real_solve,
        )

        kw = dict(captured["kw"])
        if args.no_tighten:
            kw["tighten_domains"] = False
        if args.budget is not None:
            kw["time_budget_s"] = args.budget
        if args.cancel_slack is not None:
            kw["cancel_slack"] = args.cancel_slack
        kw["log_search_progress"] = True
        print(
            f"[descent] real solve: tighten_domains={kw['tighten_domains']} "
            f"budget={kw['time_budget_s']}s "
            f"cancel_slack={kw.get('cancel_slack', 2)} "
            f"workers={os.environ.get('TW_CPSAT_WORKERS', '16')}",
            flush=True,
        )
        assignment, stats = real_solve(
            captured["output_node"], captured["pos_encoding"], **kw
        )
        print(
            f"[descent] status={stats.status_name} "
            f"objective={stats.objective_value} "
            f"bound={stats.best_objective_bound} "
            f"n_layers={getattr(assignment, 'n_layers', None)} "
            f"wall={stats.wall_time_s:.1f}s",
            flush=True,
        )
        for line in stats.solver_log.splitlines():
            if line.startswith("#") or "hint" in line.lower():
                print(f"  | {line}")
        return
    _audit(
        captured["output_node"],
        captured["pos_encoding"],
        captured["kw"],
        fix_budget_s=args.fix_budget,
        core_budget_s=args.core_budget,
    )


if __name__ == "__main__":
    main()
