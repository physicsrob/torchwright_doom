"""Diagnose the CP-SAT "INFEASIBLE under width pressure" bug on doom forward().

Schedule-only (no weights, ~1.3-2 GB): replicates ``forward_compile``'s CP-SAT
path — heuristic warm-start, then ``solve_schedule`` with
``solver_max_layers = hint_n_layers + 1`` — and reports the solver STATUS, so a
silent heuristic-fallback can't pass as a solve.

Phase A (``--mode status``): build the model exactly as the compiler does and
report ``stats.status_name`` for each d.  This is the cheap OPTIMAL→INFEASIBLE
flip finder.

Phase B (``--mode localize``): feed the heuristic's known-feasible schedule into
the CP-SAT model as HARD constraints and confirm INFEASIBLE, then bisect the
constraint families (disable one at a time) to find which one rejects the valid
schedule.  Requires ``build_cpsat_model`` (the extracted model builder).

Usage:
    python -m scripts.cpsat_diagnose --mode status   --d 11200 6400 4800 3040 2560
    python -m scripts.cpsat_diagnose --mode localize --d 2560
    python -m scripts.cpsat_diagnose --mode status   --d 2560 --solver-log
"""

from __future__ import annotations

import argparse
import copy
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

from torchwright.compiler.forward.compile import _run_heuristic_warm_start
from torchwright.compiler.forward.cpsat_scheduler import (
    ATTN,
    CONSTRAINT_FAMILIES,
    build_cpsat_model,
    solve_schedule,
)
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
from torchwright.ops.inout_nodes import create_pos_encoding
from torchwright_doom.embedding import build_doom_embedding
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward


def _build_graph():
    emb = build_doom_embedding("token_ids")
    pos = create_pos_encoding()
    nt = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)
    return nt, pos


def _setup(output_node, pos, d, assume_zero_init=True):
    """Replicate forward_compile's pre-loop init (schedule-only).

    ``assume_zero_init`` defaults to True to match ``compile_to_onnx`` (the
    ONNX runtime always zero-inits the residual stream): the initially-free
    pool is marked clean so the heuristic warm-start skips BIRTH-dirty cancels.
    """
    graph = GraphAnalyzer(output_node)
    output_node = graph.get_output_node()
    input_nodes = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]
    rmap = ResidualStreamMap(d)
    rmap.allocate(pos)
    rmap.mark_clean(rmap.get_indices(pos))
    for n in input_nodes:
        if n is pos:
            continue
        rmap.allocate(n)
        rmap.mark_clean(rmap.get_indices(n))
    if assume_zero_init:
        rmap.mark_clean(set(rmap._free))
    computed = set(input_nodes)
    return graph, output_node, rmap, computed


def _warm_start(graph, output_node, pos, d, d_head, rmap, computed, max_layers):
    return _run_heuristic_warm_start(
        graph=graph,
        d=d,
        d_head=d_head,
        pos_encoding=pos,
        d_hidden=d,
        residual_map=rmap,
        computed=computed,
        clusters=None,
        admission_budget_fraction=0.4,
        policy=SchedulingPolicy(),
        overlay_pinned_inputs=set(),
        output_node=output_node,
        max_layers=max_layers,
    )


def mode_status(output_node, pos, d, d_head, time_budget_s, solver_log,
                horizon_margin=1):
    graph, output_node, rmap, computed = _setup(output_node, pos, d)
    t0 = time.perf_counter()
    hint_layers, hint_routing, hint_cancel, hint_n_layers = _warm_start(
        graph, output_node, pos, d, d_head, rmap, computed, max_layers=400
    )
    t_hint = time.perf_counter() - t0
    solver_max_layers = (
        min(400, hint_n_layers + horizon_margin) if hint_n_layers > 0 else 400
    )

    t1 = time.perf_counter()
    assignment, stats = solve_schedule(
        output_node,
        pos,
        d=d,
        d_head=d_head,
        d_hidden=d,
        time_budget_s=time_budget_s,
        max_layers=solver_max_layers,
        hint_layers=hint_layers if hint_layers else None,
        hint_routing=hint_routing if hint_routing else None,
        hint_cancel=hint_cancel if hint_cancel else None,
        log_search_progress=solver_log,
        assume_zero_init=True,
    )
    t_solve = time.perf_counter() - t1

    fell_back = assignment is None
    cpsat_layers = getattr(assignment, "n_layers", None)
    tag = ">>> INFEASIBLE/FALLBACK <<<" if fell_back else "OK"
    print(
        f"  d={d:>6}  heur={hint_n_layers:>3}L ({t_hint:.0f}s)  "
        f"solver_max_layers={solver_max_layers:>3}  "
        f"status={stats.status_name:<12} cpsat_layers={cpsat_layers}  "
        f"({t_solve:.0f}s)  {tag}"
    )
    if solver_log and stats.solver_log:
        for line in stats.solver_log.splitlines()[-25:]:
            print(f"        | {line}")
    return stats, fell_back


def _fix_schedule(built, hint_layers, hint_routing, hint_cancel):
    """Add hard equality constraints pinning the model to the heuristic
    schedule.  ``hint_cancel`` may be None to leave cancel layers free.
    Returns (n_layer_fixed, n_route_fixed, n_cancel_fixed, n_schedulable)."""
    model = built.model
    lv, ca, ia = built.layer_var, built.cancel_layer, built.is_attn
    nlf = nrf = ncf = 0
    for nid, L in hint_layers.items():
        if nid in lv:
            model.Add(lv[nid] == L)
            nlf += 1
    if hint_routing:
        for nid, route in hint_routing.items():
            if nid in ia:
                model.Add(ia[nid] == (1 if route == ATTN else 0))
                nrf += 1
    if hint_cancel:
        for nid, L in hint_cancel.items():
            if nid in ca:
                model.Add(ca[nid] == L)
                ncf += 1
    return nlf, nrf, ncf, len(lv)


def _solve_status(model, time_budget_s, log=False):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_budget_s
    solver.parameters.num_search_workers = 16
    solver.parameters.log_search_progress = log
    status = solver.Solve(model)
    return solver.StatusName(status)


def localize(output_node, pos, d, d_head, time_budget_s):
    """Phase B: pin the heuristic schedule into the model as HARD constraints,
    confirm INFEASIBLE, then disable one constraint family at a time to find
    which family rejects the known-feasible point."""
    graph, output_node, rmap, computed = _setup(output_node, pos, d)
    hint_layers, hint_routing, hint_cancel, hint_n_layers = _warm_start(
        graph, output_node, pos, d, d_head, rmap, computed, max_layers=400
    )
    solver_max_layers = min(400, hint_n_layers + 1) if hint_n_layers > 0 else 400
    print(
        f"\n=== localize d={d}: heuristic={hint_n_layers}L, "
        f"solver_max_layers={solver_max_layers} ==="
    )

    def _build(disabled=frozenset(), fix_cancel=False):
        b = build_cpsat_model(
            output_node,
            pos,
            d=d,
            d_head=d_head,
            d_hidden=d,
            max_layers=solver_max_layers,
            _disabled_families=disabled,
        )
        cov = _fix_schedule(
            b, hint_layers, hint_routing, hint_cancel if fix_cancel else None
        )
        return b, cov

    # Experiment 1: fix layer + routing, leave cancel free, all families on.
    # A restriction of the (infeasible) production model — must be INFEASIBLE.
    b1, cov = _build()
    st1 = _solve_status(b1.model, time_budget_s)
    print(
        f"  fixed layers={cov[0]}/{cov[3]}, routing={cov[1]}, cancel=free  "
        f"[all families ON]                 -> {st1}"
    )

    # Experiment 2: additionally fix cancel to the heuristic's actual frees.
    b2, cov2 = _build(fix_cancel=True)
    st2 = _solve_status(b2.model, time_budget_s)
    print(
        f"  fixed layers+routing+cancel={cov2[2]}  "
        f"[all families ON]                 -> {st2}"
    )

    # Confirmation: drop the BIRTH-dirty intervals (assume_zero_init=True) and
    # keep ONLY the attention cumulative.  If this flips INFEASIBLE->feasible,
    # the BIRTH-dirty over-count is the culprit.
    all_fams_no_attn = frozenset(CONSTRAINT_FAMILIES) - {"attn_cumulative"}
    bz = build_cpsat_model(
        output_node, pos, d=d, d_head=d_head, d_hidden=d,
        max_layers=solver_max_layers, _disabled_families=all_fams_no_attn,
        assume_zero_init=True,
    )
    _fix_schedule(bz, hint_layers, hint_routing, None)
    stz = _solve_status(bz.model, time_budget_s)
    print(
        f"  [only attn_cumulative, assume_zero_init=True (no BIRTH-dirty)] -> {stz}"
    )

    # Sanity baseline: disable ALL toggleable families.  With layers+routing
    # fixed and cancel free and no capacity/ordering family, only definitional
    # always-on constraints remain — this MUST be feasible.  If it is not, the
    # culprit is an always-on constraint (or the fixing itself conflicts).
    all_fams = frozenset(CONSTRAINT_FAMILIES)
    b0, _ = _build(disabled=all_fams)
    st0 = _solve_status(b0.model, time_budget_s)
    print(f"  [ALL families OFF, fix layer+routing, cancel free] -> {st0}")
    b0b, _ = _build(disabled=all_fams)
    # layer-only fix (no routing) with all off — isolates a routing conflict.
    b0b2 = build_cpsat_model(
        output_node, pos, d=d, d_head=d_head, d_hidden=d,
        max_layers=solver_max_layers, _disabled_families=all_fams,
    )
    _fix_schedule(b0b2, hint_layers, None, None)
    st0b = _solve_status(b0b2.model, time_budget_s)
    print(f"  [ALL families OFF, fix layer ONLY, cancel free]    -> {st0b}")

    # Dual bisect: ENABLE exactly one family (all others off).  Each family
    # that is INFEASIBLE on its own independently rejects the heuristic point.
    print("  --- dual: ENABLE ONE family (rest off), fix layer+routing, cancel free ---")
    for fam in sorted(CONSTRAINT_FAMILIES):
        disabled = all_fams - {fam}
        bb, _ = _build(disabled=disabled)
        s = _solve_status(bb.model, time_budget_s)
        flag = " <== REJECTS the heuristic schedule" if s == "INFEASIBLE" else ""
        print(f"    only {fam:<20} -> {s}{flag}")

    # Original bisect: disable exactly one family (rest on).
    print("  --- bisect: disable ONE family (rest on), fix layer+routing, cancel free ---")
    for fam in sorted(CONSTRAINT_FAMILIES):
        bb, _ = _build(disabled=frozenset({fam}))
        s = _solve_status(bb.model, time_budget_s)
        flag = " <== RESTORES FEASIBILITY" if s in ("OPTIMAL", "FEASIBLE") else ""
        print(f"    disable {fam:<20} -> {s}{flag}")


def measure(output_node, pos, d, d_head):
    """Deterministic per-layer load measurement of the model's demand functions
    evaluated on the heuristic's fixed schedule.  Quantifies the attn-cumulative
    over-count (compute + DEATH-cancel + BIRTH-dirty vs capacity) and the
    residual peak, and breaks the peak attention layer into its three terms."""
    from collections import defaultdict

    from torchwright.compiler.forward.cpsat_scheduler import (
        build_graph_model,
        heads_for,
        is_flex,
        routing,
        uses_residual,
    )
    from torchwright.compiler.forward.scheduling_policy import LEGACY_POLICY
    from torchwright.graph import Add, Attn, Concatenate

    graph, output_node, rmap, computed = _setup(output_node, pos, d)
    hint_layers, hint_routing, hint_cancel, hint_n_layers = _warm_start(
        graph, output_node, pos, d, d_head, rmap, computed, max_layers=400
    )
    gm = build_graph_model(output_node, pos)

    n_heads = d // d_head
    capacity = n_heads * d_head
    input_residual = sum(len(n) for n in gm.input_nodes)
    if gm.pos_encoding not in set(gm.input_nodes):
        input_residual += len(gm.pos_encoding)
    avail = d - input_residual
    n_layers = hint_n_layers
    layer_of = dict(hint_layers)
    pinned = gm.pinned_nodes

    def cancel_of(n):
        return hint_cancel.get(n.node_id, n_layers)

    def cancel_model(n):
        """Tightest cancel the CP-SAT model allows: max(consumer layer)+1,
        or n_layers for pinned / keep-forever (terminal-Concatenate consumer)
        nodes.  Mirrors the model's cancel lower bound — the residual-minimising
        choice the solver would make with cancel free."""
        if n in pinned:
            return n_layers
        cmax = layer_of[n.node_id] + 1
        for c in gm.consumers_eff.get(n, set()):
            if isinstance(c, Concatenate):
                return n_layers
            if c.node_id in layer_of:
                cmax = max(cmax, layer_of[c.node_id] + 1)
        return cmax

    def attn_routed(n):
        if isinstance(n, (Add, Attn)):
            return True
        if is_flex(n, gm):
            return hint_routing.get(n.node_id, routing(n, gm, LEGACY_POLICY)) == ATTN
        return routing(n, gm, LEGACY_POLICY) == ATTN

    def free_add(A):
        for E in A.inputs:
            if (
                isinstance(E, Concatenate)
                or E in pinned
                or E.node_id not in layer_of
            ):
                continue
            other = [c for c in gm.consumers_eff.get(E, set()) if c is not A]
            if any(
                isinstance(c, Concatenate) or c.node_id not in layer_of
                for c in other
            ):
                continue
            if all(layer_of[c.node_id] < layer_of[A.node_id] for c in other):
                return True
        return False

    compute = defaultdict(int)
    dirty = defaultdict(int)
    death = defaultdict(int)
    resid_events = []
    freeadd_saving = defaultdict(int)  # free-add reuse credit at the add layer
    n_freeadds = 0
    for n in gm.schedulable:
        if n.node_id not in layer_of:
            continue
        L = layer_of[n.node_id]
        h = heads_for(n, d_head)
        ru = uses_residual(n, gm) and n not in pinned
        if isinstance(n, Add):
            fr = free_add(n)
            compute[L] += (h if fr else 2 * h) * d_head
            if ru and not fr:
                dirty[L] += len(n)
            if fr:
                # Free-add reuses a dead addend's columns: the addend and the
                # Add output occupy the SAME len(A) cols, overlapping only at
                # the add layer L.  Credit that overlap back.
                freeadd_saving[L] += len(n)
                n_freeadds += 1
        else:
            if attn_routed(n) and h > 0:
                compute[L] += h * d_head
            if ru:
                dirty[L] += len(n)
        if ru:
            death[cancel_of(n)] += len(n)
        if uses_residual(n, gm):
            cm = cancel_model(n)
            resid_events.append((L, len(n)))
            resid_events.append((cm, -len(n)))

    # Per-layer attention load, three ways.
    print(
        f"\n=== measure d={d}: n_heads={n_heads}, attn capacity={capacity} cols, "
        f"residual avail={avail} cols, heuristic={n_layers}L ==="
    )
    peak_full = peak_full_L = 0
    peak_nodirty = peak_nodirty_L = 0
    for L in range(n_layers):
        full = compute[L] + dirty[L] + death[L]
        nod = compute[L] + death[L]
        if full > peak_full:
            peak_full, peak_full_L = full, L
        if nod > peak_nodirty:
            peak_nodirty, peak_nodirty_L = nod, L
    over = "OVERFLOW" if peak_full > capacity else "ok"
    over2 = "OVERFLOW" if peak_nodirty > capacity else "ok"
    print(
        f"  attn load WITH BIRTH-dirty:  peak {peak_full} cols @L{peak_full_L} "
        f"(capacity {capacity}) -> {over} by {peak_full - capacity}"
    )
    print(
        f"  attn load NO BIRTH-dirty:    peak {peak_nodirty} cols @L{peak_nodirty_L} "
        f"(capacity {capacity}) -> {over2}"
    )
    print(
        f"  peak layer L{peak_full_L} breakdown: compute={compute[peak_full_L]} "
        f"dirty={dirty[peak_full_L]} death={death[peak_full_L]} cols"
    )

    # Residual occupancy sweep.
    births_at = defaultdict(int)
    deaths_at = defaultdict(int)
    for ev, w in resid_events:
        if w > 0:
            births_at[ev] += w
        else:
            deaths_at[ev] += -w
    occ = 0
    peak_occ = peak_occ_L = 0
    peak_merged = peak_merged_L = 0
    for L in range(n_layers + 1):
        # node live at L iff birth <= L < cancel; a death at L (cancel==L)
        # is no longer live at L, so apply births and deaths before recording.
        occ += births_at[L] - deaths_at[L]
        if occ > peak_occ:
            peak_occ, peak_occ_L = occ, L
        merged = occ - freeadd_saving[L]
        if merged > peak_merged:
            peak_merged, peak_merged_L = merged, L
    rover = "OVERFLOW" if peak_occ > avail else "ok"
    mover = "OVERFLOW" if peak_merged > avail else "ok"
    print(
        f"  residual occupancy (model count):    peak {peak_occ} cols @L{peak_occ_L} "
        f"(avail {avail}) -> {rover} by {peak_occ - avail}"
    )
    print(
        f"  residual w/ free-add reuse merged:   peak {peak_merged} cols @L{peak_merged_L} "
        f"(avail {avail}) -> {mover} ({n_freeadds} free-adds)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["status", "localize", "measure"], default="status"
    )
    ap.add_argument("--d", type=int, nargs="+", default=[11200, 2560])
    ap.add_argument("--d-head", type=int, default=160)
    ap.add_argument("--time-budget", type=float, default=30.0)
    ap.add_argument("--horizon-margin", type=int, default=1,
                    help="solver_max_layers = hint_n_layers + this (default 1)")
    ap.add_argument("--solver-log", action="store_true")
    args = ap.parse_args()

    nt, pos = _build_graph()
    print(f"cpsat_diagnose mode={args.mode} d_head={args.d_head}")

    if args.mode == "status":
        for d in args.d:
            if d % args.d_head != 0:
                print(f"  d={d}  skipped (not a multiple of d_head={args.d_head})")
                continue
            mode_status(nt, pos, d, args.d_head, args.time_budget,
                        args.solver_log, args.horizon_margin)
    elif args.mode == "localize":
        for d in args.d:
            if d % args.d_head != 0:
                print(f"  d={d}  skipped (not a multiple of d_head={args.d_head})")
                continue
            localize(nt, pos, d, args.d_head, args.time_budget)
    else:
        for d in args.d:
            if d % args.d_head != 0:
                print(f"  d={d}  skipped (not a multiple of d_head={args.d_head})")
                continue
            measure(nt, pos, d, args.d_head)


if __name__ == "__main__":
    main()
