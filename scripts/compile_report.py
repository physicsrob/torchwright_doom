"""Compile the doom ``forward()`` and report SOLVER PROVENANCE, not just layers.

Why this exists (a lesson learned the hard way): ``compile_to_onnx(optimize>0)``
SILENTLY falls back to the heuristic ``LayerScheduler`` when CP-SAT returns
INFEASIBLE / no feasible incumbent — and returns a perfectly valid ONNX with no
signal that the optimizer never ran. So ``optimize=2`` and ``optimize=0`` can
produce identical artifacts, and a layer-count-only measurement cannot tell a
real CP-SAT solve from a heuristic fallback. (The codebase names the hazard:
``test_cpsat_chain_overlap.py`` — "INFEASIBLE in presolve, silently masked by
the heuristic-fallback".)

This tool hooks the warm-start and the solver and prints, for each compile:
the heuristic warm-start layer count, the CP-SAT ``SolveStats``
(status / is_optimal / objective-vs-best-bound gap / walltime), **whether it
fell back**, and the final compiled layer count. A fallback can no longer pass
as a solve.

Reading the report:
  * ``solver=OPTIMAL``  -> CP-SAT proved the layer-count optimum. Trust it.
  * ``solver=FEASIBLE`` -> CP-SAT found *a* schedule but ran out of budget; the
    ``gap`` (best_bound..objective) shows how far from proven-optimal.
  * ``FELL_BACK``       -> CP-SAT produced nothing usable (INFEASIBLE / no
    incumbent); FINAL is the **heuristic**, not the optimizer. ``opt2==opt0``
    here is a bug signature, not an optimality result.

Usage:
    python -m scripts.compile_report --d 6400 11200 --optimize 0 2
    python -m scripts.compile_report --d 2560 --optimize 2 --solver-log
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torchwright.compiler.forward.compile as _cmp
from torchwright.compiler.export import compile_to_onnx
from torchwright.ops.inout_nodes import create_rope_config
from torchwright_doom.model.embedding import build_doom_embedding
from torchwright_doom.model.past import GraphPast
from torchwright_doom.model.render_main import forward


def _build(d_head: int):
    emb = build_doom_embedding("token_ids")
    # rope d_head MUST match the compiled d_head (the compile entry point asserts
    # it); d_rot = d_head // 2 mirrors the production 128/64 ratio.
    rope = create_rope_config(d_head=d_head, max_positions=65536, d_rot=d_head // 2)
    nt = forward(emb, GraphPast(input_vec=emb, rope=rope))
    return nt, rope, emb


def compile_with_provenance(
    d: int,
    d_head: int,
    optimize: int,
    max_layers: int = 400,
    *,
    n_heads: int | None = None,
    d_hidden: int | None = None,
    solver_seed: int | None = None,
    rms_norm_const_exp: int | None = None,
):
    """Compile once; return a dict capturing solver provenance.

    Hooks ``_run_heuristic_warm_start`` (the heuristic baseline / warm-start hint)
    and ``solve_schedule`` (the CP-SAT ``SolveStats`` + whether it returned no
    assignment, i.e. the compiler will fall back)."""
    cap: dict = {
        "heur_layers": None,
        "stats": None,
        "fell_back": None,
        "cpsat_layers": None,
        "solve_calls": 0,
    }

    orig_warm = _cmp._build_heuristic_schedule_trace
    orig_solve = _cmp.solve_schedule

    def warm_hook(*a, **k):
        trace = orig_warm(*a, **k)
        cap["heur_layers"] = trace.n_layers
        return trace

    def solve_hook(*a, **k):
        # optimize=3 iterated descent may invoke the solver several times per
        # compile; keep the last stats, but count a fallback only if NO call
        # ever produced an assignment.
        assignment, stats = orig_solve(*a, **k)
        cap["solve_calls"] += 1
        cap["stats"] = stats
        if assignment is not None:
            cap["fell_back"] = False
            cap["cpsat_layers"] = getattr(assignment, "n_layers", None)
        elif cap["fell_back"] is None:
            cap["fell_back"] = True
        return assignment, stats

    nt, rope, emb = _build(d_head)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, f"d{d}_o{optimize}.onnx")
    _cmp._build_heuristic_schedule_trace = warm_hook
    _cmp.solve_schedule = solve_hook
    t0 = time.perf_counter()
    try:
        artifact = compile_to_onnx(
            nt,
            embedding=emb,
            output_path=path,
            d=d,
            d_head=d_head,
            max_layers=max_layers,
            optimize=optimize,
            verbose=False,
            n_heads=n_heads,
            d_hidden=d_hidden,
            _solver_seed=solver_seed,
            rms_norm_const_exp=rms_norm_const_exp,
        )
        cap["final_layers"] = artifact.n_layers
        cap["per_layer_heads"] = list(artifact.per_layer_n_heads)
        cap["error"] = None
    except Exception as e:  # noqa: BLE001 — surface, don't swallow
        cap["final_layers"] = None
        cap["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"
    finally:
        cap["wall"] = time.perf_counter() - t0
        _cmp._build_heuristic_schedule_trace = orig_warm
        _cmp.solve_schedule = orig_solve
        # A d=8192 artifact is ~90 GB; a multi-seed sweep would fill the
        # container's ephemeral disk if these accumulated.
        shutil.rmtree(tmp, ignore_errors=True)
    return cap


def _fmt_row(d: int, opt: int, cap: dict) -> str:
    if cap["error"]:
        return f"  {d:>6} {opt:>3}   FAILED: {cap['error']}  ({cap['wall']:.0f}s)"
    heur = cap["heur_layers"]
    final = cap["final_layers"]
    if opt == 0 or cap["stats"] is None:
        prov = "heuristic (no solver invoked)"
    else:
        st = cap["stats"]
        status = getattr(st, "status_name", "?")
        is_opt = getattr(st, "is_optimal", None)
        obj = getattr(st, "objective_value", None)
        bound = getattr(st, "best_objective_bound", None)
        if cap["fell_back"]:
            prov = (
                f"solver={status}  >>> FELL BACK TO HEURISTIC <<< "
                f"(FINAL is heuristic, not CP-SAT)"
            )
        else:
            gap = ""
            if obj not in (None, -1) and bound is not None:
                gap = f" gap=[{bound:.0f}..{obj}]"
            tag = "OPTIMAL✓" if is_opt else "feasible(not proven optimal)"
            prov = f"solver={status} {tag} cpsat_layers={cap['cpsat_layers']}{gap}"
    heur_s = f"{heur}" if heur is not None else "-"
    final_s = f"{final}" if final is not None else "-"
    return f"  {d:>6} {opt:>3}   heur={heur_s:>4}  FINAL={final_s:>4}  ({cap['wall']:>4.0f}s)  {prov}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[6400])
    ap.add_argument("--optimize", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--d-head", type=int, default=160)
    ap.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="decouple the per-layer head capacity from d // d_head",
    )
    ap.add_argument(
        "--d-hidden",
        type=int,
        default=None,
        help="MLP hidden width (production e1m1 passes 16384; omitting it "
        "measures a different machine — the finding-14a sweep artifact)",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="CP-SAT solver seeds; each seed runs one full compile per (d, opt)",
    )
    ap.add_argument(
        "--rms-const-exp",
        type=int,
        default=None,
        help="pinned RMS constant exponent (production e1m1 passes 63)",
    )
    ap.add_argument(
        "--solver-log",
        action="store_true",
        help="dump the CP-SAT solver log tail for each opt>0 run",
    )
    args = ap.parse_args()

    from torchwright_doom.model.constants import SCREEN_HEIGHT, SCREEN_WIDTH

    seeds = list(args.seeds) if args.seeds else [None]
    print(
        f"compile provenance report (d_head={args.d_head} n_heads={args.n_heads} "
        f"d_hidden={args.d_hidden} screen={SCREEN_WIDTH}x{SCREEN_HEIGHT} "
        f"rms_const_exp={args.rms_const_exp})",
        flush=True,
    )
    print(f"  {'d':>6} {'opt':>3}   layers + SOLVER PROVENANCE", flush=True)
    for d in args.d:
        if d % args.d_head != 0:
            print(
                f"  {d:>6}  -- skipped: d must be a multiple of d_head"
                f"={args.d_head} for the real compile"
            )
            continue
        for opt in args.optimize:
            for seed in seeds:
                cap = compile_with_provenance(
                    d,
                    args.d_head,
                    opt,
                    n_heads=args.n_heads,
                    d_hidden=args.d_hidden,
                    solver_seed=seed,
                    rms_norm_const_exp=args.rms_const_exp,
                )
                tag = f"  seed={seed}" if seed is not None else ""
                print(_fmt_row(d, opt, cap) + tag, flush=True)
                heads = cap.get("per_layer_heads")
                if heads:
                    head_cap = args.n_heads or (d // args.d_head)
                    at_cap = sum(1 for h in heads if h >= head_cap)
                    print(
                        f"        heads/layer: max={max(heads)} "
                        f"at_cap={at_cap}/{len(heads)} total={sum(heads)}",
                        flush=True,
                    )
                if args.solver_log and cap.get("stats") is not None:
                    log = getattr(cap["stats"], "solver_log", "") or ""
                    for line in log.splitlines()[-8:]:
                        print(f"        | {line}")


if __name__ == "__main__":
    main()
