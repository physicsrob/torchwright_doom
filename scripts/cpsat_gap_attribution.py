"""DOOM CP-SAT gap-attribution sweep (expressiveness plan §0, Unit 0).

Measures, on the PRODUCTION DOOM graph, how much each CP-SAT constraint
family costs in compiled depth — the DOOM half of the G0 attribution table.

Runs THROUGH THE REAL COMPILE PATH (plan decision #2): the production graph
builder ``build_graph`` (with ``apply_screen_env`` first, so the token vocab
is screen-sized correctly — NOT the 60x50 hud-off default) plus
``forward_compile`` in solve-only mode.  It is NOT a standalone replica of
the solve flow — the graph and the solve are the production building blocks.
The graph-identity gate (critical path + the fingerprint ``forward_compile``
prints) must match a plain ``make compile`` on the same torchwright commit.

Each cell is one (family, seed): a fresh solve with that constraint family
disabled (or the sound model), read solve-only (no replay / no 28 GB
weight-writing / no export).  ``sound - relaxed`` attributes the family's
bindingness.  Both the incumbent draw AND the proven bound are printed per
cell (plan §0 honest-ledger rule).

Emits parseable ``[cell] ...`` lines to stdout (``make modal-run`` returns
stdout only — nothing written to the container disk comes back).

Run on a 64-CPU Modal container, one family (x all seeds) per invocation so
several run concurrently:

    MODAL_RUN_CPU=64 MODAL_RUN_MEMORY=65536 MODAL_RUN_TIMEOUT=7200 \
        make modal-run MODULE=scripts.cpsat_gap_attribution CPU_ONLY=1 \
        ARGS="--config configs/e1m1.yaml --sound --seeds 0,1,2,3,4"

    ... ARGS="--config configs/e1m1.yaml --families residual_cumulative --seeds 0,1,2,3,4"

``TW_CPSAT_WORKERS`` does NOT survive ``make modal-run``'s env allowlist, so
it is set here (before the solver is imported/used) from ``--workers`` and
printed back so the container's worker count is verifiable.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a render config yaml")
    ap.add_argument(
        "--families",
        default="",
        help="comma-separated constraint families to disable (each alone); "
        "empty means none",
    )
    ap.add_argument("--seeds", default="0", help="comma-separated solver seeds")
    ap.add_argument(
        "--sound",
        action="store_true",
        help="also measure the sound model (no family disabled) solve-only",
    )
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--optimize", type=int, default=2, help="production is 2")
    args = ap.parse_args()

    # Must be set BEFORE the solver runs; the modal-run env allowlist drops it.
    os.environ["TW_CPSAT_WORKERS"] = str(args.workers)
    print(f"[env] TW_CPSAT_WORKERS={os.environ['TW_CPSAT_WORKERS']}", flush=True)

    from torchwright_doom.inference.config import apply_screen_env, load_render_config

    config_path = Path(args.config)
    config = load_render_config(config_path)
    # BEFORE any graph module import: the token vocab is screen-sized at import.
    apply_screen_env(config)

    from torchwright.compiler.forward.compile import forward_compile
    from torchwright.compiler.forward.cpsat_scheduler import critical_path_layers
    from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
    from torchwright.compiler.lower import lower
    from torchwright_doom.inference.compile_cache import resolve_wad_path
    from torchwright_doom.inference.compiled_model import build_graph

    wad_path = resolve_wad_path(config, base_dir=config_path.parent)
    m = config.model
    print(
        f"[gate] config={args.config} scale={m.scale} detail={m.detail} "
        f"hud={m.hud} d={m.d} d_head={m.d_head} d_hidden={m.d_hidden} "
        f"bias={m.bias} optimize={args.optimize}",
        flush=True,
    )

    next_token, _rope, _emb, _banks = build_graph(
        d_head=m.d_head,
        max_positions=m.max_seq_len,
        d_rot=m.d_rot,
        asset_config=config.asset_config(),
        wad_path=wad_path,
    )

    # Graph-identity gate: critical path is topology-only; the fingerprint is
    # printed by forward_compile (verbose) on the first cell.  Both must match
    # a plain `make compile` of this config on the same torchwright commit.
    # forward_compile computes the critical path on the LOWERED graph (after
    # the collapse passes cut depth), so lower here with the identical kwargs
    # (compile.py: collapse_univariate/collapse_pl on, lane_cap=d_hidden//4)
    # before measuring — the raw-graph critical path is several layers higher
    # and would not match production's print.
    cp_raw = critical_path_layers(
        next_token, None, policy=SchedulingPolicy(), flex_routing=True
    )
    lowered = lower(
        next_token,
        verbose=False,
        collapse_univariate=True,
        collapse_pl=True,
        collapse_lane_cap=(m.d_hidden if m.d_hidden else m.d) // 4,
    )
    cp = critical_path_layers(
        lowered.output_node, None, policy=SchedulingPolicy(), flex_routing=True
    )
    print(f"[gate] critical_path_layers(lowered)={cp} (raw={cp_raw})", flush=True)

    # Mirror compile_cache -> compile_to_onnx geometry.  rms_norm is on for the
    # production power-of-two d (compile_to_onnx's rms_norm=None default);
    # rms_norm_const_exp=63 is compile_to_onnx_path's default.  The rms
    # reservation shrinks the residual budget, so it must be present or the
    # residual cumulative (the resource we are attributing) is mis-sized.
    geom = dict(
        d=m.d,
        d_head=m.d_head,
        d_hidden=m.d_hidden,
        bias=m.bias,
        rms_norm=True,
        rms_norm_const_exp=63,
        max_layers=m.max_layers,
        optimize=args.optimize,
    )

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cells = []
    if args.sound:
        cells.append(("sound", frozenset()))
    for fam in families:
        cells.append((fam, frozenset({fam})))

    for label, disabled in cells:
        for i, seed in enumerate(seeds):
            t0 = time.perf_counter()
            kw = dict(_solver_seed=seed, _force_resolve=True)
            if disabled:
                kw["_disabled_families"] = disabled
            else:
                kw["_solve_only"] = True
            try:
                net = forward_compile(
                    output_node=next_token,
                    device="cpu",
                    # Print the fingerprint (identity gate) once per run.
                    verbose=(label == cells[0][0] and i == 0),
                    **geom,
                    **kw,
                )
                asg = net.cpsat_assignment
                st = net.cpsat_solve_stats
                n = None if asg is None else asg.n_layers
                print(
                    f"[cell] config={args.config} family={label} seed={seed} "
                    f"n_layers={n} bound={st.best_objective_bound} "
                    f"optimal={st.is_optimal} status={st.status_name} "
                    f"wall_s={time.perf_counter() - t0:.1f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - one bad cell shouldn't kill the sweep
                print(
                    f"[cell] config={args.config} family={label} seed={seed} "
                    f"ERROR {type(exc).__name__}: {str(exc)[:300]} "
                    f"wall_s={time.perf_counter() - t0:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
