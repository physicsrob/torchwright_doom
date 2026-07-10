"""Generate + prove the d=8192 production CP-SAT scheduling fixture on Modal.

    # generate the fixture AND prove the fast load path (two containers):
    uv run modal run modal_cpsat_fixture.py --mode both --config configs/e1m1.yaml

    # just generate, or just re-prove an existing fixture:
    uv run modal run modal_cpsat_fixture.py --mode generate --config configs/e1m1.yaml
    uv run modal run modal_cpsat_fixture.py --mode load --fingerprint <sha256>

Why a dedicated ``modal_*.py`` (not ``make modal-run``).  ``make modal-run``'s
CPU path mounts no volume and its GPU path never commits, so neither can
durably write a fixture.  Persisting an artifact to a Modal volume is the one
sanctioned reason for a purpose-built entrypoint (torchwright_doom/CLAUDE.md);
this one imports the image + cache volume from ``modal_image.py`` (no
duplication) and mirrors ``modal_render.compile_remote``'s reload/commit dance.

**generate** runs the REAL production compile (``build_graph`` + the pre-solve
half of ``forward_compile``: lower, RMSNorm reservation, eager warm-start),
intercepts the single ``solve_schedule`` call to capture the exact scheduling
problem + geometry + frozen warm-start hint, snapshots it, and writes the
fixture (+ the built CP-SAT model proto) into the cache volume, keyed by the
production schedule-cache fingerprint.  It prints the graph-identity gate
(fingerprint + lowered critical path) so a consumer can compare to
``make compile``.

**load** proves the point: a fresh container loads the fixture from the volume
— NO graph build — rebuilds the identical CP-SAT model in seconds, and runs a
short sound solve with the frozen hint, reporting load/build/solve timings and
the depth.

Measurement tooling only: production compiles still build fresh, and nothing
here feeds a schedule back into a production compile.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

from modal_image import ASSETS_IMAGE, CACHE_VOLUME

app = modal.App("torchwright-doom-cpsat-fixture", image=ASSETS_IMAGE)

# CP-SAT's parallel search scales with cores; match TW_CPSAT_WORKERS to this.
_CPUS = 64
_MOUNT = "/root/.cache/torchwright_doom/compiled"
# Fixtures live under the compile-cache volume so future containers load them.
_FIXTURE_SUBDIR = "cpsat_fixtures"


# A BaseException (not Exception) so forward_compile's own `except Exception`
# fallback handling can never swallow the capture abort.
class _CaptureDone(BaseException):
    pass


@app.function(
    cpu=_CPUS,
    # build_graph + lower for the d=8192/h16384 flagship graph is memory-heavy
    # (no weight-writing/export here, so far below compile_remote's 256 GB).
    memory=131072,
    timeout=10800,
    volumes={_MOUNT: CACHE_VOLUME},
)
def run_fixture(
    mode: str,
    config_name: str,
    config_text: str,
    fingerprint_hint: str,
    budget: float,
    seeds: tuple = (),
) -> dict:
    import os
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    os.environ["TW_CPSAT_WORKERS"] = str(_CPUS)
    print(f"[env] TW_CPSAT_WORKERS={os.environ['TW_CPSAT_WORKERS']}", flush=True)

    CACHE_VOLUME.reload()
    if mode == "generate":
        result = _generate(config_name, config_text)
        CACHE_VOLUME.commit()
        return result
    if mode == "load":
        return _load_and_prove(fingerprint_hint, budget)
    if mode == "feascheck":
        return _feascheck(fingerprint_hint, int(seeds[0]) if seeds else 1, budget)
    raise ValueError(f"unknown mode {mode!r} (expected generate|load|feascheck)")


def _feascheck(fingerprint_hint: str, off_seed: int, budget: float) -> dict:
    """C5 soundness-at-scale: hard-fix a real rows-OFF assignment into a
    rows-ON model and confirm FEASIBLE — proves the redundant rows are slow,
    not wrong (they can't cut a schedule the cumulative admits)."""
    from ortools.sat.python import cp_model

    from torchwright.compiler.forward.cpsat_scheduler import (
        ATTN,
        MLP,
        build_model_from_snapshot,
        solve_schedule_from_snapshot,
    )
    from torchwright.compiler.forward.cpsat_snapshot import SchedulingProblem
    from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy

    fixture_dir = Path(_MOUNT) / _FIXTURE_SUBDIR
    path = (
        fixture_dir / f"{fingerprint_hint}.json"
        if fingerprint_hint
        else sorted(fixture_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)[-1]
    )
    problem = SchedulingProblem.load(
        path, expected_fingerprint=fingerprint_hint or path.stem
    )
    geom = problem.identity.geometry
    policy = SchedulingPolicy(**geom["policy"]) if geom.get("policy") else None
    hint = problem.hint
    build_kw = dict(
        d=geom["d"],
        d_head=geom["d_head"],
        d_hidden=geom["d_hidden"],
        flex_routing=geom["flex_routing"],
        cancel_slack=geom["cancel_slack"],
        policy=policy,
        reserve_residual=geom["reserve_residual"],
        reserve_heads=geom["reserve_heads"],
        max_layers=geom["max_layers"],
        tighten_domains=True,
    )

    # 1. A real rows-OFF solve → a genuine cumulative-feasible assignment.
    asg, stats = solve_schedule_from_snapshot(
        problem,
        time_budget_s=budget,
        solver_params={"random_seed": off_seed},
        hint_layers=hint.layers if hint else None,
        hint_routing=hint.routing if hint else None,
        hint_cancel=hint.cancel if hint else None,
        hint_cancel_mech=hint.cancel_mech if hint else None,
        _residual_capacity_rows=False,
        **build_kw,
    )
    if asg is None:
        raise RuntimeError("rows-OFF solve found no assignment to hard-fix")
    print(
        f"[feascheck] rows-OFF seed={off_seed} draw={asg.n_layers} "
        f"status={stats.status_name}",
        flush=True,
    )

    # 2. Hard-fix that assignment into a rows-ON model; feasibility is the test.
    built = build_model_from_snapshot(problem, _residual_capacity_rows=True, **build_kw)
    model = built.model
    for nid, L in asg.node_to_layer.items():
        if nid in built.layer_var:
            model.Add(built.layer_var[nid] == L)
    for nid, L in asg.node_to_cancel_layer.items():
        if nid in built.cancel_layer:
            model.Add(built.cancel_layer[nid] == L)
        elif nid in built.input_cancel_layer:
            model.Add(built.input_cancel_layer[nid] == L)
    for nid, r in asg.node_to_routing.items():
        if nid in built.is_attn:
            model.Add(built.is_attn[nid] == (1 if r == ATTN else 0))
    for nid, mech in asg.node_to_cancel_mech.items():
        if nid in built.cancel_in_mlp:
            model.Add(built.cancel_in_mlp[nid] == (1 if mech == MLP else 0))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0
    solver.parameters.num_search_workers = _CPUS
    status = solver.StatusName(solver.Solve(model))
    print(
        f"[feascheck] rows-ON hard-fixed to off's {asg.n_layers}-layer "
        f"assignment: status={status} (FEASIBLE/OPTIMAL == rows are sound) "
        f"literals={built.residual_capacity_literals}",
        flush=True,
    )
    return {
        "off_draw": asg.n_layers,
        "off_status": stats.status_name,
        "fixed_status": status,
        "literals": built.residual_capacity_literals,
    }


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def _generate(config_name: str, config_text: str) -> dict:
    from torchwright_doom.inference.config import apply_screen_env, load_render_config

    config_path = _write_shipped_config(config_name, config_text)
    config = load_render_config(config_path)
    apply_screen_env(config)  # screen-sized token vocab BEFORE any graph import

    import torchwright.compiler.forward.compile as cmod
    from torchwright.compiler.forward.cpsat_scheduler import (
        build_graph_model,
        build_model_from_snapshot,
        critical_path_layers,
        dump_model_proto,
    )
    from torchwright.compiler.forward.cpsat_snapshot import (
        FrozenHint,
        snapshot_from_graph_model,
    )
    from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy
    from torchwright_doom.inference.compile_cache import resolve_wad_path
    from torchwright_doom.inference.compiled_model import build_graph

    wad_path = resolve_wad_path(config, base_dir=Path(config_path).parent)
    m = config.model
    print(
        f"[gate] config={config_name} scale={m.scale} d={m.d} d_head={m.d_head} "
        f"d_hidden={m.d_hidden} bias={m.bias}",
        flush=True,
    )

    t_build0 = time.perf_counter()
    next_token, _rope, _emb, _banks = build_graph(
        d_head=m.d_head,
        max_positions=m.max_seq_len,
        d_rot=m.d_rot,
        asset_config=config.asset_config(),
        wad_path=wad_path,
    )
    t_graph = time.perf_counter() - t_build0

    # Intercept the single production solve_schedule call to capture the exact
    # lowered graph + geometry + frozen warm-start hint forward_compile feeds
    # it, then abort before the (expensive) solve — we only need the problem.
    captured: dict = {}
    real_solve = cmod.solve_schedule

    def _spy(output_node, pos_encoding=None, **kw):
        captured["output_node"] = output_node
        captured["pos_encoding"] = pos_encoding
        captured["kw"] = kw
        raise _CaptureDone()

    cmod.solve_schedule = _spy
    t_pre0 = time.perf_counter()
    try:
        cmod.forward_compile(
            output_node=next_token,
            device="cpu",
            verbose=True,  # prints the CP-SAT schedule fingerprint (identity gate)
            d=m.d,
            d_head=m.d_head,
            d_hidden=m.d_hidden,
            bias=m.bias,
            rms_norm=True,
            rms_norm_const_exp=63,
            max_layers=m.max_layers,
            optimize=2,
            _force_resolve=True,
            _solve_only=True,
        )
    except _CaptureDone:
        pass
    finally:
        cmod.solve_schedule = real_solve
    t_presolve = time.perf_counter() - t_pre0

    if "kw" not in captured:
        raise RuntimeError(
            "forward_compile never reached solve_schedule — the compile call "
            "signature or cache-bypass behavior changed; capture is stale."
        )
    kw = captured["kw"]
    lowered_out = captured["output_node"]
    for req in ("d", "d_head", "d_hidden", "flex_routing", "max_layers", "policy"):
        if req not in kw:
            raise RuntimeError(
                f"solve_schedule was called without {req!r}; the production "
                f"solve signature changed — capture logic must be updated."
            )

    cp = critical_path_layers(
        lowered_out, None, policy=kw["policy"], flex_routing=kw["flex_routing"]
    )
    print(f"[gate] critical_path_layers(lowered)={cp}", flush=True)
    print(
        f"[gate] graph_build={t_graph:.1f}s pre-solve(lower+warmstart)="
        f"{t_presolve:.1f}s total_build={t_graph + t_presolve:.1f}s",
        flush=True,
    )

    problem = snapshot_from_graph_model(build_graph_model(lowered_out, None))

    hint_layers = kw.get("hint_layers") or {}
    hint = FrozenHint(
        layers=dict(hint_layers),
        routing=dict(kw.get("hint_routing") or {}),
        cancel=dict(kw.get("hint_cancel") or {}),
        cancel_mech=dict(kw.get("hint_cancel_mech") or {}),
        n_layers=(max(hint_layers.values()) + 1 if hint_layers else 0),
    )
    solver_d_hidden = kw["d_hidden"]  # d_hidden-1 for bias=False (constant lane)
    problem = problem.with_identity(
        output_node=lowered_out,
        d=kw["d"],
        d_head=kw["d_head"],
        d_hidden=solver_d_hidden,
        flex_routing=kw["flex_routing"],
        cancel_slack=kw.get("cancel_slack", 2),
        policy=kw["policy"],
        reserve_residual=kw.get("reserve_residual", 0),
        reserve_heads=kw.get("reserve_heads", 0),
        max_layers=kw["max_layers"],
        bias=m.bias,
        # The schedule-cache fingerprint keys on the RAW d_hidden (m.d_hidden),
        # while the model solves with solver_d_hidden — match forward_compile.
        fingerprint_d_hidden=(m.d_hidden if m.d_hidden else m.d),
        critical_path_layers=cp,
    )
    problem = problem.with_hint(hint).canonicalized(lowered_out)
    fp = problem.identity.fingerprint

    fixture_dir = Path(_MOUNT) / _FIXTURE_SUBDIR
    fixture_path = fixture_dir / f"{fp}.json"
    problem.save(fixture_path)

    # Also dump the built CP-SAT model proto (C1's zero-rebuild re-solve path).
    proto_path = fixture_dir / f"{fp}.cpproto"
    built = build_model_from_snapshot(
        problem,
        d=kw["d"],
        d_head=kw["d_head"],
        d_hidden=solver_d_hidden,
        flex_routing=kw["flex_routing"],
        cancel_slack=kw.get("cancel_slack", 2),
        policy=SchedulingPolicy(**problem.identity.geometry["policy"]),
        reserve_residual=kw.get("reserve_residual", 0),
        reserve_heads=kw.get("reserve_heads", 0),
        max_layers=kw["max_layers"],
        tighten_domains=True,
        hint_layers=problem.hint.layers,
        hint_cancel=problem.hint.cancel,
    )
    dump_model_proto(built, proto_path)

    size = fixture_path.stat().st_size
    proto_size = proto_path.stat().st_size
    print(
        f"[fixture] wrote {fixture_path} ({size / 1e6:.2f} MB) + proto "
        f"({proto_size / 1e6:.2f} MB)",
        flush=True,
    )
    print(
        f"[fixture] fingerprint={fp} nodes={len(problem.nodes)} "
        f"schedulable={len(problem.schedulable_ids)} edges={len(problem.edges)} "
        f"hinted={len(hint.layers)} hint_n_layers={hint.n_layers} cp={cp}",
        flush=True,
    )
    return {
        "fingerprint": fp,
        "fixture_path": str(fixture_path),
        "fixture_bytes": size,
        "proto_bytes": proto_size,
        "n_nodes": len(problem.nodes),
        "n_schedulable": len(problem.schedulable_ids),
        "n_edges": len(problem.edges),
        "critical_path_layers": cp,
        "hint_n_layers": hint.n_layers,
        "geometry": problem.identity.geometry,
        "graph_build_s": t_graph,
        "presolve_s": t_presolve,
        "total_build_s": t_graph + t_presolve,
    }


# ---------------------------------------------------------------------------
# load + prove
# ---------------------------------------------------------------------------


def _load_and_prove(fingerprint_hint: str, budget: float) -> dict:
    from torchwright.compiler.forward.cpsat_scheduler import (
        _validate_hint,
        build_model_from_snapshot,
        solve_schedule_from_snapshot,
    )
    from torchwright.compiler.forward.cpsat_snapshot import SchedulingProblem
    from torchwright.compiler.forward.scheduling_policy import SchedulingPolicy

    fixture_dir = Path(_MOUNT) / _FIXTURE_SUBDIR
    if fingerprint_hint:
        path = fixture_dir / f"{fingerprint_hint}.json"
    else:
        fixtures = sorted(fixture_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not fixtures:
            raise FileNotFoundError(f"no fixtures under {fixture_dir}")
        path = fixtures[-1]
    print(f"[load] fixture {path}", flush=True)

    t0 = time.perf_counter()
    # Fail loudly on a stale/mismatched fixture: when the caller knows the
    # expected fingerprint (generate hands it to the proof container), the
    # loaded fixture's stored fingerprint must match or load() raises — a stale
    # fixture must never be silently measured.  With no hint we still verify the
    # stored identity matches the filename it was keyed under.
    expected = fingerprint_hint or path.stem
    problem = SchedulingProblem.load(path, expected_fingerprint=expected)
    t_load = time.perf_counter() - t0
    geom = problem.identity.geometry
    policy = SchedulingPolicy(**geom["policy"]) if geom.get("policy") else None
    print(
        f"[load] loaded in {t_load * 1000:.0f} ms — nodes={len(problem.nodes)} "
        f"schedulable={len(problem.schedulable_ids)} fingerprint="
        f"{problem.identity.fingerprint} cp={problem.identity.critical_path_layers}",
        flush=True,
    )

    hint = problem.hint
    build_kw = dict(
        d=geom["d"],
        d_head=geom["d_head"],
        d_hidden=geom["d_hidden"],
        flex_routing=geom["flex_routing"],
        cancel_slack=geom["cancel_slack"],
        policy=policy,
        reserve_residual=geom["reserve_residual"],
        reserve_heads=geom["reserve_heads"],
        max_layers=geom["max_layers"],
        tighten_domains=True,
    )

    # Rebuild the identical CP-SAT model — the step that replaces the ~10-min
    # build+lower.  Validate the frozen hint against it (zero violations proves
    # the hint is a ready incumbent, not silently dropped).
    t1 = time.perf_counter()
    built = build_model_from_snapshot(
        problem,
        hint_layers=hint.layers if hint else None,
        hint_cancel=hint.cancel if hint else None,
        **build_kw,
    )
    t_build = time.perf_counter() - t1
    violations = _validate_hint(
        built,
        hint.layers if hint else None,
        hint.routing if hint else None,
        hint.cancel if hint else None,
        hint.cancel_mech if hint else None,
        max_layers=geom["max_layers"],
    )
    print(
        f"[load] model rebuilt in {t_build:.2f}s (vs ~10-min build+lower); "
        f"hint violations={len(violations)}",
        flush=True,
    )

    # Short sound solve from the frozen hint.
    t2 = time.perf_counter()
    asg, stats = solve_schedule_from_snapshot(
        problem,
        time_budget_s=budget,
        hint_layers=hint.layers if hint else None,
        hint_routing=hint.routing if hint else None,
        hint_cancel=hint.cancel if hint else None,
        hint_cancel_mech=hint.cancel_mech if hint else None,
        log_search_progress=False,
        **build_kw,
    )
    t_solve = time.perf_counter() - t2
    n_layers = None if asg is None else asg.n_layers
    print(
        f"[load] solve {stats.status_name} in {t_solve:.1f}s: n_layers={n_layers} "
        f"bound={stats.best_objective_bound} optimal={stats.is_optimal}",
        flush=True,
    )
    print(
        f"[proof] load={t_load * 1000:.0f}ms + build={t_build:.2f}s "
        f"(seconds, vs ~10-min production build+lower); depth={n_layers}, "
        f"hint_violations={len(violations)}",
        flush=True,
    )
    return {
        "fingerprint": problem.identity.fingerprint,
        "load_ms": t_load * 1000,
        "build_s": t_build,
        "solve_s": t_solve,
        "n_layers": n_layers,
        "status": stats.status_name,
        "best_bound": stats.best_objective_bound,
        "is_optimal": stats.is_optimal,
        "hint_violations": len(violations),
        "hint_n_layers": hint.n_layers if hint else None,
    }


# ---------------------------------------------------------------------------
# helpers + local entrypoint
# ---------------------------------------------------------------------------


def _write_shipped_config(config_name: str, config_text: str) -> str:
    dest = Path("/tmp/shipped_configs") / Path(config_name).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(config_text)
    return str(dest)


@app.local_entrypoint()
def main(
    mode: str = "both",
    config: str = "configs/e1m1.yaml",
    budget: float = 180.0,
    fingerprint: str = "",
):
    config_path = Path(config)
    config_text = config_path.read_text() if config_path.exists() else ""
    config_name = config_path.name

    if mode == "feascheck":
        res = run_fixture.remote(
            mode="feascheck",
            config_name=config_name,
            config_text=config_text,
            fingerprint_hint=fingerprint,
            budget=budget,
            seeds=(1,),
        )
        print("[local] feascheck:", res, flush=True)
        return

    if mode in ("generate", "both"):
        gen = run_fixture.remote(
            mode="generate",
            config_name=config_name,
            config_text=config_text,
            fingerprint_hint="",
            budget=budget,
        )
        print("[local] generate:", gen, flush=True)
        fingerprint = gen["fingerprint"]

    if mode in ("load", "both"):
        # A FRESH container: proves the load path never builds the graph.
        proof = run_fixture.remote(
            mode="load",
            config_name=config_name,
            config_text=config_text,
            fingerprint_hint=fingerprint,
            budget=budget,
        )
        print("[local] proof:", proof, flush=True)
