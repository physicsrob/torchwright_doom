"""Measure whether a Doom config can fit stock Phi-3 inference in a VRAM budget.

By default this runs the same graph construction and CP-SAT scheduler as
bundle publication, then stops before replaying the schedule into dense
weights.  It conservatively prices every layer at the configured attention
and MLP caps, so a PASS remains safe if publication trims some layers below
those caps.  ``--full-replay`` instead materializes and discards each layer
to measure the exact replay-plan shapes.

The stock target pads every layer to the largest used attention-head count
and MLP width in the schedule.  Its fp32 generation cache is likewise uniform
across layers, so those maxima determine both checkpoint and terminal-cache
memory.  The report includes an extra one-layer cache-sized transient for the
``DynamicCache`` growth copy used by Transformers generation.

Run on the production compile container, overriding a mounted base config::

    MODAL_RUN_CPU=64 MODAL_RUN_MEMORY=262144 MODAL_RUN_TIMEOUT=10800 \
        make modal-run CPU_ONLY=1 MODULE=scripts.consumer_profile \
        ARGS="--config configs/e1m1_lowres.yaml --scale 4 --d 4096"
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

_FP32_BYTES = 4
_GIB = 1024**3


class _ShapeSink:
    """Capture the announced replay shapes and discard streamed weights."""

    def __init__(self) -> None:
        self.header: Any | None = None

    def begin(self, header: Any) -> None:
        self.header = header

    def write_layer(self, _index: int, _weights: Any) -> None:
        return

    def finalize(self, _spec: Any, _weights: Any) -> None:
        return


def _gib(value: int) -> float:
    return value / _GIB


def _dense_fp32_memory(
    *,
    n_layers: int,
    max_heads: int,
    max_hidden: int,
    d: int,
    d_head: int,
    vocab_size: int,
    cap_positions: int,
) -> dict[str, int | float]:
    """Price the globally padded stock Phi-3 weights and terminal cache."""
    attention_params_per_layer = 4 * d * max_heads * d_head
    mlp_params_per_layer = 3 * d * max_hidden
    norm_params = 2 * n_layers * d + d
    embedding_params = vocab_size * d
    total_params = (
        n_layers * (attention_params_per_layer + mlp_params_per_layer)
        + norm_params
        + embedding_params
    )
    weight_bytes = total_params * _FP32_BYTES
    kv_bytes_per_position = 2 * n_layers * max_heads * d_head * _FP32_BYTES
    cap_kv_bytes = cap_positions * kv_bytes_per_position
    cache_growth_transient_bytes = 2 * max_heads * d_head * cap_positions * _FP32_BYTES
    accounted_peak_bytes = weight_bytes + cap_kv_bytes + cache_growth_transient_bytes
    return {
        "parameters": total_params,
        "weights_bytes": weight_bytes,
        "weights_gib": _gib(weight_bytes),
        "kv_bytes_per_position": kv_bytes_per_position,
        "cap_kv_bytes": cap_kv_bytes,
        "cap_kv_gib": _gib(cap_kv_bytes),
        "cache_growth_transient_bytes": cache_growth_transient_bytes,
        "cache_growth_transient_gib": _gib(cache_growth_transient_bytes),
        "accounted_peak_bytes": accounted_peak_bytes,
        "accounted_peak_gib": _gib(accounted_peak_bytes),
    }


def _profile(
    config_path: Path,
    *,
    d: int | None = None,
    d_head: int | None = None,
    d_rot: int | None = None,
    scale: int | None = None,
    d_hidden: int | None = None,
    n_heads: int | None = None,
    max_seq_len: int | None = None,
    optimize: int | None = None,
    max_new_tokens: int | None = None,
    solver_seed: int | None = None,
    solver_workers: int | None = None,
    force_resolve: bool = False,
    export_schedule: bool = False,
    full_replay: bool = False,
) -> dict[str, Any]:
    if solver_workers is not None:
        if solver_workers < 1:
            raise ValueError("solver_workers must be positive")
        os.environ["TW_CPSAT_WORKERS"] = str(solver_workers)

    from torchwright.compiler.forward.compile import forward_compile
    from torchwright.compiler.token_model import (
        CompileHeader,
        make_layer_callback,
        schedule_provenance,
    )

    from torchwright_doom.config import (
        apply_screen_env,
        load_render_config,
        resolve_wad_path,
    )

    config = load_render_config(config_path)
    model_overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "d": d,
            "d_head": d_head,
            "d_rot": d_rot,
            "scale": scale,
            "d_hidden": d_hidden,
            "n_heads": n_heads,
            "max_seq_len": max_seq_len,
            "optimize": optimize,
        }.items()
        if value is not None
    }
    config = replace(
        config,
        model=replace(cast(Any, config.model), **model_overrides),
        run=(
            config.run
            if max_new_tokens is None
            else replace(config.run, max_new_tokens=max_new_tokens)
        ),
    )
    apply_screen_env(config)
    wad_path = resolve_wad_path(config, base_dir=config_path.parent)

    # Graph-reaching imports must happen after apply_screen_env().
    from torchwright_doom.model_graph import build_graph
    from torchwright_doom.prompt.scene import (
        load_render_scene,
        pose_from_world,
        prefill_rows_for,
    )

    next_token, _rope, embedding, _asset_banks = build_graph(
        d_head=config.model.d_head,
        max_positions=config.model.max_seq_len,
        d_rot=config.model.d_rot,
        asset_config=config.asset_config(),
        wad_path=wad_path,
    )

    head_cap = config.model.n_heads or config.model.d // config.model.d_head
    sink = _ShapeSink()
    callback = make_layer_callback(
        CompileHeader(
            config.model.d,
            config.model.d_head,
            config.model.trim_heads,
            False,
            n_heads=head_cap,
        ),
        sink,
    )
    solve_only = config.model.optimize > 0 and not full_replay
    compiled = forward_compile(
        d=config.model.d,
        d_head=config.model.d_head,
        n_heads=head_cap,
        output_node=next_token,
        max_layers=config.model.max_layers,
        verbose=False,
        on_layer_compiled=callback,
        trim_heads=config.model.trim_heads,
        optimize=config.model.optimize,
        bias=False,
        output_layout_source=embedding,
        d_hidden=config.model.d_hidden,
        rms_norm=True,
        rms_norm_eps=1e-5,
        rms_norm_const_exp=63,
        machine="swish",
        _solver_seed=solver_seed,
        _force_resolve=force_resolve,
        _solve_only=solve_only,
    )
    if solve_only:
        assignment = cast(Any, compiled).cpsat_assignment
        if assignment is None:
            stats = cast(Any, compiled).cpsat_solve_stats
            status = getattr(stats, "status_name", "UNKNOWN")
            raise RuntimeError(f"CP-SAT produced no feasible incumbent ({status})")
        n_layers = int(assignment.n_layers)
        # The stock checkpoint is globally padded to the largest per-layer
        # shape. Pricing all layers at both configured caps is conservative
        # without paying for replay merely to learn which layers trim.
        max_heads = head_cap
        max_hidden = config.model.d_hidden or config.model.d
        heads_per_layer: list[int] | None = None
        mlp_per_layer: list[int] | None = None
        shape_basis = "configured_caps_conservative"
    else:
        if sink.header is None or not sink.header.layer_shapes:
            raise RuntimeError("compiler did not announce replay-plan layer shapes")
        shapes = sink.header.layer_shapes
        n_layers = len(shapes)
        max_heads = max(shape.n_heads for shape in shapes)
        max_hidden = max(shape.d_hidden for shape in shapes)
        heads_per_layer = [shape.n_heads for shape in shapes]
        mlp_per_layer = [shape.d_hidden for shape in shapes]
        shape_basis = "exact_replay_plan"
    d = config.model.d
    d_head = config.model.d_head
    vocab_size = len(cast(Any, embedding).tokenizer.vocab)

    scene = load_render_scene(config, base_dir=wad_path.parent)
    pose = pose_from_world(scene)
    prompt_rows = prefill_rows_for(scene, pose)
    from torchwright_doom.interpret.reference import pydoom_scene_for
    from torchwright_doom.pydoom import expected_ar_tokens

    py_scene = pydoom_scene_for(scene, pose)
    expected_rollout_rows = len(expected_ar_tokens(py_scene, py_scene.test_poses[0]))
    cap_positions = len(prompt_rows) + config.run.max_new_tokens
    fp32 = _dense_fp32_memory(
        n_layers=n_layers,
        max_heads=max_heads,
        max_hidden=max_hidden,
        d=d,
        d_head=d_head,
        vocab_size=vocab_size,
        cap_positions=cap_positions,
    )

    schedule_export = None
    if solve_only:
        stats = cast(Any, compiled).cpsat_solve_stats
        schedule = {
            "optimize": config.model.optimize,
            "selected_origin": "solver",
            "delivery": "solve_only",
            "selected_objective": n_layers,
            "solver_seed": solver_seed,
            "solver_workers": solver_workers,
            "solver_status": getattr(stats, "status_name", None),
            "solver_objective": getattr(stats, "objective_value", None),
            "solver_best_bound": getattr(stats, "best_objective_bound", None),
            "solver_is_optimal": getattr(stats, "is_optimal", None),
        }
        schedule = {key: value for key, value in schedule.items() if value is not None}
        if export_schedule:
            payload = cast(Any, compiled).cpsat_assignment_payload
            fingerprint = cast(Any, compiled).schedule_fingerprint
            if payload is None or not fingerprint:
                raise RuntimeError("compiler did not export the solve-only schedule")
            solver_attempt = {
                "status_name": getattr(stats, "status_name", None),
                "objective_value": getattr(stats, "objective_value", None),
                "best_objective_bound": getattr(stats, "best_objective_bound", None),
                "total_attn_heads": getattr(stats, "total_attn_heads", None),
                "total_mlp_bypass_slots": getattr(
                    stats, "total_mlp_bypass_slots", None
                ),
                "is_optimal": getattr(stats, "is_optimal", None),
            }
            solver_attempt = {
                key: value for key, value in solver_attempt.items() if value is not None
            }
            objective_scale = int(getattr(stats, "objective_scale", 1))
            payload["meta"] = {
                "status_name": getattr(stats, "status_name", "UNKNOWN"),
                "best_objective_bound": getattr(stats, "best_objective_bound", -1),
                "is_optimal": bool(getattr(stats, "is_optimal", False)),
                "origin": "solver",
                "optimize": config.model.optimize,
                "d": config.model.d,
                "d_head": config.model.d_head,
                "d_hidden": config.model.d_hidden or config.model.d,
                "realized_objective": n_layers,
                "realized_objective_blocks": [n_layers, 0],
                "objective_scale": objective_scale,
                "costs": [1, 0, 0, 0, 0],
                "selected": {
                    "origin": "solver",
                    "is_optimal": bool(getattr(stats, "is_optimal", False)),
                    "realized_objective": n_layers,
                    "realized_objective_blocks": [n_layers, 0],
                },
                "solver_attempt": solver_attempt,
            }
            schedule_export = {
                "filename": f"{fingerprint}.json",
                "schedule_text": json.dumps(payload),
            }
    else:
        schedule = {
            key: value
            for key, value in asdict(
                schedule_provenance(compiled, config.model.optimize)
            ).items()
            if value is not None
        }
    report = {
        "config": str(config_path),
        "screen": list(config.screen),
        "model": asdict(config.model),
        "prompt_rows": len(prompt_rows),
        "expected_rollout_rows": expected_rollout_rows,
        "generation_cap_rows": config.run.max_new_tokens,
        "terminal_headroom_rows": config.run.max_new_tokens - expected_rollout_rows,
        "cap_positions": cap_positions,
        "compile": {
            "n_layers": n_layers,
            "head_cap": head_cap,
            "priced_heads": max_heads,
            "priced_mlp": max_hidden,
            "shape_basis": shape_basis,
            "heads_per_layer": heads_per_layer,
            "mlp_per_layer": mlp_per_layer,
            "schedule": schedule,
        },
        "fp32": fp32,
    }
    if schedule_export is not None:
        report["schedule_export"] = schedule_export
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile dense fp32 Doom Phi-3 weight and generation memory"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--d", type=int)
    parser.add_argument("--d-head", type=int)
    parser.add_argument("--d-rot", type=int)
    parser.add_argument("--scale", type=int)
    parser.add_argument("--d-hidden", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--optimize", type=int)
    parser.add_argument("--solver-seed", type=int)
    parser.add_argument("--solver-workers", type=int)
    parser.add_argument(
        "--force-resolve",
        action="store_true",
        help="bypass a matching schedule-cache entry and take a fresh solver draw",
    )
    parser.add_argument(
        "--full-replay",
        action="store_true",
        help="materialize/discard weights to measure exact trimmed shapes",
    )
    parser.add_argument(
        "--budget-gib",
        type=float,
        default=24.0,
        help="physical VRAM target",
    )
    parser.add_argument(
        "--runtime-reserve-gib",
        type=float,
        default=2.0,
        help=(
            "unaccounted allocator/workspace reserve after weights, terminal KV, "
            "and one full cache-growth copy (default: 2 GiB)"
        ),
    )
    args = parser.parse_args()
    if args.budget_gib <= 0:
        parser.error("--budget-gib must be positive")
    if not 0 <= args.runtime_reserve_gib < args.budget_gib:
        parser.error("--runtime-reserve-gib must be nonnegative and below the budget")
    report = _profile(
        args.config,
        d=args.d,
        d_head=args.d_head,
        d_rot=args.d_rot,
        scale=args.scale,
        d_hidden=args.d_hidden,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        optimize=args.optimize,
        max_new_tokens=args.max_new_tokens,
        solver_seed=args.solver_seed,
        solver_workers=args.solver_workers,
        force_resolve=args.force_resolve,
        full_replay=args.full_replay,
    )
    accounted_limit_gib = args.budget_gib - args.runtime_reserve_gib
    memory_passes = report["fp32"]["accounted_peak_gib"] <= accounted_limit_gib
    terminal_passes = report["terminal_headroom_rows"] >= 0
    report["acceptance"] = {
        "physical_budget_gib": args.budget_gib,
        "runtime_reserve_gib": args.runtime_reserve_gib,
        "accounted_limit_gib": accounted_limit_gib,
        "memory_passes": memory_passes,
        "terminal_passes": terminal_passes,
        "passes": memory_passes and terminal_passes,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        compile_report = report["compile"]
        fp32 = report["fp32"]
        acceptance = report["acceptance"]
        print(
            f"consumer profile: screen={report['screen'][0]}x{report['screen'][1]} "
            f"layers={compile_report['n_layers']} "
            f"heads={compile_report['priced_heads']} "
            f"mlp={compile_report['priced_mlp']}"
        )
        print(
            f"  prompt={report['prompt_rows']} "
            f"expected_new={report['expected_rollout_rows']} "
            f"cap_new={report['generation_cap_rows']} "
            f"headroom={report['terminal_headroom_rows']} "
            f"cap_positions={report['cap_positions']}"
        )
        print(
            f"  fp32 weights={fp32['weights_gib']:.2f} GiB "
            f"KV@cap={fp32['cap_kv_gib']:.2f} GiB "
            f"cache-grow={fp32['cache_growth_transient_gib']:.2f} GiB "
            f"accounted={fp32['accounted_peak_gib']:.2f} GiB"
        )
        print(
            f"  {args.budget_gib:g}-GiB gate "
            f"({args.runtime_reserve_gib:g}-GiB runtime reserve): "
            f"{'PASS' if acceptance['passes'] else 'FAIL'}"
        )
        schedule = compile_report["schedule"]
        if schedule:
            print(f"  schedule={json.dumps(schedule, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
