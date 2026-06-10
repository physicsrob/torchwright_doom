"""YAML-config render CLI.

Primary commands:

``python -m torchwright_doom.render compile --config job.yaml``
``python -m torchwright_doom.render run --config job.yaml --x 1056 --y -3616 --angle 64``

The old fixture CLI is still available by calling this module with no subcommand.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import apply_screen_env, load_render_config


def compile_config(
    *,
    config_path: str | Path,
    verbose_compile: bool = False,
    cache_dir: str | Path | None = None,
    compile_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)
    from .cache import compile_cached

    cache_dir = compile_cached(
        config,
        base_dir=config_path.parent,
        verbose=verbose_compile,
        cache_dir=cache_dir,
        compile_payload=compile_payload,
    )
    return {"cache_dir": str(cache_dir)}


def run_config(
    *,
    config_path: str | Path,
    x: float | None = None,
    y: float | None = None,
    angle: int | None = None,
    viewz: float | None = None,
    mode: str = "spec_decode",
    out_dir: str | Path = "out/render",
    max_positions: int = 10240,
    draft_window: int = 0,
    prefill_chunk_size: int = 1024,
    progress_every: int = 250,
    png: bool = False,
    compare_images: bool = False,
    png_zoom: int = 8,
    verbose_compile: bool = False,
    cache_dir: str | Path | None = None,
    profile: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)

    from ..vocab import DONE
    from . import artifacts, compare as compare_mod
    from .cache import compile_cached, load_cached_runtime
    from .decode import decode_rows_to_pixels
    from .inference import pure_ar_rollout
    from .tokens_bridge import row_index
    from .wad_scene import (
        load_render_scene,
        pose_from_world,
        prefill_rows_for,
        sandbox_scene_for,
    )

    if mode not in ("spec_decode", "pure_ar", "both"):
        raise ValueError(
            f"unknown mode {mode!r} (expected spec_decode | pure_ar | both)"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_scene = load_render_scene(config, base_dir=config_path.parent)
    pose = pose_from_world(render_scene, x=x, y=y, angle=angle, viewz=viewz)
    prefill_ids = prefill_rows_for(render_scene, pose)
    terminal_row = row_index(DONE, {})

    print(
        f"[run] wad={render_scene.wad_path.name} map={config.map} "
        f"screen={config.screen[0]}x{config.screen[1]} prefill={len(prefill_ids)} "
        f"mode={mode}",
        flush=True,
    )
    if cache_dir is not None:
        # Precompiled path (e.g. Modal renders a locally-compiled ONNX that was
        # uploaded to the mounted cache volume): skip compilation entirely and
        # render the handed-over directory directly.
        cache_dir = Path(cache_dir)
        print(f"[run] using precompiled ONNX from {cache_dir}", flush=True)
    else:
        cache_dir = compile_cached(
            config,
            base_dir=config_path.parent,
            verbose=verbose_compile,
        )
    print(f"[run] loading ONNX runtime from {cache_dir}", flush=True)
    compiled = load_cached_runtime(
        cache_dir, enable_profiling=profile, profile_dir=out_dir if profile else None
    )
    print("[run] ONNX runtime ready", flush=True)

    pure = None
    spec = None
    spec_stats: dict[str, Any] | None = None
    if mode in ("pure_ar", "both"):
        pure = pure_ar_rollout(
            compiled,
            prefill_ids,
            max_positions=max_positions,
            terminal_row=terminal_row,
            progress_every=progress_every,
            prefill_chunk_size=prefill_chunk_size,
        )
        print(
            f"[run] pure-AR: {len(pure.emitted_rows)} tokens in {pure.seconds:.0f}s, "
            f"{pure.n_forward_passes} forward passes, stopped={pure.stopped}",
            flush=True,
        )

    sb_scene = None
    sb_pose = None
    if mode in ("spec_decode", "both") or png or compare_images:
        sb_scene = sandbox_scene_for(render_scene, pose)
        sb_pose = sb_scene.test_poses[0]

    if mode in ("spec_decode", "both"):
        from doom_sandbox.implementation.reference_drafter import ARDrafter

        from .inference import spec_decode_rollout

        spec, spec_stats = spec_decode_rollout(
            compiled,
            prefill_ids,
            ARDrafter(sb_scene, sb_pose),
            max_positions=max_positions,
            terminal_row=terminal_row,
            draft_window=draft_window,
            progress_every=progress_every,
            prefill_chunk_size=prefill_chunk_size,
        )
        attempted = spec_stats.get("attempted_drafts", 0)
        accepted = spec_stats.get("accepted_drafts", 0)
        n_tokens = len(spec.emitted_rows)
        # Per-token acceptance is the canonical metric: the fraction of emitted
        # tokens that came from an accepted draft (accepted_drafts / tokens).
        # Do NOT headline accepted/attempted — on a mispredict the reuse buffer
        # re-offers the window's tail as fresh drafts, so attempted_drafts
        # inflates well past the token count at low acceptance (e.g. 18,647
        # attempts for 8,000 tokens), deflating accepted/attempted into a
        # misleading number. Keep accepted/attempted as a secondary
        # draft-efficiency stat.
        per_token_accept = accepted / n_tokens if n_tokens else 0.0
        draft_efficiency = accepted / attempted if attempted else 0.0
        print(
            f"[run] spec-decode: {n_tokens} tokens in {spec.seconds:.0f}s, "
            f"{spec.n_forward_passes} forward passes "
            f"({n_tokens / max(1, spec.n_forward_passes):.2f} tok/pass), "
            f"accept rate {per_token_accept:.1%} per-token ({accepted}/{n_tokens}); "
            f"draft efficiency {draft_efficiency:.1%} ({accepted}/{attempted}), "
            f"stopped={spec.stopped}",
            flush=True,
        )

    if mode == "both":
        assert spec is not None and pure is not None
        assert spec.emitted_rows == pure.emitted_rows, (
            "spec-decode is not bit-identical to pure-AR (first diff at "
            f"{_first_diff(spec.emitted_rows, pure.emitted_rows)})"
        )
        print(
            f"[run] spec-decode used {spec.n_forward_passes} forward passes vs "
            f"{pure.n_forward_passes} pure-AR "
            f"({pure.n_forward_passes / max(1, spec.n_forward_passes):.2f}x fewer); "
            "bit-identical OK",
            flush=True,
        )

    if profile:
        # All step() calls are done; flush ORT's trace and write a parsed
        # summary next to it so both sync back from Modal (out_dir is mirrored).
        from .profile_analysis import summarize_profile

        profile_json = compiled.end_profiling()
        if profile_json:
            try:
                summary = summarize_profile(profile_json)
            except Exception as exc:  # never let analysis crash the render
                summary = f"[profile] analysis failed: {exc!r}"
            print("\n" + summary, flush=True)
            (out_dir / "profile_summary.txt").write_text(summary)
            print(
                f"[run] wrote profile trace {Path(profile_json).name} + "
                f"profile_summary.txt to {out_dir}",
                flush=True,
            )

    rollout = pure if mode == "pure_ar" else spec
    assert rollout is not None
    emitted_rows = rollout.emitted_rows
    gen = decode_rows_to_pixels(emitted_rows, palette=render_scene.asset_book.palette)

    report = None
    pngs: list[Path] = []
    if png or compare_images:
        ref = compare_mod.reference_pixels(sb_scene, sb_pose)
        options = compare_mod.reference_options(sb_scene, sb_pose)
        report = compare_mod.compare(gen, ref, options)
        if compare_images:
            print(report.format_short(), flush=True)
        if png:
            pngs = compare_mod.write_pngs(
                gen, ref, out_dir, options=options, scale=png_zoom
            )

    world_x = float(x if x is not None else 1056.0)
    world_y = float(y if y is not None else -3616.0)
    pose_payload = {
        "x": world_x,
        "y": world_y,
        "angle": int(angle if angle is not None else 64),
        "viewz": float(viewz if viewz is not None else 41.0),
    }
    dump = artifacts.build_token_dump(
        fixture=f"{render_scene.wad_path.name}:{config.map}",
        pose_index=0,
        pose=pose_payload,
        prefill_rows=prefill_ids,
        emitted_rows=emitted_rows,
        mode=mode,
        spec_decode_stats=spec_stats,
        label=f"{config.map.lower()}__{config.screen[0]}x{config.screen[1]}",
        config={
            "path": str(config_path),
            "cache_dir": str(cache_dir),
            "wall_textures": list(config.textures.wall),
            "flat_textures": list(config.textures.flat),
        },
    )
    dump_path = artifacts.write_token_dump(out_dir / "token_dump.json", dump)
    print(
        f"[run] wrote token_dump.json"
        f"{' + ' + ', '.join(p.name for p in pngs) if pngs else ''} to {out_dir}",
        flush=True,
    )
    return {
        "token_dump": str(dump_path),
        "pngs": [str(p) for p in pngs],
        "report": asdict(report) if report is not None else None,
        "report_text": report.format_short() if report is not None else None,
        "rollout": {
            "mode": mode,
            "n_tokens": len(emitted_rows),
            "n_forward_passes": rollout.n_forward_passes,
            "seconds": rollout.seconds,
            "stopped": rollout.stopped,
        },
        "spec_decode_stats": spec_stats,
        "cache_dir": str(cache_dir),
        "out_dir": str(out_dir),
    }


def run_render(
    *,
    fixture: str = "e1m1_subset_textured",
    pose_index: int = 0,
    mode: str = "spec_decode",
    out_dir: str | Path = "out/render",
    max_positions: int = 10240,
    d: int = 4096,
    d_head: int = 32,
    scale: int = 8,
    draft_window: int = 0,
    prefill_chunk_size: int = 1024,
    progress_every: int = 250,
    verbose_compile: bool = False,
) -> dict[str, Any]:
    """Legacy fixture-driven Plan-K render path."""
    import torch

    from ..constants import SCREEN_HEIGHT, SCREEN_WIDTH
    from ..vocab import DONE
    from . import artifacts, compare
    from .compiled_model import build_compiled
    from .decode import decode_rows_to_pixels
    from .inference import pure_ar_rollout
    from .tokens_bridge import row_index, sandbox_token_to_row
    from .wad_scene import _ensure_doom_sandbox

    _ensure_doom_sandbox()
    from doom_sandbox import fixtures
    from doom_sandbox.implementation import prefill as sb_prefill
    from doom_sandbox.implementation import reference as sb_ref

    if (sb_ref.SCREEN_WIDTH, sb_ref.SCREEN_HEIGHT) != (SCREEN_WIDTH, SCREEN_HEIGHT):
        raise RuntimeError(
            f"screen-dim mismatch: torchwright_doom {SCREEN_WIDTH}x{SCREEN_HEIGHT} "
            f"vs doom_sandbox {sb_ref.SCREEN_WIDTH}x{sb_ref.SCREEN_HEIGHT}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = fixtures.load_fixture(fixture)
    pose = scene.test_poses[pose_index]
    prefill_tokens = list(sb_prefill.get_prefill(scene, pose))
    prefill_ids = [sandbox_token_to_row(t) for t in prefill_tokens]
    terminal_row = row_index(DONE, {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[legacy] fixture={fixture} pose={pose_index} prefill={len(prefill_ids)} "
        f"device={device} d={d} d_head={d_head}",
        flush=True,
    )
    t_compile = time.time()
    compiled, _output_node = build_compiled(
        device, d=d, d_head=d_head, verbose=verbose_compile
    )
    print(f"[legacy] compiled in {time.time() - t_compile:.0f}s", flush=True)

    pure = None
    spec = None
    spec_stats = None
    if mode in ("pure_ar", "both"):
        pure = pure_ar_rollout(
            compiled,
            prefill_ids,
            max_positions=max_positions,
            terminal_row=terminal_row,
            progress_every=progress_every,
            prefill_chunk_size=prefill_chunk_size,
        )
    if mode in ("spec_decode", "both"):
        from .inference import spec_decode_rollout

        spec, spec_stats = spec_decode_rollout(
            compiled,
            prefill_ids,
            _make_drafter(scene, pose),
            max_positions=max_positions,
            terminal_row=terminal_row,
            draft_window=draft_window,
            progress_every=progress_every,
            prefill_chunk_size=prefill_chunk_size,
        )
    if mode == "both":
        assert spec.emitted_rows == pure.emitted_rows
    rollout = pure if mode == "pure_ar" else spec
    emitted_rows = rollout.emitted_rows
    gen = decode_rows_to_pixels(emitted_rows)
    ref = compare.reference_pixels(scene, pose)
    options = compare.reference_options(scene, pose)
    report = compare.compare(gen, ref, options)
    print(report.format_short(), flush=True)
    pngs = compare.write_pngs(gen, ref, out_dir, options=options, scale=scale)
    dump = artifacts.build_token_dump(
        fixture=fixture,
        pose_index=pose_index,
        pose={
            "x": float(pose.x),
            "y": float(pose.y),
            "angle": int(pose.angle),
            "viewz": float(pose.viewz),
        },
        prefill_rows=prefill_ids,
        emitted_rows=emitted_rows,
        mode=mode,
        spec_decode_stats=spec_stats,
    )
    dump_path = artifacts.write_token_dump(out_dir / "token_dump.json", dump)
    return {
        "report": asdict(report),
        "report_text": report.format_short(),
        "pngs": [str(p) for p in pngs],
        "token_dump": str(dump_path),
        "rollout": {
            "mode": mode,
            "n_tokens": len(emitted_rows),
            "n_forward_passes": rollout.n_forward_passes,
            "seconds": rollout.seconds,
            "stopped": rollout.stopped,
        },
        "spec_decode_stats": spec_stats,
        "out_dir": str(out_dir),
    }


def _make_drafter(scene, pose):
    from doom_sandbox.implementation.reference_drafter import ARDrafter

    return ARDrafter(scene, pose)


def _first_diff(a: list[int], b: list[int]) -> int:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("compile", "run", "-h", "--help"):
        return _legacy_main(argv)

    p = argparse.ArgumentParser(description="Compile and run YAML DOOM render jobs")
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("compile", help="compile config to the ONNX cache")
    pc.add_argument("--config", required=True, dest="config_path")
    pc.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")

    pr = sub.add_parser("run", help="render one pose from a YAML config")
    pr.add_argument("--config", required=True, dest="config_path")
    pr.add_argument("--x", type=float)
    pr.add_argument("--y", type=float)
    pr.add_argument("--angle", type=int)
    pr.add_argument("--viewz", type=float)
    pr.add_argument(
        "--mode", default="spec_decode", choices=["spec_decode", "pure_ar", "both"]
    )
    pr.add_argument("--out-dir", default="out/render", dest="out_dir")
    pr.add_argument("--max-positions", type=int, default=10240, dest="max_positions")
    pr.add_argument("--draft-window", type=int, default=0, dest="draft_window")
    pr.add_argument(
        "--prefill-chunk-size", type=int, default=1024, dest="prefill_chunk_size"
    )
    pr.add_argument("--progress-every", type=int, default=250, dest="progress_every")
    pr.add_argument("--png", action="store_true")
    pr.add_argument("--compare", action="store_true", dest="compare_images")
    pr.add_argument("--png-zoom", type=int, default=8, dest="png_zoom")
    pr.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")
    pr.add_argument(
        "--profile",
        action="store_true",
        help="enable ORT per-node profiling + INFO logging; write trace + "
        "summary to out_dir (phase-1 CUDA-graph measurement)",
    )

    args = p.parse_args(argv)
    command = args.command
    values = vars(args)
    values.pop("command", None)
    if command == "compile":
        compile_config(**values)
    else:
        run_config(**values)
    return 0


def _legacy_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Legacy Plan-K fixture render CLI")
    p.add_argument("--fixture", default="e1m1_subset_textured")
    p.add_argument("--pose", type=int, default=0, dest="pose_index")
    p.add_argument(
        "--mode",
        default="spec_decode",
        choices=["spec_decode", "pure_ar", "both"],
        help="spec_decode (default) | pure_ar | both",
    )
    p.add_argument("--out-dir", default="out/render", dest="out_dir")
    p.add_argument("--max-positions", type=int, default=10240, dest="max_positions")
    p.add_argument("--d", type=int, default=4096)
    p.add_argument("--d-head", type=int, default=32, dest="d_head")
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--draft-window", type=int, default=0, dest="draft_window")
    p.add_argument(
        "--prefill-chunk-size", type=int, default=1024, dest="prefill_chunk_size"
    )
    p.add_argument("--progress-every", type=int, default=250, dest="progress_every")
    p.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")
    args = p.parse_args(argv)
    run_render(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
