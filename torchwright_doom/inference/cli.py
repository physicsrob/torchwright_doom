"""YAML-config render CLI.

Commands:

``python -m torchwright_doom.inference compile --config job.yaml``
``python -m torchwright_doom.inference run --config job.yaml --x 1056 --y -3616 --angle 64``
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import apply_screen_env, load_render_config, resolve_run_args


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
    from .compile_cache import compile_cached

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
    out_dir: str | Path = "out/render",
    max_positions: int | None = None,
    prefill_chunk_size: int | None = None,
    progress_every: int = 250,
    png: bool = False,
    compare_images: bool = False,
    png_zoom: int = 8,
    verbose_compile: bool = False,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)
    # Render-job knobs resolve CLI flag > config ``run:`` section (whose
    # dataclass defaults cover configs without one) — see resolve_run_args.
    args = resolve_run_args(
        config,
        x=x,
        y=y,
        angle=angle,
        viewz=viewz,
        max_positions=max_positions,
        prefill_chunk_size=prefill_chunk_size,
    )
    x, y, angle, viewz = args.x, args.y, args.angle, args.viewz
    max_positions, prefill_chunk_size = args.max_positions, args.prefill_chunk_size

    from ..vocab import DONE
    from . import artifacts, compare as compare_mod
    from .compile_cache import compile_cached
    from .decode import decode_rows_to_pixels
    from .hf_runtime import load_hf_runtime
    from .tokens_bridge import row_index
    from .wad_scene import (
        load_render_scene,
        pose_from_world,
        prefill_rows_for,
        pydoom_scene_for,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_scene = load_render_scene(config, base_dir=config_path.parent)
    pose = pose_from_world(render_scene, x=x, y=y, angle=angle, viewz=viewz)
    prefill_ids = prefill_rows_for(render_scene, pose)
    terminal_row = row_index(DONE, {})

    print(
        f"[run] wad={render_scene.wad_path.name} map={config.map} "
        f"screen={config.screen[0]}x{config.screen[1]} prefill={len(prefill_ids)}",
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
    print(f"[run] loading HF runtime from {cache_dir} (device={device})", flush=True)
    compiled = load_hf_runtime(cache_dir, device=device)
    print("[run] HF runtime ready", flush=True)

    rollout = compiled.pure_ar_rollout(
        prefill_ids,
        max_positions=max_positions,
        terminal_row=terminal_row,
        progress_every=progress_every,
        prefill_chunk_size=prefill_chunk_size,
    )
    print(
        f"[run] {len(rollout.emitted_rows)} tokens in {rollout.seconds:.0f}s, "
        f"{rollout.n_forward_passes} forward passes, stopped={rollout.stopped}",
        flush=True,
    )

    emitted_rows = rollout.emitted_rows
    gen = decode_rows_to_pixels(emitted_rows, palette=render_scene.asset_book.palette)

    report = None
    pngs: list[Path] = []
    if png or compare_images:
        py_scene = pydoom_scene_for(render_scene, pose)
        py_pose = py_scene.test_poses[0]
        ref = compare_mod.reference_pixels(py_scene, py_pose)
        options = compare_mod.reference_options(py_scene, py_pose)
        report = compare_mod.compare(gen, ref, options)
        if compare_images:
            print(report.format_short(), flush=True)
        if png:
            pngs = compare_mod.write_pngs(
                gen, ref, out_dir, options=options, scale=png_zoom
            )

    pose_payload = {"x": x, "y": y, "angle": angle, "viewz": viewz}
    dump = artifacts.build_token_dump(
        fixture=f"{render_scene.wad_path.name}:{config.map}",
        pose_index=0,
        pose=pose_payload,
        prefill_rows=prefill_ids,
        emitted_rows=emitted_rows,
        mode="pure_ar",
        spec_decode_stats=None,
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
            "mode": "pure_ar",
            "n_tokens": len(emitted_rows),
            "n_forward_passes": rollout.n_forward_passes,
            "seconds": rollout.seconds,
            "stopped": rollout.stopped,
        },
        "cache_dir": str(cache_dir),
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
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
    # Run-knob defaults live in the config's ``run:`` section (run_config
    # resolves None there) — argparse must NOT restate them.
    pr.add_argument("--out-dir", default="out/render", dest="out_dir")
    pr.add_argument("--max-positions", type=int, default=None, dest="max_positions")
    pr.add_argument(
        "--prefill-chunk-size", type=int, default=None, dest="prefill_chunk_size"
    )
    pr.add_argument("--progress-every", type=int, default=250, dest="progress_every")
    pr.add_argument("--png", action="store_true")
    pr.add_argument("--compare", action="store_true", dest="compare_images")
    pr.add_argument("--png-zoom", type=int, default=8, dest="png_zoom")
    pr.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")
    pr.add_argument(
        "--device",
        default="cpu",
        help="torch device for the HF model (cpu | cuda); the full frame "
        "needs a big GPU",
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


if __name__ == "__main__":
    raise SystemExit(main())
