"""YAML-config render CLI.

Commands:

``python -m torchwright_doom.inference compile --config job.yaml``
``python -m torchwright_doom.inference run --config job.yaml --x 1056 --y -3616 --angle 64``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import (
    apply_screen_env,
    canonical_compile_payload,
    hf_bundle_cache_dir,
    load_render_config,
    resolve_run_args,
    resolve_wad_path,
)


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
    from .hf_bundle import compile_phi3_bundle, is_complete_hf_bundle

    wad_path = resolve_wad_path(config, base_dir=config_path.parent)
    payload = compile_payload or canonical_compile_payload(config, wad_path)
    destination = (
        Path(cache_dir)
        if cache_dir is not None
        else hf_bundle_cache_dir(config, wad_path)
    )
    if is_complete_hf_bundle(destination, expected_payload=payload):
        print(f"[compile] direct-HF cache hit {destination}", flush=True)
        return {"cache_dir": str(destination), "cache_hit": True}
    print(f"[compile] direct-HF cache miss {destination}", flush=True)
    report = compile_phi3_bundle(
        config,
        wad_path=wad_path,
        destination=destination,
        compile_payload=payload,
        verbose=verbose_compile,
    )
    provenance = report.manifest["schedule"]
    print(
        f"[compile] {report.n_layers} layers "
        f"(selected={provenance.get('selected_origin')}, "
        f"delivery={provenance.get('delivery')}, "
        f"objective={provenance.get('selected_objective')}) -> {destination}",
        flush=True,
    )
    return {**report.to_dict(), "cache_dir": str(destination), "cache_hit": False}


def compile_onnx_debug_config(
    *, config_path: str | Path, verbose_compile: bool = False
) -> dict[str, Any]:
    """Explicit diagnostic backend; its output is never accepted by rendering."""
    config_path = Path(config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)
    from .compile_cache import compile_onnx_debug_cached

    cache_dir = compile_onnx_debug_cached(
        config, base_dir=config_path.parent, verbose=verbose_compile
    )
    return {"cache_dir": str(cache_dir), "artifact_kind": "onnx_debug"}


def run_config(
    *,
    config_path: str | Path,
    x: float | None = None,
    y: float | None = None,
    angle: int | None = None,
    viewz: float | None = None,
    out_dir: str | Path = "out/render",
    max_new_tokens: int | None = None,
    png: bool = False,
    compare_images: bool = False,
    png_zoom: int = 8,
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
        max_new_tokens=max_new_tokens,
    )
    x, y, angle, viewz = args.x, args.y, args.angle, args.viewz
    max_new_tokens = args.max_new_tokens

    from ..tokenizer.codec import raw_text_from_rows, rows_from_raw_text
    from ..vocab import DONE
    from . import artifacts, compare as compare_mod
    from .decode import decode_rows_to_pixels
    from .hf_bundle import validate_bundle_manifest
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
    if cache_dir is None:
        wad_path = resolve_wad_path(config, base_dir=config_path.parent)
        cache_dir = hf_bundle_cache_dir(config, wad_path)
    else:
        cache_dir = Path(cache_dir)
    bundle_manifest = validate_bundle_manifest(cache_dir)
    expected_screen = {
        "width": config.screen[0],
        "height": config.screen[1],
        "scale": config.model.scale,
        "detail": config.model.detail,
        "hud": config.model.hud,
    }
    if bundle_manifest.get("screen") != expected_screen or bundle_manifest.get(
        "model"
    ) != asdict(config.model):
        raise ValueError("validated bundle does not match the requested render config")

    # Prompt construction is Doom-specific and happens before inference.  The
    # bundle's frozen canonical words are authoritative, and the exact text is
    # retained as a run artifact.
    vocab = json.loads((cache_dir / "doom_vocab.json").read_text(encoding="utf-8"))
    words = list(vocab["words"])
    prompt_text = raw_text_from_rows(words, prefill_ids) + "\n"
    prompt_path = out_dir / "prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    # This is deliberately the exact portable program shipped at the bundle
    # root, executed as a process boundary.  Production therefore has no
    # private model loader, cache implementation, or generation loop.
    infer_script = cache_dir / "infer.py"
    command = [
        sys.executable,
        str(infer_script),
        "--model",
        str(cache_dir),
        "--prompt",
        str(prompt_path),
        "--output",
        str(out_dir),
        "--device",
        device,
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    print(f"[run] executing portable inference: {infer_script}", flush=True)
    subprocess.run(command, check=True)

    ids_path = out_dir / "output.ids.json"
    raw_text_path = out_dir / "output.txt"
    inference = json.loads(ids_path.read_text(encoding="utf-8"))
    if inference.get("format") != "torchwright_doom.output_ids.v1":
        raise ValueError("portable inference wrote an unsupported ids artifact")
    if inference.get("bundle") != bundle_manifest.get("bundle_identity"):
        raise ValueError("portable inference output names a different bundle")
    if inference.get("prompt", {}).get("row_ids") != prefill_ids:
        raise ValueError("portable inference prompt ids differ from the render prompt")
    emitted_rows = [int(row) for row in inference.get("emitted_row_ids", [])]
    if not emitted_rows:
        raise ValueError("portable inference emitted no rows")
    termination = inference.get("generation", {}).get("termination_reason")
    if termination == "terminal" and emitted_rows[-1] != terminal_row:
        raise ValueError("portable inference claimed terminal without the DONE row")
    if termination == "cap" and len(emitted_rows) != max_new_tokens:
        raise ValueError("portable inference claimed cap at the wrong row count")
    # Protocol validation goes through the contract (the raw-word codec over
    # the bundle's frozen words), not the prettifier.
    text_rows = rows_from_raw_text(words, raw_text_path.read_text(encoding="utf-8"))
    if text_rows != emitted_rows:
        raise ValueError("portable output.txt differs from output.ids.json")
    print(
        f"[run] portable inference emitted {len(emitted_rows)} rows; "
        f"stopped={termination}",
        flush=True,
    )

    # Everything below is post-processing over the two portable inference
    # artifacts.  No model is loaded and no additional inference occurs.
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
        mode="transformers_generate",
        label=f"{config.map.lower()}__{config.screen[0]}x{config.screen[1]}",
        config={
            "path": str(config_path),
            "cache_dir": str(cache_dir),
            "wall_textures": list(config.textures.wall),
            "flat_textures": list(config.textures.flat),
        },
        bundle_manifest=bundle_manifest,
    )
    dump_path = artifacts.write_token_dump(out_dir / "token_dump.json", dump)
    print(
        f"[run] retained prompt.txt + {ids_path.name} + {raw_text_path.name}; "
        f"wrote token_dump.json"
        f"{' + ' + ', '.join(p.name for p in pngs) if pngs else ''} to {out_dir}",
        flush=True,
    )
    return {
        "token_dump": str(dump_path),
        "pngs": [str(p) for p in pngs],
        "report": asdict(report) if report is not None else None,
        "report_text": report.format_short() if report is not None else None,
        "inference": {
            "mode": "transformers_generate",
            "n_tokens": len(emitted_rows),
            "timing_seconds": inference.get("timing_seconds"),
            "stopped": termination,
        },
        "cache_dir": str(cache_dir),
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="Compile and run YAML DOOM render jobs")
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("compile", help="compile config to a complete Phi-3 bundle")
    pc.add_argument("--config", required=True, dest="config_path")
    pc.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")

    pod = sub.add_parser(
        "compile-onnx-debug", help="compile the explicit diagnostic ONNX artifact"
    )
    pod.add_argument("--config", required=True, dest="config_path")
    pod.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")

    pr = sub.add_parser("run", help="render one pose from a YAML config")
    pr.add_argument("--config", required=True, dest="config_path")
    pr.add_argument("--x", type=float)
    pr.add_argument("--y", type=float)
    pr.add_argument("--angle", type=int)
    pr.add_argument("--viewz", type=float)
    # Run-knob defaults live in the config's ``run:`` section (run_config
    # resolves None there) — argparse must NOT restate them.
    pr.add_argument("--out-dir", default="out/render", dest="out_dir")
    pr.add_argument("--max-new-tokens", type=int, default=None, dest="max_new_tokens")
    pr.add_argument("--png", action="store_true")
    pr.add_argument("--compare", action="store_true", dest="compare_images")
    pr.add_argument("--png-zoom", type=int, default=8, dest="png_zoom")
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
    elif command == "compile-onnx-debug":
        compile_onnx_debug_config(**values)
    else:
        run_config(**values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
