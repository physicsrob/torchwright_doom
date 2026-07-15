"""Render-job orchestration: the end-to-end ``run_config`` driver.

The conductor may see everything; nothing sees the conductor. Its phases:

    load config -> load WAD scene and pose -> construct prompt row ids
    -> rows to canonical prompt text (tokenizer/codec + bundle vocab)
    -> execute <bundle>/infer.py as a subprocess
    -> validate output.ids.json / output.txt
    -> interpret: decode / compare / write optional artifacts

It never loads a model or tokenizer through Transformers, never calls
``.generate()``, never compiles, never imports diagnostics, and never
modifies the canonical inference artifacts after the subprocess returns.
Prompt text, ``output.ids.json``, and ``output.txt`` are the retained
artifacts; everything else is reproducible downstream.

Graph-reaching imports stay inside ``run_config`` after
``apply_screen_env`` (the import-time screen-environment rule).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import (
    apply_screen_env,
    load_render_config,
    resolve_run_args,
    resolve_wad_path,
)
from .identity import hf_bundle_cache_dir


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

    from .bundle.manifest import validate_bundle_manifest
    from .interpret import artifacts, compare as compare_mod
    from .interpret.decode import decode_rows_to_pixels
    from .interpret.reference import pydoom_scene_for
    from .prompt.scene import load_render_scene, pose_from_world, prefill_rows_for
    from .tokenizer.codec import raw_text_from_rows, rows_from_raw_text
    from .tokenizer.rows import row_index
    from .model.vocab import DONE

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
    # Host-side allocator config only (no computation moves): the stock
    # DynamicCache grows by a per-layer torch.cat each step, whose transient
    # second copy needs one layer's full K/V (~1 GiB at 320x200 end-of-frame)
    # as a CONTIGUOUS block.  A full-resolution frame fills the render GPU to
    # within a few GiB, and ~30 min of grow-free cycles fragments what's
    # left — the 2026-07-14 production render died at ~61k/61,440 tokens with
    # 5.2 GiB stranded in reserved-but-unallocated segments.  Expandable
    # segments lets those segments grow in place instead.
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, check=True, env=env)

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
