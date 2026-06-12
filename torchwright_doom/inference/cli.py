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
    mode: str = "spec_decode",
    out_dir: str | Path = "out/render",
    max_positions: int = 10240,
    draft_window: int = 0,
    prefill_chunk_size: int = 128,
    progress_every: int = 250,
    png: bool = False,
    compare_images: bool = False,
    png_zoom: int = 8,
    verbose_compile: bool = False,
    cache_dir: str | Path | None = None,
    profile: bool = False,
    attention_buckets: str | list[int] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)
    # Attention-window bucket table (stride bucketing) — a RUNTIME knob:
    # the symbolic-dim graph serves any table without recompiling, so this
    # deliberately does NOT enter the compile-cache key.  None = quarters
    # of cache_stride.  CLI hands it over as a comma-separated string.
    if isinstance(attention_buckets, str):
        attention_buckets = (
            [int(tok) for tok in attention_buckets.replace(",", " ").split()]
            if attention_buckets.strip()
            else None
        )

    from ..vocab import DONE
    from . import artifacts, compare as compare_mod
    from .compile_cache import compile_cached, load_cached_runtime
    from .decode import decode_rows_to_pixels
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
        cache_dir,
        enable_profiling=profile,
        profile_dir=out_dir if profile else None,
        attention_buckets=attention_buckets,
    )
    print("[run] ONNX runtime ready", flush=True)

    pure = None
    spec = None
    spec_stats: dict[str, Any] | None = None
    if mode in ("pure_ar", "both"):
        pure = compiled.pure_ar_rollout(
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

        spec, spec_stats = compiled.spec_decode_rollout(
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
        ties = _assert_streams_equivalent(spec.emitted_rows, pure.emitted_rows)
        tie_note = (
            "token-identical OK"
            if not ties
            else (
                f"token-identical up to {len(ties)} value-token bin-boundary "
                f"ties at {ties[:6]}{'…' if len(ties) > 6 else ''} (fp32 "
                f"kernel-shape ulp between batched and single-row attention; "
                f"pixel tokens exact)"
            )
        )
        print(
            f"[run] spec-decode used {spec.n_forward_passes} forward passes vs "
            f"{pure.n_forward_passes} pure-AR "
            f"({pure.n_forward_passes / max(1, spec.n_forward_passes):.2f}x fewer); "
            f"{tie_note}",
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


def _first_diff(a: list[int], b: list[int]) -> int:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def _assert_streams_equivalent(spec_rows: list[int], pure_rows: list[int]) -> list[int]:
    """Assert spec/pure render equivalence; return positions of allowed ties.

    fp32 cannot promise bitwise stream equality across kernel shapes: the
    batched (spec verify) and single-row (captured decode) attention kernels
    reduce in different orders, and a scene value lying within an ulp of a
    vocabulary quantization-bin boundary argmaxes into ADJACENT value bins
    (observed: 13/9739 positions, all the same wall-geometry scratch value,
    pixels identical — see plan_cuda_graph_decode.md).  The load-bearing
    contract is therefore: equal length, and every mismatch is a pair of
    ``value``-type tokens (protocol scratch).  Any pixel/geometry/protocol
    token diff, or a length diff, is a hard failure.
    """
    from ..embedding import TOKEN_VOCAB
    from ..vocab import VALUE

    if len(spec_rows) != len(pure_rows):
        raise RuntimeError(
            f"spec-decode and pure-AR emitted different lengths: "
            f"{len(spec_rows)} vs {len(pure_rows)} (first diff at "
            f"{_first_diff(spec_rows, pure_rows)})"
        )
    ties: list[int] = []
    propagated: list[tuple[int, str, str]] = []
    for i, (s, p) in enumerate(zip(spec_rows, pure_rows)):
        if s == p:
            continue
        s_type, s_vals = TOKEN_VOCAB.row_to_token[s]
        p_type, p_vals = TOKEN_VOCAB.row_to_token[p]
        # Positionwise TYPE alignment is the hard invariant: a spec-decode
        # machinery bug (wrong commit, stale cache row) cascades — every
        # subsequent token shifts and the types misalign immediately.
        if s_type is not p_type:
            raise RuntimeError(
                f"spec-decode structurally diverged from pure-AR at {i}: "
                f"{s_type.name} vs {p_type.name} (rows {s} vs {p})"
            )
        if s_type is VALUE:
            ties.append(i)
        else:
            propagated.append((i, f"{p_type.name}{p_vals}", f"{s_type.name}{s_vals}"))
    # Measured full-frame tie census (d4096/S=65536/B200): 52/9739 diffs =
    # 50 value-token adjacent-bin ties + 2 downstream propagations (one
    # pixel color via a colormap lookup, one wallColU off-by-one).  A small
    # number of propagated ties per frame is fp32-expected; more than a
    # handful means a real divergence.
    _PROPAGATED_TIE_BUDGET = 8
    for i, p_txt, s_txt in propagated:
        print(
            f"[run] WARNING propagated fp32 tie at {i}: pure {p_txt} vs "
            f"spec {s_txt}",
            flush=True,
        )
    if len(propagated) > _PROPAGATED_TIE_BUDGET:
        raise RuntimeError(
            f"{len(propagated)} non-value token diffs exceeds the propagated-tie "
            f"budget ({_PROPAGATED_TIE_BUDGET}) — treat as a real divergence: "
            f"{propagated[:4]}"
        )
    return ties + [i for i, _, _ in propagated]


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
    pr.add_argument(
        "--mode", default="spec_decode", choices=["spec_decode", "pure_ar", "both"]
    )
    pr.add_argument("--out-dir", default="out/render", dest="out_dir")
    pr.add_argument("--max-positions", type=int, default=10240, dest="max_positions")
    pr.add_argument("--draft-window", type=int, default=0, dest="draft_window")
    pr.add_argument(
        "--prefill-chunk-size", type=int, default=128, dest="prefill_chunk_size"
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
    pr.add_argument(
        "--attention-buckets",
        default=None,
        dest="attention_buckets",
        help="comma-separated attention-window bucket table (stride "
        "bucketing), e.g. '16384,32768,49152,65536'; runtime knob, no "
        "recompile; default = quarters of cache_stride",
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
