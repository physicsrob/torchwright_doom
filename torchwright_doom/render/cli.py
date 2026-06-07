"""Plan K CLI — generate a DOOM frame with the compiled transformer.

``python -m torchwright_doom.render.cli --fixture e1m1_subset_textured --pose 0
--mode pure_ar --out-dir out/run``

Orchestrates: load scene -> build prefill rows -> compile the token-id forward ->
autoregressively *generate* the rollout (pure-AR, or spec-decode asserted
bit-identical) -> decode to pixels -> compare to the reference render -> write
generated/reference/diff PNGs + token_dump.json. This is the only layer that
touches argparse, disk, and ``doom_sandbox``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..constants import SCREEN_HEIGHT, SCREEN_WIDTH
from ..vocab import DONE
from . import artifacts, compare
from .compiled_model import DEFAULT_D, DEFAULT_D_HEAD, build_compiled
from .decode import decode_rows_to_pixels
from .pure_ar import pure_ar_rollout
from .tokens_bridge import row_index, sandbox_token_to_row


def _ensure_doom_sandbox() -> None:
    """Make the sibling ``doom_sandbox`` checkout importable (umbrella on path)."""
    try:
        import doom_sandbox  # noqa: F401

        return
    except ImportError:
        pass
    umbrella = Path(__file__).resolve().parents[3]
    if (umbrella / "doom_sandbox").is_dir():
        sys.path.insert(0, str(umbrella))
    import doom_sandbox  # noqa: F401  - raise clearly if still missing


def _pick_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _check_screen_dims() -> None:
    from doom_sandbox.implementation import reference as sb_ref

    if (sb_ref.SCREEN_WIDTH, sb_ref.SCREEN_HEIGHT) != (SCREEN_WIDTH, SCREEN_HEIGHT):
        raise RuntimeError(
            f"screen-dim mismatch: torchwright_doom {SCREEN_WIDTH}x{SCREEN_HEIGHT} "
            f"vs doom_sandbox {sb_ref.SCREEN_WIDTH}x{sb_ref.SCREEN_HEIGHT}. Set "
            f"DOOM_SANDBOX_SCREEN_WIDTH/HEIGHT to match, or (x,y) will misalign."
        )


def run_render(
    *,
    fixture: str = "e1m1_subset_textured",
    pose_index: int = 0,
    mode: str = "pure_ar",
    out_dir: str | Path = "out/render",
    max_positions: int = 8000,
    d: int = DEFAULT_D,
    d_head: int = DEFAULT_D_HEAD,
    scale: int = 8,
    draft_window: int = 8,
    progress_every: int = 250,
    verbose_compile: bool = False,
) -> dict[str, Any]:
    """Generate one frame and write artifacts. Returns a summary dict."""
    import torch

    _ensure_doom_sandbox()
    from doom_sandbox import fixtures
    from doom_sandbox.implementation import prefill as sb_prefill

    _check_screen_dims()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = fixtures.load_fixture(fixture)
    pose = scene.test_poses[pose_index]
    prefill_tokens = list(sb_prefill.get_prefill(scene, pose))
    prefill_ids = [sandbox_token_to_row(t) for t in prefill_tokens]
    terminal_row = row_index(DONE, {})

    device = _pick_device()
    print(
        f"[cli] fixture={fixture} pose={pose_index} prefill={len(prefill_ids)} "
        f"device={device} d={d} d_head={d_head}",
        flush=True,
    )
    t_compile = time.time()
    compiled, output_node = build_compiled(
        device, d=d, d_head=d_head, verbose=verbose_compile
    )
    print(f"[cli] compiled in {time.time() - t_compile:.0f}s", flush=True)

    pure = pure_ar_rollout(
        compiled,
        prefill_ids,
        max_positions=max_positions,
        terminal_row=terminal_row,
        progress_every=progress_every,
    )
    emitted_rows = pure.emitted_rows
    print(
        f"[cli] pure-AR: {len(emitted_rows)} tokens in {pure.seconds:.0f}s "
        f"({pure.seconds / max(1, len(emitted_rows)) * 1000:.0f} ms/tok), "
        f"stopped={pure.stopped}",
        flush=True,
    )

    spec_stats: dict[str, Any] | None = None
    if mode == "both":
        from .spec_decode import spec_decode_rollout

        spec, spec_stats = spec_decode_rollout(
            compiled,
            prefill_ids,
            _make_drafter(scene, pose),
            max_positions=max_positions,
            terminal_row=terminal_row,
            draft_window=draft_window,
            progress_every=progress_every,
        )
        assert spec.emitted_rows == emitted_rows, (
            "spec-decode is not bit-identical to pure-AR — a correctness bug, not "
            "a speedup question (first diff at "
            f"{_first_diff(spec.emitted_rows, emitted_rows)})"
        )
        print(
            f"[cli] spec-decode: {spec.n_forward_passes} forward passes vs "
            f"{pure.n_forward_passes} pure-AR "
            f"({pure.n_forward_passes / max(1, spec.n_forward_passes):.2f}x fewer); "
            f"bit-identical OK",
            flush=True,
        )
    elif mode != "pure_ar":
        raise ValueError(f"unknown --mode {mode!r} (expected pure_ar | both)")

    # Decode -> compare -> artifacts.
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

    footprint = {}
    if device == "cuda":
        footprint = {
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        print(
            f"[cli] GPU peak allocated {footprint['peak_allocated_gib']:.2f} GiB / "
            f"reserved {footprint['peak_reserved_gib']:.2f} GiB",
            flush=True,
        )

    print(f"[cli] wrote {[p.name for p in pngs]} + token_dump.json to {out_dir}")
    return {
        "report": asdict(report),
        "report_text": report.format_short(),
        "pngs": [str(p) for p in pngs],
        "token_dump": str(dump_path),
        "pure_ar": {
            "n_tokens": len(emitted_rows),
            "n_forward_passes": pure.n_forward_passes,
            "seconds": pure.seconds,
            "stopped": pure.stopped,
        },
        "spec_decode_stats": spec_stats,
        "footprint": footprint,
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
    p = argparse.ArgumentParser(description="Plan K — compiled transformer renders DOOM")
    p.add_argument("--fixture", default="e1m1_subset_textured")
    p.add_argument("--pose", type=int, default=0, dest="pose_index")
    p.add_argument("--mode", default="pure_ar", choices=["pure_ar", "both"])
    p.add_argument("--out-dir", default="out/render", dest="out_dir")
    p.add_argument("--max-positions", type=int, default=8000, dest="max_positions")
    p.add_argument("--d", type=int, default=DEFAULT_D)
    p.add_argument("--d-head", type=int, default=DEFAULT_D_HEAD, dest="d_head")
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--draft-window", type=int, default=8, dest="draft_window")
    p.add_argument("--progress-every", type=int, default=250, dest="progress_every")
    p.add_argument("--verbose-compile", action="store_true", dest="verbose_compile")
    args = p.parse_args(argv)
    run_render(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
