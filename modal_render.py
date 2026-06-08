"""Render on Modal A100-80GB, with artifact sync-back.

    uv run modal run modal_render.py --config configs/e1m1_start_room.yaml
    uv run modal run modal_render.py --config configs/e1m1_start_room.yaml --mode both --png --compare

The full frame's KV footprint is too large for the local L4, so generation runs
on an A100-80GB. Artifacts (generated/reference/diff PNGs + token_dump.json) are
written to a ``modal.Volume`` (durable; inspect later with ``modal volume get``)
*and* returned so the local entrypoint mirrors them to ``out/<run>/`` on disk —
``make modal-run`` only captures stdout, which is exactly why this dedicated
entrypoint exists (the one sanctioned new root ``modal_*.py``).

The shared image includes ``numba`` for the ``doom_sandbox`` runtime. This module
mounts the sibling ``doom_sandbox`` checkout (code + fixture JSONs + WAD) at
``/root/doom_sandbox`` so the reference renderer and drafter import there.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

from modal_image import IMAGE

_HERE = Path(__file__).resolve().parent
_DOOM_SANDBOX = _HERE.parent / "doom_sandbox"
_CONFIGS = _HERE / "configs"

_IGNORE_PARTS = {"__pycache__", "token_dumps", "specs", "scripts", ".git", ".venv"}


def _ignore(p: Path) -> bool:
    return any(part in _IGNORE_PARTS for part in p.parts) or p.suffix == ".pyc"


RENDER_IMAGE = (
    IMAGE.add_local_dir(str(_DOOM_SANDBOX), "/root/doom_sandbox", ignore=_ignore)
    .add_local_dir(str(_CONFIGS), "/root/configs", ignore=_ignore)
    .add_local_file(str(_HERE / "doom1.wad"), "/root/configs/doom1.wad")
)

app = modal.App("torchwright-doom-render", image=RENDER_IMAGE)
RENDER_VOLUME = modal.Volume.from_name(
    "torchwright-doom-render", create_if_missing=True
)
CACHE_VOLUME = modal.Volume.from_name(
    "torchwright-doom-render-cache", create_if_missing=True
)


@app.function(
    gpu="a100-80gb",
    cpu=8,
    memory=65536,
    timeout=5400,
    volumes={
        "/artifacts": RENDER_VOLUME,
        "/root/.cache/torchwright_doom/compiled": CACHE_VOLUME,
    },
)
def render_remote(run_id: str, kwargs: dict) -> dict:
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")  # make /root/doom_sandbox importable

    from torchwright_doom.render.cli import run_config

    out_dir = f"/artifacts/{run_id}"
    summary = run_config(out_dir=out_dir, **kwargs)
    CACHE_VOLUME.commit()
    RENDER_VOLUME.commit()

    files = {
        p.name: p.read_bytes() for p in sorted(Path(out_dir).glob("*")) if p.is_file()
    }
    return {"summary": summary, "files": files}


def _remote_config_path(config: str) -> Path:
    path = Path(config)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts[:2] == ("torchwright_doom", "configs"):
        return Path("/root/configs", *parts[2:])
    if parts and parts[0] == "configs":
        return Path("/root", *parts)
    return Path("/root/configs") / path


@app.local_entrypoint()
def main(
    config: str = "configs/e1m1_start_room.yaml",
    x: float = 1056.0,
    y: float = -3616.0,
    angle: int = 64,
    viewz: float = 41.0,
    mode: str = "spec_decode",
    out_dir: str = "out/render",
    run_name: str = "",
    max_positions: int = 8000,
    draft_window: int = 0,
    prefill_chunk_size: int = 65536,
    progress_every: int = 250,
    png: bool = False,
    compare: bool = False,
    png_zoom: int = 8,
    verbose_compile: bool = False,
):
    config_path = Path(config)
    remote_config = _remote_config_path(config)

    run_id = (
        run_name
        or f"{config_path.stem}__x{x:g}_y{y:g}_a{angle}__{mode}__{int(time.time())}"
    )
    kwargs = dict(
        config_path=str(remote_config),
        x=x,
        y=y,
        angle=angle,
        viewz=viewz,
        mode=mode,
        max_positions=max_positions,
        draft_window=draft_window,
        prefill_chunk_size=prefill_chunk_size,
        progress_every=progress_every,
        png=png,
        compare_images=compare,
        png_zoom=png_zoom,
        verbose_compile=verbose_compile,
    )
    print(f"[local] launching render_remote run_id={run_id} kwargs={kwargs}")
    result = render_remote.remote(run_id, kwargs)

    local_dir = _HERE / out_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        (local_dir / name).write_bytes(data)
    if result["summary"].get("report_text"):
        print("\n" + result["summary"]["report_text"])
    print(f"\n[local] artifacts -> {local_dir}")
    print(f"[local] files: {sorted(result['files'])}")
    print(f"[local] remote volume path: /artifacts/{run_id}")
