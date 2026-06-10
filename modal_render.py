"""Compile + render on Modal, with artifact sync-back.

    uv run modal run modal_render.py --config configs/e1m1_start_room.yaml
    uv run modal run modal_render.py --config configs/e1m1_start_room.yaml --mode both --png --compare

Two remote stages: ``compile_remote`` (CPU-only, 64 cores so CP-SAT's parallel
search gets real width; writes the ONNX into the cache volume) and
``render_remote`` (A100-80GB — the full frame's KV footprint is too large for
the local L4). The compile cache key is computed on the LOCAL machine because
it embeds submodule git SHAs that don't exist inside a Modal container.
Artifacts (generated/reference/diff PNGs + token_dump.json) are written to a
``modal.Volume`` (durable; inspect later with ``modal volume get``) *and*
returned so the local entrypoint mirrors them to ``out/<run>/`` on disk —
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
# Solved CP-SAT schedules keyed by graph topology (torchwright
# schedule_cache): a schedule win is durable across compiles whose graph
# construction is unchanged, so the solver runs at most once per shape.
SCHEDULE_VOLUME = modal.Volume.from_name(
    "torchwright-doom-schedule-cache", create_if_missing=True
)

# CPU-only compile container. CP-SAT's parallel search scales with cores, so
# the compile gets its own high-CPU function (the GPU render container stays
# at 8 CPUs). TW_CPSAT_WORKERS below must match this number.
_COMPILE_CPUS = 64

# Render GPU, read at (local) import time — pass as an env var, not a make
# variable: RENDER_GPU=b200 make run.  The captured decode step is
# KV-bandwidth-bound, so GPU HBM bandwidth maps ~directly to step time
# (A100-80GB ~2 TB/s; B200 ~8 TB/s and 192 GB fits the 64k cache + 1024-row
# prefill chunks comfortably).
import os as _os

_RENDER_GPU = _os.environ.get("RENDER_GPU", "a100-80gb")


@app.function(
    cpu=_COMPILE_CPUS,
    # Building + exporting holds the full weight set plus the ONNX
    # serialization copy in RAM: ~23 GB fp32 at d=4096, ~4x that at the
    # d=8192/h16384 flagship config — size for the latter.
    memory=262144,
    timeout=10800,
    volumes={
        "/root/.cache/torchwright_doom/compiled": CACHE_VOLUME,
        "/root/.cache/torchwright_doom/schedules": SCHEDULE_VOLUME,
    },
)
def compile_remote(
    config_path: str,
    cache_subdir: str,
    compile_payload: dict,
    verbose_compile: bool,
) -> dict:
    import os
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    # CP-SAT reads this at solve time (torchwright cpsat_scheduler); point it
    # at the container's full CPU allocation instead of the 16-worker default.
    os.environ["TW_CPSAT_WORKERS"] = str(_COMPILE_CPUS)
    # Durable schedule cache (see SCHEDULE_VOLUME above).
    os.environ["TW_SCHEDULE_CACHE_DIR"] = "/root/.cache/torchwright_doom/schedules"

    from torchwright_doom.render.cli import compile_config

    # The cache key + payload come from the LOCAL machine: this container has
    # no ``.git``, so deriving them here would collapse the git SHAs to
    # "unknown" and silently reuse a stale model across code changes.
    CACHE_VOLUME.reload()
    SCHEDULE_VOLUME.reload()
    cache_dir = f"/root/.cache/torchwright_doom/compiled/{cache_subdir}"
    result = compile_config(
        config_path=config_path,
        verbose_compile=verbose_compile,
        cache_dir=cache_dir,
        compile_payload=compile_payload,
    )
    CACHE_VOLUME.commit()
    SCHEDULE_VOLUME.commit()
    return result


def _volume_has_compiled(cache_subdir: str) -> bool:
    """Local cache-hit probe so a hit skips the compile container entirely."""
    try:
        names = {
            entry.path.rsplit("/", 1)[-1]
            for entry in CACHE_VOLUME.listdir(f"/{cache_subdir}")
        }
    except Exception:
        return False
    return {"model.onnx", "model.meta.json"} <= names


@app.function(
    gpu=_RENDER_GPU,
    cpu=8,
    memory=65536,
    timeout=5400,
    volumes={
        "/artifacts": RENDER_VOLUME,
        "/root/.cache/torchwright_doom/compiled": CACHE_VOLUME,
    },
)
def render_remote(run_id: str, kwargs: dict, cache_subdir: str) -> dict:
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")  # make /root/doom_sandbox importable

    from torchwright_doom.render.cli import run_config

    # Compilation happens in ``compile_remote`` (a separate 64-CPU container),
    # which writes the ONNX into CACHE_VOLUME under ``cache_subdir``. Reload so
    # those files are visible, then point the renderer straight at them — this
    # container is render-only and never compiles.
    CACHE_VOLUME.reload()
    cache_dir = f"/root/.cache/torchwright_doom/compiled/{cache_subdir}"
    kwargs = {**kwargs, "cache_dir": cache_dir}

    out_dir = f"/artifacts/{run_id}"
    summary = run_config(out_dir=out_dir, **kwargs)
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
    prefill_chunk_size: int = 128,
    progress_every: int = 250,
    png: bool = False,
    compare: bool = False,
    png_zoom: int = 8,
    verbose_compile: bool = False,
    profile: bool = False,
):
    config_path = Path(config)
    remote_config = _remote_config_path(config)

    # --- Compile on Modal (64-CPU container), key computed LOCALLY -------
    # The cache key embeds the submodule git SHAs, and Modal containers have
    # no ``.git`` (``_git_sha`` collapses to "unknown" there, so a
    # remotely-derived key would never change on a code edit — silently
    # reusing a stale model).  The local machine computes the canonical
    # payload + key and hands both to ``compile_remote``, which compiles
    # straight into CACHE_VOLUME under that key with CP-SAT fanned out
    # across the container's 64 CPUs.
    #
    # Importing ``render.config`` here is safe: it has no dependency on the
    # screen-sized token vocab (the import-order trap that forces
    # ``compile_config`` to call ``apply_screen_env`` before touching
    # ``render.cache``).
    from torchwright_doom.render.config import (
        cache_key_from_payload,
        canonical_compile_payload,
        load_render_config,
        resolve_wad_path,
    )

    render_config = load_render_config(config_path)
    wad_path = resolve_wad_path(render_config, base_dir=config_path.parent)
    compile_payload = canonical_compile_payload(render_config, wad_path)
    cache_subdir = cache_key_from_payload(compile_payload)

    if _volume_has_compiled(cache_subdir):
        print(f"[local] compile cache HIT CACHE_VOLUME:/{cache_subdir}", flush=True)
    else:
        print(
            f"[local] compile cache MISS — compiling {config_path} on Modal "
            f"({_COMPILE_CPUS} CPUs) -> CACHE_VOLUME:/{cache_subdir}",
            flush=True,
        )
        compile_remote.remote(
            str(remote_config), cache_subdir, compile_payload, verbose_compile
        )

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
        profile=profile,
    )
    print(f"[local] launching render_remote run_id={run_id} kwargs={kwargs}")
    result = render_remote.remote(run_id, kwargs, cache_subdir)

    local_dir = _HERE / out_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        (local_dir / name).write_bytes(data)
    if result["summary"].get("report_text"):
        print("\n" + result["summary"]["report_text"])
    print(f"\n[local] artifacts -> {local_dir}")
    print(f"[local] files: {sorted(result['files'])}")
    print(f"[local] remote volume path: /artifacts/{run_id}")
