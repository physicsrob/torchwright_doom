"""Compile + render on Modal, with artifact sync-back.

    uv run modal run modal_render.py --config configs/e1m1.yaml
    uv run modal run modal_render.py --config configs/e1m1.yaml --png --compare

Two remote stages: ``compile_remote`` (CPU-only, 64 cores so CP-SAT's parallel
search gets real width; writes the ONNX into the cache volume) and
``render_remote`` (a big GPU — the native HF model is ~28 GB fp32 weights plus
an unbounded KV cache over the full frame). The render loads the artifact as a
native ``TorchwrightForCausalLM`` (the production runtime). The compile cache
key is computed on the LOCAL machine because it embeds submodule git SHAs that
don't exist inside a Modal container.
Artifacts (generated/reference/diff PNGs + token_dump.json) are written to a
``modal.Volume`` (durable; inspect later with ``modal volume get``) *and*
returned so the local entrypoint mirrors them to ``out/<run>/`` on disk —
``make modal-run`` only captures stdout, which is exactly why this dedicated
entrypoint exists (the one sanctioned new root ``modal_*.py``).

The shared image includes ``numba`` for the reference renderer (pydoom)
runtime; ``modal_image.ASSETS_IMAGE`` mounts the ``torchwright_doom`` package
(which vendors pydoom) plus the WAD so the reference renderer and drafter
import there.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

from modal_image import ASSETS_IMAGE, CACHE_VOLUME

_HERE = Path(__file__).resolve().parent

app = modal.App("torchwright-doom-render", image=ASSETS_IMAGE)
RENDER_VOLUME = modal.Volume.from_name(
    "torchwright-doom-render", create_if_missing=True
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

# Render GPU — an env var (NOT a config field) because it parameterizes the
# ``@app.function(gpu=...)`` decorator at module-import time, before --config
# is parsed.  The Makefile exports it (``RENDER_GPU ?= b200``); the fallback
# here must match.  B200 is the default: the captured decode step is
# KV-bandwidth-bound, so GPU HBM bandwidth maps ~directly to step time
# (A100-80GB ~2 TB/s; B200 ~8 TB/s and 192 GB fits the 64k cache + 1024-row
# prefill chunks comfortably).  The A100 also has slack for the windowed
# production config (~11.4 GB KV + ~17 GB weights): RENDER_GPU=a100-80gb.
import os as _os

_RENDER_GPU = _os.environ.get("RENDER_GPU", "b200")


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
    config_name: str,
    config_text: str,
    cache_subdir: str,
    compile_payload: dict,
    verbose_compile: bool,
    disable_cache: bool = False,
) -> dict:
    import os
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    config_path = _write_shipped_config(config_name, config_text)

    # CP-SAT reads this at solve time (torchwright cpsat_scheduler); point it
    # at the container's full CPU allocation instead of the 16-worker default.
    os.environ["TW_CPSAT_WORKERS"] = str(_COMPILE_CPUS)
    if disable_cache:
        # DISABLE_CACHE: neither durable cache is read or written.  Both the
        # ONNX artifact and the schedule entry go to a per-call scratch dir
        # (mkdtemp, not a fixed path: a warm container must not hand a later
        # no-cache run an earlier one's artifact or schedule as a hit), and
        # the volume reload/commit pairs are skipped — the mounts stay
        # untouched.
        import tempfile

        scratch = tempfile.mkdtemp(prefix="compile-nocache-", dir="/tmp")
        os.environ["TW_SCHEDULE_CACHE_DIR"] = f"{scratch}/schedules"
        cache_dir = f"{scratch}/compiled/{cache_subdir}"
        print(f"[compile] [nocache] volumes bypassed; scratch={scratch}", flush=True)
    else:
        # Durable schedule cache (see SCHEDULE_VOLUME above).
        os.environ["TW_SCHEDULE_CACHE_DIR"] = "/root/.cache/torchwright_doom/schedules"
        # The cache key + payload come from the LOCAL machine: this container
        # has no ``.git``, so deriving them here would collapse the git SHAs
        # to "unknown" and silently reuse a stale model across code changes.
        CACHE_VOLUME.reload()
        SCHEDULE_VOLUME.reload()
        cache_dir = f"/root/.cache/torchwright_doom/compiled/{cache_subdir}"

    from torchwright_doom.inference.cli import compile_config

    result = compile_config(
        config_path=config_path,
        verbose_compile=verbose_compile,
        cache_dir=cache_dir,
        compile_payload=compile_payload,
    )
    if disable_cache:
        # Ship the sampled schedule back by VALUE so the local side can
        # preserve the draw without any volume write (schedules enter the
        # durable cache exactly one way — a caching compile solve).
        sched_dir = Path(os.environ["TW_SCHEDULE_CACHE_DIR"])
        result["nocache_schedules"] = {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(sched_dir.glob("*.json"))
        }
        print(
            f"[compile] [nocache] done — durable caches untouched; "
            f"{len(result['nocache_schedules'])} schedule entry(ies) "
            f"returned inline",
            flush=True,
        )
    else:
        CACHE_VOLUME.commit()
        SCHEDULE_VOLUME.commit()
    return result


def _write_shipped_config(config_name: str, config_text: str) -> str:
    """Materialize the LOCAL config file inside the container.

    The config is shipped by VALUE (its text), not by path: /tmp variant
    configs (the one-config discipline's A/B mechanism) don't exist in
    the baked image, and shipping the committed configs the same way
    removes the path-translation special case.  WAD resolution is
    unaffected: resolve_wad_path falls back to the repo root
    (/root/doom1.wad) when the config's own directory has no WAD.
    """
    from pathlib import Path as _P

    dest = _P("/tmp/shipped_configs") / _P(config_name).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(config_text)
    return str(dest)


def _volume_has_compiled(cache_subdir: str) -> bool:
    """Local cache-hit probe so a hit skips the compile container entirely."""
    try:
        names = {
            entry.path.rsplit("/", 1)[-1]
            for entry in CACHE_VOLUME.listdir(f"/{cache_subdir}")
        }
    except (FileNotFoundError, modal.exception.NotFoundError):
        return False  # genuinely no entry under that key
    except Exception as exc:
        # A volume/auth failure is NOT a cache miss — treating it as one
        # spins up a 64-CPU compile container that immediately rediscovers
        # the hit.  Degrade, but say what happened.
        print(
            f"[local] volume cache probe failed ({exc!r}) — treating as miss",
            flush=True,
        )
        return False
    return {"model.onnx", "model.meta.json"} <= names


def _save_nocache_schedules(result: dict) -> None:
    """Mirror a no-cache run's sampled schedule entries to local /tmp.

    The no-cache path never writes SCHEDULE_VOLUME, so a good draw would
    otherwise die with the container.  Filenames carry a timestamp + pid
    (concurrent `make compile` runs are separate processes and must not
    collide) plus the drawn n_layers so draws compare at a glance.
    """
    import json
    import os

    entries = result.get("nocache_schedules") or {}
    if not entries:
        # optimize=0 and UNKNOWN/INFEASIBLE fallbacks store no entry.
        print("[local] [nocache] no schedule entry to save", flush=True)
        return
    save_dir = Path("/tmp/torchwright_doom-nocache-schedules")
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for name, text in sorted(entries.items()):
        n_layers = json.loads(text).get("n_layers")
        dest = save_dir / f"{stamp}-p{os.getpid()}_L{n_layers}_{name}"
        dest.write_text(text, encoding="utf-8")
        print(f"[local] [nocache] schedule draw saved -> {dest}", flush=True)


def _compile_on_modal(
    config_path: Path, verbose_compile: bool, disable_cache: bool = False
) -> str:
    """Compute the compile-cache key LOCALLY and compile on Modal on a miss.

    Shared by the render entrypoint (``main``) and the compile-only
    entrypoint (``compile_only``), so ``make compile`` and the implicit
    compile inside ``make run`` are byte-for-byte the same job: the same
    64-CPU ``compile_remote`` container, the same key, the same artifact in
    CACHE_VOLUME.

    The key embeds the submodule git SHAs, and Modal containers have no
    ``.git`` (``_git_sha`` collapses to "unknown" there, so a remotely-derived
    key would never change on a code edit — silently reusing a stale model).
    The local machine computes the canonical payload + key and hands both to
    ``compile_remote``, which compiles straight into CACHE_VOLUME under that
    key with CP-SAT fanned out across the container's 64 CPUs.

    Importing ``inference.config`` here is safe: it has no dependency on the
    screen-sized token vocab (the import-order trap that forces
    ``compile_config`` to call ``apply_screen_env`` before touching
    ``inference.compile_cache``).

    ``disable_cache`` (the ``DISABLE_CACHE=1 make compile`` path) skips the
    HIT probe (a volume read), compiles into container-local scratch instead
    of CACHE_VOLUME, and mirrors the sampled schedule to local /tmp — the
    durable caches are neither read nor written.

    Returns the cache subdir (the volume-relative key).
    """
    from torchwright_doom.inference.config import (
        cache_key_from_payload,
        canonical_compile_payload,
        load_render_config,
        resolve_wad_path,
    )

    # Shipped by VALUE (see _write_shipped_config): /tmp variant configs work
    # on Modal, and the legs of an A/B differ only in the file text.
    config_text = config_path.read_text()
    render_config = load_render_config(config_path)
    wad_path = resolve_wad_path(render_config, base_dir=config_path.parent)
    compile_payload = canonical_compile_payload(render_config, wad_path)
    cache_subdir = cache_key_from_payload(compile_payload)

    if disable_cache:
        print(
            f"[local] [nocache] DISABLE_CACHE — cache probe skipped; compiling "
            f"{config_path} on Modal ({_COMPILE_CPUS} CPUs) -> container-local "
            f"scratch (volumes untouched)",
            flush=True,
        )
        result = compile_remote.remote(
            config_path.name,
            config_text,
            cache_subdir,
            compile_payload,
            verbose_compile,
            True,
        )
        _save_nocache_schedules(result)
    elif _volume_has_compiled(cache_subdir):
        print(f"[local] compile cache HIT CACHE_VOLUME:/{cache_subdir}", flush=True)
    else:
        print(
            f"[local] compile cache MISS — compiling {config_path} on Modal "
            f"({_COMPILE_CPUS} CPUs) -> CACHE_VOLUME:/{cache_subdir}",
            flush=True,
        )
        compile_remote.remote(
            config_path.name,
            config_text,
            cache_subdir,
            compile_payload,
            verbose_compile,
        )
    return cache_subdir


@app.local_entrypoint()
def compile_only(
    config: str = "configs/e1m1.yaml",
    verbose_compile: bool = False,
    disable_cache: bool = False,
):
    """``make compile`` — compile a config to the Modal cache volume, no render.

    Runs the SAME 64-CPU ``compile_remote`` path ``make run`` uses on a cache
    miss, so the production artifact is built once with the wide CP-SAT search
    and a later ``make run`` is a cache hit.  The compiled ONNX lives in
    CACHE_VOLUME (durable), exactly where ``render_remote`` reads it — there is
    no local-disk copy (that is what ``make run-local`` would build instead).

    ``--disable-cache`` (``DISABLE_CACHE=1 make compile``) runs the same
    production compile but touches neither durable cache: no HIT probe, the
    artifact dies with the container, and the sampled schedule is mirrored to
    local /tmp instead of SCHEDULE_VOLUME (see _save_nocache_schedules).
    """
    config_path = Path(config)
    cache_subdir = _compile_on_modal(config_path, verbose_compile, disable_cache)
    if disable_cache:
        print(
            "[local] [nocache] compile complete — CACHE_VOLUME/SCHEDULE_VOLUME "
            "untouched (artifact discarded with the container)",
            flush=True,
        )
    else:
        print(f"[local] compile complete -> CACHE_VOLUME:/{cache_subdir}", flush=True)


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
def render_remote(
    run_id: str, kwargs: dict, cache_subdir: str, config_name: str, config_text: str
) -> dict:
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")  # make the /root-mounted packages importable

    import torch

    from torchwright_doom.inference.cli import run_config

    kwargs = {**kwargs, "config_path": _write_shipped_config(config_name, config_text)}

    # Compilation happens in ``compile_remote`` (a separate 64-CPU container),
    # which writes the ONNX into CACHE_VOLUME under ``cache_subdir``. Reload so
    # those files are visible, then point the renderer straight at them — this
    # container is render-only and never compiles.
    CACHE_VOLUME.reload()
    cache_dir = f"/root/.cache/torchwright_doom/compiled/{cache_subdir}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {**kwargs, "cache_dir": cache_dir, "device": device}

    out_dir = f"/artifacts/{run_id}"
    summary = run_config(out_dir=out_dir, **kwargs)
    RENDER_VOLUME.commit()

    files = {
        p.name: p.read_bytes() for p in sorted(Path(out_dir).glob("*")) if p.is_file()
    }
    return {"summary": summary, "files": files}


@app.function(
    gpu=_RENDER_GPU,
    cpu=8,
    memory=131072,
    timeout=5400,
    volumes={
        "/artifacts": RENDER_VOLUME,
        "/root/.cache/torchwright_doom/compiled": CACHE_VOLUME,
    },
)
def hf_export_remote(
    config_name: str,
    config_text: str,
    cache_subdir: str,
) -> dict:
    """Convert the compiled DOOM artifact to a native HF bundle.

    Runs on the render GPU: the densified fp32 weights (~28 GB) need a big card.
    The full bundle (safetensors) is written to RENDER_VOLUME (durable, ~28 GB —
    never synced back); only ``config.json`` and the export report return to the
    local entrypoint.
    """
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from torchwright_doom.inference.config import (
        apply_screen_env,
        load_render_config,
    )

    config_path = _write_shipped_config(config_name, config_text)
    config = load_render_config(config_path)
    # Screen env BEFORE any vocab/embedding/scene/fixture import (the DOOM vocab
    # is screen-sized at import time).
    apply_screen_env(config)

    from torchwright_doom.inference import hf_export

    CACHE_VOLUME.reload()
    cache_dir = f"/root/.cache/torchwright_doom/compiled/{cache_subdir}"
    onnx_path = f"{cache_dir}/model.onnx"
    save_dir = f"/artifacts/hf/{cache_subdir}"

    export = hf_export.export_bundle(onnx_path, save_dir, config)
    RENDER_VOLUME.commit()

    from pathlib import Path as _P

    config_json = (_P(save_dir) / "config.json").read_text()
    files = sorted(p.name for p in _P(save_dir).glob("*") if p.is_file())
    return {
        "export": export,
        "config_json": config_json,
        "bundle_files": files,
        "remote_bundle": save_dir,
    }


@app.local_entrypoint()
def hf_export(
    config: str = "configs/e1m1.yaml",
    out_dir: str = "out/hf_export",
    verbose_compile: bool = False,
):
    """``make hf-export`` — convert the compiled DOOM artifact to a native HF
    bundle on Modal (the Hub publish path).

    Compiles on a cache miss (same 64-CPU path as ``make compile``), then runs
    the GPU export. The 28 GB safetensors bundle stays in RENDER_VOLUME;
    ``config.json`` + the report mirror to ``out_dir`` locally. Correctness is
    gated separately by ``make hf-render`` (the full render vs pydoom).
    """
    config_path = Path(config)
    config_text = config_path.read_text()
    cache_subdir = _compile_on_modal(config_path, verbose_compile)

    print(f"[local] launching hf_export_remote cache_subdir={cache_subdir}")
    result = hf_export_remote.remote(config_path.name, config_text, cache_subdir)

    local_dir = _HERE / out_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "config.json").write_text(result["config_json"])
    import json as _json

    (local_dir / "report.json").write_text(
        _json.dumps({"export": result["export"]}, indent=2)
    )
    print("\n=== HF export ===")
    export = result["export"]
    print(f"  vocab_size: {export['vocab_size']}  n_layers: {export['n_layers']}")
    print(
        f"  bos_token_id: {export['bos_token_id']}  eos_token_id: {export['eos_token_id']}"
    )
    print(
        f"\n[local] bundle files (remote {result['remote_bundle']}): {result['bundle_files']}"
    )
    print(f"[local] config.json + report.json -> {local_dir}")


@app.local_entrypoint()
def main(
    # Run-knob defaults live in the config's ``run:`` section — run_config
    # resolves None there, so this entrypoint must NOT restate them.
    config: str = "configs/e1m1.yaml",
    x: float | None = None,
    y: float | None = None,
    angle: int | None = None,
    viewz: float | None = None,
    out_dir: str = "out/render",
    run_name: str = "",
    max_positions: int | None = None,
    prefill_chunk_size: int | None = None,
    progress_every: int = 250,
    png: bool = False,
    compare: bool = False,
    png_zoom: int = 8,
    verbose_compile: bool = False,
):
    config_path = Path(config)
    # Shipped by VALUE (see _write_shipped_config): /tmp variant configs
    # work on Modal, and the legs of an A/B differ only in the file text.
    config_text = config_path.read_text()

    # Compile on Modal (64-CPU CP-SAT) if the cache misses — the SAME job
    # ``make compile`` (the ``compile_only`` entrypoint) runs, so the render
    # below is a cache hit whenever it was pre-compiled.
    cache_subdir = _compile_on_modal(config_path, verbose_compile)

    # Resolve the run section for the run id + pose defaults below.
    from torchwright_doom.inference.config import load_render_config

    render_config = load_render_config(config_path)

    # Display-only resolution for the run id (run_config re-resolves the
    # real values remotely from the same flag > config.run order).
    run = render_config.run
    rid_x = run.pose.x if x is None else x
    rid_y = run.pose.y if y is None else y
    rid_angle = run.pose.angle if angle is None else angle
    run_id = (
        run_name
        or f"{config_path.stem}__x{rid_x:g}_y{rid_y:g}_a{rid_angle}"
        f"__{int(time.time())}"
    )
    kwargs = dict(
        x=x,
        y=y,
        angle=angle,
        viewz=viewz,
        max_positions=max_positions,
        prefill_chunk_size=prefill_chunk_size,
        progress_every=progress_every,
        png=png,
        compare_images=compare,
        png_zoom=png_zoom,
        verbose_compile=verbose_compile,
    )
    print(f"[local] launching render_remote run_id={run_id} kwargs={kwargs}")
    result = render_remote.remote(
        run_id, kwargs, cache_subdir, config_path.name, config_text
    )

    local_dir = _HERE / out_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        (local_dir / name).write_bytes(data)
    if result["summary"].get("report_text"):
        print("\n" + result["summary"]["report_text"])
    print(f"\n[local] artifacts -> {local_dir}")
    print(f"[local] files: {sorted(result['files'])}")
    print(f"[local] remote volume path: /artifacts/{run_id}")
