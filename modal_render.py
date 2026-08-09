"""Compile + render on Modal, with artifact sync-back.

    uv run modal run modal_render.py --config configs/e1m1.yaml
    uv run modal run modal_render.py --config configs/e1m1.yaml --png --compare

Two remote stages: ``compile_remote`` (CPU-only, 64 cores so CP-SAT's parallel
search gets real width; writes a complete direct-HF bundle into its volume) and
``render_remote`` (a big GPU — the dense native HF checkpoint is ~86 GB fp32
plus the generation cache over the full frame). The render executes the
artifact's isolated stock-Transformers bundle-root ``infer.py``. The compile
cache key is computed on the LOCAL machine because it embeds submodule git SHAs
that don't exist inside a Modal container.
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

import json
import os as _os
import time
from pathlib import Path

import modal

from modal_image import ASSETS_IMAGE, HF_BUNDLE_VOLUME, STOCK_HF_IMAGE

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
# here must match. B200 is the default: its 192 GB HBM fits the ~86 GB dense
# fp32 model plus the growing stock generation cache, and its high bandwidth
# makes autoregressive decode fast.
_RENDER_GPU = _os.environ.get("RENDER_GPU", "b200")


@app.function(
    cpu=1,
    volumes={"/bundle-volume": HF_BUNDLE_VOLUME},
)
def _volume_publication_probe_write(token: str) -> str:
    """Exercise the exact non-empty-directory replacement used by publication."""
    import os
    import shutil
    from pathlib import Path

    HF_BUNDLE_VOLUME.reload()
    root = Path("/bundle-volume/.publication-probe") / token
    current, staging, backup = root / "current", root / "staging", root / "backup"
    root.mkdir(parents=True)
    current.mkdir()
    staging.mkdir()
    (current / "identity").write_text("old")
    (staging / "identity").write_text("new")
    os.replace(current, backup)
    try:
        os.replace(staging, current)
        if (current / "identity").read_text() != "new":
            raise RuntimeError("directory replacement did not expose staged content")
        # Simulate rollback of the newly published directory.
        os.replace(current, staging)
        os.replace(backup, current)
        if (current / "identity").read_text() != "old":
            raise RuntimeError("directory rollback did not restore prior content")
        shutil.rmtree(staging)
        (root / "committed").write_text("visible")
        HF_BUNDLE_VOLUME.commit()
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return str(root)


@app.function(
    cpu=1,
    volumes={"/bundle-volume": HF_BUNDLE_VOLUME},
)
def _volume_publication_probe_read(token: str) -> None:
    import shutil
    from pathlib import Path

    HF_BUNDLE_VOLUME.reload()
    root = Path("/bundle-volume/.publication-probe") / token
    if (root / "committed").read_text() != "visible":
        raise RuntimeError("committed publication probe is not visible after reload")
    shutil.rmtree(root)
    HF_BUNDLE_VOLUME.commit()


@app.local_entrypoint()
def probe_volume_publication():
    """One-off production-volume rename/rollback/commit/reload gate."""
    import uuid

    token = uuid.uuid4().hex
    _volume_publication_probe_write.remote(token)
    _volume_publication_probe_read.remote(token)
    print("Modal HF bundle volume publication semantics: PASS")


@app.function(
    cpu=_COMPILE_CPUS,
    # Direct bundle compilation streams each completed layer to its final shard.
    memory=262144,
    timeout=10800,
    volumes={
        "/root/.cache/torchwright_doom/hf_phi3": HF_BUNDLE_VOLUME,
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
    solver_seed: int | None = None,
    force_resolve: bool = False,
    solver_workers: int | None = None,
) -> dict:
    import os
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    config_path = _write_shipped_config(config_name, config_text)

    # CP-SAT reads this at solve time (torchwright cpsat_scheduler). Production
    # normally uses the full allocation; release draws may explicitly retain
    # a narrower worker portfolio, which is recorded in the bundle manifest.
    effective_solver_workers = (
        _COMPILE_CPUS if solver_workers is None else solver_workers
    )
    if not 1 <= effective_solver_workers <= _COMPILE_CPUS:
        raise ValueError(
            f"solver_workers must be between 1 and {_COMPILE_CPUS}, "
            f"got {effective_solver_workers}"
        )
    os.environ["TW_CPSAT_WORKERS"] = str(effective_solver_workers)
    if disable_cache:
        # DISABLE_CACHE: neither durable cache is read or written.  Both the
        # HF bundle and the schedule entry go to a per-call scratch dir
        # (mkdtemp, not a fixed path: a warm container must not hand a later
        # no-cache run an earlier one's artifact or schedule as a hit), and
        # the volume reload/commit pairs are skipped — the mounts stay
        # untouched.
        import tempfile

        scratch = tempfile.mkdtemp(prefix="compile-nocache-", dir="/tmp")
        os.environ["TW_SCHEDULE_CACHE_DIR"] = f"{scratch}/schedules"
        cache_dir = f"{scratch}/hf_phi3/{cache_subdir}"
        print(f"[compile] [nocache] volumes bypassed; scratch={scratch}", flush=True)
    else:
        # Durable schedule cache (see SCHEDULE_VOLUME above).
        os.environ["TW_SCHEDULE_CACHE_DIR"] = "/root/.cache/torchwright_doom/schedules"
        # The cache key + payload come from the LOCAL machine: this container
        # has no ``.git``, so deriving them here would collapse the git SHAs
        # to "unknown" and silently reuse a stale model across code changes.
        HF_BUNDLE_VOLUME.reload()
        SCHEDULE_VOLUME.reload()
        cache_dir = f"/root/.cache/torchwright_doom/hf_phi3/{cache_subdir}"

    from torchwright_doom.bundle.build import compile_config

    result = compile_config(
        config_path=config_path,
        verbose_compile=verbose_compile,
        cache_dir=cache_dir,
        compile_payload=compile_payload,
        solver_seed=solver_seed,
        force_resolve=force_resolve,
        solver_workers=effective_solver_workers,
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
        HF_BUNDLE_VOLUME.commit()
        SCHEDULE_VOLUME.commit()
    return result


@app.function(
    cpu=_COMPILE_CPUS,
    memory=262144,
    timeout=10800,
)
def consumer_schedule_draw_remote(
    config_name: str,
    config_text: str,
    solver_seed: int,
    solver_workers: int,
    target_layers: int,
) -> dict:
    """Take one target-bounded consumer schedule draw and return it by value.

    The lower bound is search-only: the selected assignment is serialized in
    the clean compiler's ordinary cache format and later passes its unmodified
    directed-replay validation. This lets a release select an exact narrative
    depth without hand-editing or padding an assignment.
    """
    import importlib
    import os
    import sys
    from pathlib import Path as _P

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    if not 1 <= solver_workers <= _COMPILE_CPUS:
        raise ValueError(
            f"solver_workers must be between 1 and {_COMPILE_CPUS}, "
            f"got {solver_workers}"
        )

    os.environ.pop("TW_SCHEDULE_CACHE_DIR", None)
    config_path = _write_shipped_config(config_name, config_text)

    from scripts.consumer_profile import _profile

    scheduler = importlib.import_module("torchwright.compiler.forward.cpsat_scheduler")
    original_build = getattr(scheduler, "build_cpsat_model")

    def build_with_target_floor(*args, **kwargs):
        built = original_build(*args, **kwargs)
        built.model.add(built.n_layers_var >= target_layers)
        return built

    setattr(scheduler, "build_cpsat_model", build_with_target_floor)
    try:
        report = _profile(
            _P(config_path),
            solver_seed=solver_seed,
            solver_workers=solver_workers,
            force_resolve=True,
            export_schedule=True,
        )
    finally:
        setattr(scheduler, "build_cpsat_model", original_build)
    exported = report.pop("schedule_export")
    schedule_payload = json.loads(exported["schedule_text"])
    meta = schedule_payload["meta"]
    constrained_result = {
        "status_name": meta.get("status_name"),
        "best_objective_bound": meta.get("best_objective_bound"),
        "is_optimal": meta.get("is_optimal"),
    }
    meta["selection_constraint"] = {"minimum_n_layers": target_layers}
    meta["constrained_solver_result"] = constrained_result
    # The proof applies only to the explicitly bounded search. The assignment
    # remains sound for ordinary replay, but must not claim global optimality
    # in the unconstrained production cache.
    meta["status_name"] = "FEASIBLE"
    meta["best_objective_bound"] = -1
    meta["is_optimal"] = False
    meta["selected"]["is_optimal"] = False
    meta["solver_attempt"]["status_name"] = "FEASIBLE"
    meta["solver_attempt"]["best_objective_bound"] = -1
    meta["solver_attempt"]["is_optimal"] = False
    exported["schedule_text"] = json.dumps(schedule_payload)
    return {
        "filename": exported["filename"],
        "schedule_text": exported["schedule_text"],
        "report": report,
    }


@app.function(
    cpu=1,
    volumes={"/root/.cache/torchwright_doom/schedules": SCHEDULE_VOLUME},
)
def install_consumer_schedule_remote(filename: str, schedule_text: str) -> dict:
    """Ratchet one selected schedule into the durable cache."""
    import json
    from pathlib import Path as _P

    if _P(filename).name != filename or not filename.endswith(".json"):
        raise ValueError(f"invalid schedule filename: {filename!r}")
    candidate = json.loads(schedule_text)
    candidate_layers = int(candidate["n_layers"])
    cache_dir = _P("/root/.cache/torchwright_doom/schedules")
    SCHEDULE_VOLUME.reload()
    dest = cache_dir / filename
    prior_layers = None
    if dest.exists():
        prior_layers = int(json.loads(dest.read_text(encoding="utf-8"))["n_layers"])
        if prior_layers <= candidate_layers:
            return {
                "installed": False,
                "n_layers": prior_layers,
                "candidate_layers": candidate_layers,
                "filename": filename,
            }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(schedule_text, encoding="utf-8")
    SCHEDULE_VOLUME.commit()
    return {
        "installed": True,
        "n_layers": candidate_layers,
        "prior_layers": prior_layers,
        "filename": filename,
    }


@app.local_entrypoint()
def search_consumer_schedule(
    config: str = "configs/e1m1_lowres.yaml",
    draws: int = 4,
    solver_seed: int = 3,
    solver_workers: int = 16,
    target_layers: int = 70,
):
    """Run independent solve-only draws, then durably install only the best."""
    if draws < 1:
        raise ValueError("draws must be positive")
    config_path = Path(config)
    config_text = config_path.read_text()
    calls = [
        consumer_schedule_draw_remote.spawn(
            config_path.name,
            config_text,
            solver_seed,
            solver_workers,
            target_layers,
        )
        for _ in range(draws)
    ]
    results = [call.get() for call in calls]
    for index, result in enumerate(results, 1):
        n_layers = result["report"]["compile"]["n_layers"]
        print(f"[schedule-search] draw {index}/{draws}: {n_layers} layers")
    filenames = {result["filename"] for result in results}
    if len(filenames) != 1:
        raise RuntimeError(
            f"schedule draws produced different fingerprints: {sorted(filenames)}"
        )
    for result in results:
        schedule_layers = int(json.loads(result["schedule_text"])["n_layers"])
        report_layers = int(result["report"]["compile"]["n_layers"])
        if schedule_layers != report_layers:
            raise RuntimeError(
                "schedule draw report disagrees with its serialized assignment"
            )
    exact = [
        result
        for result in results
        if int(result["report"]["compile"]["n_layers"]) == target_layers
    ]
    best = min(results, key=lambda row: row["report"]["compile"]["n_layers"])
    best_layers = int(best["report"]["compile"]["n_layers"])
    print(f"[schedule-search] best draw: {best_layers} layers")
    if exact:
        selected = exact[0]
    elif best_layers > target_layers:
        # A better-but-still-high incumbent helps the next search. Never
        # install a below-target draw: the cache ratchet would then reject the
        # narratively selected target as a regression.
        selected = best
    else:
        raise RuntimeError(
            f"best schedule used {best_layers} layers, below the exact "
            f"{target_layers}-layer release target; durable cache unchanged"
        )
    installed = install_consumer_schedule_remote.remote(
        selected["filename"], selected["schedule_text"]
    )
    print(f"[schedule-search] durable cache: {json.dumps(installed, sort_keys=True)}")
    selected_layers = int(selected["report"]["compile"]["n_layers"])
    if selected_layers != target_layers:
        raise RuntimeError(
            f"best schedule used {best_layers} layers; exact target is "
            f"{target_layers}"
        )


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


@app.function(
    cpu=1,
    volumes={"/root/.cache/torchwright_doom/hf_phi3": HF_BUNDLE_VOLUME},
)
def _bundle_cache_probe_remote(cache_subdir: str, compile_payload: dict) -> bool:
    """Parse the manifest and structural index before declaring a cache hit."""
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from torchwright_doom.bundle.manifest import is_complete_hf_bundle

    HF_BUNDLE_VOLUME.reload()
    directory = f"/root/.cache/torchwright_doom/hf_phi3/{cache_subdir}"
    return is_complete_hf_bundle(directory, expected_payload=compile_payload)


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
    config_path: Path,
    verbose_compile: bool,
    disable_cache: bool = False,
    solver_seed: int | None = None,
    force_resolve: bool = False,
    spawn: bool = False,
    solver_workers: int | None = None,
) -> str:
    """Compute the compile-cache key LOCALLY and compile on Modal on a miss.

    Shared by the render entrypoint (``main``) and the compile-only
    entrypoint (``compile_only``), so ``make compile`` and the implicit
    compile inside ``make run`` are byte-for-byte the same job: the same
    64-CPU ``compile_remote`` container, the same key, the same artifact in
    HF_BUNDLE_VOLUME.

    The key embeds the submodule git SHAs, and Modal containers have no
    ``.git`` (``_git_sha`` collapses to "unknown" there, so a remotely-derived
    key would never change on a code edit — silently reusing a stale model).
    The local machine computes the canonical payload + key and hands both to
    ``compile_remote``, which compiles straight into HF_BUNDLE_VOLUME under that
    key with CP-SAT fanned out across the container's 64 CPUs.

    Importing the root ``config`` / ``identity`` modules here is safe: they
    have no dependency on the screen-sized token vocab (the import-order trap
    that forces ``compile_config`` to call ``apply_screen_env`` before
    touching graph-reaching modules).

    ``disable_cache`` (the ``DISABLE_CACHE=1 make compile`` path) skips the
    HIT probe (a volume read), compiles into container-local scratch instead
    of HF_BUNDLE_VOLUME, and mirrors the sampled schedule to local /tmp — the
    durable caches are neither read nor written.

    Returns the cache subdir (the volume-relative key).
    """
    from torchwright_doom.config import load_render_config, resolve_wad_path
    from torchwright_doom.identity import (
        cache_key_from_payload,
        canonical_compile_payload,
    )

    # Shipped by VALUE (see _write_shipped_config): /tmp variant configs work
    # on Modal, and the legs of an A/B differ only in the file text.
    config_text = config_path.read_text()
    render_config = load_render_config(config_path)
    wad_path = resolve_wad_path(render_config, base_dir=config_path.parent)
    compile_payload = canonical_compile_payload(render_config, wad_path)
    cache_subdir = cache_key_from_payload(compile_payload)

    if disable_cache and spawn:
        raise ValueError("spawned compile is incompatible with --disable-cache")
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
            solver_seed,
            force_resolve,
            solver_workers,
        )
        _save_nocache_schedules(result)
    elif not force_resolve and _bundle_cache_probe_remote.remote(
        cache_subdir, compile_payload
    ):
        print(f"[local] compile cache HIT HF_BUNDLE_VOLUME:/{cache_subdir}", flush=True)
    else:
        print(
            f"[local] compile cache MISS — compiling {config_path} on Modal "
            f"({_COMPILE_CPUS} CPUs) -> HF_BUNDLE_VOLUME:/{cache_subdir}",
            flush=True,
        )
        compile_args = (
            config_path.name,
            config_text,
            cache_subdir,
            compile_payload,
            verbose_compile,
            False,
            solver_seed,
            force_resolve,
            solver_workers,
        )
        if spawn:
            call = compile_remote.spawn(*compile_args)
            print(
                f"[local] spawned compile function call {call.object_id} "
                f"for HF_BUNDLE_VOLUME:/{cache_subdir}",
                flush=True,
            )
        else:
            compile_remote.remote(*compile_args)
    return cache_subdir


@app.local_entrypoint()
def compile_only(
    config: str = "configs/e1m1.yaml",
    verbose_compile: bool = False,
    disable_cache: bool = False,
    solver_seed: int | None = None,
    force_resolve: bool = False,
    spawn: bool = False,
    solver_workers: int | None = None,
):
    """``make compile`` — compile a config to the Modal cache volume, no render.

    Runs the SAME 64-CPU ``compile_remote`` path ``make run`` uses on a cache
    miss, so the production artifact is built once with the wide CP-SAT search
    and a later ``make run`` is a cache hit. The complete Phi-3 bundle lives in
    HF_BUNDLE_VOLUME (durable), exactly where ``render_remote`` reads it.

    ``--spawn`` submits the compile without tying its lifetime to the local
    Modal client; use it with ``modal run --detach`` and inspect app logs.
    ``--disable-cache`` (``DISABLE_CACHE=1 make compile``) runs the same
    production compile but touches neither durable cache: no HIT probe, the
    bundle dies with the container, and the sampled schedule is mirrored to
    local /tmp instead of SCHEDULE_VOLUME (see _save_nocache_schedules).
    """
    config_path = Path(config)
    cache_subdir = _compile_on_modal(
        config_path,
        verbose_compile,
        disable_cache,
        solver_seed,
        force_resolve,
        spawn,
        solver_workers,
    )
    if disable_cache:
        print(
            "[local] [nocache] compile complete — HF_BUNDLE_VOLUME/SCHEDULE_VOLUME "
            "untouched (artifact discarded with the container)",
            flush=True,
        )
    elif spawn:
        print(
            f"[local] compile submitted -> HF_BUNDLE_VOLUME:/{cache_subdir}",
            flush=True,
        )
    else:
        print(
            f"[local] compile complete -> HF_BUNDLE_VOLUME:/{cache_subdir}",
            flush=True,
        )


@app.function(
    cpu=8,
    memory=32768,
    timeout=86400,
    volumes={"/root/.cache/torchwright_doom/hf_phi3": HF_BUNDLE_VOLUME},
    secrets=[modal.Secret.from_name("huggingface")],
)
def publish_private_remote(
    cache_subdir: str,
    compile_payload: dict,
    repo_id: str,
    expected_layers: int,
    expected_solver_seed: int,
    expected_solver_workers: int,
) -> dict:
    """Validate one cached release bundle and upload it to a private Hub repo."""
    import sys

    from huggingface_hub import HfApi

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from torchwright_doom.bundle.manifest import (
        MANIFEST_NAME,
        validate_bundle_manifest,
    )

    HF_BUNDLE_VOLUME.reload()
    bundle = Path("/root/.cache/torchwright_doom/hf_phi3") / cache_subdir
    manifest = validate_bundle_manifest(bundle, expected_payload=compile_payload)
    actual_layers = int(manifest["compile"]["n_layers"])
    actual_seed = manifest["compile"].get("solver_seed")
    actual_workers = manifest["compile"].get("solver_workers")
    if actual_layers != expected_layers:
        raise RuntimeError(
            f"refusing Hub upload: expected {expected_layers} layers, "
            f"bundle has {actual_layers}"
        )
    if actual_seed != expected_solver_seed:
        raise RuntimeError(
            f"refusing Hub upload: expected solver seed {expected_solver_seed}, "
            f"bundle records {actual_seed!r}"
        )
    if actual_workers != expected_solver_workers:
        raise RuntimeError(
            f"refusing Hub upload: expected {expected_solver_workers} solver workers, "
            f"bundle records {actual_workers!r}"
        )

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo_id).private:
        raise RuntimeError(f"refusing upload because Hub repo is public: {repo_id}")

    commit = api.upload_folder(
        folder_path=str(bundle),
        repo_id=repo_id,
        repo_type="model",
        commit_message=(
            f"Publish E1M1 {actual_layers}-layer seed-{actual_seed} bundle"
        ),
        # If a retry emits fewer shards, do not leave stale model weights in
        # the private staging repo.
        delete_patterns=["model-*.safetensors"],
    )
    remote_files = set(api.list_repo_files(repo_id, repo_type="model"))
    expected_files = {MANIFEST_NAME, *manifest["files"]}
    missing = sorted(expected_files - remote_files)
    if missing:
        raise RuntimeError(f"Hub upload is missing files: {missing}")
    unexpected = sorted(remote_files - expected_files - {".gitattributes"})
    if unexpected:
        raise RuntimeError(f"Hub upload has unexpected files: {unexpected}")
    wad_files = sorted(name for name in remote_files if name.lower().endswith(".wad"))
    if wad_files:
        raise RuntimeError(
            f"refusing Hub publication containing WAD files: {wad_files}"
        )
    info = api.model_info(repo_id)
    if not info.private:
        raise RuntimeError(f"Hub repo became public during upload: {repo_id}")
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "private": True,
        "commit": str(commit.commit_url),
        "revision": commit.oid,
        "n_layers": actual_layers,
        "solver_seed": actual_seed,
        "solver_workers": actual_workers,
        "n_files": len(remote_files),
    }


@app.local_entrypoint()
def publish_private(
    repo_id: str,
    config: str = "configs/e1m1.yaml",
    solver_seed: int = 0,
    expected_layers: int = 38,
    solver_workers: int = _COMPILE_CPUS,
    verbose_compile: bool = False,
    force_resolve: bool = False,
):
    """Compile the selected release draw, validate it, and stage it privately."""
    from torchwright_doom.config import load_render_config, resolve_wad_path
    from torchwright_doom.identity import canonical_compile_payload

    if "/" not in repo_id:
        raise ValueError("repo_id must include its Hugging Face namespace")
    config_path = Path(config)
    cache_subdir = _compile_on_modal(
        config_path,
        verbose_compile,
        False,
        solver_seed,
        force_resolve,
        False,
        solver_workers,
    )
    render_config = load_render_config(config_path)
    wad_path = resolve_wad_path(render_config, base_dir=config_path.parent)
    compile_payload = canonical_compile_payload(render_config, wad_path)
    result = publish_private_remote.remote(
        cache_subdir,
        compile_payload,
        repo_id,
        expected_layers,
        solver_seed,
        solver_workers,
    )
    print("PRIVATE_HUB_PUBLICATION " + json.dumps(result, sort_keys=True), flush=True)


@app.function(
    gpu=_RENDER_GPU,
    cpu=8,
    memory=65536,
    timeout=5400,
    image=STOCK_HF_IMAGE,
    secrets=[modal.Secret.from_name("huggingface")],
)
def smoke_hub_remote(
    repo_id: str,
    revision: str,
    expected_layers: int,
    max_new_tokens: int = 1,
) -> dict:
    """Download a Hub revision into a clean container and run its infer.py."""
    import hashlib
    import importlib.util
    import subprocess
    import sys
    import tempfile

    from huggingface_hub import HfApi, snapshot_download

    leaked = [
        name
        for name in ("torchwright", "torchwright_doom")
        if importlib.util.find_spec(name) is not None
    ]
    if leaked:
        raise RuntimeError(f"Hub smoke image contains workspace packages: {leaked}")

    scratch = Path(tempfile.mkdtemp(prefix="hub-smoke-", dir="/tmp"))
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=scratch / "bundle",
        )
    )
    resolved_revision = HfApi().model_info(repo_id, revision=revision).sha
    manifest_path = snapshot / "doom_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("validation", {}).get("complete"):
        raise RuntimeError("Hub smoke downloaded an incomplete Doom bundle")
    for name, expected in manifest["files"].items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != int(expected["size"]):
            raise RuntimeError(f"Hub smoke file size mismatch: {name}")
        expected_sha = expected.get("sha256")
        if expected_sha:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_sha:
                raise RuntimeError(f"Hub smoke file digest mismatch: {name}")
    actual_layers = int(manifest["compile"]["n_layers"])
    if actual_layers != expected_layers:
        raise RuntimeError(
            f"Hub smoke expected {expected_layers} layers, found {actual_layers}"
        )
    output = scratch / "output"
    subprocess.run(
        [
            sys.executable,
            str(snapshot / "infer.py"),
            "--model",
            str(snapshot),
            "--prompt",
            str(snapshot / "examples/e1m1_prompt.txt"),
            "--output",
            str(output),
            "--device",
            "cuda",
            "--max-new-tokens",
            str(max_new_tokens),
        ],
        check=True,
    )
    payload = json.loads((output / "output.ids.json").read_text(encoding="utf-8"))
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "bundle_identity": manifest["bundle_identity"],
        "n_layers": actual_layers,
        "solver_seed": manifest["compile"].get("solver_seed"),
        "solver_workers": manifest["compile"].get("solver_workers"),
        "generated_rows": len(payload["emitted_row_ids"]),
        "termination_reason": payload["generation"]["termination_reason"],
        "timing_seconds": payload["timing_seconds"],
        "cuda_memory": payload["cuda_memory"],
        "workspace_imports_absent": True,
    }


@app.local_entrypoint()
def smoke_hub(
    repo_id: str,
    revision: str = "main",
    expected_layers: int = 38,
    max_new_tokens: int = 1,
):
    """Clean-room load and short pipeline generation for one Hub revision."""
    result = smoke_hub_remote.remote(
        repo_id,
        revision,
        expected_layers,
        max_new_tokens,
    )
    print("HUB_PIPELINE_SMOKE " + json.dumps(result, sort_keys=True), flush=True)


@app.function(
    gpu=_RENDER_GPU,
    cpu=8,
    memory=65536,
    timeout=5400,
    volumes={
        "/artifacts": RENDER_VOLUME,
        "/root/.cache/torchwright_doom/hf_phi3": HF_BUNDLE_VOLUME,
    },
)
def render_remote(
    run_id: str, kwargs: dict, cache_subdir: str, config_name: str, config_text: str
) -> dict:
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")  # make the /root-mounted packages importable

    import torch

    from torchwright_doom.run import run_config

    kwargs = {**kwargs, "config_path": _write_shipped_config(config_name, config_text)}

    # Compilation happens in ``compile_remote`` (a separate 64-CPU container),
    # which writes the validated bundle under ``cache_subdir``. Reload so
    # those files are visible, then point the renderer straight at them — this
    # container is render-only and never compiles.
    HF_BUNDLE_VOLUME.reload()
    cache_dir = f"/root/.cache/torchwright_doom/hf_phi3/{cache_subdir}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {**kwargs, "cache_dir": cache_dir, "device": device}

    out_dir = f"/artifacts/{run_id}"
    summary = run_config(out_dir=out_dir, **kwargs)
    RENDER_VOLUME.commit()

    files = {
        p.name: p.read_bytes() for p in sorted(Path(out_dir).glob("*")) if p.is_file()
    }
    return {"summary": summary, "files": files}


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
    max_new_tokens: int | None = None,
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
    from torchwright_doom.config import load_render_config

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
        max_new_tokens=max_new_tokens,
        png=png,
        compare_images=compare,
        png_zoom=png_zoom,
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
