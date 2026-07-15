"""Run an arbitrary Python module or script on Modal.

Usage (via Makefile):
    make modal-run MODULE=scripts.compile_report
    make modal-run MODULE=scripts.foo ARGS="--flag x"
    make modal-run SCRIPT=path/to/one_shot.py ARGS="..."
    make modal-run MODULE=scripts.cpu_only_thing CPU_ONLY=1

Direct:
    uv run modal run modal_run.py --module scripts.compile_report

The GPU container mounts the compile cache volume (read-mostly) plus
``configs/`` and the WAD, so the artifact-debugging scripts run against the real
diagnostic ONNX — the cache key must be computed LOCALLY (it embeds git SHAs the
container can't derive) and handed over explicitly:

    make modal-run MODULE=scripts.compile_report \\
        ARGS="--config configs/e1m1.yaml --cache-dir /root/.cache/torchwright_doom/onnx_debug/<key>"

The CPU-only container is sized via env vars read at (local) import time —
they must be in the environment, not make variables:

    MODAL_RUN_CPU=64 MODAL_RUN_MEMORY=65536 MODAL_RUN_TIMEOUT=7200 \\
        make modal-run MODULE=scripts.analyze_forward_cost CPU_ONLY=1
"""

import os
import shlex
import subprocess
import sys
import time

import modal

from modal_image import (
    HF_BUNDLE_VOLUME,
    IMAGE,
    ONNX_DEBUG_VOLUME,
    ONNX_DIAGNOSTIC_IMAGE,
)

# Per-run render artifacts (modal_render.py's RENDER_VOLUME): reference
# token streams for bundle-debugging scripts.  from_name resolves the same
# durable volume; modal dedupes by name.
RENDER_ARTIFACTS_VOLUME = modal.Volume.from_name(
    "torchwright-doom-render", create_if_missing=True
)

app = modal.App("torchwright-doom-run", image=IMAGE)

_CPU = int(os.environ.get("MODAL_RUN_CPU", "4"))
_MEMORY = int(os.environ.get("MODAL_RUN_MEMORY", "8192"))
_TIMEOUT = int(os.environ.get("MODAL_RUN_TIMEOUT", "1800"))


def _build_cmd(module: str, script: str, args: str) -> list[str]:
    cmd = [sys.executable]
    if module:
        cmd += ["-m", module]
    else:
        cmd.append(script)
    if args:
        cmd += shlex.split(args)
    return cmd


# Local env vars forwarded verbatim into the remote process. The screen/graph
# shape vars are load-bearing for every probe that rebuilds the production
# graph (the screen-env trap: without them a probe silently measures the
# 60x50 hud-off graph — see scripts/critical_path_extract.py).
_FORWARD_ENV_PREFIXES = ("TORCHWRIGHT_DOOM_",)
_FORWARD_ENV_NAMES = ("CRIT_PATH", "OPT_GRAPH", "FUSE_FILTER")


def _forwarded_env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(_FORWARD_ENV_PREFIXES) or k in _FORWARD_ENV_NAMES
    }
    return env


@app.function(
    # GPU / memory / timeout are env-overridable (like the CPU container): e.g.
    #   MODAL_RUN_GPU=B200 MODAL_RUN_TIMEOUT=7200 MODAL_RUN_GPU_MEMORY=65536 \
    #       make modal-run MODULE=... for a heavy full-frame oracle eval that
    # needs more VRAM and wall time than the default A100.
    gpu=os.environ.get("MODAL_RUN_GPU", "a100-80gb"),
    cpu=8,
    memory=int(os.environ.get("MODAL_RUN_GPU_MEMORY", "32768")),
    timeout=_TIMEOUT,
    # The configs-augmented image: the artifact-debugging scripts need the
    # committed configs + WAD next to the mounted compile cache.  Production
    # HF bundles and per-run render artifacts mount read-alongside at their
    # production paths so bundle-debugging scripts (teacher forcing, logit
    # probes) can load the exact published model and reference streams.
    image=ONNX_DIAGNOSTIC_IMAGE,
    volumes={
        "/root/.cache/torchwright_doom/onnx_debug": ONNX_DEBUG_VOLUME,
        "/root/.cache/torchwright_doom/hf_phi3": HF_BUNDLE_VOLUME,
        "/artifacts": RENDER_ARTIFACTS_VOLUME,
    },
)
def run_gpu(module: str, script: str, args: str, env: dict[str, str]) -> int:
    ONNX_DEBUG_VOLUME.reload()
    HF_BUNDLE_VOLUME.reload()
    RENDER_ARTIFACTS_VOLUME.reload()
    cmd = _build_cmd(module, script, args)
    if env:
        print(
            f"[remote/gpu] env {' '.join(f'{k}={v}' for k, v in sorted(env.items()))}"
        )
    print(f"[remote/gpu] {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, env={**os.environ, **env}).returncode
    print(f"[remote/gpu] exit {rc} in {time.time() - t0:.0f}s")
    return rc


@app.function(
    cpu=_CPU,
    memory=_MEMORY,
    timeout=_TIMEOUT,
    # Same configs-augmented image as run_gpu: CPU scripts that rebuild the
    # production graph load configs/e1m1.yaml from /root/configs.
    image=ONNX_DIAGNOSTIC_IMAGE,
)
def run_cpu(module: str, script: str, args: str, env: dict[str, str]) -> int:
    cmd = _build_cmd(module, script, args)
    if env:
        print(
            f"[remote/cpu] env {' '.join(f'{k}={v}' for k, v in sorted(env.items()))}"
        )
    print(f"[remote/cpu] {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, env={**os.environ, **env}).returncode
    print(f"[remote/cpu] exit {rc} in {time.time() - t0:.0f}s")
    return rc


@app.local_entrypoint()
def main(
    module: str = "",
    script: str = "",
    args: str = "",
    cpu_only: bool = False,
):
    if not module and not script:
        print(
            "error: pass --module <dotted.name> or --script <path>",
            file=sys.stderr,
        )
        sys.exit(2)
    if module and script:
        print("error: pass --module OR --script, not both", file=sys.stderr)
        sys.exit(2)
    fn = run_cpu if cpu_only else run_gpu
    sys.exit(fn.remote(module=module, script=script, args=args, env=_forwarded_env()))
