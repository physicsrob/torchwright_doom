"""Run an arbitrary Python module or script on Modal.

Usage (via Makefile):
    make modal-run MODULE=scripts.compile_report
    make modal-run MODULE=scripts.foo ARGS="--flag x"
    make modal-run SCRIPT=path/to/one_shot.py ARGS="..."
    make modal-run MODULE=scripts.cpu_only_thing CPU_ONLY=1

Direct:
    uv run modal run modal_run.py --module scripts.compile_report

The GPU container mounts the compile cache volume (read-mostly) plus the
``doom_sandbox`` checkout and ``configs/``, so the artifact-debugging scripts
run against the real compiled ONNX — the cache key must be computed LOCALLY
(it embeds git SHAs the container can't derive) and handed over explicitly:

    make modal-run MODULE=scripts.k_localize_divergence \\
        ARGS="--config configs/e1m1.yaml --cache-dir /root/.cache/torchwright_doom/compiled/<key>"

The CPU-only container is sized via env vars read at (local) import time —
they must be in the environment, not make variables:

    MODAL_RUN_CPU=64 MODAL_RUN_MEMORY=65536 MODAL_RUN_TIMEOUT=7200 \\
        make modal-run MODULE=scripts.cpsat_space_experiments CPU_ONLY=1
"""

import os
import shlex
import subprocess
import sys
import time

import modal

from modal_image import ASSETS_IMAGE, CACHE_VOLUME, IMAGE

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


@app.function(
    gpu="a100-80gb",
    cpu=8,
    memory=32768,
    timeout=1800,
    # The sandbox/configs-augmented image: the artifact-debugging scripts
    # (k_probe/k_localize) need the reference renderer and the committed
    # configs next to the mounted compile cache.
    image=ASSETS_IMAGE,
    volumes={"/root/.cache/torchwright_doom/compiled": CACHE_VOLUME},
)
def run_gpu(module: str, script: str, args: str) -> int:
    CACHE_VOLUME.reload()
    cmd = _build_cmd(module, script, args)
    print(f"[remote/gpu] {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"[remote/gpu] exit {rc} in {time.time() - t0:.0f}s")
    return rc


@app.function(cpu=_CPU, memory=_MEMORY, timeout=_TIMEOUT)
def run_cpu(module: str, script: str, args: str) -> int:
    cmd = _build_cmd(module, script, args)
    print(f"[remote/cpu] {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
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
    sys.exit(fn.remote(module=module, script=script, args=args))
