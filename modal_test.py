"""Run pytest on Modal with GPU access.

The full suite is split across parallel containers (one A100, cpu=8
each); wall-clock is the slowest shard. Heavy files get their own
container via ``_HEAVY_FILES``; mid-weight files are batched in
``_MEDIUM_FILE_GROUPS``; anything unlisted falls into the catch-all
shard automatically.

The wall-clock floor is the full-frame flat-pass oracle
(``test_flat_pixel_oracle.py``): its ``reference_eval`` is a CPU-bound,
O(n_pos^2) attention oracle that runs ~190-260s and is dominated by
Modal host-to-host variance (NOT by the cpu reservation — uncapped
torch already bursts onto many host cores, so reserving 16/32 cores
does not help; capping OMP/MKL threads to 8, however, slows it to
~370s, so leave threads uncapped). Pushing below that floor reliably
needs ``reference_eval`` itself to run on the GPU — a torchwright-side
change, since op weights/literals are constructed on CPU.

Usage (via Makefile):
    make test                    # full suite, all shards in parallel
    make test FILE=tests/foo.py  # single container, single file
    make test ARGS="-k test_foo" # filter applied to all shards
"""

import shlex
import subprocess
import sys
import time

import modal

from modal_image import TEST_IMAGE

app = modal.App("torchwright-doom-test", image=TEST_IMAGE)

# ── Shard definitions ─────────────────────────────────────────────
# Simple file-level sharding.  Heavy compiled-test files get their
# own container; everything else is batched together.
# New test files are caught by the catch-all shard automatically.
#
# Both lists are empty: the whole suite fits one container today.
# When a heavy compiled-test file needs its own container, list it in
# _HEAVY_FILES; medium-weight groups go in _MEDIUM_FILE_GROUPS.
# Anything not listed is picked up by the catch-all shard at the
# bottom of SHARDS.

_HEAVY_FILES: list[str] = [
    # The full-frame flat-pass oracle: the single heaviest gate (a whole-frame
    # reference_eval through DONE, ~190-260s). The suite's wall-clock floor.
    "tests/scene/test_flat_pixel_oracle.py",
    # Free-running compiled AR rollout — in-process compile_headless, ~90s.
    "tests/scene/test_forward_ar_rollout.py",
]

_MEDIUM_FILE_GROUPS: list[list[str]] = [
    # Whole-forward ONNX compile + a couple of light oracle/layout gates.
    [
        "tests/scene/test_forward_compiles.py",
        "tests/scene/test_radix_successor_oracle.py",
        "tests/embedding/test_shared_slot_layout.py",
    ],
    # The emit-graph correctness gate. The wall/projection/traversal oracle
    # gates that used to share these shards were folded into the pydoom
    # whole-frame routing gate (test_flat_pixel_oracle) and deleted with the
    # sandbox.
    [
        "tests/embedding/test_emit_graph_correctness.py",
    ],
]

_ALL_NAMED_FILES = _HEAVY_FILES + [f for g in _MEDIUM_FILE_GROUPS for f in g]

SHARDS = [
    *_HEAVY_FILES,
    *(" ".join(group) for group in _MEDIUM_FILE_GROUPS),
    "tests "
    + (" ".join(f"--ignore={f}" for f in _ALL_NAMED_FILES) if _ALL_NAMED_FILES else ""),
]


# ── Remote function ───────────────────────────────────────────────


# timeout: the cross-submodule oracle gates (reference_eval over the full
# forward graph, O(n_pos^2)) push a shard well past the old 30-minute budget.
#
# Leave torch's thread pool UNCAPPED: in a Modal container torch bursts onto the
# host's cores, and capping OMP/MKL threads to the reservation measurably slows
# the reference_eval GEMMs (~190s -> ~370s on the flat gate). Raising the cpu
# reservation past 8 does NOT help — bursting already uses what the host has.
@app.function(gpu="a100-80gb", cpu=8, memory=32768, timeout=3600)
def run_pytest(pytest_args: str, shard_id: int = 0, extra_args: str = "") -> int:
    tag = f"[shard {shard_id}]"
    t0 = time.time()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *shlex.split(pytest_args),
        "-v",
        "--tb=short",
        "--no-header",
        "--durations=0",
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    print(f"{tag} {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"{tag} {line}", end="")
    proc.wait()
    elapsed = time.time() - t0
    print(f"\n{tag} finished in {elapsed:.0f}s (exit {proc.returncode})")
    return proc.returncode


# ── Entrypoint ────────────────────────────────────────────────────


@app.local_entrypoint()
def main(file: str = "tests", args: str = ""):
    if file != "tests":
        rc = run_pytest.remote(pytest_args=file, shard_id=0, extra_args=args)
        sys.exit(rc)

    shards = SHARDS
    print(f"Running {len(shards)} shards in parallel:")
    for i, s in enumerate(shards):
        label = s[:90] + "…" if len(s) > 90 else s
        print(f"  shard {i}: {label}")

    t0 = time.time()
    results = list(
        run_pytest.map(shards, range(len(shards)), kwargs={"extra_args": args})
    )
    elapsed = time.time() - t0

    failed = sum(1 for rc in results if rc != 0)
    print(f"\n{'=' * 60}")
    print(f"All shards finished in {elapsed:.0f}s")
    for i, rc in enumerate(results):
        status = "PASS" if rc == 0 else "FAIL"
        print(f"  shard {i}: {status} (rc={rc})")
    if failed:
        print(f"{failed}/{len(results)} shards failed")
    print(f"{'=' * 60}")

    sys.exit(1 if failed else 0)
