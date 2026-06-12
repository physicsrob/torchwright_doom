"""Run pytest on Modal with GPU access.

The full suite currently runs as a single catch-all shard (one A100
container).  The shard tables below exist so heavy compiled-test files
can be split into their own containers when the suite needs it — both
lists are empty today, so listing a file there is how you opt in.

Usage (via Makefile):
    make test                    # full suite (one container today)
    make test FILE=tests/foo.py  # single container, single file
    make test ARGS="-k test_foo" # filter applied to all shards
"""

import os
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

_HEAVY_FILES: list[str] = []

_MEDIUM_FILE_GROUPS: list[list[str]] = []

_ALL_NAMED_FILES = _HEAVY_FILES + [f for g in _MEDIUM_FILE_GROUPS for f in g]

SHARDS = [
    *_HEAVY_FILES,
    *(" ".join(group) for group in _MEDIUM_FILE_GROUPS),
    "tests "
    + (" ".join(f"--ignore={f}" for f in _ALL_NAMED_FILES) if _ALL_NAMED_FILES else ""),
]


# ── Remote function ───────────────────────────────────────────────


# timeout: the cross-submodule oracle gates (reference_eval over the full
# forward graph, O(n_pos^2)) push the catch-all shard well past the old
# 30-minute budget.
@app.function(gpu="a100-80gb", cpu=8, memory=32768, timeout=3600)
def run_pytest(pytest_args: str, shard_id: int = 0, extra_args: str = "") -> int:
    tag = f"[shard {shard_id}]"
    t0 = time.time()
    # The oracle gates must RUN here, not skip: tests/sandbox_support.py
    # fails loud (and tests/test_sandbox_gates_guard.py goes red) if the
    # doom_sandbox sibling shipped in TEST_IMAGE stops being importable.
    os.environ["TWDOOM_REQUIRE_SANDBOX_GATES"] = "1"
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
