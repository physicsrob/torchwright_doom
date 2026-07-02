"""Flagship forward divergence: FINAL block-IR model vs the ORIGINAL pre-refactor
chain-mined baseline.

Compiles the production e1m1 graph on two code states and reports the max/mean
absolute logit divergence and per-position argmax agreement over an identical
token prefill (the production token-level equivalence bar):

- **baseline** = the 2a tip (torchwright 77ea7e9 + torchwright_doom 6048410),
  whose ``linear_relu_linear`` builds ``Linear -> ReLU -> Linear`` chains and
  whose width-safe fusion is the original one -> the 64-layer chain-mined model.
- **HEAD** = native Blocks + block-aware fusion + chain machinery deleted -> the
  57-layer model.

The two states cannot coexist in one process (both packages are named
``torchwright``; the HEAD copy is PEP-660 editable-installed via a meta-path
finder that beats ``sys.path``).  So each side compiles+runs in its OWN
subprocess via :mod:`scripts._divergence_driver`:

- baseline: ``python -S`` (skips ``.pth`` processing, so the editable finder is
  never installed) with ``TW_SNAPSHOT_PATHS`` = the 2a worktrees and
  ``TW_VENV_SITE`` = the venv site-packages; the driver rebuilds ``sys.path`` so
  ``import torchwright`` resolves to the 2a snapshot.
- HEAD: launched normally.

Subprocessing also keeps only one ~d=8192 model resident at a time (the driver
saves its logits to disk; this orchestrator loads both and diffs).  Runs the
original Gate-C compile settings (optimize=0, assume_zero_init=True,
rms_norm_const_exp=63) and the same deterministic prefill fallback on both
sides, so the numbers are comparable to the earlier 1.22e-04 / 9-of-9 run.

Why not onnxruntime: at d=8192 the folded ``embed_table`` initializer is
~3.06 GB dense; ORT densifies sparse initializers past its 2 GiB limit.  The
driver mirrors production: convert the ONNX to HF weights and run under torch.

GPU + wall-time + host-RAM heavy (two flagship compiles) — run on Modal:
    make modal-run MODULE=scripts.block_forward_divergence \
        MODAL_RUN_GPU_MEMORY=131072 MODAL_RUN_TIMEOUT=10800

Snapshot dirs default to the Modal layout (/root/baseline_2a/...); override with
BASELINE_TW / BASELINE_TWD for a local run.  Config overrides mirror
block_equivalence_flagship.py (D, D_HEAD, D_ROT, D_HIDDEN, CONFIG, MAX_SEQ_LEN);
PREFILL sets the prefill string after <bos>.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# The 2a-tip commits the baseline snapshot must be checked out at (printed in the
# run header for self-certification; also verified against the driver's reported
# torchwright paths and layer counts).
BASELINE_TW_SHA = "77ea7e9"
BASELINE_TWD_SHA = "6048410"


def _driver_env(out_path: str, onnx_path: str) -> dict:
    env = dict(os.environ)
    env["OUT_PATH"] = out_path
    env["ONNX_PATH"] = onnx_path
    # Config knobs pass straight through (the driver reads the same names).
    return env


def _run_side(label: str, *, snapshot: bool, out_path: str, onnx_path: str) -> dict:
    """Run one compile+prefill side in a subprocess; return the saved dict."""
    import torch

    driver = str(_HERE / "_divergence_driver.py")
    env = _driver_env(out_path, onnx_path)
    cmd = [sys.executable]
    if snapshot:
        base_tw = os.environ.get("BASELINE_TW", "/root/baseline_2a/torchwright")
        base_twd = os.environ.get("BASELINE_TWD", "/root/baseline_2a/torchwright_doom")
        for p, what in ((base_tw, "torchwright"), (base_twd, "torchwright_doom")):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"baseline {what} snapshot not found at {p}; set BASELINE_TW/"
                    f"BASELINE_TWD or add_local_dir it into the Modal image."
                )
        # site-packages holding torch/onnx/ortools/... (the driver appends it
        # under -S, where site.py did not add it).
        venv_site = str(Path(torch.__file__).resolve().parents[1])
        env["TW_SNAPSHOT_PATHS"] = os.pathsep.join([base_tw, base_twd])
        env["TW_VENV_SITE"] = venv_site
        cmd.append("-S")  # skip .pth so the HEAD editable finder never installs
    else:
        env.pop("TW_SNAPSHOT_PATHS", None)
    cmd.append(driver)

    print(f"\n=== compiling {label} side (subprocess) ===", flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, env=env, check=True)
    print(
        f"[{label}] subprocess finished in {time.perf_counter() - t0:.1f}s", flush=True
    )
    return torch.load(out_path, weights_only=False)


def main() -> None:
    import torch

    prefill = os.environ.get("PREFILL", "e1m1")
    print("=" * 70, flush=True)
    print(
        "Flagship forward divergence: FINAL (block-IR) vs ORIGINAL (chain)", flush=True
    )
    print("=" * 70, flush=True)
    print(
        f"  baseline snapshot: torchwright@{BASELINE_TW_SHA} "
        f"torchwright_doom@{BASELINE_TWD_SHA} (chain-mined, 64-layer expected)",
        flush=True,
    )
    print(
        "  HEAD: native Blocks + block-aware fusion + chains deleted "
        "(57-layer expected)",
        flush=True,
    )
    print(
        f"  config: d={os.environ.get('D', '8192')} "
        f"d_head={os.environ.get('D_HEAD', '128')} prefill={prefill!r}",
        flush=True,
    )

    tmp = Path(tempfile.mkdtemp(prefix="fwd_div_"))
    baseline = _run_side(
        "baseline-chain",
        snapshot=True,
        out_path=str(tmp / "chain.pt"),
        onnx_path=str(tmp / "chain.onnx"),
    )
    head = _run_side(
        "head-block",
        snapshot=False,
        out_path=str(tmp / "block.pt"),
        onnx_path=str(tmp / "block.onnx"),
    )

    chain_logits = baseline["logits"]
    block_logits = head["logits"]
    print("\n" + "=" * 70, flush=True)
    print("Provenance (self-certifying):", flush=True)
    print(
        f"  baseline: {baseline['n_layers']} layers | tw={baseline['tw_file']}",
        flush=True,
    )
    print(
        f"  HEAD:     {head['n_layers']} layers | tw={head['tw_file']}",
        flush=True,
    )
    print(f"  prefill tokens: {baseline['tokens']}", flush=True)
    assert (
        baseline["tokens"] == head["tokens"]
    ), f"prefill mismatch: {baseline['tokens']} vs {head['tokens']}"
    assert (
        chain_logits.shape == block_logits.shape
    ), f"shape mismatch {tuple(chain_logits.shape)} vs {tuple(block_logits.shape)}"

    diff = (chain_logits - block_logits).abs()
    max_div = float(diff.max())
    mean_div = float(diff.mean())
    chain_arg = chain_logits.argmax(dim=-1)
    block_arg = block_logits.argmax(dim=-1)
    n_pos = chain_logits.shape[0]
    n_match = int((chain_arg == block_arg).sum())

    print("\n" + "=" * 70, flush=True)
    print("Forward divergence (ORIGINAL chain-64 vs FINAL block-57)", flush=True)
    print("=" * 70, flush=True)
    print(
        f"  logits shape: {tuple(chain_logits.shape)}  (positions x vocab)", flush=True
    )
    print(f"  max  |logit divergence|: {max_div:.6e}", flush=True)
    print(f"  mean |logit divergence|: {mean_div:.6e}", flush=True)
    print(
        f"  per-position argmax agreement: {n_match == n_pos} "
        f"({n_match}/{n_pos} positions match)",
        flush=True,
    )


if __name__ == "__main__":
    main()
