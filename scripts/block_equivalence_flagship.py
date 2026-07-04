"""Gate-C equivalence run for the block-IR step-1 refactor on the flagship
DOOM graph (torchwright ``docs/block_ir_step1_plan.md``, "Equivalence harness"
+ Gate C).

Builds the production ``e1m1`` graph exactly as ``compile_to_onnx_path`` does
(``build_graph`` then the production width-safe fusion), then runs
torchwright's :mod:`scripts.ffn_equivalence` harness in schedule-only mode
over both the chain-mined path and the blockified path, and prints the
compile-metrics tuple for each.  Any cost regression (layers/heads/hidden) is a
Gate-C stop-and-report.

Schedule-only (no d*d weight tensors) so it runs on CPU at d=8192.  Both paths
use identical seeding (the same const-1 self-match column + inputs, same
``assume_zero_init``), so the chain-vs-block delta is what the report measures
— RMSNorm's 1-2 reserved columns are omitted here because they cost both paths
the same and cancel in the delta.  The scheduler is the eager heuristic, which
is what production's optimize=2 CP-SAT falls back to at flagship scale (cold
timeout); both paths are therefore compared under one scheduler mode.

Usage (CPU, in-process — build is heavy but schedule-only is memory-safe):
    python -m scripts.block_equivalence_flagship
    # or on Modal:
    make modal-run MODULE=scripts.block_equivalence_flagship CPU_ONLY=1

Config overrides (default to e1m1.yaml production values):
    D=8192 D_HEAD=128 D_ROT=64 D_HIDDEN=16384 CONFIG=e1m1 MAX_SEQ_LEN=65536
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _resolve_wad(cfg):
    from torchwright_doom.inference.config import resolve_wad_path

    try:
        return resolve_wad_path(cfg, base_dir=str(_UMBRELLA / "torchwright_doom"))
    except Exception:
        for cand in (
            _UMBRELLA / "torchwright_doom" / "doom1.wad",
            _UMBRELLA / "doom1.wad",
            Path.home() / "Downloads" / "doom1.wad",
        ):
            if cand.exists():
                return str(cand)
        return None


def build_flagship(config_name: str, d_head: int, d_rot, max_seq_len: int):
    """Production flagship graph builder: ``build_graph`` (which builds MLP
    Blocks natively since Phase 2b) then the production block-aware fusion (same
    as ``compile_to_onnx_path`` before every compile)."""
    from torchwright.graph.optimize import fuse_consecutive_linears
    from torchwright_doom.inference.compiled_model import build_graph
    from torchwright_doom.inference.config import load_render_config

    cfg = load_render_config(
        str(_UMBRELLA / "torchwright_doom" / "configs" / f"{config_name}.yaml")
    )
    next_token, _rope, _emb, _banks = build_graph(
        d_head=d_head,
        max_positions=max_seq_len,
        d_rot=d_rot,
        asset_config=cfg.asset_config(),
        wad_path=_resolve_wad(cfg),
    )
    fuse_consecutive_linears({next_token}, verbose=False)
    return next_token


# The Phase-2a Gate-C block-path baseline (native Blocks must not regress it).
BASELINE = {
    "n_layers": 64,
    "total_heads": 813,
    "peak_hidden": 16384,
    "residual_peak": 8157,
}


def main() -> None:
    import importlib.util

    import torch  # noqa: F401  (imported for side-effect parity with the builder)

    from torchwright.compiler.lower import lower

    # The reusable harness lives in torchwright/scripts/, whose package name
    # ("scripts") collides with torchwright_doom/scripts/ on sys.path — load it
    # by explicit file path under a unique module name to sidestep the clash.
    _harness_path = _UMBRELLA / "torchwright" / "scripts" / "ffn_equivalence.py"
    _spec = importlib.util.spec_from_file_location("_tw_ffn_equivalence", _harness_path)
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod  # dataclass type resolution reads sys.modules
    _spec.loader.exec_module(_mod)
    schedule_metrics = _mod.schedule_metrics

    config_name = os.environ.get("CONFIG", "e1m1")
    d = int(os.environ.get("D", "8192"))
    d_head = int(os.environ.get("D_HEAD", "128"))
    d_rot = int(os.environ["D_ROT"]) if os.environ.get("D_ROT") else 64
    d_hidden = int(os.environ.get("D_HIDDEN", "16384"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "65536"))

    print(
        f"Flagship metrics: config={config_name} d={d} d_head={d_head} "
        f"d_rot={d_rot} d_hidden={d_hidden} max_seq_len={max_seq_len}"
    )

    # Since Phase 2b the op layer builds Blocks natively; fusion (Phase 2c) is
    # block-aware.  There is a single path now — build it, certify it at the
    # lowering boundary (closed vocabulary: zero raw chains; fresh derived
    # caches), schedule it, and compare the metrics tuple to the Phase-2a
    # Gate-C block-path baseline.
    t0 = time.perf_counter()
    out = build_flagship(config_name, d_head, d_rot, max_seq_len)
    lowered = lower(out, verbose=True)  # certification: raises on raw chains
    print(
        f"  pre-schedule demand: {lowered.cost_summary(d_head=d_head).format_short()}"
    )
    print(f"  built block-native graph in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    metrics = schedule_metrics(
        out, d=d, d_head=d_head, d_hidden=d_hidden, assume_zero_init=True
    )
    print(f"  scheduled in {time.perf_counter() - t0:.1f}s")

    got = {
        "n_layers": metrics.n_layers,
        "total_heads": metrics.total_heads,
        "peak_hidden": metrics.peak_hidden,
        "residual_peak": metrics.residual_peak,
    }
    print()
    print("  metric        baseline(2a)   native(2c)   delta")
    regressed = []
    for key in ("n_layers", "total_heads", "peak_hidden", "residual_peak"):
        base = BASELINE[key]
        cur = got[key]
        delta = cur - base
        flag = ""
        # Layers/heads/peak_hidden are the cost gate; residual_peak is
        # informational.  A positive delta on a gated metric is a regression.
        if key != "residual_peak" and delta > 0:
            flag = "  <-- REGRESSION"
            regressed.append(key)
        elif delta < 0:
            flag = "  (improvement)"
        print(f"  {key:<13} {base:>10}   {cur:>10}   {delta:+d}{flag}")

    print()
    print(
        f"  tuple (n_layers, total_heads, peak_hidden, residual_peak) = {metrics.as_tuple()}"
    )
    if regressed:
        print(f"\nGATE FAIL: cost regression in {regressed} — STOP and report.")
        sys.exit(1)
    print("\nGATE PASS: no cost regression vs the 2a baseline.")


if __name__ == "__main__":
    main()
