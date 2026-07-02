"""Gate-C equivalence run for the block-IR step-1 refactor on the flagship
DOOM graph (torchwright ``docs/block_ir_step1_plan.md``, "Equivalence harness"
+ Gate C).

Builds the production ``e1m1`` graph exactly as ``compile_to_onnx_path`` does
(``build_graph`` then the production width-safe fusion), then runs
torchwright's :mod:`scripts.block_equivalence` harness in schedule-only mode
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
    """Production flagship graph builder: ``build_graph`` then the production
    width-safe fusion (same as ``compile_to_onnx_path`` before every compile)."""
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
    fuse_consecutive_linears({next_token}, verbose=False, skip_relu_ejecting=True)
    return next_token


def main() -> None:
    import importlib.util

    import torch  # noqa: F401  (imported for side-effect parity with the builder)

    from torchwright.graph.blockify import blockify

    # The reusable harness lives in torchwright/scripts/, whose package name
    # ("scripts") collides with torchwright_doom/scripts/ on sys.path — load it
    # by explicit file path under a unique module name to sidestep the clash.
    _harness_path = _UMBRELLA / "torchwright" / "scripts" / "block_equivalence.py"
    _spec = importlib.util.spec_from_file_location(
        "_tw_block_equivalence", _harness_path
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod  # dataclass type resolution reads sys.modules
    _spec.loader.exec_module(_mod)
    EquivalenceReport = _mod.EquivalenceReport
    schedule_metrics = _mod.schedule_metrics

    config_name = os.environ.get("CONFIG", "e1m1")
    d = int(os.environ.get("D", "8192"))
    d_head = int(os.environ.get("D_HEAD", "128"))
    d_rot = int(os.environ["D_ROT"]) if os.environ.get("D_ROT") else 64
    d_hidden = int(os.environ.get("D_HIDDEN", "16384"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "65536"))

    print(
        f"Flagship equivalence: config={config_name} d={d} d_head={d_head} "
        f"d_rot={d_rot} d_hidden={d_hidden} max_seq_len={max_seq_len}"
    )

    t0 = time.perf_counter()
    chain_out = build_flagship(config_name, d_head, d_rot, max_seq_len)
    print(f"  built chain-path graph in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    chain_metrics = schedule_metrics(
        chain_out, d=d, d_head=d_head, d_hidden=d_hidden, assume_zero_init=True
    )
    print(
        f"  scheduled chain path in {time.perf_counter() - t0:.1f}s: "
        f"{chain_metrics.as_tuple()}"
    )

    t0 = time.perf_counter()
    block_out = blockify(build_flagship(config_name, d_head, d_rot, max_seq_len), verbose=True)
    print(f"  built + blockified block-path graph in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    block_metrics = schedule_metrics(
        block_out, d=d, d_head=d_head, d_hidden=d_hidden, assume_zero_init=True
    )
    print(
        f"  scheduled block path in {time.perf_counter() - t0:.1f}s: "
        f"{block_metrics.as_tuple()}"
    )

    report = EquivalenceReport(chain_metrics=chain_metrics, block_metrics=block_metrics)
    report.notes.append(
        "schedule-only (no weights); metrics are (n_layers, total_heads, "
        "peak_hidden, residual_peak)"
    )
    print()
    print(report.format())


if __name__ == "__main__":
    main()
