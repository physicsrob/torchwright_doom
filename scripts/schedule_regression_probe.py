"""Replay cached schedules into ONNX debug artifacts and localize where two
schedules of the SAME graph compute differently on the SAME stream.

Mode 1 (single --fingerprint): replay + debug forward; the graph's own
asserts run against compiled values (first firing printed).  NOTE: this
graph carries at least one range claim that exact math itself violates on
the production prompt (the R_RenderSegLoop staircase saturates at
protocol-inactive positions), so a firing assert is NOT evidence of a
schedule defect on its own — compare against the baseline schedule.

Mode 2 (--baseline-fingerprint): replay BOTH schedules, run both debug
forwards, then diff every graph node's compiled value between the two
artifacts.  Same graph + same inputs means every node agrees up to
schedule-level numerical noise; nodes with structural deltas mark the
divergence frontier of the regression.  No exact-math reference is used,
so pre-existing claim violations cannot confuse the result.

Schedule entries are COPIED from the read-side volume mount into a
container-local dir before TW_SCHEDULE_CACHE_DIR is set — replays can
never write back to the durable cache; a fresh solve trips the
one-entry-per-fingerprint integrity check and aborts.

    TORCHWRIGHT_DOOM_SCREEN_WIDTH=160 TORCHWRIGHT_DOOM_SCREEN_HEIGHT=100 \\
    TORCHWRIGHT_DOOM_RENDER_SCALE=2 TORCHWRIGHT_DOOM_DETAIL=low \\
    TORCHWRIGHT_DOOM_HUD=1 \\
    MODAL_RUN_GPU=B200 MODAL_RUN_GPU_MEMORY=262144 MODAL_RUN_TIMEOUT=10800 \\
    make modal-run MODULE=scripts.schedule_regression_probe \\
        ARGS="--fingerprint <bad> --baseline-fingerprint <clean> \\
              --reference-run <run-dir> --rows 110"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import deque
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_SCHEDULE_RO = Path(
    os.environ.get("TORCHWRIGHT_DOOM_SCHEDULE_RO", "/schedule-cache-ro")
)
_ARTIFACT_ROOT = Path(os.environ.get("TORCHWRIGHT_DOOM_ARTIFACT_ROOT", "/artifacts"))


def _all_nodes(root):
    seen, out = set(), []
    dq = deque([root])
    while dq:
        n = dq.popleft()
        if id(n) in seen:
            continue
        seen.add(id(n))
        out.append(n)
        dq.extend(getattr(n, "inputs", []))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint", required=True)
    ap.add_argument(
        "--baseline-fingerprint",
        default=None,
        help="second cached schedule to replay as the comparison baseline; "
        "every node's debug value is diffed between the two compiles",
    )
    ap.add_argument("--reference-run", required=True)
    ap.add_argument("--rows", type=int, default=110)
    ap.add_argument("--d", type=int, default=8192)
    ap.add_argument("--d-head", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=32)
    # None (flag absent) hashes differently from an explicit equal value in
    # the schedule fingerprint; pre-n_heads-config compiles used None.
    ap.add_argument("--baseline-n-heads", type=int, default=None)
    ap.add_argument("--d-hidden", type=int, default=16384)
    ap.add_argument("--optimize", type=int, default=3)
    ap.add_argument("--rms-const-exp", type=int, default=63)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    local_cache = Path(tempfile.mkdtemp(prefix="sched-replay-"))
    wanted = [args.fingerprint] + (
        [args.baseline_fingerprint] if args.baseline_fingerprint else []
    )
    for fp in wanted:
        src = _SCHEDULE_RO / f"{fp}.json"
        if not src.is_file():
            raise SystemExit(f"schedule entry not found: {src}")
        shutil.copy2(src, local_cache / src.name)
    os.environ["TW_SCHEDULE_CACHE_DIR"] = str(local_cache)

    from torchwright.compiler import CompileProfile
    from torchwright.compiler.export import compile_to_onnx
    from torchwright_doom.model_graph import build_graph
    from torchwright_doom.tokenizer.rows import rows_to_input

    inference = json.loads(
        (_ARTIFACT_ROOT / args.reference_run / "output.ids.json").read_text()
    )
    prompt_ids = [int(r) for r in inference["prompt"]["row_ids"]]
    good_rows = [int(r) for r in inference["emitted_row_ids"]][: args.rows]
    stream = prompt_ids + good_rows
    print(
        f"[probe] stream {len(stream)} rows (prompt {len(prompt_ids)} + "
        f"{len(good_rows)} emitted)",
        flush=True,
    )

    next_token, _rope, emb, _banks = build_graph(
        d_head=args.d_head, max_positions=65536, d_rot=64
    )
    inputs = rows_to_input(stream)

    def compile_and_debug(tag: str, n_heads: int):
        out_dir = Path(tempfile.mkdtemp(prefix=f"probe-{tag}-"))
        artifact = compile_to_onnx(
            next_token,
            embedding=emb,
            output_path=str(out_dir / f"{tag}.onnx"),
            d=args.d,
            d_head=args.d_head,
            n_heads=n_heads,
            d_hidden=args.d_hidden,
            max_seq_len=8192,
            max_layers=200,
            optimize=args.optimize,
            rms_norm_const_exp=args.rms_const_exp,
            profile=CompileProfile.PHI3,
            verbose=False,
        )
        print(f"[probe] {tag}: compiled n_layers={artifact.n_layers}", flush=True)
        sess = artifact.debug_session(next_token)
        try:
            sess.step(inputs, sess.empty_past(), past_len=0, debug=True)
            print(f"[probe] {tag}: no assert fired", flush=True)
        except AssertionError as failure:
            print(f"[probe] {tag}: ASSERT: {str(failure)[:300]}", flush=True)
        except RuntimeError as failure:
            print(f"[probe] {tag}: RUNTIME: {str(failure)[:300]}", flush=True)
        return sess

    # A fresh solve would add a new entry to the local dir; abort loudly.
    before = sorted(p.name for p in local_cache.glob("*.json"))
    sess_bad = compile_and_debug("bad", args.n_heads)
    if args.baseline_fingerprint is None:
        return 0
    sess_clean = compile_and_debug("clean", args.baseline_n_heads)
    after = sorted(p.name for p in local_cache.glob("*.json"))
    if before != after:
        raise SystemExit(
            f"replay integrity failure: schedule dir changed {before} -> {after}"
        )

    # ---- node-by-node diff between the two compiled artifacts ----------
    n_prompt = len(prompt_ids)
    rows_diff = []
    for node in _all_nodes(next_token):
        try:
            va = sess_bad.debug_value(node)
            vb = sess_clean.debug_value(node)
        except Exception:  # noqa: BLE001 — nodes without assignments etc.
            continue
        if va is None or vb is None or va.shape != vb.shape:
            continue
        delta = (va - vb).abs()
        dmax = float(delta.max())
        if dmax < 1e-9:
            continue
        # first position carrying at least half the node's peak delta —
        # always nonempty by construction
        pos_any = (delta >= dmax * 0.5).any(dim=1)
        first_pos = int(pos_any.nonzero().reshape(-1)[0])
        rows_diff.append(
            (
                dmax,
                first_pos,
                getattr(node, "name", None),
                (getattr(node, "annotation", None) or "")[:55],
                type(node).__name__,
                va.shape[1],
            )
        )
    rows_diff.sort(key=lambda r: -r[0])
    print(
        f"\n[diff] nodes with nonzero delta: {len(rows_diff)} "
        f"(prompt ends at position {n_prompt - 1})"
    )
    print("[diff] top by max |bad - clean|:")
    for dmax, first_pos, name, ann, typ, w in rows_diff[: args.top]:
        print(
            f"  d={dmax:>12.6g} first_pos={first_pos:>5} {typ}:{name} w={w} @{ann}",
            flush=True,
        )
    # earliest divergence: smallest first_pos among big deltas
    big = [r for r in rows_diff if r[0] > 1.0]
    big.sort(key=lambda r: r[1])
    print(f"\n[diff] earliest positions among deltas > 1.0 ({len(big)} nodes):")
    for dmax, first_pos, name, ann, typ, w in big[:20]:
        print(f"  pos={first_pos:>5} d={dmax:>12.6g} {typ}:{name} w={w} @{ann}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
