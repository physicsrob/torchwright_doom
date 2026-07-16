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
        return artifact, sess

    # A fresh solve would add a new entry to the local dir; abort loudly.
    before = sorted(p.name for p in local_cache.glob("*.json"))
    art_bad, sess_bad = compile_and_debug("bad", args.n_heads)
    if args.baseline_fingerprint is None:
        return 0
    art_clean, sess_clean = compile_and_debug("clean", args.baseline_n_heads)
    after = sorted(p.name for p in local_cache.glob("*.json"))
    if before != after:
        raise SystemExit(
            f"replay integrity failure: schedule dir changed {before} -> {after}"
        )

    # ---- node-by-node diff, EMITTED-REGION focus ------------------------
    # Prompt-position lanes are largely protocol-inactive garbage that may
    # legally differ between schedules; the regression's real flips live at
    # emitted positions (teacher forcing: first flip at emit[103] = stream
    # position n_prompt + 103).  For every node, measure the delta over
    # emitted positions only, and capture the exact (bad, clean) value pair
    # at the divergence site — the magnitude discriminates residue-scale
    # (~1e-9) vs PL-noise-scale (~1e-3..0.5) vs structural (>= bin).
    import torch  # noqa: F401 — tensor ops below

    n_prompt = len(prompt_ids)
    records = []
    for node in _all_nodes(next_token):
        try:
            va = sess_bad.debug_value(node)
            vb = sess_clean.debug_value(node)
        except Exception:  # noqa: BLE001 — nodes without assignments etc.
            continue
        if va is None or vb is None or va.shape != vb.shape:
            continue
        emit_a = va[n_prompt - 1 :]
        emit_b = vb[n_prompt - 1 :]
        if emit_a.numel() == 0:
            continue
        delta = (emit_a - emit_b).abs()
        dmax = float(delta.max())
        if dmax <= 0.0:
            continue
        flat = int(delta.reshape(-1).argmax())
        p, lane = divmod(flat, delta.shape[1])
        records.append(
            {
                "dmax": dmax,
                "emit_pos": p - 1,  # -1: index 0 predicts emit row 0
                "lane": lane,
                "bad": float(emit_a[p, lane]),
                "clean": float(emit_b[p, lane]),
                "name": getattr(node, "name", None),
                "ann": (getattr(node, "annotation", None) or "")[:55],
                "typ": type(node).__name__,
                "w": va.shape[1],
            }
        )
    records.sort(key=lambda r: -r["dmax"])
    print(
        f"\n[emit-diff] nodes differing in the emitted region: {len(records)}",
        flush=True,
    )
    print("[emit-diff] top by emitted-region max |bad - clean|:")
    for r in records[: args.top]:
        print(
            f"  d={r['dmax']:.3e} emit[{r['emit_pos']:>4}] lane={r['lane']:>3} "
            f"bad={r['bad']:.10g} clean={r['clean']:.10g} "
            f"{r['typ']}:{r['name']} w={r['w']} @{r['ann']}",
            flush=True,
        )

    # Angle chain: every node whose name/annotation mentions angle/atan,
    # regardless of rank — the emit[103] angleValue flip's feeders.
    print("\n[emit-diff] angle/atan-chain nodes (any nonzero emitted delta):")
    n_shown = 0
    for r in records:
        blob = f"{r['name']} {r['ann']}".lower()
        if "angle" in blob or "atan" in blob:
            print(
                f"  d={r['dmax']:.3e} emit[{r['emit_pos']:>4}] "
                f"bad={r['bad']:.10g} clean={r['clean']:.10g} "
                f"{r['typ']}:{r['name']} @{r['ann']}",
                flush=True,
            )
            n_shown += 1
            if n_shown >= 25:
                break

    # Earliest discrete flips: big deltas ordered by position, and the
    # divergence cascade's seed candidates.
    big = [r for r in records if r["dmax"] > 0.3]
    big.sort(key=lambda r: r["emit_pos"])
    print(f"\n[emit-diff] deltas > 0.3 in emit-position order ({len(big)}):")
    for r in big[:30]:
        print(
            f"  emit[{r['emit_pos']:>4}] d={r['dmax']:.3e} "
            f"bad={r['bad']:.8g} clean={r['clean']:.8g} "
            f"{r['typ']}:{r['name']} w={r['w']} @{r['ann']}",
            flush=True,
        )

    # The earliest flip's inputs: for the first 3 big-delta nodes, print
    # each input's delta at THAT position — is the seed perturbation
    # residue-scale or structural?
    by_id = {id(n): n for n in _all_nodes(next_token)}
    name_index: dict[tuple, object] = {}
    for n in by_id.values():
        name_index.setdefault(
            (
                getattr(n, "name", None),
                type(n).__name__,
                getattr(n, "annotation", None),
            ),
            n,
        )
    for r in big[:3]:
        node = name_index.get((r["name"], r["typ"], None)) or next(
            (
                n
                for n in by_id.values()
                if getattr(n, "name", None) == r["name"]
                and type(n).__name__ == r["typ"]
                and (getattr(n, "annotation", None) or "")[:55] == r["ann"]
            ),
            None,
        )
        if node is None:
            continue
        pos = r["emit_pos"] + n_prompt
        print(
            f"\n[seed] inputs of {r['typ']}:{r['name']} @{r['ann']} at emit[{r['emit_pos']}]:"
        )
        for inp in getattr(node, "inputs", []):
            try:
                ia, ib = sess_bad.debug_value(inp), sess_clean.debug_value(inp)
            except Exception:  # noqa: BLE001
                continue
            if ia is None or ib is None or ia.shape != ib.shape:
                print(
                    f"    {type(inp).__name__}:{getattr(inp,'name',None)} — no comparable value"
                )
                continue
            d_here = float((ia[pos] - ib[pos]).abs().max())
            d_hist = float((ia[: pos + 1] - ib[: pos + 1]).abs().max())
            print(
                f"    {type(inp).__name__}:{getattr(inp,'name',None)} w={ia.shape[1]} "
                f"delta@pos={d_here:.3e} delta_max_upto_pos={d_hist:.3e}",
                flush=True,
            )

    # Histogram of emitted-region delta magnitudes across all nodes.
    from collections import Counter
    import math

    hist: Counter[int] = Counter()
    for r in records:
        hist[int(math.floor(math.log10(r["dmax"])))] += 1
    print("\n[emit-diff] delta magnitude histogram (log10 bucket: count):")
    for b in sorted(hist):
        print(f"    1e{b:+d}: {hist[b]}")

    # ---- seed walk: follow the largest-delta input upstream -------------
    # Start at the earliest big attention flip (the emit[83] Attn @proj/stor)
    # and walk toward the perturbation's origin: at each level print every
    # input's max delta over ALL positions (prompt included — keys live at
    # prompt rows) and descend into the largest, until deltas die out.
    start = None
    for n in _all_nodes(next_token):
        if type(n).__name__ != "Attn":
            continue
        if (getattr(n, "annotation", None) or "") != "proj/stor":
            continue
        try:
            va, vb = sess_bad.debug_value(n), sess_clean.debug_value(n)
        except Exception:  # noqa: BLE001
            continue
        if va is None or vb is None or va.shape != vb.shape:
            continue
        if float((va[n_prompt - 1 :] - vb[n_prompt - 1 :]).abs().max()) > 5:
            start = n
            break
    if start is not None:
        print("\n[seed-walk] from the emit[83] attention flip upstream:")
        cur = start
        for depth in range(15):
            print(
                f"  [{depth}] {type(cur).__name__}:{getattr(cur, 'name', None)} "
                f"@{(getattr(cur, 'annotation', None) or '')[:45]}"
            )
            best, best_d = None, 0.0
            for inp in getattr(cur, "inputs", []):
                try:
                    ia, ib = sess_bad.debug_value(inp), sess_clean.debug_value(inp)
                except Exception:  # noqa: BLE001
                    ia = ib = None
                if ia is None or ib is None or ia.shape != ib.shape:
                    print(
                        f"      in: {type(inp).__name__}:{getattr(inp, 'name', None)}"
                        f" — no comparable value"
                    )
                    continue
                dt = (ia - ib).abs()
                dm = float(dt.max())
                flat = int(dt.reshape(-1).argmax())
                p, lane = divmod(flat, dt.shape[1])
                print(
                    f"      in: {type(inp).__name__}:{getattr(inp, 'name', None)} "
                    f"w={ia.shape[1]} dmax={dm:.3e} at pos={p} lane={lane} "
                    f"bad={float(ia[p, lane]):.8g} clean={float(ib[p, lane]):.8g}",
                    flush=True,
                )
                if dm > best_d:
                    best, best_d = inp, dm
            if best is None or best_d < 1e-4:
                print(f"  [seed-walk] frontier reached (max input delta {best_d:.3e})")
                break
            cur = best

    # ---- BOS-head input discrimination ------------------------------------
    # The seed walk above stops AT the boosted-BOS attention head: its
    # ~7.6e-6 output delta is below the 1e-4 descent threshold, so its own
    # inputs were never compared.  Descend with threshold ZERO: if every
    # input is bit-identical between the two artifacts, the weight
    # difference is generated inside the attention computation itself
    # (different packed shapes -> different kernels / reduction orders /
    # folded head constants); if an input already differs, the perturbation
    # comes from upstream accumulation and the walk continues into it.
    bos_ffn = next(
        (
            n
            for n in _all_nodes(next_token)
            if (getattr(n, "name", None) or "") == "bos_weight_to_position_0_1024"
        ),
        None,
    )
    bos_attn = (
        next(
            (
                inp
                for inp in getattr(bos_ffn, "inputs", [])
                if type(inp).__name__ == "Attn"
            ),
            None,
        )
        if bos_ffn is not None
        else None
    )
    if bos_attn is not None:
        import torch as _tb

        va, vb = sess_bad.debug_value(bos_attn), sess_clean.debug_value(bos_attn)
        dt = (va - vb).abs()
        n_diff = int((dt > 0).sum())
        mean_diff = float(dt[dt > 0].mean()) if n_diff else 0.0
        print(
            f"\n[bos-head] Attn output: dmax={float(dt.max()):.3e} "
            f"n_diff={n_diff}/{dt.numel()} mean|delta| over differing={mean_diff:.3e}",
            flush=True,
        )

        def _bitwise_descend(node, depth):
            pad = "  " * depth
            for inp in getattr(node, "inputs", []):
                try:
                    ia, ib = sess_bad.debug_value(inp), sess_clean.debug_value(inp)
                except Exception:  # noqa: BLE001
                    ia = ib = None
                nm = f"{type(inp).__name__}:{getattr(inp, 'name', None)}"
                if ia is None or ib is None or ia.shape != ib.shape:
                    print(f"[bos-head]{pad} in {nm}: no comparable value", flush=True)
                    continue
                same = bool(_tb.equal(ia, ib))
                d = (ia - ib).abs()
                print(
                    f"[bos-head]{pad} in {nm} w={ia.shape[1]}: "
                    f"bitwise_equal={same} dmax={float(d.max()):.3e} "
                    f"n_diff={int((d > 0).sum())}",
                    flush=True,
                )
                if not same and depth < 8:
                    _bitwise_descend(inp, depth + 1)

        _bitwise_descend(bos_attn, 0)

    # ---- who competed in the emit[83] read -------------------------------
    # probe_attention exposes the actual softmax weights/logits at the flip
    # query; the key node's debug value gives every candidate row's computed
    # position lane.  Together: which rows competed, their weights in each
    # schedule, and each row's true vs computed position.
    if start is not None:
        from torchwright.debug.probe import probe_attention
        import torch as _t

        # The record's emit_pos maps slice index p -> emit index p-1, so
        # emit[83] is logits at STREAM position n_prompt + 83.
        qpos = n_prompt + 83
        key_node = None
        for inp in getattr(start, "inputs", []):
            try:
                ia, ib = sess_bad.debug_value(inp), sess_clean.debug_value(inp)
            except Exception:  # noqa: BLE001
                continue
            if (
                ia is not None
                and ib is not None
                and ia.shape == ib.shape
                and ia.shape[1] == 2
                and float((ia - ib).abs().max()) > 1e-3
            ):
                key_node = inp
                break
        top_rows: set[int] = set()
        aps = {}
        for tag, sess in (("bad", sess_bad), ("clean", sess_clean)):
            try:
                aps[tag] = probe_attention(sess, inputs, start, query_pos=qpos)
            except Exception as exc:  # noqa: BLE001
                print(f"[attn:{tag}] probe_attention failed: {exc}")
        if len(aps) == 2:
            wa, wb = aps["bad"].weights, aps["clean"].weights
            for h in range(wa.shape[0]):
                if float((wa[h] - wb[h]).abs().max()) < 1e-3:
                    continue
                for tag, ap in aps.items():
                    vals, idxs = _t.topk(ap.weights[h], 6)
                    pairs = ", ".join(
                        f"row {int(i)}: w={float(v):.6f} "
                        f"logit={float(ap.logits[h, i]):.4f}"
                        for v, i in zip(vals, idxs)
                    )
                    print(
                        f"[attn:{tag}] DIFFERING head {h} @qpos {qpos}: {pairs}",
                        flush=True,
                    )
                    top_rows.update(int(i) for i in idxs[:4])
            if not top_rows:
                print(f"[attn] no head differs at qpos {qpos} (weights >1e-3)")
        if key_node is not None and top_rows:
            print("[attn] candidate rows — true vs computed position lane:")
            ka, kb = sess_bad.debug_value(key_node), sess_clean.debug_value(key_node)
            for row_i in sorted(top_rows):
                print(
                    f"    row {row_i}: match_lane bad={float(ka[row_i, 0]):+.4f} "
                    f"clean={float(kb[row_i, 0]):+.4f} | position true={row_i} "
                    f"bad={float(ka[row_i, 1]):.4f} "
                    f"(err {float(ka[row_i, 1]) - row_i:+.4f}) "
                    f"clean={float(kb[row_i, 1]):.4f} "
                    f"(err {float(kb[row_i, 1]) - row_i:+.4f})",
                    flush=True,
                )

    # ---- position-recovery error curve -----------------------------------
    # The global-position readout (bos_weight_to_position_*) recovers each
    # row's absolute index from a BOS-attention weight.  Error vs the true
    # index, across ALL positions and BOTH schedules, says whether the
    # emit[83] flip's position was typical or unlucky — i.e. how much
    # rounding margin the whole readout actually has.
    print("\n[pos-recovery] error distribution per schedule:")
    for n in _all_nodes(next_token):
        name = getattr(n, "name", None) or ""
        if not name.startswith("bos_weight_to_position"):
            continue
        for tag, sess in (("bad", sess_bad), ("clean", sess_clean)):
            try:
                v = sess.debug_value(n)
            except Exception:  # noqa: BLE001
                continue
            if v is None or v.shape[1] != 1:
                continue
            import torch as _t

            pos_idx = _t.arange(v.shape[0], dtype=v.dtype).unsqueeze(1)
            err = (v - pos_idx).reshape(-1)
            frac = (v - v.round()).abs().reshape(-1)
            margin = 0.5 - frac  # distance to the rounding cliff
            n_over = int((err.abs() > 0.5).sum())
            n_near = int((margin < 0.1).sum())
            print(
                f"  {name} [{tag}]: err mean={float(err.mean()):+.4f} "
                f"min={float(err.min()):+.4f} max={float(err.max()):+.4f} | "
                f"|err|>0.5: {n_over}/{len(err)} | margin<0.1: {n_near}",
                flush=True,
            )

    # ---- unmask behavioral asserts ---------------------------------------
    # The claimed-range checks ("matches NodeValueType") include at least one
    # that exact math itself violates on this prompt, and it aborts every
    # debug run before the behavioral asserts (picked_from, score gaps) can
    # testify.  Strip range claims from the rebuilt graph (rebuild-side
    # metadata, explicitly legal to edit), rebuild the sessions, and rerun.
    stripped = 0
    for n in _all_nodes(next_token):
        cks = getattr(n, "checks", None)
        if not cks:
            continue
        kept = [
            c
            for c in cks
            if not str(getattr(c, "message", "")).startswith("matches NodeValueType")
        ]
        stripped += len(cks) - len(kept)
        n.checks = kept
    print(f"\n[unmask] stripped {stripped} claimed-range checks; behavioral kept")
    for tag, art in (("bad", art_bad), ("clean", art_clean)):
        fresh = art.debug_session(next_token)
        for attempt in range(3):
            try:
                fresh.step(inputs, fresh.empty_past(), past_len=0, debug=True)
                print(f"[unmask:{tag}] no behavioral assert fired", flush=True)
                break
            except AssertionError as failure:
                msg = str(failure)
                print(f"[unmask:{tag}] ASSERT: {msg[:600]}", flush=True)
                if "matches NodeValueType" not in msg:
                    break  # a behavioral assert — the testimony we wanted
                site = msg.split("Assert failed at ", 1)[1].split(":", 1)[0]
                for n in _all_nodes(next_token):
                    if getattr(n, "annotation", None) == site and getattr(
                        n, "checks", None
                    ):
                        n.checks = [
                            c
                            for c in n.checks
                            if not str(getattr(c, "message", "")).startswith("matches")
                        ]
                fresh = art.debug_session(next_token)
            except RuntimeError as failure:
                print(f"[unmask:{tag}] RUNTIME: {str(failure)[:300]}", flush=True)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
