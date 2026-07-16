"""Derisking probe for the smooth-recency-rank plan (smooth_recency_rank.md).

Builds a MINIMAL graph containing exactly the machinery the plan touches —
the swiglu ``global_position_from_bos`` recovery (raw), the exact zero-Q/K
causal mean over the same raw node (``attend_causal_mean``), the production
``smoothed=True`` op path as an independent end-to-end instance, and a
fixed-offset read — compiles it at production head geometry
(d_head=128, d_rot=64 partial, max_positions=65536), and measures on real
GPU kernels:

  [raw]      the recovered-position error curve err[t] = raw[t] - t and its
             adjacent steps, per position bucket (envelope shape receipt);
  [realize]  the same graph compiled at a second residual width — the
             cross-realization noise family the n_heads=32 regression
             exposed (FINDINGS finding 9);
  [rerun]    same artifact forward twice — run-to-run GPU nondeterminism
             (FINDINGS open question 6, tiny-scale);
  [meanhead] the compiled uniform mean vs a float64 host mean of the SAME
             compiled raw — the mean head's OWN fp32 accumulation noise,
             cleanly separated from input noise (Version A's crux);
  [uniform]  softmax weight spread of the mean head at distance under
             partial rotary (is attend_mean_where exactly uniform?);
  [offset]   attend_to_offset read fidelity (off[t] vs raw[t+delta]) and
             pre-BOS startup values (bounded-blend receipt);
  [decode]   prefill vs token-by-token cached decode on a prefix (parity);
  [ranks]    candidate recency ranks — raw, all-history mean x2, fixed-K
             window means for K in {4,8,16,32} — adjacent-step minima,
             monotonicity violations, and worst two-candidate softmax
             hardness at RECENCY_GAIN=8, per bucket.

Run (GPU, low-res-scale length):

    make modal-run MODULE=scripts.recency_rank_probe ARGS="--n 21000"
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_UMBRELLA = Path(__file__).resolve().parents[2]
for p in (_UMBRELLA, _UMBRELLA / "torchwright_doom"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import math

import torch

RECENCY_GAIN = 8.0
KS = (4, 8, 16, 32)
BUCKETS = (
    (0, 64),
    (64, 1024),
    (1024, 3724),
    (3724, 8000),
    (8000, 14000),
    (14000, 21000),
    (21000, 32000),
    (32000, 43000),
    (43000, 54000),
    (54000, 65536),
)


def _pack(module, named, n):
    total = sum(w for _, _, w in module._input_specs)
    out = torch.zeros(n, total)
    for name, start, w in module._input_specs:
        if name in named:
            out[:, start : start + w] = named[name]
    return out


def _find_attn(node, want_annotation=None):
    """First Attn node in the input closure (unwraps Assert wrappers)."""
    from collections import deque

    seen, dq = set(), deque([node])
    while dq:
        n = dq.popleft()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if type(n).__name__ == "Attn":
            return n
        dq.extend(getattr(n, "inputs", []))
    return None


def _bucket_stats(vec: torch.Tensor, lo: int, hi: int):
    seg = vec[lo : min(hi, len(vec))]
    if seg.numel() == 0:
        return None
    return (
        float(seg.mean()),
        float(seg.min()),
        float(seg.max()),
        float(seg.abs().max()),
    )


def _print_step_report(tag: str, rank: torch.Tensor, start: int, n: int):
    """Adjacent-step minima / monotonicity / worst hardness at gain 8."""
    r = rank[start:n].to(torch.float64)
    steps = r[1:] - r[:-1]
    if steps.numel() == 0:
        return
    n_nonpos = int((steps <= 0).sum())
    print(
        f"[ranks:{tag}] positions {start}..{n - 1}: "
        f"min step={float(steps.min()):.6f} p0.1={float(torch.quantile(steps, 0.001)):.6f} "
        f"max step={float(steps.max()):.6f} non-positive steps={n_nonpos}"
    )
    for lo, hi in BUCKETS:
        lo2 = max(lo, start)
        seg = steps[max(0, lo2 - start) : max(0, min(hi, n - 1) - start)]
        if seg.numel() == 0:
            continue
        mn = float(seg.min())
        margin = RECENCY_GAIN * mn * 0.98  # slow-plane attenuation floor cos<=0.197rad
        weight = 1.0 / (1.0 + math.exp(margin)) if margin < 30 else 0.0
        print(
            f"    bucket [{lo2:>6},{min(hi, n - 1):>6}): min step={mn:+.6f} "
            f"worst 2-cand margin={margin:.3f} runner-up weight={weight:.2e} "
            f"nonpos={int((seg <= 0).sum())}"
        )


def build_graph():
    from torchwright.graph import Concatenate
    from torchwright.ops.attention_ops import attend_causal_mean, attend_to_offset
    from torchwright.ops.inout_nodes import create_input, create_rope_config
    from torchwright.ops.swiglu.global_recency import global_position_from_bos

    rope = create_rope_config(d_head=128, max_positions=65536, d_rot=64)
    bos = create_input("bos", 1)
    raw = global_position_from_bos(rope, bos)
    # The exact zero-Q/K mean on the SAME raw node (isolates head self-noise
    # against a float64 host mean of lane 0)...
    mean = attend_causal_mean(rope, raw, claim_range=False)
    # ...and the full production path as an independent op instance (its own
    # BOS head + PWL + mean), end to end.
    sm_op = global_position_from_bos(rope, bos, smoothed=True)
    off1 = attend_to_offset(rope, raw, delta_pos=-1)
    out = Concatenate([raw, mean, sm_op, off1])
    return out, mean


def run_realization(out_node, d: int, n: int, named):
    from torchwright.compiler.export import compile_headless

    m = compile_headless(out_node, d=d, d_head=128, verbose=False)
    dev = m._net.device
    packed = _pack(m, named, n).to(dev)
    with torch.no_grad():
        y1 = m(packed).detach().to("cpu", torch.float32)
        y2 = m(packed).detach().to("cpu", torch.float32)
    return m, packed, y1, y2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=21000)
    ap.add_argument(
        "--d1", type=int, default=2048, help="residual width of the primary realization"
    )
    ap.add_argument(
        "--n-dec",
        type=int,
        default=4000,
        help="prefix length for the step-by-step decode parity check",
    )
    ap.add_argument(
        "--d2",
        type=int,
        default=0,
        help="second residual width (realization diversity); 0 skips",
    )
    args = ap.parse_args()
    n = args.n

    torch.manual_seed(0)
    out_node, mean_node = build_graph()

    named = {"bos": torch.zeros(n, 1)}
    named["bos"][0, 0] = 1.0

    print(
        f"[probe] n={n} d_head=128 d_rot=64 max_positions=65536 "
        f"cuda={torch.cuda.is_available()}",
        flush=True,
    )

    # ---- realization 1 (production-ish packing width) --------------------
    m1, packed1, y1, y1b = run_realization(out_node, args.d1, n, named)
    raw1, mean1, smop1, off1_1 = (y1[:, i] for i in range(4))
    print(f"[probe] realization A (d={args.d1}): compiled, forward ok", flush=True)

    # [rerun] same artifact, same input, forward twice
    same = bool(torch.equal(y1, y1b))
    d_rr = (y1 - y1b).abs()
    print(
        f"[rerun] bitwise_equal={same} dmax={float(d_rr.max()):.3e} "
        f"n_diff={int((d_rr > 0).sum())}/{d_rr.numel()}",
        flush=True,
    )

    # [uniform] mean-head softmax spread at distance under partial rotary
    try:
        from torchwright.debug.probe import probe_attention

        attn = _find_attn(mean_node)
        qpos = n - 1
        ap_ = probe_attention(m1, packed1, attn, query_pos=qpos)
        w = ap_.weights[0, : qpos + 1]
        print(
            f"[uniform] mean-head weights at qpos={qpos}: "
            f"min={float(w.min()):.3e} max={float(w.max()):.3e} "
            f"uniform=1/{qpos + 1}={1.0 / (qpos + 1):.3e} "
            f"rel spread={(float(w.max()) - float(w.min())) * (qpos + 1):.3e}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[uniform] probe_attention failed: {exc}", flush=True)

    # ---- [decode] cached decode: path noise AND step stats along the path --
    try:
        nd = min(args.n_dec, n)
        dev = m1._net.device
        past = m1.empty_past()
        dec_rows = []
        with torch.no_grad():
            for t in range(nd):
                row = _pack(m1, {k: v[t : t + 1] for k, v in named.items()}, 1).to(dev)
                out_t, past = m1.step(row, past)
                dec_rows.append(out_t.reshape(-1).to("cpu", torch.float32))
        dec = torch.stack(dec_rows)
        dd = (dec - y1[:nd]).abs()
        print(
            f"[decode] prefix {nd}: max |prefill-decode| per lane "
            f"raw={float(dd[:, 0].max()):.3e} mean={float(dd[:, 1].max()):.3e} "
            f"smop={float(dd[:, 2].max()):.3e} off1={float(dd[:, 3].max()):.3e}",
            flush=True,
        )
        # adjacent steps ALONG the decode-published sequence (what production
        # emitted rows put in the cache), decode-path mean-head self-noise,
        # and rank_A steps along the decode path.
        raw_d = dec[:, 0].to(torch.float64)
        mean_d = dec[:, 1].to(torch.float64)
        sd = raw_d[1:] - raw_d[:-1]
        print(
            f"[decode] raw steps along decode path: min={float(sd.min()):+.4f} "
            f"max={float(sd.max()):+.4f} nonpos={int((sd <= 0).sum())}"
        )
        t_d = torch.arange(nd, dtype=torch.float64)
        host_mean_d = torch.cumsum(raw_d, dim=0) / (t_d + 1.0)
        dm = (mean_d - host_mean_d).abs()
        print(
            f"[decode] mean-head self-noise along decode path: "
            f"max={float(dm.max()):.3e}"
        )
        _print_step_report("all-history x2 (decode path)", 2.0 * mean_d, 0, nd)
    except Exception as exc:  # noqa: BLE001
        print(f"[decode] step-loop failed: {exc}", flush=True)

    # ---- [mixed] prefill past, then decode — the production shape ----------
    # Production publishes prompt rows via ONE prefill and emitted rows via
    # per-step decode; the cache mixes both.  Feed the first half as a bulk
    # step (prefill kernels), then single steps: the boundary pair's rank
    # step is the one place the two kernel paths meet.
    try:
        nb = min(args.n_dec, n) // 2
        nm = min(args.n_dec, n)
        past = m1.empty_past()
        bulk = _pack(m1, {k: v[:nb] for k, v in named.items()}, nb).to(dev)
        with torch.no_grad():
            out_b, past = m1.step(bulk, past)
            mix_rows = [out_b.reshape(nb, -1).to("cpu", torch.float32)]
            for t in range(nb, nm):
                row = _pack(m1, {k: v[t : t + 1] for k, v in named.items()}, 1).to(dev)
                out_t, past = m1.step(row, past)
                mix_rows.append(out_t.reshape(1, -1).to("cpu", torch.float32))
        mix = torch.cat(mix_rows)
        mean_m = mix[:, 1].to(torch.float64)
        rank_m = 2.0 * mean_m
        sm = rank_m[1:] - rank_m[:-1]
        b_step = float(rank_m[nb] - rank_m[nb - 1])
        print(
            f"[mixed] prefill {nb} rows + decode to {nm}: rank_A step at the "
            f"prefill/decode boundary = {b_step:+.6f}"
        )
        print(
            f"[mixed] rank_A steps overall: min={float(sm.min()):+.4f} "
            f"nonpos={int((sm <= 0).sum())}"
        )
        raw_m = mix[:, 0].to(torch.float64)
        sraw = raw_m[1:] - raw_m[:-1]
        print(
            f"[mixed] raw step at boundary={float(raw_m[nb] - raw_m[nb - 1]):+.4f} "
            f"raw min step={float(sraw.min()):+.4f} nonpos={int((sraw <= 0).sum())}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mixed] bulk-step prefill failed: {exc}", flush=True)

    # free realization A's GPU state before compiling B
    del m1, packed1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- realization 2 (different packing width) ---------------------------
    try:
        if not args.d2:
            raise RuntimeError("skipped (--d2 0)")
        m2, packed2, y2, _ = run_realization(out_node, args.d2, n, named)
        raw2 = y2[:, 0]
        print(f"[probe] realization B (d={args.d2}): compiled, forward ok", flush=True)
        del m2, packed2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] realization B failed: {exc}", flush=True)
        raw2 = None

    # =======================================================================
    # analysis (host, float64)
    # =======================================================================
    t_idx = torch.arange(n, dtype=torch.float64)
    err1 = raw1.to(torch.float64) - t_idx

    print(f"\n[raw] err[t] = raw[t] - t, realization A (d={args.d1}):")
    for lo, hi in BUCKETS:
        s = _bucket_stats(err1, lo, hi)
        if s:
            print(
                f"    bucket [{lo:>6},{min(hi, n):>6}): mean={s[0]:+.4f} "
                f"min={s[1]:+.4f} max={s[2]:+.4f} |err|max={s[3]:.4f}"
            )

    steps_raw = err1[1:] - err1[:-1] + 1.0
    print("[raw] adjacent steps raw[t]-raw[t-1], realization A:")
    for lo, hi in BUCKETS:
        seg = steps_raw[lo : min(hi, n - 1)]
        if seg.numel():
            print(
                f"    bucket [{lo:>6},{min(hi, n - 1):>6}): min={float(seg.min()):+.4f} "
                f"max={float(seg.max()):+.4f} nonpos={int((seg <= 0).sum())}"
            )

    if raw2 is not None:
        err2 = raw2.to(torch.float64) - t_idx
        print("\n[raw] err[t], realization B (d={}):".format(args.d2))
        for lo, hi in BUCKETS:
            s = _bucket_stats(err2, lo, hi)
            if s:
                print(
                    f"    bucket [{lo:>6},{min(hi, n):>6}): mean={s[0]:+.4f} "
                    f"min={s[1]:+.4f} max={s[2]:+.4f} |err|max={s[3]:.4f}"
                )
        dr = err1 - err2
        print("[realize] raw cross-realization delta (A-B):")
        for lo, hi in BUCKETS:
            s = _bucket_stats(dr, lo, hi)
            if s:
                print(
                    f"    bucket [{lo:>6},{min(hi, n):>6}): mean={s[0]:+.2e} "
                    f"min={s[1]:+.4f} max={s[2]:+.4f} |d|max={s[3]:.4f}"
                )

    # ---- [meanhead] compiled mean vs float64 host mean of compiled raw ----
    cum = torch.cumsum(raw1.to(torch.float64), dim=0)
    host_mean = cum / (t_idx + 1.0)
    dmean = mean1.to(torch.float64) - host_mean
    print(
        "\n[meanhead] compiled attend_mean_where minus float64 cumulative mean "
        "of the SAME compiled raw (the head's own fp32 error):"
    )
    for lo, hi in BUCKETS:
        s = _bucket_stats(dmean, lo, hi)
        if s:
            print(
                f"    bucket [{lo:>6},{min(hi, n):>6}): mean={s[0]:+.2e} "
                f"min={s[1]:+.2e} max={s[2]:+.2e} |d|max={s[3]:.2e}"
            )
    print(
        f"[meanhead] rank_A self-noise = 2*|d|: global max = "
        f"{2 * float(dmean.abs().max()):.3e}"
    )

    # ---- [smop] the production smoothed op, end to end ---------------------
    d_op = (smop1.to(torch.float64) - 2.0 * mean1.to(torch.float64)).abs()
    print(
        f"\n[smop] |smoothed op - 2*mean(shared raw)| (independent op instance "
        f"vs composed lane): max={float(d_op.max()):.3e} "
        f"mean={float(d_op.mean()):.3e}"
    )
    err_op = smop1.to(torch.float64) - t_idx
    print("[smop] err[t] = smoothed_op[t] - t:")
    for lo, hi in BUCKETS:
        s = _bucket_stats(err_op, lo, hi)
        if s:
            print(
                f"    bucket [{lo:>6},{min(hi, n):>6}): mean={s[0]:+.4f} "
                f"min={s[1]:+.4f} max={s[2]:+.4f} |err|max={s[3]:.4f}"
            )

    # ---- [offset] read fidelity + startup ---------------------------------
    o1 = off1_1.to(torch.float64)
    d_o1 = (o1[1:] - raw1.to(torch.float64)[:-1]).abs()
    print(
        f"\n[offset] |off1[t] - raw[t-1]| (t>=1): max={float(d_o1.max()):.3e} "
        f"mean={float(d_o1.mean()):.3e}"
    )
    print(f"[offset] startup off1[0]={float(o1[0]):+.4f} (pre-BOS read)")
    print(
        "[offset] raw[0..8]:          " + " ".join(f"{float(v):+.3f}" for v in raw1[:9])
    )
    print(
        "[offset] mean[0..8]:         "
        + " ".join(f"{float(v):+.3f}" for v in mean1[:9])
    )
    print(
        "[offset] smoothed_op[0..8]:  "
        + " ".join(f"{float(v):+.3f}" for v in smop1[:9])
    )

    # ---- [ranks] candidate ranks ------------------------------------------
    print(
        "\n[ranks] adjacent-step report at RECENCY_GAIN=8 "
        "(margin uses 0.98 slow-plane attenuation floor):"
    )
    _print_step_report("raw", raw1.to(torch.float64), 0, n)
    _print_step_report("smoothed op (production path)", smop1.to(torch.float64), 0, n)
    rank_a = 2.0 * mean1.to(torch.float64)
    _print_step_report("all-history x2 (compiled mean)", rank_a, 0, n)
    rank_a_exact_input = 2.0 * host_mean  # isolates: input smoothing w/o head noise
    _print_step_report("all-history x2 (float64 mean of raw)", rank_a_exact_input, 0, n)
    raw64 = raw1.to(torch.float64)
    for k in KS:
        w = torch.stack([raw64[k - 1 - j : n - j] for j in range(k)]).mean(0)
        rank_k = torch.full((n,), float("nan"), dtype=torch.float64)
        rank_k[k - 1 :] = w + (k - 1) / 2.0
        _print_step_report(
            f"K={k} window mean (host, from compiled raw)", rank_k, k - 1, n
        )

    # downsampled err curve for the report
    print("\n[curve] err[t] every 1024 positions (realization A):")
    idx = list(range(0, n, 1024)) + [n - 1]
    print("    " + " ".join(f"{i}:{float(err1[i]):+.3f}" for i in idx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
