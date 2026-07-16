"""E(t) envelope model + synthetic rank replay for smooth_recency_rank derisking.

Closed-form forward encoding (torchwright/ops/swiglu/global_recency.py):
    theta  = 5e5 ** (-(d_rot-2)/d_rot),  d_rot = 64
    C(m)   = 65536 ** cos(m * theta)
    w(m)   = C / (C + m)                      # exact softmax weight at BOS
Recovery gain = |dm/dw|^-1 ... i.e. position error per unit weight error is
    g(m) = 1 / |dw/dm|   (numerical derivative; includes the dC/dm term
    that finding 11's (C+m)^2/C approximation drops).

Calibration anchors (measured, ONNX debug replay of the production low-res
bundles, stream = 3,724 positions, log 2026-07-15):
    clean artifact: err in [-0.2263, +0.0923]   (spread 0.319)
    bad   artifact: err in [-0.6230, +0.3149]   (spread 0.938)
    cross-artifact BOS weight noise: mean 1.0e-6, max 7.6e-6 (finding 9)
"""

import math

import numpy as np

N_CAP = 65536
THETA = 5e5 ** (-62.0 / 64.0)
GAIN8 = 8.0


def w_of_m(m):
    C = N_CAP ** np.cos(m * THETA)
    return C / (C + m)


def gain(m):
    m = np.asarray(m, dtype=np.float64)
    h = 1.0
    dw = (w_of_m(m + h) - w_of_m(m - h)) / (2 * h)
    return np.abs(1.0 / dw)


def main():
    ms = np.array(
        [
            1,
            100,
            1024,
            3530,
            3724,
            8000,
            14000,
            21000,
            32000,
            43000,
            53748,
            61440,
            65535,
        ],
        dtype=np.float64,
    )
    g = gain(ms)
    print("== recovery gain g(m) = position error per unit BOS-weight error ==")
    print(
        f"   theta_slow={THETA:.4e}  cap*theta={N_CAP * THETA:.4f} rad "
        f"(cos at cap={math.cos(N_CAP * THETA):.4f})"
    )
    g_ref = gain(np.array([3724.0]))[0]
    for m, gg in zip(ms, g):
        print(f"   m={int(m):>6}  g={gg:,.0f}   ratio vs m=3724: {gg / g_ref:.3f}")

    print("\n== E(t) extrapolation from measured envelopes at <=3,724 ==")
    for name, e_meas in (("clean-realization", 0.2263), ("bad-realization", 0.6230)):
        print(f"   anchor {name}: |err|max={e_meas} at m<=3724")
        for m in (17336 + 3614, 53748, 61440):
            e = e_meas * gain(np.array([float(m)]))[0] / g_ref
            print(f"      -> E({m}) ~= {e:.3f}   [same weight-noise scale]")

    print("\n== guaranteed min adjacent step 1 - 2E/K (fixed-K) and raw 1-2E ==")
    for e in (0.45, 0.63, 1.0, 1.5, 2.0):
        row = [f"raw={1 - 2 * e:+.3f}"]
        for k in (4, 8, 16, 32):
            row.append(f"K{k}={1 - 2 * e / k:.4f}")
        margins = [
            f"(margin@8x0.98: K{k}={GAIN8 * (1 - 2 * e / k) * 0.98:.2f})"
            for k in (8, 16, 32)
        ]
        print(f"   E={e:.2f}: " + "  ".join(row) + "  " + " ".join(margins))
    print("   hardness gate 0.999 needs margin >= ln(999) = 6.907")

    # ------------------------------------------------------------------
    # synthetic replay: worst-case error fixtures -> rank quality metrics
    # ------------------------------------------------------------------
    n = 21000
    t = np.arange(n, dtype=np.float64)
    e_prof = 0.63 * gain(t + 1) / g_ref  # bad-realization scale, position-shaped
    rng = np.random.default_rng(7)

    fixtures = {
        "iid uniform +-E(t)": rng.uniform(-1, 1, n) * e_prof,
        "alternating +-E(t)": np.where(t % 2 == 0, 1.0, -1.0) * e_prof,
        "block-alt period 2K (K=8 worst)": np.where((t // 8) % 2 == 0, 1.0, -1.0)
        * e_prof,
        "block-alt period 2K (K=32 worst)": np.where((t // 32) % 2 == 0, 1.0, -1.0)
        * e_prof,
        "isolated spikes (1%)": np.where(
            rng.random(n) < 0.01, rng.choice([-1.0, 1.0], n), 0.0
        )
        * e_prof,
        "bias step at n/2": np.where(t < n / 2, -0.3, +0.3) * e_prof / 0.63,
        "slow sinusoid (period 4k)": np.sin(2 * np.pi * t / 4000.0) * e_prof,
    }

    def metrics(rank, start):
        r = rank[start:]
        s = np.diff(r)
        if len(s) == 0:
            return "n/a"
        margin = GAIN8 * s.min() * 0.98
        wt = 1.0 / (1.0 + math.exp(min(margin, 700.0))) if margin < 30 else 0.0
        return (
            f"min step={s.min():+.4f} nonpos={int((s <= 0).sum())} "
            f"worst margin={margin:+.2f} runner-up wt={wt:.2e}"
        )

    print("\n== synthetic replay (n=21000, bad-realization scale, gain 8) ==")
    for name, err in fixtures.items():
        raw = t + err
        print(f"   fixture: {name}   (|err|max={np.abs(err).max():.3f})")
        print(f"      raw:            {metrics(raw, 1)}")
        cum = np.cumsum(raw) / (t + 1)
        print(f"      all-hist x2:    {metrics(2 * cum, 1)}")
        for k in (8, 16, 32):
            w = np.convolve(
                raw, np.ones(k) / k, mode="valid"
            )  # w[i]=mean raw[i..i+k-1]
            rank_k = w + (k - 1) / 2.0
            print(f"      K={k:<2} window:   {metrics(rank_k, 1)}")

    # ------------------------------------------------------------------
    # content dominance at N=65,536
    # ------------------------------------------------------------------
    print("\n== content dominance recompute ==")
    for name, mg in (("MATCH_GAIN_LONG", 600_000.0), ("MATCH_GAIN_CLIP", 600_000.0)):
        need = GAIN8 * N_CAP
        print(
            f"   {name}={mg:,.0f} vs RECENCY_GAIN*N = 8*65536 = {need:,.0f}"
            f"  headroom={mg - need:,.0f} logits ({(mg - need) / mg * 100:.1f}%)"
        )
    print(
        f"   naive interval for 2*mean: 0..2N -> 8*131072 = {8 * 2 * N_CAP:,.0f}"
        f"  EXCEEDS 600,000 -> tightened range claim is mandatory"
    )
    print(
        f"   fp32 ULP at logit 600,000 = {600000 * 2**-23:.4f} "
        f"(vs min adjacent margin ~7)"
    )


if __name__ == "__main__":
    main()
