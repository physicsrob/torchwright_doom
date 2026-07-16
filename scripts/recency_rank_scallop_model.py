"""Reconstruct the PWL inversion's static bias (scallop) curve in float64.

Same construction as torchwright/ops/swiglu/global_recency.py: 1024
log-uniform breakpoints on w in [w(65536), 1], targets from 64-step
bisection of the exact forward w(m).  Ideal piecewise-LINEAR interpolation
(the swiglu fillets bend toward the true curve, so the real op sits at or
below this chord error).  Validated against the GPU probe's measured curve
at m <= 21,000; then read off the full-res tail predictions.
"""

import numpy as np

MAX_LEN = 65536
THETA = 5e5 ** (-62.0 / 64.0)
N_BPS = 1024


def w_of_m(m):
    if np.isscalar(m):
        m = np.array([m], dtype=np.float64)
    cos_m = np.cos(m * THETA)
    eff = MAX_LEN**cos_m
    return np.where(m <= 0, 1.0, eff / (eff + m))


def bisect_m(w_target):
    lo, hi = 0.0, float(MAX_LEN)
    if w_target >= 1.0:
        return 0.0
    if w_target <= w_of_m(MAX_LEN)[0]:
        return float(MAX_LEN)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if w_of_m(mid)[0] > w_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    w_min = w_of_m(MAX_LEN)[0]
    ratio = (1.0 / w_min) ** (1.0 / (N_BPS - 1))
    w_bps = np.array([w_min * ratio**k for k in range(N_BPS)])
    w_bps[-1] = 1.0
    m_bps = np.array([bisect_m(w) for w in w_bps])
    print(
        f"w_min={w_min:.6f} smallest w-gap={w_bps[1]-w_bps[0]:.3e} "
        f"largest m-gap between breakpoints={np.max(np.abs(np.diff(m_bps))):.1f}"
    )

    m = np.arange(0, MAX_LEN, 1, dtype=np.float64)
    w = w_of_m(m)
    # ideal PL interpolation on the (w_bps, m_bps) grid (w increasing)
    m_hat = np.interp(w, w_bps, m_bps)
    err = m_hat - m

    print("\nstatic PWL bias (ideal-PL chord) per bucket:")
    for lo, hi in (
        (0, 1024),
        (1024, 3724),
        (3724, 8000),
        (8000, 14000),
        (14000, 21000),
        (21000, 32000),
        (32000, 43000),
        (43000, 53748),
        (53748, 61440),
        (61440, 65536),
    ):
        seg = err[lo:hi]
        print(
            f"   [{lo:>6},{hi:>6}): min={seg.min():+8.3f} max={seg.max():+8.3f} "
            f"|err|max={np.abs(seg).max():8.3f}"
        )

    steps = np.diff(m_hat)
    print("\nadjacent steps of the ideal-PL recovery (static part only):")
    for lo, hi in (
        (0, 3724),
        (3724, 21000),
        (21000, 43000),
        (43000, 61440),
        (61440, 65535),
    ):
        seg = steps[lo:hi]
        print(
            f"   [{lo:>6},{hi:>6}): min={seg.min():+.4f} max={seg.max():+.4f} "
            f"nonpos={int((seg <= 0).sum())}"
        )

    # all-history-mean rank on the static curve, out to the cap
    cum = np.cumsum(m_hat) / (m + 1.0)
    rank_a = 2.0 * cum
    sa = np.diff(rank_a)
    print("\nall-history x2 on the static curve (input error only):")
    for lo, hi in (
        (0, 3724),
        (3724, 21000),
        (21000, 43000),
        (43000, 61440),
        (61440, 65535),
    ):
        seg = sa[lo:hi]
        print(
            f"   [{lo:>6},{hi:>6}): min step={seg.min():+.6f} nonpos={int((seg <= 0).sum())}"
        )
    for k in (8, 16, 32):
        wm = np.convolve(m_hat, np.ones(k) / k, mode="valid")
        sk = np.diff(wm)
        worst = min(
            sk[lo:hi].min()
            for lo, hi in ((3724, 21000), (21000, 43000), (43000, 61435))
        )
        print(
            f"K={k:>2} window on static curve: min step in emitted region = {worst:+.4f}"
        )


if __name__ == "__main__":
    main()
