"""Pure-Python radix decomposition oracle for ``next_start_after``.

Derisk artifact for the Task 3 d_head reduction (radix successor). It proves
the *logic* of the radix successor decomposition

    H1  same bucket, local digit strictly above the query's local digit
    H2  smallest bucket strictly greater than the query's bucket
    H3  minimum start within the bucket H2 found

against brute force, independent of any graph or fp32 numerics, BEFORE the
graph port exists. This is the standalone ``next_start_after`` oracle the
overnight report calls the unblock step (no such oracle existed before).

Two decompositions are validated against brute force:

* ``radix_next_start_after`` — the clean three-stage logic.
* ``radix_with_sentinel``    — the graph-faithful variant that carries an
  ``INVALID_HI = N_BUCKETS`` fallback row through H2 and recomputes H1/H3
  presence from selected predicate features, mirroring
  ``dhead_task3_torchwright_doom_radix_successor.md``. This is what the graph
  port must implement; proving it equals brute force here pins the design.

These helpers (``hi`` / ``lo`` / ``radix_params`` / the two decompositions)
are the reference the graph port should mirror op-for-op.

Pure Python: no torch, no graph, no GPU. Runs as a plain script
(``python tests/scene/test_radix_successor_oracle.py``) or under pytest.
"""

from __future__ import annotations

import math
import random

# The two project scales: fixture (W=60) and deferred real scale (W=160).
SCALES = (60, 160)
# Expected radix constants per scale, for an independent formula cross-check.
_EXPECTED = {60: (8, 8), 160: (13, 13)}  # W -> (B, N_BUCKETS)


def radix_params(W: int) -> tuple[int, int]:
    """``B = ceil(sqrt(W+1))``; ``N_BUCKETS = floor(W/B) + 1``."""
    B = math.isqrt(W)  # floor(sqrt(W))
    if (B + 1) * (B + 1) <= W + 1:
        B += 1
    # math.ceil(sqrt(W+1)) without float rounding risk:
    while B * B < W + 1:
        B += 1
    while (B - 1) * (B - 1) >= W + 1 and B > 1:
        B -= 1
    N_BUCKETS = W // B + 1
    return B, N_BUCKETS


def hi(v: int, B: int) -> int:
    return v // B


def lo(v: int, B: int) -> int:
    return v - B * (v // B)


def brute_next_start_after(starts, column: int, W: int) -> int:
    """Ground truth: nearest start strictly greater than ``column``, else W."""
    later = [s for s in starts if s > column]
    return min(later) if later else W


def radix_next_start_after(starts, column: int, W: int) -> int:
    """Clean three-stage radix logic (no sentinel bookkeeping)."""
    B, _ = radix_params(W)
    qh, ql = hi(column, B), lo(column, B)

    # H1: smallest start in the query's bucket whose local digit > ql.
    same = [s for s in starts if hi(s, B) == qh and lo(s, B) > ql]
    if same:
        return min(same)  # within a bucket, min start == min local digit

    # H2: smallest bucket strictly above qh that holds any start.
    higher = [hi(s, B) for s in starts if hi(s, B) > qh]
    if not higher:
        return W

    # H3: minimum start (== minimum local digit) in that bucket.
    hh = min(higher)
    carry = [s for s in starts if hi(s, B) == hh]
    return min(carry, key=lambda s: lo(s, B))


def radix_with_sentinel(starts, column: int, W: int) -> int:
    """Graph-faithful decomposition mirroring the port design.

    H2 selects over hi-digits with an always-present ``INVALID_HI = N_BUCKETS``
    fallback row (the publish-on-every-row sentinel), so H2 never has an empty
    candidate pool. H1/H3 presence is recomputed from selected predicate
    features (here: exact membership), not from a scalar average.
    """
    B, N_BUCKETS = radix_params(W)
    INVALID_HI = N_BUCKETS  # sits exactly one past the max real bucket
    SENTINEL_START = W
    qh, ql = hi(column, B), lo(column, B)

    # --- H1 same bucket, strictly above local digit ---
    h1_cands = [s for s in starts if hi(s, B) == qh and lo(s, B) > ql]
    same_present = len(h1_cands) > 0
    same_s = min(h1_cands, key=lambda s: lo(s, B)) if same_present else None

    # --- H2 next higher bucket (INVALID_HI fallback is always a candidate) ---
    # qh <= N_BUCKETS-1 < INVALID_HI, so the sentinel is always strictly above qh.
    h2_pool = [hi(s, B) for s in starts] + [INVALID_HI]
    higher_hi = min(h for h in h2_pool if h > qh)

    # --- H3 minimum start in the carried bucket ---
    h3_cands = [s for s in starts if hi(s, B) == higher_hi]  # empty if INVALID_HI
    carry_present = (higher_hi < INVALID_HI) and len(h3_cands) > 0
    carry_s = min(h3_cands, key=lambda s: lo(s, B)) if carry_present else None

    # --- combine: select(same_present, same_s, select(carry_present, carry_s, W))
    if same_present:
        return same_s
    if carry_present:
        return carry_s
    return SENTINEL_START


# --------------------------------------------------------------------------- #
# Validation drivers
# --------------------------------------------------------------------------- #

def _check(starts, column, W, stats):
    expected = brute_next_start_after(starts, column, W)
    got_clean = radix_next_start_after(starts, column, W)
    got_sent = radix_with_sentinel(starts, column, W)
    stats["n"] += 1
    if got_clean != expected:
        stats["mismatches"].append(("clean", W, list(starts), column, expected, got_clean))
    if got_sent != expected:
        stats["mismatches"].append(("sentinel", W, list(starts), column, expected, got_sent))


def _run_scale(W, stats, *, n_random, pair_cap, seed):
    B, N_BUCKETS = radix_params(W)
    # Formula cross-check against the spec's hand-computed constants.
    assert (B, N_BUCKETS) == _EXPECTED[W], (W, B, N_BUCKETS, _EXPECTED[W])
    # one_hot(hi, N_BUCKETS) never overflows; INVALID_HI=N_BUCKETS is one past.
    assert max(hi(v, B) for v in range(W + 1)) == N_BUCKETS - 1

    cols = range(W + 1)

    # Exhaustive singles x all columns.
    for s in range(W + 1):
        for c in cols:
            _check([s], c, W, stats)

    # Pairs x all columns (exhaustive for W=60, capped sample for W=160).
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(W + 1) for b in range(a, W + 1)]
    if len(pairs) > pair_cap:
        pairs = rng.sample(pairs, pair_cap)
    for a, b in pairs:
        for c in cols:
            _check([a, b], c, W, stats)

    # Random multisets (incl. empty + duplicates) x all columns.
    for _ in range(n_random):
        k = rng.randint(0, 24)
        starts = [rng.randint(0, W) for _ in range(k)]
        for c in cols:
            _check(starts, c, W, stats)

    # Targeted edge cases at the bucket boundaries.
    edge_cols = sorted({0, 1, B - 1, B, B + 1, 2 * B - 1, 2 * B, W - 1, W})
    edge_starts = [
        [],                                   # empty
        [W],                                  # only the top column
        [0, B, 2 * B, 3 * B],                 # one per low bucket (lo==0)
        [B - 1, 2 * B - 1, 3 * B - 1],        # top-of-bucket starts
        [5, 5, 5],                            # duplicates
        [c for c in range(0, W + 1, B)],      # bucket-aligned sweep
        [qh_b for qh_b in range(W + 1)],      # every column present
    ]
    for starts in edge_starts:
        for c in edge_cols:
            _check(starts, c, W, stats)

    # Symmetric wrong-bucket case (the graph blend hazard, here as pure logic):
    # query in bucket 1, starts only in buckets 0 and 2. The successor must be
    # the smaller start strictly above the column, never a bucket-1 phantom.
    if N_BUCKETS >= 3:
        for c in range(B, 2 * B):  # all of bucket 1
            _check([0, 2 * B], c, W, stats)        # both wrong buckets
            _check([B - 1, 2 * B + 1], c, W, stats)


def run_all(*, n_random=8000, seed=1234):
    stats = {"n": 0, "mismatches": []}
    for W in SCALES:
        _run_scale(
            W, stats,
            n_random=n_random,
            pair_cap=4000 if W > 60 else 10 ** 9,
            seed=seed + W,
        )
    return stats


# --------------------------------------------------------------------------- #
# pytest entry points
# --------------------------------------------------------------------------- #

def test_radix_matches_brute_force():
    stats = run_all()
    assert not stats["mismatches"], stats["mismatches"][:20]
    # Reproduce (exceed) the ~1.4M-case brute-force coverage claim.
    assert stats["n"] >= 1_400_000, stats["n"]


def test_radix_params_match_spec():
    assert radix_params(60) == (8, 8)
    assert radix_params(160) == (13, 13)


if __name__ == "__main__":
    s = run_all()
    print(f"cases checked: {s['n']:,}")
    if s["mismatches"]:
        print(f"MISMATCHES: {len(s['mismatches'])}")
        for m in s["mismatches"][:20]:
            print(" ", m)
        raise SystemExit(1)
    print("OK: radix (clean + sentinel) == brute force on every case")
