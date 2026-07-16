# Smooth Recency Rank — Derisked Proposal (2026-07-15)

Companion to `smooth_recency_rank.md`. Every load-bearing claim below was
measured today; each number cites its source. Experiment tooling (all in
`scripts/`): `recency_rank_probe.py` (GPU, via `make modal-run`),
`recency_rank_envelope_model.py` (gain model + synthetic replay, local CPU),
`recency_rank_scallop_model.py` (exact reconstruction of the
position-recovery lookup table, local CPU).

## Decision — IMPLEMENTED 2026-07-15 (upstream, whole-signal swap)

Adopt **Version A — the all-history causal mean**, applied to
`global_position` itself so both consumer classes (the recency tiebreak
AND the `pixel_index = pos − span_v0.pos − 1` arithmetic) inherit it.
Shipped shape (supersedes the earlier tiebreak-only sketch below):

- **torchwright** — `attend_causal_mean(rope, value, output_scale=,
  claim_range=)` in `ops/attention_ops.py`: a genuinely zero-Q/K head, so
  the logits are exactly 0 and the softmax exactly uniform on *every*
  rotary layout (zero vectors rotate to zero — stronger than
  `attend_mean_where`'s NoPE-tail placement, which is exact only under
  partial rotary).  Both `global_position_from_bos` twins (swiglu + relu)
  take `smoothed: bool = False`; the smoothed path is
  `2 × attend_causal_mean(raw)` with the ×2 folded into the O projection
  and a justified closing claim `assert_in_range(0, max_len, atol=8)`.
  Both twins' docstrings now record the measured fp32 wander.
- **doom** — `GraphPast.global_position()` passes `smoothed=True`
  (`_SMOOTHED_GLOBAL_POSITION` module flag in `model/past.py` kept for
  A/B probes).  `pick_most_recent` is untouched mechanically —
  `RECENCY_GAIN` stays 8, the tiebreak inherits the smoothed signal
  through the existing plumbing, and the earlier `recency_scale = 16` /
  `assert_in_range(rank, 0, 33_000)` machinery is unnecessary: the
  smoothed node's semantic range is `[0, max_positions]` like raw's.
- Net graph cost: **one attention head + one sublayer of depth** on every
  `global_position` consumer's chain.

Why the whole-signal swap (not tiebreak-only): both sides of the
`pos − marker` subtraction are the same node, so the swap is atomic, and
the smoothed signal dominates raw on every measured axis — absolute error
±10.4 → ~±1.7 at 54k, adjacent steps 0.53 → ≥0.965, sustained
differencing drift over a ~300-row span at tail positions ~1.2 → ~0.11
(the pixel-index budget is 0.5, so raw was eating the whole margin at
full-res tail; plausibly part of the 3.5% non-exact full-res colors).

**Version B (fixed-K window) is rejected**, not deferred: on the real
compiled error curve at full-res tail scale (n=54,000) **every K ≤ 16 fails
the 0.999 hardness gate** (runner-up weight 1.4e-3 – 2.1e-3 in the 43k–54k
bucket; K=32 is marginal at 9.3e-4), while costing K−1 extra full-width
heads plus an unresolved startup policy. The all-history mean passes the
same bucket at 5.2e-4 with one head. The reference implementation is not
worth building.

**No startup policy is needed** for Version A. The mean is defined from
t=0 and measured strictly monotone from the first position; pre-BOS
offset reads (Version B's startup hazard) are not used at all.

## What the measurements changed about the original plan

### 1. The `E = 0.45` bounded-error model was wrong in kind, not just size

The raw recovered position's dominant error at scale is a **deterministic,
bit-stable fp32-evaluation wander of the compiled PWL machine** — not
realization noise, and not the lookup table's design error:

| positions | measured raw `err[t] = raw[t] − t` (compiled, GPU) | ideal-table error (float64 reconstruction) |
|---|---|---|
| 0–1k | ±0.09 | ≤ 0.010 |
| 1k–3.7k | −0.49 .. +0.11 | ≤ 0.010 |
| 3.7k–8k | −3.26 .. −0.19 | ≤ 0.010 |
| 8k–14k | ±4.34 | ≤ 0.010 |
| 14k–21k | **±9.22** | ≤ 0.010 |
| 43k–54k (54k run, d=1024) | **±10.36** | ≤ 0.007 |

(Compiled numbers: `recency_rank_probe --n 21000`, production head geometry
d_head=128 / d_rot=64 / cap 65,536. Ideal-table: exact re-construction of the
1024-breakpoint log-uniform inversion grid, float64 chord interpolation.)

Two independent GPU compiles (d=2048 and d=1536) produced **bit-identical**
curves, and re-running one artifact is bit-identical too — so this wander is
static per compiled expression graph. It is smooth: adjacent raw steps stay
positive everywhere measured, but dip to **+0.53 (decode path) / +0.59
(prefill)** — at gain 8 that is a 4.2-logit margin, i.e. ~1% of softmax
weight on the runner-up. That is exactly the production failure class
(FINDINGS finding 12 measured a 0.52 gap and a 1.5% blend on the bad
schedule). The production-artifact anchor agrees: the ONNX replay of the
real bundles measured err ∈ [−0.226, +0.092] (clean) and [−0.623, +0.315]
(bad) over the first 3,724 positions.

Consequences: (a) `raw` can *never* be made safe by raising gain — the
plan's premise is confirmed with real curves; (b) any K-table derived from
constant E is fiction — the derisked comparison below uses the real curve.

### 2. Version A's feared risks are all measured small

The review flagged three open risks for the all-history mean. All three
resolved in its favor:

- **The mean head's own fp32 accumulation noise** (the crux): compiled
  `attend_mean_where` output vs a float64 cumulative mean of the *same*
  compiled raw values: |error| ≤ 4.1e-3 at 21k positions (prefill) and
  4.4e-3 (decode path). After the ×2, rank self-noise ≤ 8.3e-3 — two orders
  of magnitude below the ~0.5 that would threaten the hardness gate.
- **Uniformity under partial RoPE**: measured softmax weights at query
  20,999 are *exactly* 1/21000, relative spread 0.0e+00. Under
  `d_rot < d_head` the validity column rides the unrotated NoPE tail
  (`rotary_content_head` → `place_on_nope_tail`), so equal-logit is exact by
  construction. (The shipped design goes one step further: the new
  `attend_causal_mean` primitive uses zero Q/K projections outright, making
  the uniformity exact on full rotary too — layout-proof for any
  torchwright user, no guard needed. Verified in the oracle at 2.4e-7 under
  full rotary.)
- **Prefill/decode boundary** (prompt rows publish via prefill kernels,
  emitted rows via per-step decode): measured rank_A step across a
  2,000-row-prefill → decode boundary = **+0.9994**. The kernel paths do
  differ (same position computed both ways differs by up to 0.20 in raw),
  but the cache freezes each row once, and the mean head's own path
  difference is ~1e-3 — the boundary pair is unremarkable.

### 3. Head-to-head on the real compiled curve (the decision table)

Adjacent-step minima / worst two-candidate runner-up weight at gain 8
(0.98 slow-plane attenuation floor), consumed region (t ≥ 64), n=21,000:

| rank | min step | worst runner-up weight | extra heads | startup |
|---|---|---|---|---|
| raw (today) | **0.59 prefill / 0.53 decode** | **9.7e-3 / ~1.5e-2** | — | — |
| all-history ×2 | **0.988** (0.985 decode) | **4.4e-4** | +1 | none needed |
| K=4 window | 0.942 | 6.2e-4 | +3 | required |
| K=8 window | 0.958 | 5.5e-4 | +7 | required |
| K=16 window | 0.963 | 5.2e-4 | +15 | required |
| K=32 window | 0.970 | 5.0e-4 | +31 | required |

A perfect integer rank at gain 8 would give 3.9e-4. The all-history mean is
within 12% of that ceiling, at the lowest cost, with zero monotonicity
violations across 21,000 prefill positions, 4,000 decode positions, and the
mixed prefill/decode stream. Version B's windows are *worse* because the
window mean inherits 1/K of the local wander slope while the all-history
mean divides it by t.

**Tail behavior (n=54,000, production full-res rollout scale, B200, d=1024)**
— this run settles the K question decisively. Worst consumed-region bucket
(43k–54k), runner-up weight at gain 8:

| rank | min step @43k–54k | worst runner-up weight | vs 0.999 gate |
|---|---|---|---|
| raw | 0.738 (21k–32k) | 3.1e-3 | fails (and margin is realization luck) |
| all-history ×2 (compiled) | **0.965** | **5.2e-4** | **passes, 2× margin** |
| K=4 | 0.788 | 2.1e-3 | fails |
| K=8 | 0.804 | 1.8e-3 | fails |
| K=16 | 0.834 | 1.4e-3 | fails |
| K=32 | 0.891 | 9.3e-4 | marginal |

The raw wander grows to ±10.4 at 43k–54k, and the fixed windows inherit its
local slope divided only by K — **every K ≤ 16 fails the hardness gate at
full-res tail scale**. The all-history mean divides the same wander by t and
keeps 96%+ of the ideal step everywhere; its only tail cost is its own fp32
accumulation (measured 1.5e-2 on the mean at 54k → 3.0e-2 on the rank —
eating 0.035 of the step, exactly the gap between the float64-input control
at 0.9994 and the compiled 0.965). Zero non-positive steps across all 54,000
positions; uniformity still exact (weights ≡ 1/54,000); prefill/decode
boundary step +1.00003; rerun bit-identical.

Synthetic worst-case fixtures (iid, alternating, block-alternating, spikes,
bias step, sinusoid at the bad-realization envelope) confirm the same
ordering: raw produces thousands of *inversions* (non-positive steps —
argmax flips no gain can fix); all-history and K≥16 stay monotone
everywhere, and the all-history worst margins (4.9–5.4, alternating/iid
fixtures) occur only at t < ~64, inside the prompt (earliest consumed read
is at the first emitted row, ~position 3,614; header-row candidates at
positions 1–60 are correctly ordered — measured min step 0.968 from t=0).

### 4. Content dominance at the real cap, and the range claim

`RECENCY_GAIN·N` grew past the documented budget: render_constants sized
`MATCH_GAIN_* = 600,000` against `8 × 61,440 = 491,520`, but the RoPE cap
is 65,536 → the binding number is `8 × 65,536 = 524,288`.
**Headroom is 75.7k logits (12.6%), not 18%** — still passing, with fp32
ULP at 600k = 0.07 (100× under the adjacent-step margin ~7).

With the whole-signal swap the bookkeeping simplified: the smoothed op's
output is a position in `[0, max_positions]` with the same claim shape as
raw (closing `assert_in_range(0, max_len, atol=8)` inside the op — the
atol covers the measured 2×running-mean bias ±1.7 plus head noise, and the
top end exceeding `max_len` by 2×bias on a cap-filling rollout), so no
special mean-range claim and no `recency_scale=16` form exist anywhere.
The render_constants dominance comments are updated to the 65,536 cap
(done).

## Implementation status (updated after landing)

**DONE (2026-07-15):**

1. torchwright `attend_causal_mean` (`ops/attention_ops.py`) — zero-Q/K
   exact causal mean, `output_scale` folded into O, `claim_range` opt-out
   for callers whose inputs carry assert slack.
2. torchwright `global_position_from_bos(..., smoothed=True)` on both
   twins (`ops/swiglu/global_recency.py`, `ops/relu/global_recency.py`),
   docstrings recording the measured fp32 wander (the relu twin's
   "~0.15 max error" claim corrected to short-rollout-only).
3. doom `GraphPast.global_position()` → `smoothed=True` behind
   `_SMOOTHED_GLOBAL_POSITION = True` (`model/past.py`), docstring updates
   on `global_position` / `pick_most_recent`, dominance comments at the
   65,536 cap in `render_constants.py`, and the `pos_scalar` comment in
   `render_main.py` (same-node cancellation note).
4. Unit tests: 3 oracle tests for `attend_causal_mean` (incl. full-rotary
   exactness and the ×2 identity) in `tests/ops/test_attention_ops.py`;
   2 compiled tests for the smoothed position (tracking + step bounds,
   prefill/decode parity) in
   `tests/compile/forward/test_rope_global_recency.py`. All 57 tests in
   the touched files pass locally; GPU probe re-run of the real
   production op below.

**Gate results:**

5. Full `make test`, both submodules — **PASSED (2026-07-15 19:4x)**.
   torchwright: 207 + 1,389 passed; the only failure is the pre-existing
   `tests/docs/test_numerical_noise_drift.py` staircase-measurement drift
   from the Jul-14 routing commits (identical failure on the pre-change
   baseline run at 22:54 Jul 14; nothing about the new op). doom: all 5
   shards green. (Lint fixes along the way: black/ruff on the three new
   probe scripts, and two pre-existing mypy errors in
   `scripts/schedule_regression_probe.py`.)

6. `make compile CONFIG=configs/e1m1_lowres.yaml` — **PASSED (19:55)**:
   fresh CP-SAT solve (selected=solver, delivery=fresh), **39 layers** vs
   the 38L raw-tiebreak baseline. NOTE: a subsequent 5-seed no-cache sweep
   (below) showed solver-seed variance of 36–39 layers on this same graph,
   so "+1 layer vs 38" compares two single draws — the smoothing's real
   depth cost is *within seed noise*, not a measured +1.
6b. **5-seed no-cache sweep** (`scripts/compile_report.py --d 8192
   --d-head 128 --n-heads 32 --d-hidden 16384 --rms-const-exp 63
   --optimize 3 --seeds 1..5`, 64-CPU each, production low-res geometry):
   seeds → **39, 39, 37, 36, 39** layers; heuristic warm-start 45; all
   real solves (`FEASIBLE`, no fallback), all with proven lower bound 33.
   The shipped 39L bundle sits at the top of the seed range — a seed sweep
   before publication could recover 2–3 layers if depth matters.
8. Low-res `make run COMPARE=1` at n_heads=32 — **PASSED (20:2x)**:
   **17,336 rows (exact row-count parity), 100.0% coverage, 100.0%
   within-option** — the regression configuration is fixed (the raw
   tiebreak gave 99.5% within-option and +13 rows here). Exact color
   **93.0%**, vs 91.6% (clean-old) / 90.8% (de-confounder) — a 1.4–2.2
   point *improvement* over both prior artifacts, consistent with the
   prediction that the raw wander was consuming pixel-index margin.

9. Full-res `make run COMPARE=1` — **PASSED (21:1x)**: compile 40 layers
   (fresh solve); render 53,747 rows, **100.0% coverage, 100.0%
   within-option (63,482/63,490), exact color 96.8%** vs the certified
   baseline's 96.5% — parity-or-better on every axis, with the predicted
   exact-color improvement showing up (+0.3 at full-res, +1.4–2.2 at
   low-res).

All acceptance gates are green. The smoothed global position is live in
both committed configs' compiled bundles.

Acceptance: unchanged from the original plan's gates 1–8, with these
measured expectations: min winning weight ≥ 0.9995 (vs gate 0.999), zero
non-positive steps, and *expected exact-color churn* in the render diff —
any recompile re-randomizes boundary-sitting picks (FINDINGS finding 5);
judge on within-option/coverage/row-count, not exact-color %.

## Receipts index

| # | experiment | where | key result |
|---|---|---|---|
| R1 | production bundle replay (pre-existing) | modal-run log 2026-07-15 17:22, `[pos-recovery]` | clean err ∈ [−0.226,+0.092], bad ∈ [−0.623,+0.315] @ ≤3,724; bad has 1 row \|err\|>0.5 |
| R2 | tiny-graph GPU probe, n=21k | `scripts/recency_rank_probe.py` run 18:09 | raw err ±9.2 @21k, steps ≥0.59; rank_A steps ≥0.988 consumed, self-noise ≤4.1e-3; uniformity exact; offsets bitwise-exact; bit-identical across d=2048/1536 and across reruns |
| R3 | decode/mixed probe, n=21k + 4k steps | same script, run 18:2x | decode raw steps ≥0.53; decode rank_A ≥0.985; boundary step +0.9994; decode self-noise 4.4e-3 |
| R4 | tail probe, n=54k | same script, B200, run 18:23 | raw err ±10.4, steps ≥0.74; rank_A ≥0.965 / ≤5.2e-4 leak; K≤16 all FAIL the 0.999 gate at the tail; mean self-noise 1.5e-2; uniformity exact at 54k |
| R5 | exact table reconstruction | `scripts/recency_rank_scallop_model.py` | ideal-PL bias ≤0.01 everywhere → the ±9 wander is fp32 evaluation, not table design |
| R6 | gain model + synthetic replay | `scripts/recency_rank_envelope_model.py` | exact recovery gain 65k→185k over the cap (finding 11's 247k corrected); raw inversion counts in the thousands under envelope noise; dominance recompute |
| R7 | call-site census | grep, `model/` | 7 read families, all through `GraphPast.pick_most_recent`; existing `mean_where` facade already in place |
| R8 | shipped-op GPU verification, n=21k | probe run 18:5x (post-implementation) | the end-to-end `smoothed=True` op instance is **bit-identical** to the composed `2×attend_causal_mean(raw)` lane (max delta 0.0); production-path rank min step 0.968 (t<64) / ≥0.9988 consumed; decode path min 0.964, boundary +0.999; smoothed_op[0..8] tracks t from the first token; zero-Q/K uniformity exact on GPU; full DOOM graph builds with the swap |

## Residual risks (what this hour did NOT close)

- **R-a: runtime representativeness.** The probe runs torchwright's compiled
  torch module; production is HF `Phi3ForCausalLM` (also torch fp32, but
  fused 32-head QKV at d=8192 — different kernel shapes). The mean-head
  self-noise has ~50× margin to the gate, so a kernel-family factor would
  have to be enormous to matter; the low-res COMPARE gate is the check.
- **R-b: in-situ packing.** In the full DOOM graph the mean head shares
  fused QKV tensors with other heads; the probe's head was nearly alone.
  Same argument as R-a; covered by the phase-5 gates. An optional heavier
  receipt: one low-res bundle replay dumping the rank node
  (`schedule_regression_probe`-style, ~1–3h B200) — recommended only if the
  low-res gate shows anything odd.
- **R-c: schedule/depth cost.** +1 sublayer on the raw→pick dependency
  chain; CP-SAT may absorb it or add a layer. Known only at `make compile`.
- **R-d: gate churn.** The recompile will flip some unrelated
  boundary-sitting picks (finding 5) — pre-agreed to judge on
  within-option/coverage, so this doesn't stall the rollout.

## FINDINGS follow-ups this closes

- Open question 5 (design budget of the position tiebreak): answered — at
  gain 8 the raw tiebreak has **zero** realization margin at scale (measured
  0.53 min step ⇒ ~1.5% blends), and the smoothed rank restores the design
  margin (≤4.4e-4 leak, 12% off the theoretical ceiling).
- Open question 6 (run-to-run stability): the tiny-graph artifact is
  bitwise-stable across reruns on one GPU; cross-*kernel-path* variation
  (prefill vs decode) is the real noise family and is measured above.
- New finding, now recorded in both `global_position_from_bos` twins'
  docstrings: the compiled PWL machine's fp32 evaluation wander
  (±9 positions at 21k, deterministic) — mechanically distinct from finding
  11's realization-noise amplification, and invisible to it because all
  prior measurements stopped at 3,724 positions. An entry in torchwright's
  `docs/numerical_noise_findings.md` via the op-noise measurement workflow
  is a sensible follow-up.
