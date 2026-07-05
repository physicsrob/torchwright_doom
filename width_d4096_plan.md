# Width plan — make d=4096 schedulable (the KV play)

Execution plan for one of two parallel tracks (the other is
`depth_flatten_plan.md`). Written 2026-07-05 from the measurements in
`swiglu_opportunities_findings.md` — that doc is the evidence base;
this one is the work order. Self-contained for a fresh session.

**Branch**: doom `width-d4096`, plus a torchwright branch for the new
ops (this track is the ONLY writer to torchwright — the depth track
touches doom only). See "Consolidation contract" at the bottom.

## Goal and why

The production 320×200 certification is blocked on KV-cache size
(~221 GB total demand vs a B200's 178 GiB). The KV cache is
`layers × d_model × positions` — at 57.4k positions and d=8192,
~3.8 GB per layer, ~184 GB at 49 layers. **d_model must be a power of
two** (RMSNorm cancellation — Rob's constraint, stated not
re-verified), so the one narrower legal width is **d=4096** (32
uniform heads at d_head=128). At d=4096, KV is 92–113 GB at 49–60
layers — under the ceiling with margin, and it also shrinks the
attention weight blocks.

Today the graph does NOT schedule at d=4096: the heuristic scheduler
deadlocks (measured). The bracket (production graph and env, heuristic
schedule-only, fusion pre-pass, d_hidden=16,384): d=8192 → 65 layers,
7168 → 64, 6144 → 66, 5120 → 67, 4608 → 80, 4096 → deadlock. Depth is
flat down to ~5120, so the residual-width demand sits somewhere in
(4096, 4608]. This plan removes enough resident width to move the
knee below 4096.

## What occupies the width (measured, production env)

At the heuristic's peak (residual fully used at 8192):

- **`ray_scaled` — 2 × 1,024 columns.** Stage 1 of the graph's atan2
  (`render_ops.py::signed_world_angle` / `_ray_count`): the angle of a
  vector is computed as a count of threshold crossings against 1,024
  candidate angles per half-quadrant, and the 1,024-wide indicator
  vector of each half is materialized in the residual so stage 2 can
  count it. Both halves are live simultaneously: **2,048 columns —
  half of a d=4096 stream on its own.**
- **`floor_int` stage-1 step chunks — ~2,600 columns at the peak.** A
  floor over N boundaries materializes a step vector (up to 512
  columns per chunk) between its two stages; the emit digit-quad and
  paint floors were the residents at the measured peak.
- **~600 one-wide glue nodes** (compares, gates, emit scalars) plus
  `in_range` mask vectors (1,024-wide instances exist:
  `proj/pmrk`, `pmrk/R_CheckPlane`) elsewhere in the schedule.

## W0 — resident-width floor census (go/no-go, do first)

Before building anything: measure the width that can never be
reclaimed — the embedding row's fixed columns, the never-freed
const-one column, input nodes, and any node the schedule must hold
live across the whole pass. If (irreducible residents + the widest
unavoidable transient working set) already crowds 4,096, this plan
stops here and reports that instead.

Method: instrument `ResidualStreamMap` the way
`scripts/analyze_forward_cost.py::schedule_only_capture` does, but
report allocations that are never freed (no cancel layer) separately
from transients; alternatively read the allocation of input nodes +
embedding width directly at warm-start setup. Output: one number
(permanent columns) + the top permanent residents by name.

**RESULT (2026-07-04, `scripts/width_census.py`, production env,
measured at d=8192 and d=4608): M1 GO.** The permanent residents
occupy three *time-disjoint* windows, none close to 4,096:

- **whole-pass: 1 column** (the RoPE self-match const-one; a real
  compile adds 1–2 reserved RMSNorm pinned-constant columns the
  schedule-only path doesn't reserve),
- **input window: 1,409 columns** (the embedding row), freed at L1,
- **end window: 1,447 columns** (the output token row —
  `emit_derived_zero` 1,390 + three 19-wide heads), born in the last
  two layers; one 19-wide output leaf (a `cond_gate`) is held from
  mid-schedule to the end.

The measured d=8192 peak live-set (L15) is 1 permanent + 8,191
transient columns, led by exactly the residents this plan targets:
2×1,024 `ray_scaled` (W2) and ~2,600 columns of `floor_int` step
chunks (W1). The d=4096 deadlock is transient working-set pressure,
not a structural floor — proceed.

## W1 — radix-decomposed floors (shared with the lane story)

`floor_int` over N boundaries costs 3N lanes, 2 sublayers, and an
N-up-to-512-per-chunk-wide residual intermediate. The radix form —
`hi = floor(x/D)`, snap `hi` to an exact integer, `lo = floor(x −
D·hi)` with D ≈ √N — costs ~9√N lanes, ~6 sublayers, and **~√N-wide
intermediates**. This is the emit digit-quad generalized
(`emit.py::_digit_quad_payload` is the existing instance), and it
must inherit the digit-quad's boundary-sliver fix: an input just under
a multiple of D lands in the hi floor's ramp and the fractional hi is
amplified ×D in the low part; the integer snap caps it at ±1 step
(cutover find #3; regression template
`test_two_digit_boundary_sliver_snaps_to_one_step`).

Work items:

1. Torchwright: a `radix_floor_int` op (or a `levels=2` mode on
   `floor_int`), with the ramp/flat-zone contract re-derived for the
   composed form (the outer floor's legal-input contract must hold for
   `x − D·hi`), the sharpness/spacing audit at each site's input
   scale, and a **measured noise entry** in `docs/op_noise_data.json`
   before doom leans on it (repo rule).
2. Doom: convert the floor sites in slack order. Site list, lanes, and
   per-site slack are tabulated in the findings doc (R5 section);
   phase 1 = every site with slack ≥ 5 (`pix/R_DrawSpan`'s four
   N=2046 native-coordinate floors first — 24.5k lanes, slack 9).
   NOTE: the slack table was computed at the 49-layer floor at d=8192;
   re-run `scripts/critical_chain.py` after the depth track lands
   anything, and expect the schedule to reshuffle at d=4096 — the
   oracle below is the real arbiter.

Payoff for THIS plan: the step-chunk residents shrink ~10×. (The
~41k-lane saving is the same work's other dividend.)

**PROGRESS + a measured correction (2026-07-05).**

- Landed: torchwright `radix_floor_int` (branch `width-d4096`,
  commit 4dd0bbe) with tests + noise entries — the divisor-boundary
  sliver measures exactly 0 (the snap + extended-lo compensation is
  exact, not just ±1). Landed: the four `pix/R_DrawSpan` N=2046 floors
  (doom f7f19d1, flat-pixel oracle green; DAG floor stays 49; d=8192
  heuristic 65→64).
- **The "~4 sublayers" cost estimate was wrong: a composed
  `radix_floor_int` costs ~8 *layers* on its own chain** (3 floors ×
  2 sublayers + the affine glue the fusion pre-pass can't fold into
  FFN gates), and an emit digit-quad converts TWO chained floors
  (hi + snap) — ~16 layers. Converting `_digit_quad_payload`
  unconditionally regressed the DAG floor 49 → 55; the witness chain
  ran through `pix/emit emit_dq_setCursorX` L33–L49. Reverted.
- Digit-quad radix therefore needs per-site opt-in at slack ≥ ~16 (a
  `radix` flag threaded `make_token_head → emit_token →
  _digit_quad_payload`), deferred: W2 may make it unnecessary for
  d=4096 feasibility — re-probe the oracle after W2 and only build the
  plumbing if the deadlock persists and the census still names emit
  step chunks as top residents.

## W2 — two-level ray count (mandatory: 2,048 → ~64 columns)

Replace each 1,024-threshold thermometer with a two-level count:

1. **Coarse**: 32 thresholds at every 32nd candidate angle — same
   exact-count machinery as today (`_ray_count`), 32-wide
   intermediate, exact by the same argument (each term exactly 0/1 in
   fp32).
2. **Segment select**: the coarse count picks which 32-angle segment
   the true angle falls in; the 32 fine slopes for that segment come
   from a compile-time 32×32 constant table (a `broadcast_select` /
   `onehot_lookup` row pick — constants, cheap).
3. **Fine**: 32 ray tests with the selected slopes. The test becomes
   `|dy| − slope·|dx|` with a **runtime** slope — a gated product,
   which the swiglu multiply supports (this design was impossible on
   relu grids).

Derivation this plan owes before landing (the real research risk of
the track): today's exactness argument runs on *constant* slopes — the
smallest nonzero fixture ray (~1.5e-4) clears the hinge's unsaturated
band ~36×, so every indicator is exactly 0/1
(`render_ops.py:232-237`, pinned by `tests/scene/test_ray_count.py`).
With runtime slopes the ray value carries multiply noise (~2 ulp
relative), so the clearance argument must be re-derived: bound
|product noise| against the band at the fixture-extreme rays, and
extend `test_ray_count.py` to the two-level form (including
segment-boundary angles, the analogue of the digit-quad's
boundary-sliver hazard: a true angle exactly at a coarse threshold).

Cost estimate: ~3×32 lanes coarse + ~32 gated lanes select + ~2×32
fine + counts ≈ ~250 lanes (vs 4,096 today) and ~64-wide residents
(vs 2,048); +3–4 sublayers on the angle chain — check
`scripts/critical_chain.py` slack for `proj` first (the angle chain
feeds the BSP walk; it was NOT on the 49-layer zero-slack spine, but
verify at the current floor).

**RESULT (2026-07-05): W2 landed, and M2 PASSED with it — via the
dispatch fanout dial.** The two-level count computes fine rays as
``v_base + Δ·op`` — the active segment's coarse ray value (constant
Linear, picked through the segment one-hot) plus a constant-table
slope delta times the live coordinate — which is *algebraically* the
flat form's ray, so every threshold stays in today's half-integer set
and the fixture clearance transfers. Delta products stay ≤ ~160 in
magnitude (fp32 rounding ~1e-5 ≪ the 1.5e-4 clearance).
`test_ray_count.py` pins the two forms bit-equal on angle sweeps
(incl. segment-boundary-adjacent angles and quadrant reflections).
`ray_scaled` residents: 2×1,024 → 2×31. DAG floor stays 49.

After W2 the d=4096 deadlock (at the default dispatch `max_fanout=8`)
is no longer any single wide node — it's aggregate crowding at ~L45
(11 `in_range` masks ~1.3k cols + 20 held `Attn` outputs ~1.2k cols +
emit glue; the census's `DEADLOCK` dump shows it). The admission
budget doesn't unjam it; **the dispatch `type_switch` fanout does**:

| dispatch max_fanout | d=4096 heuristic | DAG floor |
|---|---|---|
| 8 (production) | deadlock | 49 |
| 4 | deadlock | — |
| 3 | **81 layers** | — |
| 2 | **100 layers** | 59 |

KV at 57.4k positions: 81×4096 ≈ 76 GB, and CP-SAT can push toward
the 59 floor (~55 GB) — far under the B200 ceiling. M3 should compile
with `max_fanout=2` or `3` (a one-line `render_main.py` change at
consolidation, output-identical per `check_fanout_equivalence`).

## W3 — if the oracle still says no

Next widest residents, in order: the 1,024-wide `in_range` mask
instances (`proj/pmrk`, `pmrk/R_CheckPlane` — can the consumer
consume 2×32 radix-split masks instead of one 1,024-wide mask?), the
`bos_weight_to_position` 1,024-lane PWL's resident output, emit
digit-quad intermediates not covered by W1. Re-census at that point
(`scripts/lane_census.py` + the peak `LIVE_DUMP`) rather than trusting
this list.

**M3 RESULT (2026-07-05): d=4096 COMPILES — 85 layers, production
CP-SAT container, artifact cached.** `make compile
CONFIG=/tmp/e1m1_d4096.yaml` (fanout=3 graph): 431 s total, 3.64 B
params used (14.5 GB fp32 weights, vs ~28 GB at d=8192); KV at 57.4k
positions ≈ 80 GB. Two caveats, both root-caused:

- **The schedule is the EAGER heuristic's 85 layers, not a CP-SAT
  optimum.** The no-eager warm start deadlocks on the production
  compile — measured cause: the RMSNorm pinned-constant reservation.
  At d=4096/fanout=3 the no-eager schedule is feasible at exactly
  4,096 free columns and infeasible at 4,095 (probe `RESERVE=1`,
  now supported by `analyze_forward_cost` / `width_census`); fanout=2
  doesn't restore the margin. With no hint, CP-SAT's floor probe
  (horizon 53) is infeasible and the 180 s descent finds nothing, so
  the compile falls back to eager — valid but unoptimized.
- **Path to a CP-SAT-optimized (~60-layer, ~56 GB KV) compile**: buy
  the no-eager form a few columns — the emit digit-quad radix
  conversion is the next lever, and the slack tables must be re-run
  on the fanout=3 graph (floor 52, schedule 85: pix/emit chains have
  far more slack than the old 49-floor table's 5–7).

The local oracle command below now needs `RESERVE=1` to match
production feasibility.

**M4 progress (2026-07-05):**

- Full `make test` suite on the branch: **green** (all 5 shards,
  277 tests incl. the flat-pixel oracle and AR rollout).
- **Lowres 160×100 COMPARE at d=4096: PASS** — coverage 100.0%,
  within-option 100.0%, exact 91.1%; 17,336 tokens in 703 s
  (~40 ms/token). Bonus datapoint: on the smaller lowres graph the
  no-eager warm start passes (75 layers) and **CP-SAT optimizes to 64
  layers** in 184 s — confirming the CP-SAT path works at d=4096
  whenever the no-eager hint exists; only the production graph's
  margin blocks it.
- **Production 320×200 COMPARE: NOT YET RUN — the one remaining
  gate.** The first attempt recompiled instead of reusing the M3
  artifact (the cache key includes the doom git SHA, and two
  formatting commits had landed since M3), and that recompile
  deadlocked while reporting a critical path of 49 (vs M3's 52 on
  the same doom code). Not chased down: torchwright was moving
  concurrently (the radix_floor_int commit was being landed to
  torchwright main at the time), so the two compiles likely shipped
  different torchwright states — Rob's call. The M3 85-layer artifact
  is still in the Modal CACHE_VOLUME under its original key
  (b8bf94dc87…). **Next session: with torchwright settled, recompile
  fresh (`make compile CONFIG=/tmp/e1m1_d4096.yaml`, recreate the
  /tmp config with `d: 4096` if gone) and run
  `make run COMPARE=1 CONFIG=/tmp/e1m1_d4096.yaml`. If the eager
  schedule flakes again on a settled tree, THEN treat the
  compile-determinism question as real and escalate per D1.**
  After the production PASS: consolidation per the contract below
  (flip both committed configs to `d: 4096` + fanout/cert-comment
  refresh in lockstep, land branches in merge order).

## The oracle (run after every landing, ~70 s local, CPU)

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \
    TORCHWRIGHT_DOOM_HUD=1 \
    SCHED_ONLY=1 OPT_GRAPH=1 D=4096 D_HEAD=128 DH=16384 \
    uv run python -m scripts.analyze_forward_cost

Deadlock (`RuntimeError: heuristic deadlocked`) = not yet; a layer
count = feasible, and the printed depth prices the KV. **The env vars
are load-bearing** — without them you measure the 60×50 hud-off graph
(the screen-env trap; findings doc, Method).

## Milestones and gates

- M1: W0 census says d=4096 is not structurally impossible.
- M2: first oracle PASS (any layer count) after W1/W2.
- M3: production CP-SAT compile at d=4096 via a **/tmp config
  variant** (copy `configs/e1m1.yaml`, set `d: 4096`; two-committed-
  configs rule — no third YAML). Record layers + solve time.
- M4: full gate stack — `make test`, the flat-pixel oracle + AR
  rollout, then `make run COMPARE=1` at lowres and production. The
  production 320×200 cert on the B200 is the finish line.
- Config lockstep: only at consolidation do `configs/e1m1.yaml` /
  `e1m1_lowres.yaml` change (`d`, and the certified-numbers comments),
  in the same commit.

Every torchwright op lands with its noise entry; every doom conversion
lands gate-green on its branch. No host-side computation anywhere
(dumb host principle).

## Consolidation contract (mirrored in `depth_flatten_plan.md`)

- **File ownership** — width track: torchwright `ops/swiglu/*` (new
  ops only), doom `render_ops.py` (`_ray_count` /
  `signed_world_angle` / floor shims), `emit.py` (digit-quad floors),
  `uv_compute.py`, `flat_state.py`, floor call sites. Depth track:
  `pixel_dispatcher.py`, `lighting.py`, `assets.py`,
  `statusbar_renderer.py`, `wall_column_renderer.py`,
  `doom_lighting.py`. Shared, additive-only: `std.py`, small
  `render_ops.py` helpers — keep those diffs minimal and coordinate.
- **Each branch stays independently green** (`make test` + its own
  oracle numbers recorded per landing) so either can land first.
- **Merge order**: whichever certifies first lands to `main`; the
  other rebases and re-runs its oracle (depth wins change the floor
  and the slack tables; width wins change nothing the depth oracle
  measures). The umbrella pointer bumps only on `main` landings.
- **The joint finish**: production CP-SAT at d=4096 with both tracks
  merged; KV multiplies (e.g. 45 layers × 4096 ≈ 85 GB at 57.4k
  positions). Config + doc refresh in lockstep.
