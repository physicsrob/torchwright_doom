# Depth experiments ledger (worktree: depth-experiments)

Baseline: main @ 93bb98e, measured on Modal (CPU): `fused=1209 floor=37`.
All measurements: `analyze_forward_cost` with the production screen env
(`RENDER_SCALE=1 320x200 DETAIL=low HUD=1`), `CRIT_PATH=1 OPT_GRAPH=1`,
via `make modal-run` with `UV_PROJECT=<umbrella> PYTHONPATH=$PWD`.
None of these are validated (no `make test` / compile gates / renders) —
experiment mode; validation deferred until we like the total.

## 1. 37 → 36 — flatten pixel-priority forks into the dispatch switch (ae262f8)

The three shared pixel branches return exclusive (mask, arm) pairs;
dispatch ANDs each mask with its transition predicate (all early) and
folds the arms into one flat `type_switch` (max_fanout 8 → 16 keeps the
sum single-level at ~15 heads). Removes one gate+sum level from the tail.
`fused=1215`.

## 2. 36 → 35 — transpose the wall texel tables (821d561)

`table_lookup_2d` consumes the row index at stage 1 and the column index
only at the final col_gate — the column operand may arrive ~2 layers late
for free. The wall banks had the late operand (row address = local·H +
v_mod, ready L24) on the early axis and the early one (u_mod, ~L21) on
the late axis. Tables now fed transposed `(W, n·H)`. `fused=1215`.
Validation note: live row vector per bank widens from W to n·H
(bank 6: 128 → 384) — recheck the d=4608 compile gate.

## 3. 35 → 36 — colormap operand flip: NEGATIVE, reverted

Materialize `COLORMAP[cmap_row]` early (constant-table read; cmap_row is
published span state) and extract by the late palette index. Measured +1
(`fused=1223 floor=36`) and reverted. **Root cause, worth keeping:**
every "index a runtime vector by a late scalar" form (`pick_by_index`,
`one_hot`+`pick_by_one_hot`) lowers through `in_range` on the late
operand — 3 late-path FFN stages. The existing 32-PWL form is
late-stage-OPTIMAL: the PWL bank converts the late palette in ONE FFN
(the PWL is the value map) and the selection gates come from the EARLY
operand (cmap one-hot). Its ~21k lanes are the price of that minimality.
This also explains the old D2 "+1" measurement — same rewrite class,
same mechanism. Corollary rule: **on the depth track, keep value maps on
the late operand and gate construction on the early operand.**

## 4. 35 → 34 — CEIL_Y via floor_int(output_map=−k) (a88583c)

Fresh-eyes provenance trace of the current cascade (chain_provenance on
Modal) showed spine layer L10 was entirely ceil_int's output affine
(add_const + negate after the saturate stage). Rebuilt CEIL_Y /
CEIL_Y_WIDE doom-side as floor_int(−x, output_map=−k): the per-step
output constants fold into the saturate weights (FLOOR_MOD64
precedent) — zero width delta, exact-math equivalent (14-point check
incl. saturation edges). The parallel FLOOR_Y_WIDE path already ended a
layer earlier, so the ceil tail was the sole binder. `fused=1215`.

## 5. 34 → 33 — max/min_screen as (a+b±|a−b|)/2 (8a22eab)

The witness's gt→select pairs (max_screen/min_screen, 2 scheduled
layers each) replaced by the exact algebraic form: |a−b| is a
3-breakpoint PWL (kink at 0), sub fuses in, half-sum Linear fuses out —
ONE FFN sublayer. Caught in testing: the PWL saturates outside its
input range (no slope extension) — range set to ±4096 to cover all real
diffs; junk beyond is bounded-wrong (permitted). All call sites upgrade
via the shared defs. `fused=1218`.

**Conservation law (from the L17–L19 analysis):** publish→read relay
chains are incompressible by moving work across the read — both spine
reads are value-bound, so publisher-side precompute deepens V exactly
as much as it saves the reader. Only algebra compression (exps 4, 5) or
key-side/front-end changes move the floor.

**Key-side position pin (provenance-measured):** recency queries are
already position-free (Q@L0), but every read's KEY embeds the L0–L3
recovered position — the front block taxes all reads from the K side.
The remaining multi-layer play is making pick_most_recent's monotone
score positional-native (past.py machinery, doom-side).

## 6. 33 stays — clip-variant arm flatten (6a negative, 6b hardening)

6a (operand-level clip resolve — pick effective bounds once at the clip
pick) measured **+1, reverted**: present's chain is depth-comparable to
the geometry, so the early pick serializes present in front of every
le. **Rule: an end-of-chain resolve whose condition computes in
parallel with the arms is load-bearing parallelism — don't hoist the
condition into the operands.** 6b kept the end-resolve and flattened
the arms (bool_and per variant, le-through-max expansion,
select(present, max/min, plain) bounds): floor stays 33 (fused=1213),
kept as anti-re-bind hardening per the paint-plan precedent.

## SPINE FLIP at 33 (the wall-track payout)

After exp 6b the wall texel twins are GONE from the zero-slack set.
New binder: the flat-span / visplane spine — pix/R_DrawSpan (55
zero-slack nodes) + proj/plan (16). Witness: PLANE_MARK input gating
(L0–L2) → pmrk radix key (L3–L6) → THREE chained plan reads (L7, L8,
L11) → multiply → floor (L13–L15) → in_range/broadcast_select →
R_DrawSpan. This chain had 5–40 layers of slack at 37; the six wall
experiments spent it. Next depth work attacks THIS chain (provenance
trace first — the old visplane_cascade_plan's unimplemented steps 3–4
targeted it and may now bind).

## Validation (2026-07-06, at 3553201 — everything except the production render)

- lint: green (black, mypy, ruff) after the cleanup commit.
- `make test`: 5/5 shards PASS, 0 failures.
- d=4608 structural compile gate (`test_forward_compiles_to_onnx`):
  **PASS**, real 115s run — the transpose's wider row vectors and the
  fanout-16 dispatch fit the column budget.
- lowres COMPARE (160×100, fresh production-width compile, 17,336
  tokens, stopped=terminal): **coverage 100.0%** (15,854/15,854, zero
  missing/extra), **within-option 100.0%** — matches the recorded
  "lowres cert 100/100" gate. Exact color match 91.6%; exact-match is
  not a gate metric and no same-scene baseline was captured this
  session — diff against a main-checkout run if it matters.
- NOT run: production 320×200 COMPARE (explicitly excluded).

## Production compile check (2026-07-06): floor 33, artifact 40 — SOLVER-limited

`make compile CONFIG=configs/e1m1.yaml` (d=8192, optimize=3):
**n_layers=40, status FEASIBLE** (not OPTIMAL). The CP-SAT floor probe
returned UNKNOWN at 154s; warm-start descent (heuristic seed 44)
reached 40 when the 300s budget expired with the solver's own bound
window still [32, 47]. The lowres compile (the COMPARE artifact) is the
same story: 36 FEASIBLE. Baseline compiled 37 OPTIMAL in the same
budget — the flattens made the schedule space harder to SEARCH (more
parallel freedom = wider branching), not infeasible.

Mechanics: the budget is `_OPTIMIZE_BUDGETS = {1:60, 2:180, 3:300}` in
torchwright `compiler/forward/compile.py:890` — no env/config knob;
`solve_schedule(solver_params=...)` is an escape hatch not threaded
through the doom compile path. The warm-start hint covered only
12,552/46,504 free variables.

Consequence: the depth wins are floor-real and gate-validated, but the
SHIPPED artifact currently regressed 37 → 40 until the solver can
close the gap. Follow-ups (torchwright-side, out of this series'
scope): raise the optimize=3 budget / add a budget knob, and/or feed a
denser hint (the heuristic scheduler's full assignment). The true
constrained optimum lies somewhere in [33, 40] — unproven either way.

## Retrace (2026-07-07): per-commit production compiles — the cliff is at exp 1, and it's a lottery, not a resource

Two independent production compiles per commit (opt3/300s + opt2/180s,
fresh solves — schedule-cache entries deleted where they blocked
re-solving):

| commit | floor | solves |
|---|---|---|
| baseline 93bb98e | 37 | **37 OPTIMAL (92s), 37 OPTIMAL (59s)** |
| exp1 flatten     | 36 | 40 F, 39 F |
| exp2 transpose   | 35 | **36 F**, 38 F |
| exp4 ceil        | 34 | 41 F, 39 F |
| exp5 max/min     | 33 | 42 F, 43 F |
| exp6b (HEAD)     | 33 | 40 F, 42 F |

Findings:
1. **The cliff is binary and at exp 1**: baseline proves OPTIMAL twice
   with time to spare; from the first flatten on, NO solve ever proves
   anything again. Serial chains pin schedule variables; the flattens
   freed them, and CP-SAT's proof machinery never recovers.
2. **Post-cliff outputs are lottery draws** (±1–2 per graph, range
   36–43 across the series). Non-monotone across nested commits
   (exp2 ⊃ exp1 yet drew 36 vs 40). Budget is a non-factor in
   [180s, 300s] (exp4: 41@300s vs 39@180s). Weak trend: later commits
   center ~2 higher — accumulating search hostility, thin evidence.
3. **The width theory is dead**: the heaviest width spender (transpose)
   produced the best draws in the series (36, 38); the width-neutral
   algebra commits produced the worst (41–43). Earlier "lb=38 width
   bind" was retracted (un-fused-graph artifact, see round-2 note in
   scripts/cpsat_lb_attribution.py); fused-graph probes never proved
   any capacity bind.
4. **Schedule-cache mechanics amplify the lottery**: keyed by graph
   fingerprint only (ignores optimize/budget) and never re-solves on a
   hit. A FEASIBLE draw gets PINNED until manually deleted (observed:
   HEAD pinned at 40; entry deleted at session end so HEAD is
   unpinned). The good 36-layer exp2 schedule was found under exp2's
   own fingerprint — a different key than HEAD's, so it never applied
   to HEAD; "orphaned" here means the win stayed keyed to a graph we
   moved past. CORRECTION (2026-07-07): the original entry here claimed
   writes are last-writer-wins; that is wrong — `store_assignment`
   (torchwright `schedule_cache.py`) refuses any equal-or-worse entry,
   a one-way min-ratchet. The "re-pinned at a worse 42" observation was
   confounded by this session's own manual cache deletes between draws.
   Repeated re-solves therefore already ratchet monotonically; what's
   missing upstream is only the re-solve-on-hit trigger.
5. **The remedy that follows from the data** (torchwright-side, out of
   this series' scope): make the schedule cache a RATCHET — re-solve on
   hit when budget allows and keep the better schedule. Every compile
   becomes a lottery ticket; the pin becomes monotone improvement. Six
   graphs × 2 draws already produced a 36 (< baseline 37) once. Plus:
   denser hints (production hint covers 12,552/46,504 vars; first
   incumbent 48 is WORSE than the heuristic's own 44 — hint
   transmission is demonstrably lossy) and seed variation.

Bottom line: the floor wins (37 → 33) are real and correctness-
validated; the artifact regression is a solver-search phenomenon
triggered by the very first flatten, not a resource cost of any
specific change; and the shipped number is currently whichever lottery
draw happens to be cached.

## Audit campaign (2026-07-07): production-exact re-measurement — most retrace claims revised

Instrument: `scripts/cpsat_prod_harness.py` — replicated the production
solve's model construction bit-for-bit and PROVED it by fingerprint
equality with the production compile's own prints (HEAD `5d86ed63ded1`,
baseline `89aa36948a5f` — both matched exactly) plus behavioral
replication (baseline 37 OPTIMAL @60s vs prod 59s; HEAD floor probe
UNKNOWN @153s vs prod 154s; n_vars 46,504 exact). torchwright pinned
clean at `9cbc2a3` for every run. Every earlier probe in this worktree
(cpsat_space_experiments, cpsat_lb_attribution) was NOT production-exact
— see the fidelity caveats in those files.

**DELETED (2026-07-09)** along with `cpsat_domain_stats.py` /
`cpsat_occupancy.py`: the replica rotted against two torchwright
API changes in one day (the `mark_clean` removal, the 5-tuple
warm-start return) — a standalone copy of production construction
re-diverges on every compiler change, the exact defect it was built
to fix. Production-exact measurement now goes through the real
`forward_compile` (`scripts/cpsat_gap_attribution.py`, solve-only,
same fingerprint gate) or the CP-SAT fixture layer
(`modal_cpsat_fixture.py`), which snapshots the problem from inside
the real compile.

Corrections to the retrace section above (kept in place for the record):

1. **The lowered DAG floor at HEAD is 32, not 33.** Every "floor" here
   was measured without the production collapse passes; production
   lowering shortens the spine by one more. Per-commit production
   floors: base 37, exp1 36, exp2 35, exp4 34, exp5 33, HEAD 32.
2. **"Baseline proves OPTIMAL, post-cliff never proves" is wrong in
   both directions** (5 seeds x 180s, production-exact): base
   37/37/37/38F/38F; exp2 proved 35 OPTIMAL on one seed. The real law
   (exact across all ~40 solves of this campaign): **a proof happens
   iff the found schedule's depth equals the commit's dependency
   floor.** CP-SAT's only working lower bound on this model is the
   dependency longest-path (free, from propagation); it NEVER moved
   above the floor in any run (up to 3600s, any portfolio). Proofs are
   floor-coincidences, not solver health.
3. **"Budget-insensitive" only held for [180s, 300s].** HEAD medians:
   40 @180s, 39 @300s, 35 @1800s (n=5/6/3). The 180s draw is
   descent-rate-limited (see mechanism below), so more budget = more
   rungs.
4. **The hint-transmission anomaly is resolved — transmission is
   faithful.** The "repaired first incumbent 48" IS the no-eager
   warm-start hint accepted verbatim (measured: first incumbent 48
   tagged [hint] at 13s); the "heuristic's own 44" is the EAGER
   fallback schedule, which is deliberately un-hintable
   (model-inexpressible in-layer column reuse). Nothing lossy.
5. **"Width theory is dead" is REVERSED at the frontier — and refined.**
   Family isolation at K=32 (production-exact, 64 CPU): dependency-only
   SAT/OPTIMAL in 1s; +cancel machinery 5s; +heads+MLP (only
   residual_cumulative off) 31s; ALL families on = UNKNOWN at 3600s x2.
   **Residual-column capacity is single-handedly what makes near-floor
   construction intractable.** But see the two-bottleneck mechanism:
   width governs the frontier (~33-34), NOT the 180s draw (~38).
6. **"The flattens freed the schedule variables" is dead.** Domain
   stats (scripts/cpsat_domain_stats.py) at the production horizon:
   0% of nodes pinned at ALL six commits, mean layer-domain width
   31-33 everywhere. Freedom was always maximal; nothing changed at
   the cliff.

**The mechanism, in full (replaces the phase-transition story):** the
depth experiments removed serialization without removing work, so the
packing density required at the floor rose monotonically (baseline's
optimal 37 already peaks at exactly 100.0% of column capacity, mean
66%). Two separate bottlenecks emerged:

- **The frontier (~33-34):** feasible packings become rare. A
  capacity-blind 32-layer schedule needs 177% of capacity at its worst
  layer (mean 93.7%). Construction there fails cold (all fixed-K
  probes UNKNOWN) and the bound side is blind (relaxations of the
  cumulative cannot see fragmentation), so [32, achievable] stays
  formally open. A 33-layer schedule for the PRE-W2 HEAD graph was
  CONSTRUCTED by iterated descent (48 -> 35 -> 34 -> 33, each rung a
  fresh solve hinted with the previous best at horizon best+1; hard
  K-ceilings kill the hint — measured 3x — only objective descent
  works). Two 1800s attempts at 32 both held: 32 unresolved. That
  schedule is orphaned by the W2 merge (fingerprint change) and by the
  no->300s-strategies decision; recorded here as an existence proof.
- **The 180s draw (~38):** the production solve descends from the
  48-layer warm start and runs out of budget mid-descent, far above
  the frontier. Rungs are slow everywhere (~100s+ each even at loose
  depths) because a single layer-drop relocates ~600 of 5,362 nodes
  (measured by schedule diff; only ~40 nodes live in the top-3
  layers) — local repair rarely finds such moves. Width relief cannot
  help this regime, and measurably didn't (below).

**Width-reducer merge (from width-d4096):** W2 two-level ray count
merged (ray residents 2x1024 -> 2x31; floor stays 32; new fingerprint
`b4240c2dcf67`). Result at 180s, n=15 vs n=15: **median 38 -> 38 — no
detectable depth effect** (the relieved columns were not in the
pressure band, and 180s never reaches the width-limited frontier).
Kept: costs nothing, serves the d=4096 track. W1 radix floors measured
at +7 floor layers on this branch (32 -> 39: the flat-span chain is
now the zero-slack spine; the width branch had slack 9 there) —
excluded; would need torchwright `radix_floor_int` to learn the
`output_map` fold, and even fused stays spine-hostile. Dispatch
fanout 8->3 excluded (reverses exp1).

**Pressure attribution** (scripts/cpsat_occupancy.py, band = layers at
>=95% capacity): the band is dominated by `floor_int_step` FFN
intermediates — 512-wide, in GANGS (8x pix/R_DrawSpan co-resident, 4x
pix/R_DrawColumn, 4x stor/paint) — ~50% of the feasible 36-layer
schedule's band. Depth-safe width targets = the WALL-side floor sites
(R_DrawColumn/stor: slack-rich post-flatten); the R_DrawSpan sites are
spine-locked. Caveat learned: capacity-blind schedules overstate
immovable pressure (lazy placement) — `emit_derived_zero` (1,390-col
zero tail) looked like the #1 occupant but real schedules defer it to
the tail layers (verified: born L27 in the feasible 36); sanity-check
every target against a feasible schedule.

**Decision (2026-07-07, Rob): the target is <=35 layers from a plain
<=180s compile; no compiler strategies over 300s.** Configs flipped to
`optimize: 2` (both, lockstep) — this is a deliberate cap. Status
against that bar: NOT met (180s median 38, min 36 at n=15). The
evidence-ranked path to it:

1. **Better warm start (torchwright, targeted):** the 180s budget is
   spent descending 48 -> ~38; the eager heuristic already builds 44
   but is model-inexpressible as a hint. Closing the hint gap starts
   the descent 4-5 rungs lower — the single highest-leverage item.
2. **Cache pre-seeding — mechanism replaced (2026-07-09):** manual
   injection (`modal volume put torchwright-doom-schedule-cache`) is
   closed; it was a side-channel around the compile, and the scratchpad
   36 it would have injected keyed to a topology that no longer exists.
   The schedule-cache fingerprint now includes the torchwright source
   hash (any compiler change invalidates and re-solves) and entries are
   gated on the optimize level they were validated at. Schedules enter
   the cache exactly one way: a real `make compile` solve, which
   ratchet-stores its own win. Pre-seeding is therefore "run `make
   compile` once after the code settles" — the win replays for every
   later compile of the same code + topology.
3. **Fused-radix wall-side floors (torchwright):** moves the frontier,
   not the 180s draw; relevant only after (1), or for a future 33/32
   push.

## Open next candidates (in rough value order)

- Early bank pick / shared column stage (candidate −1): bank mask is
  ready ~L22 but selects after the 8 lookups (L28–29). One shared
  col stage over a bank-picked row vector + bank-picked column address
  moves the palette from L29 to L27. Needs `table_lookup_2d` split into
  row/col stages — torchwright surgery (worktree there).
- `clamp_j` skip-rule relaxation (candidate −1): escalated item; the
  clamp on the (now column-side) address is kept only because the skip
  rule wants exact [0, top] while mod outputs carry ±ε. torchwright-side.
- v-chain floor+mod fold (candidate −1, co-binds with metadata at L24 —
  both must shorten): multi-`output_map` floor_int sharing the step
  stage; the naive form was reverted once for residual-column blowup.
- Front-end (scene L0–L3 + read hoisting via recency-biased scores):
  candidate −2..−3, doom-side but structural.
- Paint cascade flatten (L10–L19): biggest block, needs dependence trace.
