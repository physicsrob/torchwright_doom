# SwiGLU opportunities — research plan (shallower / smaller network)

Handoff for a fresh session. **This is a research plan, not an
implementation plan**: the output is a findings doc with per-item
go/no-go decisions and sized derivations — no flagship changes land
from this work. Drafted 2026-07-04, immediately post-cutover
(`swiglu_cutover_plan.md` execution record; doom `c7a4983`,
torchwright `2fb6bd6`).

## Why

The cutover's savings were mechanical. The machine's new primitive —
a gated lane computes `Swish(affine₁(residual)) × affine₂(residual)`,
a *product* of two functions of the residual with smooth curvature,
where a relu lane computed only `relu(affine)` — enables designs that
were numerically impossible or unaffordable before. Nobody has yet
asked which of doom's expensive constructions should be *re-designed*
rather than merely re-costed.

## What savings buy (read this before ranking anything)

- **Lanes** buy `d_hidden` headroom (production runs decoupled 16k).
- **Sublayers** buy layers; the production compile is 49 layers
  (CP-SAT OPTIMAL at the current topology).
- **Layers are the only currency that buys KV-cache relief**: at full
  frame (~57.4k positions, stock-Phi3 uniform 64 heads, fp32) each
  layer costs ~3.8 GB of unbounded cache — and the production 320×200
  certification is currently blocked at ~221 GB total vs a B200's
  178 GiB (cutover plan, open item 2). Depth-reducing items therefore
  outrank lane-reducing items if unblocking full-res matters.

## R0 — measure before believing anything (do this first)

Every estimate below is a session-drafted guess. Build the ground
truth:

- **Lane census by op family**: where do the 16k hidden lanes
  actually go post-cutover? Torchwright `b12bac3` ("scaling harness:
  count FFN lanes") exists — reuse it, or walk the doom graph the way
  the cutover's energy audit did (BFS over `.inputs` from
  `render_main.forward`'s output; see the execution record's find #2
  for the pattern). Bucket by op (`table_lookup_2d`, `floor_int`,
  `reciprocal`, `_ray_count`, selects, everything else).
- **Depth census**: what occupies each of the 49 layers / what chain
  is critical-path? `scripts/analyze_forward_cost.py` and
  `scripts/widest_nodes.py` exist — check what they already report
  before writing anything new.
- Rank R1–R6 by measured budget share; drop anything under ~2% of
  its budget without further work.

## The items

Each: hypothesis → what to derive/measure → decision criterion.
Prototypes live in scratch/tests, noise-measured through torchwright's
op harness (`make measure-noise` workflow, torchwright CLAUDE.md)
before any go verdict.

### R1 — Newton-iteration reciprocal (lanes; M effort)

Hypothesis: replace the geometric-breakpoint PWL `reciprocal`
(torchwright `ops/swiglu/arithmetic_ops.py:509`; ~1,500 breakpoints on
the production denominator range) with a ~32-lane PWL seed + 2 Newton
steps `x ← x(2 − a·x)` — each step two *exact* swiglu multiplies,
2 sublayers. Impossible on relu (grid-multiply noise breaks
convergence). Research: count actual doom reciprocal sites + their
real lane costs (R0); derive convergence under fp32 + the swish dip
(seed accuracy needed for 2-step fp32 floor); check the emitted
`DRAWSEG_SCALE1_DEN` carrier's 16-bit linear quantization (see the
cutover plan's D4 near/far note) doesn't already dominate the error —
if the carrier quantization is the accuracy floor, a cheaper
reciprocal loses nothing. Decision: lanes saved vs sublayers added on
the drawseg critical path; needs a new torchwright op + noise entry
if it wins.

### R2 — flatten the dispatch select ladder (DEPTH → layers → KV; M–L effort; highest strategic value)

Hypothesis: the nested select chains that pick the winning branch's
emit head (`wall_range_builder.py:289–307`, ~8 deep;
`render_main.py`'s dispatch generally) collapse to one gated sum
`Σ maskᵢ·headᵢ` in 1–2 sublayers. Newly viable: relu select error was
δ·M with M ~ the branch-range union (~±46k emit heads — fatal);
swiglu costs δ·|value|. Research: (a) map every select ladder on the
critical path + its depth; (b) the garbage-row contract — the
`broadcast_select` junk-mask guarantee requires zero in the
branch-range union; a flat sum over emit *heads* has different
discarded-row behavior — derive the bound the way the cutover derived
`_ray_count`'s; (c) mask-noise × head-magnitude vs the emit argmax
margin (~0.97 in ~46,000 — see `snap_bool`'s docstring for the
failure class); (d) prototype compile at the d=4096 gate
(`tests/scene/test_forward_compiles.py` is the cheap harness) to
measure the actual layer-count change. Decision: layers saved; each
layer is ~3.8 GB of full-frame KV.

### R3 — `table_lookup_2d` axis rebalance (lanes; S–M effort; already specced)

Already a cutover-plan D4 bullet with torchwright's spec example
(~6.5k → ~3.1k lanes on the flat bank at 512×512 axes). Research is
arithmetic + reading: per-table Pareto for the four `assets.py`
lookup sites (shapes from `asset_banks.py`), counting the
divide-and-remainder index overhead honestly. Mostly a sizing memo.

### R4 — swish curvature as a function basis (lanes; L effort; highest risk)

Hypothesis: smooth targets fit with tens of swish basis lanes instead
of hundreds of PWL breakpoints (a swish lane is a smooth ramp; relu
lanes are bends). Constraint that kills most uses: doom's spine wants
**exact integers via saturation**, not smooth values — so candidates
are only chains whose consumers hold tolerances
(lighting/colormap: `doom_lighting.py`, `lighting.py`; NOT the
BAM/floor geometry). Research: inventory PWLs with toleranced
consumers (from R0); offline fit experiments (scratch script:
lanes-vs-error curves for the real targets); sketch the torchwright
op + its noise-measurement story. Decision: only worth building if a
toleranced chain is a top-5 lane consumer in R0.

### R5 — radix-decomposed floors (lanes ↔ depth; M effort)

Hypothesis: `floor_int` at N boundaries costs ~2N lanes; a two-level
radix split costs ~4√N + 2 sublayers (2,046 boundaries → ~180 lanes).
Not swiglu-specific — newly dominant because the grids died and
staircases are now the width giants. Research: enumerate every floor
× boundary count (R0); port the digit-quad boundary-sliver analysis —
the two-level split inherits exactly the fractional-hi-digit hazard
the cutover fixed with the integer snap (`emit.py`,
`_digit_quad_payload`, and its regression test) — that precedent is
the correctness template. Decision: per-floor lane/depth table.

### R6 — the unadopted swiglu library (S effort; do early, it's cheap)

The flip only mirrored existing imports. `torchwright/ops/swiglu/`
ships modules doom uses nowhere: `onehot_table.py`,
`marker_count.py`, `scalar_encoding.py`, `sequence_ops.py`,
`embedding_arithmetic.py`. Research: read each against doom's
hand-rolled equivalents (`std.py`'s one_hot/pick machinery, the emit
digit-quad, protocol marker handling) and note where a purpose-built
op is cheaper or shallower than what doom builds by hand. This is
also where genuinely new "others" ideas will surface — the library
authors already thought about this machine.

## Constraints (non-negotiable, from the repo doctrine)

- Dumb host principle: nothing moves computation to the host.
- Two committed configs only; experiments use /tmp variants.
- Exactness-by-saturation is the design backbone: any smooth-for-exact
  trade must name its consumer's tolerance and prove the fit — the
  cutover's pattern (derivation → pinning unit test → oracle gates)
  is the template for any follow-on landing.
- Carriers quantize linearly (16-bit over the range span) — encoding
  redesigns (log carriers, the D4 near/far note) are out of scope
  here unless explicitly promoted to their own plan.
- Per-op noise budgets: torchwright `docs/op_noise_data.json` is
  canonical; new ops need measured entries before doom leans on them.

## Deliverable and gates

One findings doc (suggest `swiglu_opportunities_findings.md`, doom
root): per item — the measurement, the derivation, a go/no-go, and
for "go" items a sized implementation sketch (lanes, sublayers,
expected layer delta, which gates certify it). No flagship code
changes from the research session itself. `make test` untouched/green
throughout (research artifacts live in scratch or /tmp).

## Session-start pointers

- `swiglu_cutover_plan.md` — execution record (the five finds are the
  house style for root-causing), D4 list, open items.
- `GLOSSARY.md` — carrier, digit-quad, lifted key, visplane, etc.
- torchwright: `docs/ops_plain_english.md` (the op spec),
  `docs/op_noise_data.json` + `docs/numerical_noise_findings.md`,
  `docs/swiglu_step2_plan.md`.
- Harnesses: `make test-local FILE=…` (fast local), `make test`
  (Modal suite), `make modal-run MODULE=…` (GPU scripts),
  the d=4096 compile gate, `torchwright.debug.probe` (oracle probes).
