# SwiGLU opportunities — findings (research session, 2026-07-04)

Results for the research plan in `swiglu_opportunities_plan.md`. Every
number below was **measured on the production graph topology** (320×200
low-detail, HUD on — see "Method" for the env trap). Per the plan, no
flagship changes land from this session; the one committed change is a
repair to the measurement tooling itself (`scripts/analyze_forward_cost.py`,
see "Tooling"). Prototypes were built, measured, and reverted.

## Verdict summary

| Item | Verdict | Measured size |
|---|---|---|
| d=4096 via width reduction (post-review find) | **GO, gated** | d_model must be a power of two (RMSNorm cancellation), so the target is 4096 — today it deadlocks; ray two-level + R5 unblock it; payoff 92–113 GB KV vs 184 |
| R5 radix floors | **GO** (phased) | 73,452 lanes (50.1% of all hidden lanes); phase 1 saves ~41k at zero depth cost; also shrinks the residual peak (min-d lever) |
| Colormap → `table_lookup_2d` (new find) | **GO** | 21,168 → ~3,600 lanes (~12% of total width), depth-neutral |
| R2 flatten select ladders | **GO** (highest strategic value) | measured **49 → 47 layers** from one mechanical edit; ~7.6 GB of full-frame KV |
| R6 unadopted library | **1 adoption** (`onehot_lookup`), 4 passes | ~8–10 constant-table pick sites; 1 sublayer each, some upstream of the critical spines |
| R3 axis rebalance as specced | **NO-GO** | < ~1.5k lanes at production shapes, before paying index arithmetic |
| R4 swish function basis | **NO-GO** | top candidate is not smooth; all smooth targets < 2% |
| R1 Newton reciprocal | **NO-GO** | premise stale: no reciprocal op exists in the graph |

The two structural facts that reframe the whole plan:

1. **Production depth is purely dependency-bound — at d=8192.** The
   compiled 49 layers exactly equal the dependency-graph longest path
   after the always-on linear-fusion pre-pass (measured: unfused floor
   52, fused floor 49, production compile 49). The CP-SAT schedule sits
   AT the topological floor; at the production width the residual
   stream has slack. Consequences: **hidden-lane savings buy zero
   layers** (they buy weight memory and per-layer compute), and every
   layer saved must come from shortening a dependency chain.
   **Correction (Rob, post-review): depth is not the only KV lever.**
   The KV cache is `layers × d_model × positions` — narrowing the
   residual stream cuts it exactly like removing layers does. The
   measured d-bracket below shows the width slack survives down to
   ~d=5120; with d_model constrained to powers of two (RMSNorm
   cancellation), the actionable target is d=4096, gated on the width
   reducers. See "Width is a KV lever" below.
2. **The critical spine is narrow and known.** Only 150 of 6,608
   schedulable nodes have zero slack. The spine: BOS/global-position
   recovery (layers 0–4) → the wall lighting + branch cascade under the
   `proj/paint` annotation (layers 5–27, including two attention reads)
   → the wall-column texel fetch + colormap application
   (`pix/R_DrawColumn`, layers 28–41) → the pixel-emit select ladder and
   dispatch output assembly (layers 42–48). Everything else — the whole
   flat-span pipeline, the emit digit-quads, HUD, weapon, planes — has
   5–40 layers of slack.

## Method (read before re-measuring anything)

**The screen-env trap is real and it bit this session.** The graph
modules read screen dimensions from env vars at import; without them
you silently measure a 60×50 HUD-off graph. First-pass numbers were
wrong: 111,766 total lanes vs the true 146,655; DAG floor 51 vs 52.
Every command below ran under:

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \
    TORCHWRIGHT_DOOM_HUD=1

Instruments:

- **Depth / DAG floor**: `scripts/analyze_forward_cost.py` (repaired
  this session, see "Tooling"). `CRIT_PATH=1 OPT_GRAPH=1` gives the
  production-comparable dependency floor in ~2 minutes, no compile, no
  weights. `SCHED_ONLY=1` gives the heuristic schedule (65 layers at
  production dims — width-bound, 16 above the CP-SAT floor; useful for
  attribution only).
- **Lane census**: a scratch script (session scratchpad,
  `lane_census.py`) walking the graph exactly like
  `scripts/widest_nodes.py` and sizing each FFN node by its `gate_proj`
  row count — the torchwright `b12bac3` pattern. (Note:
  `widest_nodes.py`'s "widest MLP-hidden (ReLU nodes)" section is
  silently empty post-cutover — same blindness `b12bac3` fixed in the
  torchwright scaling harness; candidate repair, not done here.)
- **Critical-chain attribution**: a scratch script (`crit_chain.py`)
  reusing the CP-SAT scheduler's own mode-aware earliest/latest layer
  bounds (`cpsat_scheduler._compute_layer_bounds`); pinning
  `max_layers` to the floor makes "earliest == latest" exactly the
  union of all critical chains.

## Lane census (production topology)

146,655 total hidden lanes across 2,351 FFN nodes (plus 340 attention
nodes; zero legacy relu nodes — the cutover left nothing behind).

| op family | lanes | share | what it is |
|---|---|---|---|
| `floor_int` | 73,452 | 50.1% | integer-floor staircases (coordinates, indices, digit-quad bytes) |
| `colormap_row` | 21,168 | 14.4% | wall lighting: 32 COLORMAP rows as separate 256-breakpoint tables + a runtime pick |
| `table_lookup_2d` | 16,190 | 11.0% | texture/flat/HUD/weapon texel tables |
| `in_range` | 13,584 | 9.3% | interval-membership mask vectors (4 lanes per slot) |
| select-family | 11,526 | 7.9% | 1,433 mostly-tiny gates (select / cond_gate / broadcast_select / compare) |
| `ray_count` / `ray_scaled` | 4,096 | 2.8% | the cutover's swiglu-ffn ray banks |
| generic `pwl_*` | 1,902 | 1.3% | texture-coordinate wraps etc. |
| everything else | ~4,700 | ~3.2% | visplane query, BOS position, sawtooth, clamps, … |

By renderer subsystem: `pix` 56.7%, `stor` 14.2%, `proj` 13.4%,
`dispatch` 6.9%; the rest under 3% each.

## R1 — Newton-iteration reciprocal: NO-GO (stale premise)

There is no reciprocal op in the production graph. The lane census
finds zero nodes named `reciprocal` (torchwright's op names its nodes
that), and doom's only textual references are a docstring and a
comment. The reciprocal moved into the **embedding table**: every
`VALUE` carrier row bakes derived columns `v{idx}` (decoded value) and
`inv{idx}` (zero-guarded 1/value) at compile time
(`value_ranges.value_derived_columns`), and the graph reads the `inv`
columns directly (`protocol_tokens.py:301–313`, ranges R5/R6/R7 — the
scale denominators and widths). A reciprocal via embedding lookup costs
zero lanes and zero sublayers at runtime; a 2-Newton-step design cannot
beat that.

The accuracy question R1 raised is real but belongs to a different
plan: the binding constraint is the carrier's 16-bit **linear**
quantization (the cutover plan's D4 near/far note — log-encoded
carriers), explicitly out of scope here.

## R2 — flatten the select ladders: GO, measured 49 → 47

**What was measured.** The plan's guess about *which* ladder matters
was wrong. The 12-deep `wall_range_builder.py` VALUE ladder is NOT on
the critical spine (it has slack; flattening it buys zero layers). The
ladder that binds is the pixel-emit tail in `pixel_dispatcher.py`: the
three shared pixel/cursor branches each wrap a priority ladder —
`select(hud_seen, …, select(weapon_seen, …, select(flat_span_seen, …,
wall)))` — and `after_pixel_color`'s sits at layers 42–45 of the spine.

**The prototype** (built, measured, reverted — the diff shape is
recorded here so landing it is mechanical):

- The ladder's conditions are latch flags, so the ladder encodes
  *priority*, not mutual exclusion. The flat form derives exclusive
  masks with one parallel layer of boolean machinery:
  `m_hud = hud_seen`, `m_weapon = all(weapon_seen, not hud_seen)`,
  `m_flat = all(flat_span_seen, not weapon_seen, not hud_seen)`,
  `m_wall = all(not flat_span_seen, not weapon_seen, not hud_seen)`
  using `bool_all_true` / `bool_not` (torchwright `logic_ops`), then
  one `type_switch` over the four (mask, head) pairs.
- The masks derive from the early latch flags, so the mask network runs
  in parallel with the branch-arm chains and adds no depth of its own.
- Measured DAG floor: 49 baseline → **47** with `after_pixel_color`
  flattened → still 47 with all three shared branches flattened (the
  other two were never binding; flatten them anyway when landing — they
  cost nothing and remove future cliff edges).

**Why it is numerically safe** — this extends a pattern production
already certifies, not a new trick. The main dispatch already
flat-folds emit heads with `type_switch`/`cond_gate` at the same head
magnitudes. The select/cond_gate contract (op docstring): at a clean ±1
condition the losing branch contributes *exactly* zero in fp32 and the
winner passes within ~1 ulp relative — there is no additive offset, no
finite-range requirement, which is precisely what made this fatal on
the relu machine and viable now. The discipline that must hold: every
mask must be a snapped ±1 boolean (the latch flags and `bool_*` outputs
are; anything recovered through a softmax pick must keep passing
through `snap_bool` — `render_ops.py:83` documents the failure class
against the emit argmax margin of ~0.97 in ~46,000).

**What binds at 47.** Two parallel spines of equal length:
`pix/R_DrawColumn` and `stor/R_StoreWallRange` (each runs its own texel
fetch + colormap chain), both inheriting the same layers-0–27 prefix.
The roadmap, in decreasing value-per-effort:

1. **The `proj/paint` cascade (layers 5–27, ~23 layers, shared by both
   spines)** — the wall lighting selection plus wall-column branch
   logic (`wall_column_renderer.py` has 22 selects; `doom_lighting.py`
   feeds it). It interleaves compares/selects with two attention reads
   (layers 11 and 20). Research question for a follow-up: can the
   attention reads hoist (read the past unconditionally in parallel,
   select afterward), or do their queries genuinely depend on earlier
   select outputs? Every hoisted read + flattened stretch is a layer.
2. **The colormap application stack (4 layers on each spine)** — see
   the colormap section below; the compose-into-the-texture-table idea
   removes it entirely.
3. **`table_lookup_2d`'s internal 4-layer stack** (clamp → row step →
   row deltas → column gate) on each spine — irreducible without a new
   op design; note only.
4. The output-assembly Linear run (layers 46–48) — the fusion pre-pass
   already eats what it can (686 → 695 fused pairs with the flatten in
   place); likely ~1 layer of slack hiding here at most.

**Cost of landing the tail flatten**: the masks add ~10 tiny lanes; the
switch replaces the ladder's gates (lane-neutral). Gates to run: the
graph-level oracles (`test_flat_pixel_oracle`, `test_forward_ar_rollout`),
the d=4096 compile gate, then the lowres + production `COMPARE=1`
walkthroughs. The priority→exclusive-mask rewrite must preserve the
ladder's degenerate-case behavior (before the flat pass,
`flat_span_seen` is structurally false and the fork must degenerate to
the wall arm — same head, one mask evaluation).

## R3 — `table_lookup_2d` axis rebalance: NO-GO as specced

The cutover plan's D4 example (~6.5k → ~3.1k lanes on a 2048×128 flat
bank) describes a shape that does not exist at production. Measured
shapes and the exact `3·(rows + cols)` cost (formula verified against
the census: flat instance = 3·(384+64) = 1,344 ✓):

| site | table shape | lanes/instance | best rebalance | saving |
|---|---|---|---|---|
| flat (`pix/R_DrawSpan`, ×2) | 384×64 | 1,344 | 192×128 → 960 | ~384 each, before index math |
| HUD (`pix/hud`, ×2) | 359×320 | 2,037 | **359 is prime** — current split is already the divisor optimum | ~0 |
| wall banks (×2 sites) | mixed, largest 384×128 | 4,495/site total | ~192 per bank | small |
| weapon (`pix/pspr`) | 46×26 | 217 | — | ~0 |

Total honest headroom < ~1.5k lanes (~1%) before paying each
rebalance's divide-and-remainder index arithmetic (a floor + linear per
site, ~100+ lanes each), and the plan's own 2% cutoff drops it.

**The real table win found by the census is the colormap.**
`lighting.apply_colormap_row` builds all 32 COLORMAP rows as separate
256-breakpoint piecewise-linear tables and picks one at runtime
(`lighting.py:55–58`) — 21,168 lanes, the #2 consumer, and 4 layers on
both critical spines. That is a 32×256 2D constant table built the
expensive way. Replacing each of the 4 instances with
`table_lookup_2d(light_row, palette_index, COLORMAP)` costs
3·(32+256) ≈ 864 + clamps ≈ ~900 lanes/instance: **21,168 → ~3,600
lanes (~12% of total width), depth-neutral** (the lookup's 4 internal
layers replace the current 4-layer pick stack). The op's noise entry is
exact (0 abs / 0 rel on the integer grid) — same contract class as the
current path, which also requires integer row indices.

To verify before landing: (a) the light-row input is already
integer-snapped at every call site (the current `pick_by_index` design
needs that too, so this should hold — confirm); (b) a fractional row
in-band blends *adjacent COLORMAP rows'* palette indices — palette
indices are not ordered by brightness, so the blend contract is
garbage-in-band; confirm the inputs never sit in-band (same argument
the flat/wall texel lookups already make).

**Stretch idea recorded for the R2 roadmap: compose the colormap into
the texel tables.** `lit = COLORMAP[light][TEX[row, u]]` is a
compile-time-composable function of `(row·32 + light, u)` — the
composed index is exact linear arithmetic (free), the composed table is
32× the entries (weights, not lanes — a few MB), and balanced axes put
the composed wall table near 3·2·√(1.57M) ≈ 7.5k lanes vs today's
~9.8k (texel + colormap) per spine — while deleting ~4 layers from BOTH
critical spines. Needs the same integrality verification plus a
carefulness pass on which light row varies per pixel vs per column.

## R4 — swish curvature as a function basis: NO-GO

The census kills it cleanly:

- The only large "toleranced consumer" candidate — lighting/colormap —
  turned out to be a jagged integer table (palette indices are
  unordered in index space), not a smooth target. A smooth basis cannot
  represent it; the right tool is the exact 2D lookup above.
- Every genuinely smooth PWL target in the graph is tiny: the generic
  `pwl_*` family totals 1,902 lanes (1.3%), the BOS-weight→position
  curve 1,026 (0.7%), sawtooth wraps 852 (0.6%). All below the plan's
  2% cutoff individually; even a 10× compression on all of them saves
  ~3k lanes (2%) for a new op family + noise-measurement machinery.

The exact-integers-via-saturation backbone stands: nothing on the
spine tolerates smooth values.

## R5 — radix-decomposed floors: GO, phased by measured slack

**The cost model** (from `torchwright/ops/swiglu/arithmetic_ops.py::floor_int`):
a floor over N boundaries costs exactly **3N lanes** — 2 per boundary
in the step stage plus 1 per boundary in the count stage — in 2
dependent sublayers, chunked at 512 boundaries per FFN. The two-stage
shape is load-bearing for fp32 accumulation (the docstring forbids
flattening it); the census confirms the model everywhere (e.g. the
N=2046 native-coordinate floor = 4× (1024/1020 step + 512/510
saturate) = 6,138 lanes).

**Where the 73,452 floor lanes live**, with the slack each site's chain
has under the 49-layer floor (a radix rewrite adds ~4 sublayers to its
own chain, so slack ≥ ~5 means depth-free):

| site | floor lanes | min slack | phase |
|---|---|---|---|
| `pix/R_DrawSpan` (4× N=2046 native u/v floors) | 24,552 | 9 | **1** |
| `pix/R_DrawColumn` (2× N=2046 + digit-quad pairs) | 15,342 | 0 (p50 = 0) | 2 — on the spine |
| `stor/R_StoreWallRange` | 7,524 | 2 | 2 |
| `dispatch/paint`, `pix/pspr`, `pix/hud`, `pix/emit`, `dispatch/emit` (digit-quad pairs) | 3,066 each | 5–28 | **1** |
| `hud/ST_Drawer`, `paint/emit`, `pspr/…`, `pmrk/R_CheckPlane`, `dispatch/stor` | 1,533 each | 24–39 | **1** |
| `proj/paint` | 2,655 | 0 (p50 = 6) | 2 (mixed) |
| `proj/plan` | 384 | 6 | 1 |

**The sketch.** A floor over N boundaries splits radix-style with
divisor D ≈ √N: `hi_raw = floor(x/D)` (N/D boundaries) → snap `hi` to
an exact integer (a second small floor at half-integer offsets) →
`lo = floor(x − D·hi)` (D boundaries; `D·hi` is a compile-time-constant
scale, exact linear). Cost ≈ 3·(N/D + (N/D+1) + D) ≈ **9·√N lanes and
+4 sublayers** vs 3N and 2. For N=2046: ~410 vs 6,138 (15×).

Phase-1 scope (every site with slack ≥ 5): 47,931 lanes → roughly
6,500 post-radix — **~41,000 lanes saved (~28% of the machine's total
hidden width) at zero depth cost**. Phase 2 (the on-spine column/stor
floors, ~25,500 lanes) unlocks after R2 buys slack on those chains.

**Correctness template — this is the digit-quad, generalized.** The
emit path already runs exactly this structure for carrier bytes:
`floor(q/BASE)` then an integer snap then the low byte
(`emit.py::_digit_quad_payload`), and the cutover's find #3 is the
hazard this design must inherit the fix for: an input just below a
multiple of D lands in the hi floor's ramp, the fractional hi digit is
amplified ×D in the low part; the snap caps it at ±1 step
(`test_two_digit_boundary_sliver_snaps_to_one_step` is the regression
template). The derivation a landing session must produce: the ramp/
flat-zone contract for the composed op (the outer floor's legal-input
contract must hold for `x − D·hi`), the sharpness/spacing audit at each
site's input scale, and a measured noise entry in torchwright's op
harness (`docs/op_noise_data.json`) before doom leans on it — the op
does not exist in torchwright today.

**What R5 buys and doesn't.** With production depth dependency-bound,
these lanes buy weight memory and per-layer compute, not layers. If
d_hidden could drop 16,384 → 8,192 after phases 1+2 (peak-layer packing
must be re-checked), the MLP weight blocks halve. **And R5 has a
second, residual-width effect** (see "Width is a KV lever" below): a
floor's stage-1 step vector is a residual-resident intermediate up to
512 columns wide per chunk — the production peak live-set contains
2,627 columns of floor-step intermediates — and a radix floor's
intermediates are ~√N wide instead. So R5 also lowers the minimum
feasible d_model, which IS a KV lever.

## R6 — the unadopted swiglu library: one adoption, four confirmed passes

Premise verified: doom imports none of the five audited modules. The
sharper root cause the audit surfaced: doom never adopted
`map_to_table` (zero call sites), and three of the five modules are
composed on top of it — their non-use is downstream of doom's
deliberate choice to route lookups through `one_hot` / `pick_by_one_hot`
/ `pick_by_index` / `table_lookup_2d`.

**`onehot_table.onehot_lookup` — ADOPT (the one genuine win).** Maps a
one-hot to a **compile-time-constant** table; in the single-block case
it is a plain `Linear` — zero hidden lanes, zero added sublayers, and
it folds into its consumer. Doom's `pick_by_one_hot` over a constant
table pays `n·d_fill` gated lanes + a sum Linear + a sublayer for the
same job, because `broadcast_select` must support runtime tables and
cannot collapse `Σ onehotᵢ·constᵢ` into the matmul that it is.
torchwright's `const.py:24` even names the
`in_range → bool_to_01 → onehot_lookup` chain as a bit-exact target —
the op was built to consume doom's own `one_hot` and doom never picked
it up. Candidate sites are the constant-table subset of the ~35 pick
sites (~8–10): `assets._snap_index` (`assets.py:68`), the five wall
bank-metadata reads (`assets.py:94–126`: `bank_id`, `local_id`,
`width`, `height`, `h_idx_oh` — the widest, `d_fill` =
`len(wall_height_bank)`), and the statusbar constant tables
(`statusbar_renderer.py:90`). Lane savings are modest (tens per site);
the interesting part is **1 sublayer per site, and the bank-metadata
reads sit upstream of the texel fetch on both critical spines** — a
mechanical swap plus one `CRIT_PATH=1 OPT_GRAPH=1` run measures whether
it moves the 47 floor. (One correction to the audit: `lighting.py:58`
is NOT a candidate — its pick table is the 32 runtime colormap-row
outputs, not a constant; and that site disappears under the colormap
rewrite above anyway.)

The four passes, each with a confirmed reason:

- **`marker_count.count_since_marker`** — computes a position gap via
  uniform attention + a reciprocal FFN; doom subtracts two absolute
  positions off the cached `global_position_from_bos` (`past.py:72`)
  and wins by the reciprocal. Redundant.
- **`scalar_encoding`** — base-10 digit↔scalar pipeline over digit
  embeddings; doom's numbers are range-encoded VALUE carriers emitted
  through the base-256 digit-quad. Same *shape* as
  `_digit_quad_payload` (positional split via saturating staircase),
  but doom's variant is carry-free and snap-guarded for exactly the
  1-unit argmax margin the generic op misses. Inapplicable.
- **`sequence_ops`** — digit-stream parsing/emission over token
  windows; doom's emission is type-keyed protocol dispatch. One note
  worth keeping: `output_sequence`'s "trigger at P → emit seq[i] at
  P+i" is a positional fan-out primitive doom's dispatch lacks — reach
  for it if a fixed derived-token sequence ever needs spooling across
  consecutive AR steps. Inapplicable today.
- **`embedding_arithmetic`** — carry-propagated digit addition
  (~200 lanes per digit pair); doom adds scalars in one Linear.
  Inapplicable.

## Width is a KV lever: the d_model bracket (added after review)

Rob's correction to the first draft of this doc: the KV cache is
`layers × d_model × positions`, so narrowing the residual stream cuts
KV exactly like removing layers — compiling at a similar depth at
d=4096 would halve the uniform head count and halve the cache. The
first draft's "width constrains nothing" was true only of the 49-layer
schedule at d=8192.

**Measured bracket** (production graph, production env, heuristic
schedule-only with the production fusion pre-pass, d_hidden held at
16,384, d_head=128 — head counts divide cleanly at every point):

| d_model | heuristic layers | layers×d | KV at 57.4k positions |
|---|---|---|---|
| 8192 | 65 | 532k | 245 GB — production CP-SAT: 49 → 184 GB |
| 7168 | 64 | 459k | 211 GB |
| 6144 | 66 | 405k | 186 GB |
| **5120** | **67** | **343k** | **157 GB** |
| 4608 | 80 | 369k | 169 GB |
| 4096 | deadlock | — | — |

The knee is sharp: heuristic depth is essentially flat from 8192 down
to 5120 (+2 layers), then serialization bites (80 at 4608) and the
schedule deadlocks at 4096. Two readings:

- **Even the unoptimized heuristic schedule at d=5120 fits the KV
  cache under the B200's 178 GiB** (157 GB; weights add roughly
  15–25 GB at that width, so margin is thin but CP-SAT has 18 layers
  of headroom to reclaim — the dependency floor is width-independent
  and stays 49, worth 115 GB at d=5120).
- The caveat that keeps this a "measure next" and not a "done": the
  heuristic runs 16 layers above CP-SAT's floor at d=8192, and CP-SAT's
  behavior under tighter width is exactly what the bracket cannot see.
  The near-flat heuristic curve to 5120 is strong evidence the width
  slack survives, not proof.

**Constraint (Rob, stated not re-verified): d_model must be a power of
two — RMSNorm cancellation.** That removes every intermediate bracket
point from play: of the measured widths only 8192 and 4096 are legal,
and 4096 is exactly where the heuristic deadlocks. So there is **no
zero-graph-change narrow-d shortcut**; the intermediate rows in the
table above are evidence about where width slack lives, not landable
configurations. The KV-via-width path goes through **making d=4096
feasible**, and the peak arithmetic says what that takes: the two
1,024-wide `ray_scaled` halves alone are 2,048 columns — half of a
d=4096 stream — so the two-level ray count below is mandatory, not
optional, alongside the R5 step-chunk shrinkage (~2.6k columns) and a
look at the 1,024-wide `in_range` masks. The payoff: at d=4096
(32 uniform heads), 49–60 layers is 92–113 GB of KV — under the
ceiling with room for weights. Sequence for a follow-up: land the
width reducers → re-run this bracket's d=4096 point (heuristic
feasibility is the cheap gate) → production CP-SAT at d=4096.

**What pins the minimum d** (the peak live-set at the production-env
heuristic peak): the two 1,024-wide `ray_scaled` intermediates (the
atan2 thermometer's stage-1 indicator vectors — the widest single
resident, 2,048 columns), ~2,600 columns of `floor_int` step-chunk
intermediates (R5 shrinks these to ~√N), and ~600 one-wide glue nodes.
If d is to go below ~5120 later, the recorded idea for the ray banks:
a two-level count (32 coarse thresholds, then 32 fine thresholds with
gate-selected slopes — a runtime-slope ray test the swiglu multiply
makes possible) takes the live intermediate from 2,048 columns to ~64
for a few extra layers — same trade family as the radix floors.

## Others surfaced by the census (recorded, no verdict forced)

- **`in_range` (13,584 lanes, 9.3%)** — 4 lanes per tested slot,
  output is a full-width mask vector, so the width is the *output*, not
  removable by a scalar-style radix trick. The biggest sites are the
  plane/visplane column masks (`proj/plan` 5,936). If a consumer only
  ever reduces the mask (sums it, picks through it), a fused
  design could shrink it — nothing concrete proposed.
- **`widest_nodes.py` post-cutover blindness** — its MLP-hidden section
  keys on relu nodes and is silently empty; repair candidate (the
  `b12bac3` pattern).
- **Stale header comment** — `configs/e1m1.yaml` still says "51
  compiled layers"; known-stale per the cutover record (refreshes when
  the production walkthrough re-certifies).

## Tooling repaired (committed with this doc)

`scripts/analyze_forward_cost.py` had rotted against three torchwright
changes and could not run at all:

1. `GraphAnalyzer` now requires a wrapper-free graph — both capture
   paths now `lower()` first (the compiler-private copy; names and
   annotations carry over, so bucketing is unaffected).
2. The critical-path trace silently keyed source-graph node ids against
   the lowered copy's layer map — post-lowering it returned a 1-node
   "path". It now seeds from the globally deepest scheduled node and
   walks inside the copy.
3. `fuse_consecutive_linears` lost its relu-era `skip_relu_ejecting` /
   `eject_budget` kwargs; the docstring claiming the compile pipeline
   "never runs" the fusion pre-pass was wrong since fusion became
   always-on (cutover find #4) — production-comparable numbers need
   `OPT_GRAPH=1`, and the docstring now says so.

## What a landing session should do first

1. Land the R2 pixel-tail flatten (mechanical; the prototype shape is
   in the R2 section) behind the full gate stack. 49 → 47 measured.
2. Land the colormap → `table_lookup_2d` rewrite (verify the two
   integrality caveats first). ~17.6k lanes.
3. Build the radix-floor op in torchwright (derivation + noise entry),
   then convert phase-1 floor sites. ~41k lanes, and it shrinks the
   residual peak (a d=4096 enabler).
4. Build the two-level ray count (the `ray_scaled` halves are 2,048
   residual columns — half of a d=4096 stream; see "Width is a KV
   lever"). With 3 and the `in_range` masks addressed, re-run the
   d=4096 heuristic-feasibility check, then the production CP-SAT at
   d=4096 — the KV payoff (92–113 GB vs 184) is the certification
   unblock. d_model is power-of-two-only (RMSNorm cancellation), so
   there is no cheaper intermediate-width shortcut.
5. Swap the ~8–10 constant-table picks to `onehot_lookup` (mechanical;
   measure the floor before/after — the bank-metadata reads sit
   upstream of both critical spines).
6. Open the `proj/paint` cascade investigation (the attention-hoist
   question) — it gates everything below 47.
