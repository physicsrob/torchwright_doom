# Visplane-cascade plan — flatten the plane-mark / visplane spine

Work order for a single agent/session, the follow-on to
`paint_cascade_plan.md` (depth item D5) and item D6 of
`depth_flatten_plan.md`. Self-contained: everything needed to start is
in this file plus the referenced code. Written 2026-07-05 from the
43-floor witness in `paint_cascade_plan.md`'s execution record.

## Where the two branch lines stand (read cold)

The compiled depth floor's history: **49** at the common ancestor
(depth-flatten @ d61f72f) → **47** on `depth-flatten` (D1, the
pixel-tail flatten; D3 landed depth-neutral; D2 was measured +1 layer
and skipped; D4 is a documented NO-GO) → **43** on
`worktree-paint_cascade` (the paint-cascade steps 1–5, measured
standalone WITHOUT D1). The two lines are **unconsolidated**; their
only shared file is `std.py`, where both sides append helpers.

At 43 the binding chain left `proj/paint` entirely. The witness now
runs through the **plane-mark / visplane spine** — the runtime
visplane bookkeeping (`visplane_state.py`) and the flat-span boundary
logic it feeds (`flat_state.py`) — which is this work order's scope.

**Branch**: `worktree-paint_cascade`, rebased onto `depth-flatten`
(Phase 0 below does the rebase). File ownership:

- `visplane_state.py` — primary, this plan owns it.
- `flat_state.py` — secondary. **Width-track coordination note**: the
  consolidation contract in `depth_flatten_plan.md` assigns
  `flat_state.py` to the width track (`width_d4096_plan.md`). Their
  interest is `_ray_count` / the floor ops — lane width, not these
  select ladders — so the edits should not collide, but any change
  here must be rebase-checked against `width-d4096` before landing,
  and a real conflict means coordinate first (see P4).
- `std.py` / `render_ops.py` — shared, additive-only; keep diffs
  minimal.

## The oracle (run after every change, ~2 min local, CPU)

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \
    TORCHWRIGHT_DOOM_HUD=1 \
    CRIT_PATH=1 OPT_GRAPH=1 uv run python -m scripts.analyze_forward_cost

prints the dependency floor. `scripts/critical_chain.py` (same env
vars) prints the zero-slack set, the per-layer table, and the witness
chain; `scripts/chain_provenance.py` (same env vars) additionally pins
every witness node to its creation site and decomposes each attention
read's Q/K/V binding depths. **The env vars are load-bearing**:
without them you silently measure a 60×50 hud-off graph (the
screen-env trap; `swiglu_opportunities_findings.md`, Method).

## Phase 0 — consolidate the two branch lines (do first)

1. Rebase `worktree-paint_cascade` onto `depth-flatten` @ eae3c9e.
   Expected conflict: `std.py` only — both sides append helpers; keep
   both appends.
2. `make test` green, then re-baseline the oracle. The paint 43 and
   depth-flatten's D1 combine to an unknown floor: D1 flattened the
   pixel-tail selects that sit at L37–39 of the 43-floor witness, so
   expect **~41–42**, but measure, don't assume. Re-run
   `scripts/critical_chain.py` and record the new witness in the
   execution record — every layer number in P0's sketch below shifts
   by the consolidation delta.
3. Land the consolidated line to `depth-flatten`; then run the ONE
   combined runtime certification owed by both lines:
   `make run COMPARE=1` at `configs/e1m1_lowres.yaml`, then at the
   production `configs/e1m1.yaml` (covers D1 + D3 + paint steps 1–5
   in one paid pass).

## P0 — pin the post-consolidation witness (no edits)

Run `scripts/chain_provenance.py` under the production env vars and
build the layer→line map for the new witness. The sketch below is the
43-floor trace read against the code (layer numbers pre-consolidation;
**where the measured map disagrees with the sketch, the map wins** —
re-plan the phases against it, exactly as the paint session had to).

- **L0–L2** `[proj/input]` — the PLANE_MARK input gating:
  `type_matches` (the `type_matches_dot_PLANE_MARK` Linear + compare,
  `extract.py:542-558`, reached via
  `screen_range_after_plane_mark`, `protocol_tokens.py:608-612`) and
  the plane-mark side-channel selects
  (`seg_projection.py:482-493`).
- **L3–L8** `[proj/pmrk]` — `RuntimeVisplaneState.publish`
  (`visplane_state.py:310`): an attention read, then
  `_radix_plane_key` (`:181-188`) built with the serial
  `thermometer_floor_div` + `mod_const` pair, then
  one_hot/Linear/cond_gate into the published keys
  (`visplane_bounds_min_key` `:354`, `visplane_bounds_max_key`
  `:368`, `used_vp_key` `:447`).
- **L9–L13** `[proj/plan]` — `check_conflict` (`:470-578`): the H1
  same-bucket read (`:500`) and H2 next-bucket read (`:525`) are both
  query-independent of each other; the H3 carry read (`:550`) has a
  query that consumes H2's `higher_bucket_oh` (built `:544-546`,
  consumed `:555`) — a genuinely serial read pair. The
  `_lifted_instance_query` square (`:217-238`) feeds all three.
- **L14–L21** — the presence compare ladders: the
  `bool_and(snap_bool(valid), compare×3)` blocks at `:514-521`,
  `:537-543`, `:562-571` (and `next_plane_after`'s at `:615-623`,
  `:648-654`), plus `le_x2` (`:577`).
- **L22–L29** — cond_gate → read → the flat-boundary select run in
  `flat_state.py`: the R_MakeSpans `t1`/`b1`/`t2`/`b2` selects
  (`:276-279`), the opening-indicator selects (`:307-318`), and the
  chunked `flat_span_x1` compare-gated select fold (`:401-406`).
- **L30+** — the `FlatPassState` plan reads (`flat_state.py:521-580`)
  into the pixel tail.

Deliverable: the measured map appended to this file's execution
record.

## P1 — per-read dependency proof (probe, not source argument)

For every attention read on the witness — the pmrk publish-side reads,
H1/H2/H3 in `check_conflict`, `next_plane_after`'s reads, the
`column_range` / `min_x` / `max_x` argmax reads, and the L30 plan
reads — classify what places it: **query-bound** (its query waits on
earlier chain values) vs **value-bound** (its VALUE matrix contains
the current position's just-published state — publish→read within one
forward pass, the paint P1 finding, where the read moves up 1:1 as the
published value's depth drops). Use
`scripts/chain_provenance.py`'s per-Attn Q/K/V decomposition — do NOT
argue from the source alone; the graph is the truth.

Expected: H3-after-H2 is irreducible as designed (H3's query is
genuinely chained on H2's output). Write the proof either way — a
per-node record of which input binds at which layer, so nobody
re-derives it.

## P2 — designs (the measured recipe book, in expected-value order)

Every recipe below was measured in the paint session; the notation
"step N" refers to `paint_cascade_plan.md`'s P2/P3 ledger.

1. **Plane/vp radix digits → parallel sawtooth** (the step-5 recipe).
   `render_ops.radix_col_key` (`render_ops.py:757-791`) already
   computes its low digit as one sawtooth PWL
   (`x − B·floor((x+0.5)/B)`, ramp pairs bracketing each `k·B − 0.5`
   jump by ±0.05) in parallel with the bucket thermometer. Extract
   that grid builder into a shared `mod_sawtooth(scalar, base,
   max_value)` helper and apply it at every remaining serial
   `thermometer_floor_div` + `mod_const` pair:
   `_radix_plane_key:186-188`, `_publish_occupancy_radix:730-731`,
   `_publish_used_plane_successor:785-788`, `check_conflict:491-492`,
   `next_plane_after:596-597`. Constraint: the ramps must stay
   bucket-consistent with each site's thermometer (transitions at
   `k·B − 0.5`), as in the column key.
2. **Presence blocks → fewer layers.** Each block is
   `bool_and(snap_bool(valid), compare(pick_by_one_hot(...)), ...)`
   — the compares take one layer IF they run in parallel (verify with
   the probe that nothing serializes them) + `bool_all_true`'s one
   layer. Then fold the `or_` clusters — `cstar_present` (`:576`),
   `used_plane_active` (`:417`), `used_vp_active` (`:431`),
   `opening_active` (`flat_state.py:319-322`) — where
   `bool_any_true`'s TWO compare layers (torchwright
   `ops/swiglu/logic_ops.py:52`: per-input 0/1 normalize, then sum
   threshold) sit on the chain: add a clean-±1 one-layer `or_` to
   `render_ops` as `compare(sum(inputs), −(N−1))` — margin 1, but the
   **inputs must be clean ±1** (state that contract at the helper; a
   softmax-recovered input goes through `snap_bool` first).
3. **Priority ladders → exclusive masks** (the D1 recipe):
   `next_plane_after`'s 2-deep nested select (`:656-660`), the
   flat_state select run (`:276-318`), and the chunk fold
   (`flat_state.py:401-406` — a compare-gated select chain over the
   row-chunk heads; flatten with per-chunk exclusive masks +
   `pick_by_one_hot` / `type_switch`, the chunk index is exactly
   `y // CHUNK` so the masks are one parallel layer).
4. **Two-variant splits** (the step-4 recipe): where a resolved
   select's output chains into later clamps/compares, compute both
   variants in parallel and resolve once at the end. Candidates
   surface from P0's measured map, not from this sketch.
5. **The H2→H3 serial read pair** — research only if it still binds
   after 1–4; the query chaining is real and any fix is a redesign of
   the carry search, not a local flatten.

## Contracts to restate (from the paint execution record)

- **Degenerate/junk rows.** State here is built eagerly at EVERY
  position; accessors return junk off their row type. Every new mask
  needs its junk-row story derived BEFORE landing — and the
  summed-gated-one-hot lesson applies: a mask built as a SUM of gated
  one-hots must be justified **slot-wise on junk rows**, not
  condition-wise (junk defeats "the slots are distinct" arguments
  that hold on real rows; the AR-rollout gate catches violations in
  exact math, which is how the paint session found it).
- **`broadcast_select`'s mask contract is 0/1** — its retained
  value-range assert fires in exact math on violation.
- **Compare margins ≥ 1** against softmax-recovery noise ~1e-3;
  anything recovered through a softmax pick passes through
  `snap_bool` (`render_ops.py:82` documents the failure class) before
  it conds a select.
- **Exactness-by-saturation is non-negotiable** — any compression
  that would weaken an integer snap, saturation, or argmax margin is
  a stop condition, not a trade-off.

## P3 — iterate and measure

After each coherent step: the oracle, then `scripts/critical_chain.py`
to see the new binding chain. Record every (change → floor) pair in
the execution record. Stop when two consecutive design steps move the
floor by 0 (the paint precedent: that signal meant the spine had left
the plan's scope entirely — check the witness annotation before
concluding "irreducible").

Honest target from the recipe measurements: pmrk radix (−2ish),
presence blocks + ladders (−2–3), flat ladder (−1–2) ⇒ from the
post-consolidation ~41–42, a floor in the **high 30s** is success.

## P4 — landing gates

Same stack as every depth landing: `make test` green (never direct
pytest, never in background); the graph-level oracles
(`tests/scene/test_flat_pixel_oracle.py`,
`test_forward_ar_rollout.py`); the d=4096 compile gate
(`tests/scene/test_forward_compiles.py`); then the runtime gate
`make run COMPARE=1` at `configs/e1m1_lowres.yaml` first, production
config after. Two-committed-configs rule: experiments use /tmp
variants.

Stop conditions (write findings instead of pushing on) — the paint
plan's three, plus one:

- P0 map contradicts the sketch structurally → report and re-plan,
  don't force the designs.
- The reads are placed by genuinely chained queries AND the ladder
  math saves < 3 layers → record the per-node proof and stop.
- Any compression that weakens an exactness contract → stop and
  surface.
- **Any `flat_state.py` conflict with `width-d4096`** → coordinate
  with the width track before landing; do not resolve unilaterally.

## Execution record

(append here: the Phase 0 consolidation result, the P0 map, P1
answers, per-change floor deltas, gate results)

### Phase 0 — consolidation (2026-07-05)

Rebased `worktree-paint_cascade` (6 commits) onto
`depth-flatten@eae3c9e`. **No conflicts** — the expected `std.py`
collision auto-merged (the two sides' appends — paint's `compare`
re-export, depth's `pick_const_by_index` — touch non-overlapping
hunks even in the import/`__all__` regions).

**Post-consolidation floor: 41** (`fused=734
critical_path_layers=41`) — inside the predicted ~41–42. Paint's
standalone 43 gained 2 from D1: the old witness's L37–39 pixel-tail
select ladder is gone; the tail after the emit digit-quad is now just
cond_gate → dispatch Linears (L37–L40).

Zero-slack set: 80 of 6542 nodes — 48 `proj/plan`, 13 `proj/pmrk`,
6 `pix/emit`, 5 `dispatch`, 3 each `proj/input` / `pix`. The witness
is the work order's sketch shifted down 2:

- L0–L2 `[proj/input]` PLANE_MARK gating (is_type dot → compare →
  cond_gate → select).
- L3–L8 `[proj/pmrk]` Attn read ∥ `_radix_plane_key`: thermometer
  (L3) → mod_const's fused multiply-negate (L4) → Add (L5) →
  one_hot's in_range (L6) → affine/Linear (L7) → cond_gate (L8).
- L9–L13 `[proj/plan]` the three chained reads: Attn (L9) → Attn
  (L10) → split → `visplane_instance_query_square` (L11) → neg/add
  (L12) → Attn (L13).
- L14–L21 the compare/select ladder (select, add_const,
  fused_sub_compare, then compare×4 at L18–L21).
- L22–L29 cond_gate → Attn (L23) → SIX chained selects (L24–L29).
- L30 `[pix/plan]` Attn → clamp_0_2 (L31) → the setCursorX
  digit-quad emit (L32–L36: div floor_int → saturate → add →
  snap-half floor_int → saturate → add/c2) → cond_gate → dispatch
  Linears (L37–L40).

Full provenance trace: `scripts/chain_provenance.py` output archived
in the session scratchpad (`prov_41.out` / `crit_41.out`).

### P0 + P1 — the 41-floor map with per-read proofs (measured)

The map wins over the sketch, and it differs in one big way: **the
witness never touches `check_conflict`** (H1/H2/H3 and the presence
ladders all have slack). The chain is the R_MapPlane read pipeline —
`min_x`/`max_x` → `column_range` → R_MakeSpans → the chunk fold —
with every attention read bound by a SAME-PASS published key or
value, plus one genuinely query-chained read:

- **L3 read** (`attend_to_offset` for the plane-mark (p, vp),
  `visplane_state.py:329`): V-bound at L2 — the
  `plane_mark_p_or_zero` select (`seg_projection.py:488`) published
  in the same pass.
- **L3–L8** the `_radix_plane_key(occupied_p_value)` publish chain
  (`:186-188` via `:359`): thermometer (L3) → `mod_const`'s
  multiply-negate (L4) → Add (L5) → one_hot in_range (L6) → fused
  affine+scale Linear (L7) → `gate` (L8, `:356`).
- **L9 reads** (`min_x`/`max_x`, `:688`/`:699`): Q binds L6 (their
  own query-side `_radix_plane_key`), **K binds L8** — the same-pass
  `visplane_bounds_min/max_key` publish. Key-bound; the read moves up
  1:1 as the publish chain shortens.
- **L10 read** (`flat_visplane_row.pick`, `flat_state.py:259`):
  **V binds L9** — `flat_visplane_state_pub` carries the `min_x` /
  `max_x` results. Value-bound.
- **L11–L12**: `_lifted_instance_query` on the recovered
  (`flat_p`, `flat_vp`) — the instance square (L11) + neg/add-const
  (L12).
- **L13 read** (`column_range`, `visplane_state.py:712`): **Q binds
  L12** — the lifted query chains on the L10 read's output. This is
  the genuinely serial read pair of this spine (the sketch guessed
  H2→H3; it is actually flat-visplane→column_range). K (the `col_key`
  publish, `:403`) has slack at L7.
- **L14–L22** the R_MakeSpans boundary chain (`flat_state.py`):
  `t1`/`b1` selects (L14, `:276-277`) → `add_const` + `max_screen`
  (L15–L16, `:300`) → `le_span_y` (L17, `:303`) → the
  opening-indicator select (L18, `:313`) → `or_` = `bool_any_true`'s
  TWO compare layers (L19–L20, `:321`) → the `and_` (L21) →
  `gate(opening_active, chunk)` (L22, `:331`).
- **L23 read** (the chunk-head `pick_most_recent`, `:389`): **K binds
  L22** — the same-pass `flat_opening_key_k` publish. Key-bound.
- **L24–L29**: the chunk fold (`:401-406`) — SIX chained selects
  (7 chunks at 320×200). The compare conds are off-chain
  (`span_row_y` is input-derived); only the select data path binds.
- **L30 read** (`flat_span_values`, `:571`): **V binds L29** —
  `flat_span_state_pub` carries the fold output (`:402` via `:415`).
  The closure splits (`:413`) bind at only L21.
- **L31–L40**: clamp_0_2 → the setCursorX digit-quad emit (5 layers)
  → cond_gate → dispatch Linears. Rides the L30 read 1:1.

P1 answer: no hoist wins exist (all queries except L13's are
positional/off-chain); the depth is entirely in **same-pass
publish→read chains**, so every layer cut in a published key/value
chain moves the floor 1:1 until a parallel spine binds. Refined
design targets, in expected-value order:

1. Chunk fold → `pick_by_one_hot` over `one_hot(y // CHUNK)`
   (design 3): L24–L29 collapse to one pick sublayer, worth up to −5.
2. R_MakeSpans chain (designs 2+4): one-compare
   `and(x, or(a, b))` = `compare(2x + a + b, 1)` (margin 1, clean-±1
   inputs) kills the or_/and_ stack (−2); `le(max(a,c), b)` =
   `and(le(a,b), le(c,b))` restructures the max/le serial pair (−1).
3. `_radix_plane_key` publish chain → sawtooth low digit (design 1):
   −2 on L3–L8, moving the L9 K-bound reads up.
4. Precompute the instance square AT PUBLISH TIME: publish
   `flat_p·N + flat_vp` and its square in `flat_visplane_state_pub`
   (input-derived, off-chain at publish), recover both in the L10
   read, and build `_lifted_instance_query` linearly (2q fuses into
   the Q projection) — kills L11–L12, moving the L13 read up −2.

### P2/P3 — change → floor ledger

Baseline 41 (post-consolidation).

1. **41 → 39.** `render_ops.mod_sawtooth(scalar, base, max_value)` —
   the col_mod sawtooth generalized (grid extended to −1.45 holding
   the identity below the k=0 edge, so `next_plane_after`'s
   `threshold == −1` find-first digit stays −1 instead of clamping at
   −0.45; slope-1 extension, zero extra lanes). Applied at the five
   planned `mod_const` sites in `visplane_state.py` and the two
   `solid_intervals.py` twins (`:193`, `:277` — off-floor, hardened
   per the D1 all-three-branches precedent); `radix_col_key` now
   calls the shared helper. Numerics: new-vs-old worst diff 4.8e-07
   on the shared domain (identical function); integer-exactness
   worst 1.05e-02 at base 13 — the PRE-existing swish-fillet tail of
   the landed col_mod form, two orders under the 0.5 slot windows and
   unit argmin gaps its consumers resolve; `f(−1) = −1.000000` exact.
   **At 39 the spine flips**: the witness leaves the visplane suffix
   and lands on the wall texel twins — zero-slack set 314 of 6518
   with THREE co-binding spines (`stor/R_StoreWallRange` 86,
   `pix/R_DrawColumn` 86, `proj/plan`+`pmrk` 68) over a shared
   `proj/paint` prefix (L0–L19). Steps 2–4 (the plan-suffix cuts)
   can no longer move the floor alone.
