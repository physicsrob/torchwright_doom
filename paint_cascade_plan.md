# Paint-cascade plan — flatten the 23-layer shared prefix (depth D5)

Work order for a single agent/session, refining item D5 of
`depth_flatten_plan.md`. Self-contained: everything needed to start is
in this file plus the referenced code. Written 2026-07-05 from the
measurements in `swiglu_opportunities_findings.md`.

**Branch**: work on the depth track's branch (`depth-flatten`) or a
child branch off it. Files this plan owns: `wall_column_state.py`
(primary), `wall_column_renderer.py` (secondary). No torchwright
changes — every op needed (`type_switch`, `pick_by_one_hot`,
`bool_all_true` / `bool_any_true` / `bool_not`, the assert helpers)
already exists.

## Context (read cold)

The production renderer compiles to 49 layers, and 49 exactly equals
the graph's dependency depth — the longest chain of ops where each
needs the previous one's output — after the always-on linear-fusion
pre-pass. So one layer of compiled depth disappears if and only if the
longest chain gets one hop shorter. Each layer costs ~3.8 GB of
KV-cache at the full 320×200 frame (d=8192), which is what blocks the
production certification.

The longest chain has a mapped anatomy (`scripts/critical_chain.py`
prints it). Layers 5–27 — about half the depth — are one stretch,
annotated `proj/paint`: the **per-position wall-column bookkeeping
state**. It is rebuilt at every autoregressive position on the read
side (before branch dispatch), and BOTH deepest chains (the
wall-column pixel path and the store-wall-range path) consume its
outputs — so every layer cut here moves the whole floor, twice over.

What it computes: the port of DOOM's ``R_RenderSegLoop`` /
``R_StoreWallRange`` per-column state. A wall can have up to three
texture parts (middle, upper, lower). The state works out which part
the current span ordinal (0/1/2) refers to, whether each part is
visible, and the selected part's screen-y bounds, texture id,
texture-mid offset, height-bank one-hot, colormap row, and what part
comes next. Core: `WallSpanRuntimeDraft.publish`,
`wall_column_state.py:652-822`.

## The oracle (run after every change, ~2 min local, CPU)

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \
    TORCHWRIGHT_DOOM_HUD=1 \
    CRIT_PATH=1 OPT_GRAPH=1 uv run python -m scripts.analyze_forward_cost

prints the dependency floor (baseline **49**; 47 once the depth
track's pixel-tail flatten lands — rebase and re-baseline before
starting). `scripts/critical_chain.py` (same env vars) prints the
zero-slack set, the per-layer table, and the witness chain — rerun it
after every cut to see what binds next. **The env vars are
load-bearing**: without them you silently measure a 60×50 hud-off
graph (the screen-env trap; findings doc, Method).

## P0 — pin every layer to a source line (do first, no edits)

The witness chain names the stretch but five nodes are untraced. Build
the map: for each of layers 5–27, the node → file:line → plain-English
meaning. Method: `scripts/critical_chain.py` gives node names,
annotations, and earliest layers; extend it locally (scratch copy is
fine) to print full annotations and each chain node's direct inputs if
needed.

Known so far (verify, don't trust):

- L5–L10 — UNTRACED: `clamp_0_2`, `thermometer_floor_div_0_24`,
  `in_range`, affine glue. Candidates: the span-ordinal clamp (its
  range is [0,2]) and the occlusion radix bucketing
  (`solid_intervals.py:192` uses `thermometer_floor_div` on the
  cursor column). Pin them.
- L11 — attention: the recency pick recovering the active drawseg
  index (`recent_drawseg_i`); every `seg_facts.*` read keys on it.
- L12–L14 — `same_int` compares on the span ordinal
  (`wall_column_state.py:725-726`).
- L15–L19 — `selected_part` two-select ladder (`:727-731`), one_hot,
  part booleans (`:732-734`), visibility `and_`/`or_` logic
  (`:691-693, 709-719`).
- L20 — attention: the span-state read
  (`wall_column.span_values(past)` / the `span_start_state` pick).
- L21–L27 — the five payload ladders (`select(part_is_mid, mid,
  select(part_is_upper, upper, lower))` for y-start, y-end,
  texture-mid, height-bank one-hot, texture id; `:735-790`) and the
  next-span logic (`:747-760`).

Deliverable of P0: the layer→line map appended to this file's
execution record. If the map contradicts the sketch above, the map
wins — re-plan the phases against it.

## P1 — the hoist question (the biggest single unknown)

Two attention reads sit inside the stretch. An attention read
serializes only if its **query** depends on earlier chain values;
reads whose keys are marker-recency (position-shaped, not
value-shaped) can hoist to run in parallel with everything before
them.

1. **L11 (drawseg-index recency pick)**: its consumers (`seg_facts.*`
   keyed reads) genuinely wait on its result. Probably irreducible —
   confirm the pick's own query doesn't also depend on something
   late.
2. **L20 (span-state read)**: its handle (`span_start_row`,
   `wall_column_state.py:672-676`) keys on the `WALL_SPAN_META`
   marker — recency, not computed values. If its query has no data
   dependence on L12–L19, the read hoists next to L11 and the
   stretch loses the serialization between the two reads. Verify by
   walking the Attn node's `.inputs` in the graph (the witness chain
   gives the node; a scratch probe over `scripts/critical_chain.py`'s
   `id2node` does it) — do NOT argue from the source code alone; the
   graph is the truth.

Also check the `seg_facts.*` reads themselves: if they are attention
picks per fact, whether they run as one read (shared key, concat
payload) or several dependent ones.

## P2 — ladder compression (mechanical designs, existing ops)

These are selections over a value that is exactly one of three
things — NOT priority ladders — so they compress with ops doom
already trusts:

1. **Payload ladders → `pick_by_one_hot`.** `select(part_is_mid, a,
   select(part_is_upper, b, c))` is `pick_by_one_hot(part_oh,
   concat(a, b, c))` — one sublayer instead of two, and all five
   payloads share the same 3-wide `part_oh` in parallel. (`d_fill`
   handles the wide height-bank payload.)
2. **Compose ordinal → part → payload.** The three "part one-hot
   given ordinal j" vectors (`one_hot(k_part_j, 3)`) compute in
   parallel from seg facts; the true part mask is one gated sum
   `part_oh = Σ_j is_kj · part_oh_j` (a `type_switch` over 3-wide
   values — exclusivity: the ordinal is exactly one of {0,1,2} on
   span rows); every payload then picks through that one mask. The
   ordinal ladder + one_hot + boolean stages collapse to ~2 layers.
3. **Visibility / next-span logic → n-ary booleans.**
   `bool_all_true` / `bool_any_true` are single FFNs; the
   `more_after_k0` / `span_has_next` select chains flatten with
   exclusive masks exactly like the pixel-tail flatten (the D1
   recipe in `depth_flatten_plan.md`, measured and reverted in the
   research session).

**Contracts to derive before landing (state them in the PR):**

- *Degenerate rows.* This state is built eagerly at EVERY position;
  on non-span rows the ordinal accessor returns 0-ish junk. The
  current nested selects tolerate that because their conds are ±1
  booleans built from the same junk consistently. The flat gated-sum
  version must derive its discarded-row story the same way the
  pixel-tail flatten did: masks must be snapped ±1 (they come from
  `same_int` compares — verify), and a junk row's picked value must
  land somewhere harmless (the published state is only ever read
  back on span rows — verify that consumption contract and write it
  down).
- *Numerical safety.* A clean ±1 cond makes a gated branch's loser
  contribute exactly zero in fp32 (the select/cond_gate contract);
  anything recovered through a softmax pick must pass through
  `snap_bool` (`render_ops.py:83` documents the failure class
  against the ~0.97-in-~46,000 emit argmax margin). Wrap the new
  masks in `assert_onehot` / `assert_bool` (torchwright
  `graph/asserts.py`) so `debug=True` forwards and `reference_eval`
  check them.

## P3 — prototype, measure, iterate

Implement in `wall_column_state.py` (and `wall_column_renderer.py`
where the same ladder shapes appear — it has 22 selects; many are the
same part-selection pattern). After each coherent step: the oracle,
then `scripts/critical_chain.py` to see the new binding chain. Record
every (change → floor) pair in the execution record. Expect
diminishing returns as the residue (position recovery → L11 pick →
keyed reads) becomes the chain; when two consecutive design steps
move the floor by 0, stop and write up.

Honest target: the ~15 ladder layers plausibly compress to ~4–5;
the residue is ~6–8 layers of genuinely sequential reads. A floor in
the low 40s (from 47) is success; anything below 45 is excellent.

## P4 — landing gates

Same stack as every depth landing: `make test` green (Modal, never
in background, never direct pytest); the graph-level oracles
(`tests/scene/test_flat_pixel_oracle.py`,
`test_forward_ar_rollout.py`); the d=4096 compile gate
(`tests/scene/test_forward_compiles.py`); then the runtime gate
`make run COMPARE=1` at `configs/e1m1_lowres.yaml` first, production
config after. Two-committed-configs rule: experiments use /tmp
variants. If `debug=True` asserts fire intermittently on identical
inputs, read the FP-nondeterminism section in `CLAUDE.md` before
touching the op.

## Stop conditions (write findings instead of pushing on)

- P0 map shows the stretch is not what the sketch claims (e.g. the
  ladders are already parallel and the depth is all reads) → report,
  don't force the designs.
- P1 says both reads are query-dependent AND P2's ladder math saves
  < 3 layers → the cascade is effectively irreducible as designed;
  record the proof (which query depends on which value, per node) so
  nobody re-derives it.
- Any compression that requires weakening an exactness contract
  (integer snap, saturation, argmax margin) → stop and surface; the
  exactness-by-saturation backbone is not negotiable.

## Execution record

(append here: the P0 layer map, P1 answers, per-change floor deltas,
gate results)

### Baseline (2026-07-04, depth-flatten @ d61f72f)

Oracle: `fused=686 critical_path_layers=49`. D1 (pixel-tail flatten)
has NOT landed yet — only its prep commit — so the baseline is 49, not
47. Zero-slack set: 150 of 6608 nodes; through L5–L27 the set is 1–2
nodes per layer (the chain is razor-thin). `stor/R_StoreWallRange`
family min-slack is 2 — currently only the pix spine binds; the stor
twin sits 2 layers behind it.

### P0 — the layer→line map (measured, provenance-traced)

Method: scratch probe (`p0_chain_map.py` in the session scratchpad)
monkeypatches `Node.__init__` to record each node's creation stack,
inverts `lower()`'s source→clone `node_map` to attribute scheduled
clones back to user code, and re-derives the witness chain. Two
provenance caveats: (1) an op wrapped in an `Assert` (e.g. `select`'s
cond assert) can shadow the wrapped node's true creation site in the
inverted map — L24's site below was recovered from its inputs; (2) the
L14 node ("compare", created inside `select` at
`wall_column_state.py:159`) has an unresolved identity — one extra
compare layer between `present` and the clip-default selects; resolve
during the P2 rewrite of that stretch.

The map (file = `wall_column_state.py` unless noted):

- **L0–L3** `[scene]` — BOS/global-position recovery (two attention
  reads + linears). Not this plan's scope.
- **L4** `[proj]` — attention: `pick_most_recent` recovering the
  cursor-x screen coordinate (`seg_projection.py:402`), fused with
  `column_from_screen_x` (`render_ops.py:125`).
- **L5** — `SCREEN_X_CLAMP(current_x_scalar)` → `query_col`
  (`:127`, `ClipMemory.publish`).
- **L6–L8** — `_radix_col_key(query_col)` (`render_ops.py:735-743`
  via `:128`): `thermometer_floor_div` bucket (L6), `mod_const`'s
  multiply-negate + add (L7–L8).
- **L9–L10** — `one_hot(digit)` = `in_range` (L9) + `bool_to_01`
  affine (L10, unfused into the attention query projection)
  (`std.py:191` via `render_ops.py:742`).
- **L11** — attention: **the clip-memory pick** —
  `past.pick_most_recent(query, clip_range_key, clip_range_value)`
  (`:149`), recovering (ceiling, floor, col) for the current column.
- **L12–L13** — `present = same_int(recovered_col, query_col)`
  (`:157`): `ABS_SMALL_INT` PWL (L12) + compare (L13).
- **L14–L15** — the clip default selects
  (`select(present, recovered, open-clip)`, `:159-160`); L14 is the
  unresolved extra compare noted above.
- **L16–L17** `[R_RenderSegLoop]` — clamp the middle-tier span bounds
  to the recovered clip: `ceiling_min`/`gt_screen` (L16, `:302,:308`),
  `yl`/`yh` selects (L17, `:310-311`).
- **L18–L19** — `middle_span_ok_value = le_span_y(yl, yh)`
  (`render_ops.py:579` via `:476`): sub+compare (L18) + `bool_not`
  compare (L19). Published as `wall_column_span_state` (`:498-511`).
- **L20** — attention: the span-state read
  (`span_values` → `row.pick`, `:534/:538` via `:690`,
  `WallSpanRuntimeDraft.publish`).
- **L21** — `mid/upper/lower_visible = and_(has_*, *_ok)`
  (`:691-693`, `bool_all_true` = 1 compare).
- **L22–L23** — `part_visible_for`'s nested selects (`:713-716`).
- **L24** — `and_(exists, visible)` → `k1_visible`/`k2_visible`
  (`:719`).
- **L25–L26** — `more_after_k0 = or_(k1_visible, k2_visible)`
  (`:747`): `bool_any_true` is TWO compare layers (per-input 0/1
  normalize, then sum threshold — torchwright
  `ops/swiglu/logic_ops.py:52`), unlike `bool_all_true`'s one.
- **L27** — the `span_has_next` / `span_next_y` selects
  (`:748-754`). Published as `span_start_state` (`:792-808`).
- **L28** — attention `[pix/R_DrawColumn]`: reads `span_start_state`
  back; the pixel tail begins.

### P0 — where the map contradicts the plan sketch (map wins)

1. **L5–L15 is `ClipMemory.publish`** — the per-column ceiling/floor
   clip recovery (DOOM's `ceilingclip`/`floorclip`), NOT span-ordinal
   logic. The radix bucketing is the clip column key
   (`render_ops.py:741`), not `solid_intervals.py`.
2. **L11 is the clip pick, not `recent_drawseg_i`.** The drawseg pick
   and every `seg_facts.*` read are OFF the zero-slack set — they
   never appear in the 1–2-node critical layers.
3. **The `same_int` compares at L12–L13 test clip presence**
   (`recovered_col == query_col`), not the span ordinal. The
   span-ordinal compares (`:725-726`) are input-token-derived and have
   slack.
4. **L16–L19 serialize BEFORE the L20 read** because the read's VALUE
   matrix contains the current position's just-published
   `span_state` — publish→read within one forward pass, not a query
   dependence.
5. **The five payload ladders (`:735-790`) are NOT on the chain.**
   Their conds (`part_oh` via `selected_part`) derive from the input
   token + seg facts (off-chain); their data arrives at L20. They sit
   at ~read+2 with ~5 layers of slack. The binding tail L21–L27 is the
   **visibility → next-span** path (`k1_visible`/`k2_visible` →
   `or_` → `span_has_next`/`span_next_y`).

Consequence: P2's design 1 (payload ladders → `pick_by_one_hot`)
buys little to nothing on its own — the chain runs through the
visibility/next-span logic and through ClipMemory. Re-planned targets
in priority order: (a) L21–L27 flatten (~7 → ~3-4), (b) L12–L19
ClipMemory-consumer flatten (~8 → ~4-5), (c) L5–L10 clip-query
shortening, (d) payload ladders only insofar as they'd become the new
binder.

### P1 — attention-read dependency proof (graph-walked, per node)

Probe: `p1_attn_deps.py` (scratchpad) — for each witness-chain Attn,
walks each direct input (order `query_in`, `key_in`, `value`) through
unscheduled concat glue to its deepest *scheduled* dependency.

- **L11 (clip pick, `wall_column_state.py:149`)** — Q binds at L10
  (the radix column one-hot, L5–L10); K binds at L9 (the gated clip
  key, `:138`, same `_radix_col_key` shape on the publish side); V
  binds at L5 (`:143-145` selects). **Query-bound → cannot hoist.**
  The only lever is shortening `_radix_col_key` itself, which moves Q
  and K together.
- **L20 (span-state read, `:534/:538`)** — Q binds at L0 (marker
  recency: constants + the marker flag); K at L4; **V at L19** (the
  `le_span_y` ok-flags, `:476-478`). The read is placed by its VALUE —
  the same-pass `span_state` publish — exactly finding 4 above.
  "Hoisting" is a no-op; the read moves up 1:1 as the published
  value's depth drops.
- **L28 (`span_start_state` read, `:601`)** — same shape: Q at L0, K
  at L3, **V at L27** (`span_has_next`/`span_next_y`/`span_next_ordinal`
  selects, `:748-756`). The whole 21-layer pixel tail (L28–L48) rides
  on the depth of the `span_start_state` publish.

Answer to the plan's P1: both reads are already query-parallel; there
is no hoist win to collect. The entire paint-cascade depth is in the
**published-value chains** (L5→L19 feeding the L20 read's V, L21→L27
feeding the L28 read's V) — every layer cut there is a full compiled
layer off the floor. `seg_facts.*` reads are off the zero-slack set
entirely (irrelevant to depth).

### P2/P3 — change → floor ledger

Baseline 49 (depth-flatten @ d61f72f, D1 not landed).

1. **49 → 45.** Flat next-span logic in `WallSpanRuntimeDraft.publish`:
   `part_visible_for`'s nested selects + `and_` + two-layer `or_` +
   select ladders replaced by 0/1 candidate slot masks
   (`gate(exists_j, part_oh_j)`, off-chain) and one
   gated-slot-sum + compare per boolean
   (`(picked ±1 ok sum + #marked) >= 2`, threshold 1, margin 1).
   `span_next_y`/`span_next_ordinal` select on a single
   `choose_k1 = is_k0 AND k1-visible` compare of the same shape. The
   `has_*` term of the old visibility conjunction is structural
   (`vocab._K_PART_TABLES` only lists existing parts) — dropped with
   the invariant recorded at the use site. `std.compare` re-exported
   for doom call sites.
2. **45 → 43.** Two single-sublayer op forms in `render_ops`:
   `same_int` = one hat-shaped PWL on the difference (flats ±1 inside
   0.4 / outside 0.6; the sub fuses into the PWL) replacing
   abs→compare→not; `le_span_y` = `compare(y2−y1, −0.5)` replacing
   `not(y1−y2 > 0.5)`. Same half-integer integer-input contracts.
   After step 1 the witness bound through the span_y payload ladders
   (L21-23) and these two ops held three chain layers (present check
   L12-14, span-ok L18-19).
3. **43 → 43 (with 4).** Payload picks: `span_y_start/end/height` as
   `pick_by_one_hot(part_oh, ...)` with per-part height table entries
   linear in the read (fuse into the pick). Junk rows: fractional
   one-hot → bounded blend, published, never read back (span marker).
4. **43 → 43.** Two-variant clip clamp: `ClipMemory` now
   exposes `recovered_ceiling/recovered_floor/present`;
   `yl`/`yh`/`upper_mid` and the published span bounds/flags compute
   recovered-clip and open-clip variants in parallel and resolve once
   on `present`. Identities used (exact on integers, stated in code):
   `le(max(a,c), min(b,d)) = and of 4 pairwise le`; monotone
   `SPAN_Y_CLAMP` commutes with max/min; the open clip's
   ceiling_min/floor_max coincide with the clamp bounds, so the open
   variant needs no max/min. `lower_visible` now reuses the
   two-variant `lower_span_ok_value` instead of a duplicate
   `le_span_y`. The floor did not move: after step 2 the 43 floor is
   bound by at least one other chain that steps 3–4 don't touch
   (tracer identifies it below); these cuts stay because the paint
   publish must not re-bind once that chain is cut.
5. **43 → 43.** Radix column key: the low digit of
   `render_ops.radix_col_key` is one sawtooth PWL
   (`x − B·floor((x+0.5)/B)`, ramp pairs bracketing each `k·B − 0.5`
   jump by ±0.05, numerically validated over every column ± the 0.02
   leak) running in parallel with the bucket thermometer, replacing
   the serial `mod_const` (thermometer → scale → subtract). Ramps stay
   bucket-consistent with the thermometer's `k·B − 0.5` placement.

### Gate fix — junk-row mask overlap in the has-next dot

`make test` caught one real bug in step 1 (the AR-rollout gate,
`test_forward_ar_rollout`): on junk rows both next-candidate slots can
read the same junk part (junk `k_part_1 = k_part_2 = 0`, and junk 0 ≠
the PART_NONE sentinel so both `exists` flags read true), so the
SUMMED candidate mask hit 2 and `indicator_to_bool(2) = 3` violated
`broadcast_select`'s 0/1 mask contract — its retained value-range
assert fired in exact math. Fix: one pick per candidate (each single
gated one-hot stays in [0,1] on every row) with the ≥2 threshold
algebra absorbing multiplicity through the outer sum. Same depth (the
two picks run in the same layer). Lesson recorded: any mask built as a
SUM of gated one-hots must be justified against junk rows slot-wise,
not just condition-wise — junk defeats "the parts are distinct"
arguments that hold on real rows. Dead-code cleanup in the same pass
(unused `span_y_start/end` picks — the publish only ever carried the
height; unused `upper_mid`/tier aliases), lint green, `make test`
green (5/5 shards). Floor re-verified at 43 after the fix (the two
per-candidate picks run in one layer, as designed).

### P3 conclusion — the paint cascade is off the floor; 43 binds elsewhere

Two consecutive steps moved the floor by 0 → stop per the plan. The
reason is not diminishing ladder math: after step 2 the witness chain
left `proj/paint` entirely — the paint cascade (this plan's D5 scope)
no longer appears ANYWHERE in the zero-slack set (81 nodes, none
annotated `proj/paint`). D5's outcome: **49 → 43 measured**, and the
publish-side cuts of steps 3–5 additionally hardened the paint chain
against re-binding when the new spine is cut.

The 43-floor witness (handoff for the next depth item; full trace in
the session record) is the **plane-mark / visplane spine**:

- L0–L2 `[proj/input]` — PLANE_MARK input gating.
- L3–L8 `[proj/pmrk]` — an attention read, then the plane radix key
  `visplane_state._radix_plane_key` (`:181-188`): the SAME serial
  `mod_const` shape step 5 just removed from the column key
  (thermometer → scale → subtract → in_range → affine), plus the
  `cond_gate` at `:~187`. The step-5 sawtooth recipe applies verbatim
  (≈2 layers), and `visplane_state.py:492` has a third copy.
- L9–L13 `[proj/plan]` — three chained attention reads with the
  lifted-instance query square between reads 2 and 3
  (`_lifted_instance_key` / `pick_argmin_above_in_bucket`, the H1/H2
  visplane instance resolution).
- L14–L21 — a compare/select ladder (same class as the old paint
  ladder; step-1/step-4 recipes apply).
- L22–L29 — cond_gate → attention read → SIX chained selects
  (`proj/plan`) — a priority ladder, the exclusive-mask flatten's
  canonical target.
- L30–L42 — pix/plan read → setCursorX digit-quad emit → the
  pixel-tail select ladder (L37–39; this is exactly depth plan D1,
  measured 49→47 in the research session and reverted — landing D1
  now cuts real layers again) → dispatch.

Recommendation: a follow-on `plan_cascade` work order mirroring this
one, targeting `visplane_state.py` with the already-measured recipes
(sawtooth radix digit, flat priority ladders, two-variant clip-style
splits), plus landing D1.

**Follow-on written: `visplane_cascade_plan.md`** (depth item D6) —
consolidation with `depth-flatten`'s D1/D3 is its Phase 0; the P0/P1
scratch probes of this record are productized as
`scripts/chain_provenance.py`.
