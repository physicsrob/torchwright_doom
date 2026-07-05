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
