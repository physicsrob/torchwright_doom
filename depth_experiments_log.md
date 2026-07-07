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
