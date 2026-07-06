# Handoff: torchwright op-level improvements for compiled DEPTH

> **Status (2026-07-05, evening).** Written by the depth-flatten session
> against the 47-floor baseline; rescued from that session's worktree
> (the doc was never committed) with this status added. Since then:
>
> - **Item 1 LANDED** in torchwright (`083813a`) — but the skip does
>   NOT fire on doom's graph: `scripts/clamp_range_probe.py` measured
>   40/40 clamps kept. Two blockers, see the item-1 caveat below:
>   (a) `pick_by_one_hot`'s affine range rule sums entry maxima
>   (one-hot exclusivity unmodeled) — mitigated doom-side by feeding
>   each wall bank its own height's mod directly; (b) the skip
>   condition demands exactly `[0, top]` while doom's mod outputs
>   carry ±ε PWL slack — needs the torchwright condition relaxed by
>   the identity band (the commit message's ~0.37 index units).
>   **(b) is the open torchwright follow-up.**
> - **Item 2 LANDED** (`083813a`) and **adopted doom-side where it is
>   width-free** (`pwl_banks.FLOOR_MOD64`, the flat-tile sites): the
>   doom floor moved 38 → 37 together with taking the h_idx pick off
>   the row-address path. The wall-v adoption (one folded floor per
>   bank height) measured 36 but was REVERTED: four floor_ints'
>   bounded-step stages hold ~8.2k residual columns at once and blow
>   the d=4096 compile gate. **New torchwright ask**: a
>   multi-`output_map` floor_int — ONE shared bounded-step stage, K
>   saturating stages with per-wrap δ-weights (hidden lanes, not
>   residual columns) — recovers that −1 width-free.
> - **Item 3 NOT landed** — still a live request; at floor 36 the
>   dispatch tail runs two serial `cond_gate`s (L33/L34 region).
> - **Item 4 LANDED** (`6f242b4`; the Linear-over-Concatenate
>   compile-to-zero bug it introduced was fixed by `299792e`).
> - **Item 5** — still research-grade, unclaimed.

For a torchwright session. Written 2026-07-05 from measurements on the
doom `depth-flatten` branch (tip `eae3c9e`). Self-contained; every
claim below was measured this session unless marked as a caveat to
verify.

## Context in three sentences

The production DOOM graph compiles to 47 layers, and 47 exactly equals
the dependency-DAG longest path after the always-on linear-fusion
pre-pass — CP-SAT sits AT the topological floor (production compile
confirmed 47 at `optimize: 3`), so compiled depth falls if and only if
a dependency chain gets shorter. Each layer is ~3.8 GB of full-frame KV
at d=8192. The binding chain's tail (identical on the two tied spines,
`pix/R_DrawColumn` and `stor/R_StoreWallRange`) is:

    multiply → floor_int (2 sublayers) → sawtooth PWL
    → sawtooth-bank pick → texel table_lookup_2d
      (clamp_i → row_step → row_deltas → col_gate)
    → wall-bank pick → colormap (PWL → gated pick → sum)
    → emit → dispatch Linears

Doom-side oracle to re-measure after any op change (~2 min, CPU, from
the doom repo — the env vars are load-bearing):

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \
    TORCHWRIGHT_DOOM_HUD=1 \
    CRIT_PATH=1 OPT_GRAPH=1 python -m scripts.analyze_forward_cost
    # depth-flatten baseline: fused=766 critical_path_layers=47
    # scripts/critical_chain.py (same env) prints the witness chain

Ranked by (measured value) / (risk). Items 1–2 cut the floor 47 → ~45
by themselves; item 3 is for the paint-cascade work happening in
parallel; item 4 is a fusion-pass extension, not an op.

## 1. Range-aware input clamp skip in `table_lookup_2d` (−1 layer, cheapest)

`ops/swiglu/map_select.py:557` (`_upper_clamp`) — `table_lookup_2d`
(`:680`) unconditionally spends one FFN sublayer per axis
range-guarding its index (`min(g·x, top)`). That sublayer is pure
protection: when the index node's static value range already fits
`[0, top]`, the clamp is the identity.

**Measured cost of the clamp**: swapping the doom colormap stack for
`table_lookup_2d(cmap_row, palette_index, COLORMAP)` moved the floor
47 → 48 — the extra layer is exactly `clamp_j` on the late-arriving
palette-index input; the two other column-stage sublayers
(`col_step → col_gate`) match the old form's depth. (Doom findings:
`depth_flatten_plan.md`, execution record, D2 entry.)

**The improvement**: skip `_upper_clamp` (per axis) when the input
node's value-range annotation proves `0 ≤ g·x ≤ top`. Node ranges
already exist (`NodeValueType` / `Range`); this is a conditional in
the existing builder, not a new op. Exactness: clamping an in-range
value is the identity, so no noise-budget change on the skip path.

**What it buys doom**: `clamp_i` on the texel lookup's row address is
zero-slack at L34 on BOTH spines (the row address is
`snap_index·H + v_mod_h`; both terms carry provable bounds) → −1
compiled layer. It also revives the doom colormap swap at depth
parity, which deletes ~17.6k hidden lanes (the #2 lane consumer).

**Caveat to verify**: the range annotation on the actual doom inputs
must survive to the call site (the sawtooth PWL output and the snap
Linear both carry ranges today — confirm they're tight enough after
fusion, i.e. check on the lowered copy, not the source graph).

## 2. `floor_int` with per-boundary output deltas (−1 layer)

`ops/swiglu/arithmetic_ops.py:652`. The spine runs `floor_int` (its
bounded-step stage + saturating stage, 2 sublayers) and then feeds the
floored integer into a piecewise-constant "sawtooth" PWL (`v mod H`,
one more FFN; the PWL itself is doom-side, `pwl_banks.py`, lowering to
`piecewise_linear` at `arithmetic_ops.py:297`).

Floor-then-any-piecewise-constant-function is itself piecewise
constant with the SAME integer breakpoints. The saturating stage
already produces a clean 0/1 indicator per boundary; its output
weights are currently all-ones (each boundary contributes +1).
Generalize the op signature so a caller can supply per-boundary
output deltas — for the sawtooth: +1 per step, −(H−1) at multiples of
H — and the downstream PWL disappears. Reweighting exact 0/1
indicators by constants is exact, so the noise story is the
saturating stage's existing one.

**What it buys doom**: −1 sublayer on both spines (the
`floor_int → sawtooth` chain at L30–32), and the same shape likely
recurs at other floor-then-wrap sites off-spine.

**Caveat to verify**: today the sawtooth evaluates a *snapped
integer*, so inputs never sit in its transition bands; the composed
op evaluates its boundaries on the raw pre-floor value, whose
half-integer band behavior must match the current
`floor_int → sawtooth` contract (the two-stage floor design exists
precisely to manage those bands — the composition must not reopen
that).

## 3. Conjunction-gated select: `cond_gate` with a multi-condition AND (−1 sublayer per flattened ladder)

`ops/swiglu/logic_ops.py:184` (`cond_gate`), `:78` (`bool_all_true`).
The doom "priority-ladder flatten" recipe (landed as D1, and about to
be applied inside the paint cascade by a parallel session) pays one
FFN to snap `bool_all_true([c1, c2, ...])` and a second FFN to gate a
value on the result. Fused: put the conjunction affine directly in
the gate row — `gate = scale·(Σ c_i − (n−1))`. With clean ±1 inputs
the sum is exactly n (all true) or ≤ n−2, so the hinge argument is
≥ +scale/2 or ≤ −3·scale/2: saturated, exact. This is the same
argument `onehot_lookup`'s multi-block case already certifies
(`ops/swiglu/onehot_table.py`, the n_blocks > 1 lane construction) —
the op is a thin generalization of machinery you already trust.

**What it buys doom**: nothing at the already-landed pixel tail (its
mask layer is off-spine), but the paint-cascade flatten
(`paint_cascade_plan.md`, phases P2–P3, running on a parallel doom
branch) will build exclusive-mask + switch structures where the mask
FFN may sit ON the spine — there it's −1 sublayer per flattened
ladder. Coordinate with that session before building; they can
confirm from their P0 layer map whether the masks bind.

## 4. Fusion-pass extension (not an op): Linear through Concatenate

`graph/optimize.py:194` (`fuse_consecutive_linears`). The last 3–4
compiled layers are output-assembly Linears the pre-pass cannot eat
because `Concatenate` blocks the Linear→Linear fold.
`Linear(Concat(a, b)) = Linear_a(a) + Linear_b(b)` distributes
exactly (split the weight matrix by row blocks); each piece then
folds into its upstream Linear when single-consumer, and the sum is
Add hardware. Plausibly 1–2 layers off the dispatch tail; measure
with the doom oracle, the parameter-count guard should stay
(width-safety: don't fold when it grows params).

## 5. Research-grade, recorded for completeness: a two-stage 2D lookup that never materializes the row vector

`table_lookup_2d` materializes a live row vector of width =
table-columns on the residual stream between its row and column
stages. That's what killed the doom "compose the colormap into the
texel tables" idea (D4): the composed tables are 32× wider on one
axis, the row vector becomes 4096–8192 columns (one bank alone equals
the ENTIRE d=8192 residual stream), and ~10.7k columns would need to
be simultaneously live in a zero-slack window. If a lookup existed
whose column stage consumes the row stage chunk-by-chunk without ever
holding the full row (e.g. gate rows that address row-chunk partials
summed on Add hardware), composition would delete the whole colormap
stage from both spines (~3 layers each). No concrete design is
proposed; the width arithmetic that any design must beat is in the
doom `depth_flatten_plan.md` execution record (D4 entry).

## Gates and coordination

- Every op change re-measures against its exact-math reference and
  updates `docs/op_noise_data.json` per the torchwright noise
  workflow; the doom re-measure is the oracle command above plus
  `make test` in the doom repo (275 tests, includes the graph-level
  pixel oracles and the d=4096 compile gate).
- `ops/swiglu/*` is the doom width track's declared file territory
  (`width_d4096_plan.md` consolidation contract) — coordinate before
  landing so the two torchwright efforts don't collide.
- The paint-cascade doom session (item 3's consumer) is on a separate
  doom branch; its work order is doom `paint_cascade_plan.md`.
