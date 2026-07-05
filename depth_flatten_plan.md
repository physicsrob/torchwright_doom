# Depth plan — shorten the dependency spine below 49 layers

Execution plan for one of two parallel tracks (the other is
`width_d4096_plan.md`). Written 2026-07-05 from the measurements in
`swiglu_opportunities_findings.md` — that doc is the evidence base;
this one is the work order. Self-contained for a fresh session.

**Branch**: doom `depth-flatten`. This track touches doom only — every
op it needs (`type_switch`, `bool_all_true`, `bool_not`,
`table_lookup_2d`, `onehot_lookup`) already exists in torchwright.
See "Consolidation contract" at the bottom.

## Goal and why

The production compile is 49 layers, and 49 exactly equals the
dependency-DAG longest path after the always-on linear-fusion pre-pass
(measured; the CP-SAT schedule sits AT the topological floor at
d=8192). So compiled depth falls if and only if the longest dependency
chain gets shorter — and each layer is ~3.8 GB of full-frame KV at
d=8192 (~1.9 GB at the width track's d=4096 target; the two tracks'
wins multiply).

The spine is narrow and mapped: only 150 of 6,608 schedulable nodes
have zero slack. Layer by layer (the findings doc has the full
per-layer table and witness chain):

- **L0–L4**: BOS/global-position recovery (`scene`), two attention
  reads.
- **L5–L27**: the `proj/paint` cascade — wall lighting selection plus
  wall-column branch logic; compares/selects interleaved with two
  attention reads (L11, L20). ~23 layers, and it is the **shared
  prefix of both spines**.
- **L28–L41**: the texel fetch + colormap application
  (`pix/R_DrawColumn`; after the first flatten a twin spine through
  `stor/R_StoreWallRange` binds at the same depth): coordinate floor,
  sawtooth, `table_lookup_2d`'s 4 internal layers, the colormap
  stack's 4 layers.
- **L42–L48**: the pixel-emit select ladder, the dispatch gate, and
  the output-assembly Linears.

## The oracle (run after every landing, ~2 min local, CPU)

    TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 \
    TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 TORCHWRIGHT_DOOM_DETAIL=low \
    TORCHWRIGHT_DOOM_HUD=1 \
    CRIT_PATH=1 OPT_GRAPH=1 uv run python -m scripts.analyze_forward_cost

prints the fused DAG floor — the layer count CP-SAT will compile to.
`scripts/critical_chain.py` (same env) shows what binds after each win
and re-derives the slack tables. **The env vars are load-bearing** —
without them you measure the 60×50 hud-off graph (the screen-env
trap; findings doc, Method).

## D1 — land the pixel-tail flatten (measured 49 → 47)

Prototyped and measured in the research session; reverted; the shape:

- `pixel_dispatcher.py`'s three shared branches (`after_wall_column`,
  `after_set_cursor_y`, `after_pixel_color`) each wrap the priority
  ladder `select(hud_seen, …, select(weapon_seen, …,
  select(flat_span_seen, flat, wall)))`. The conds are latch flags —
  priority, not exclusivity.
- Flat form: exclusive masks in one parallel layer —
  `m_hud = hud_seen`, `m_weapon = bool_all_true([weapon_seen,
  bool_not(hud_seen)])`, `m_flat = bool_all_true([flat_span_seen,
  bool_not(weapon_seen), bool_not(hud_seen)])`, `m_wall =
  bool_all_true([three negations])` — then one
  `type_switch((m_hud, hud_arm), (m_weapon, weapon_arm),
  (m_flat, flat_arm), (m_wall, wall_arm))`. Mirror the `HUD_ENABLED`
  fork (no HUD arm at all when off — bit-identical pre-HUD contract).
- Flatten all three branches even though only `after_pixel_color`
  binds (measured: 47 either way) — the other two are one bad rebase
  away from becoming the new cliff.

Why it is numerically safe (recorded in full in the findings doc, R2):
the main dispatch already flat-folds emit heads through
`type_switch`/`cond_gate` at the same magnitudes; a clean ±1 cond
makes the losing branch contribute exactly zero in fp32. The
discipline: every mask a snapped ±1 boolean (`bool_*` outputs are;
anything softmax-recovered keeps passing through `snap_bool` —
`render_ops.py:83` documents the failure class against the ~0.97-in-
~46,000 emit argmax margin). Degenerate case to preserve: before the
flat pass `flat_span_seen` is structurally false and the fork must
degenerate to the wall arm.

Gates: `make test`, flat-pixel oracle + AR rollout, d=4096 compile
gate, lowres `COMPARE=1`; production compile to confirm 47.

## D2 — colormap → `table_lookup_2d` (enables D4; −17.6k lanes)

`lighting.apply_colormap_row` builds all 32 COLORMAP rows as separate
256-breakpoint tables and picks one at runtime (`lighting.py:55–58`) —
a 32×256 compile-time-constant table built the expensive way, 4 layers
on both spines, 21,168 lanes. Replace each of the 4 instances with
`table_lookup_2d(light_row, palette_index, COLORMAP)` — ~900 lanes
per instance, exact on the integer grid (measured 0-error noise
entry), depth-neutral (its 4 internal layers replace the 4-layer pick
stack).

Verify first (both recorded in the findings doc): (a) the light-row
input is integer-snapped at every call site — the current
`pick_by_index` needs that too, so this should hold; (b) in-band
inputs blend *adjacent rows'* palette indices (garbage for unordered
palettes) — confirm inputs never sit in-band, the same argument the
texel lookups make.

## D3 — `onehot_lookup` at the constant-table picks (possible spine cut)

`pick_by_one_hot` / `pick_by_index` over a compile-time-constant table
pays `n·d_fill` gated lanes + a sum Linear + a sublayer for what is a
matmul: `onehot_lookup` (torchwright `onehot_table.py`) folds it into
one selection Linear — zero hidden lanes, zero added depth. Sites
(~8–10): `assets._snap_index` (`assets.py:68`), the five wall
bank-metadata reads (`assets.py:94–126`; `h_idx_oh` is the widest),
the statusbar constant tables (`statusbar_renderer.py:90`). The
bank-metadata reads sit upstream of the texel fetch on both spines —
run the oracle before/after; this may be a free layer. (NOT a
candidate: `lighting.py:58` — runtime table; and it dies in D2
anyway.)

## D4 — compose the colormap into the texel tables (~4 layers, both spines)

`lit = COLORMAP[light][TEX[row, u]]` is a compile-time-composable
function: bake `TEX_LIT[row·32 + light, u]` and index it with exact
linear arithmetic (`row·32 + light` — constants-scaled add, free).
Kills the whole colormap stage (D2's 4 lookup layers) from both
spines; table entries ×32 land in weights (a few MB), and balanced
axes keep the lane cost near today's texel+colormap total. Needs: the
same integrality arguments as D2, plus a check of which light row
varies per pixel vs per column/span (DOOM: `dc_colormap` fixed per
column, `ds_colormap` per span — the AR protocol's per-pixel light
input must be confirmed at each of the fetch sites). Do after D2
certifies; D2's caveats de-risk it.

## D5 — the `proj/paint` cascade (the big one, research)

**Promoted to its own work order: `paint_cascade_plan.md`** — traced
anatomy (`WallSpanRuntimeDraft.publish`, `wall_column_state.py:652`),
the hoist question made concrete, the ladder-compression designs, and
stop conditions. The summary below stands; an agent picking up D5
starts there.

The shared L5–L27 prefix (~23 layers) is the depth prize. Composition:
lighting/paint prep (`clamp_0_2`, `thermometer_floor_div`,
`in_range`), then compares/selects with two attention reads at L11 and
L20 (`doom_lighting.py` feeds it; `wall_column_renderer.py` holds 22
selects). Research questions, in order:

1. **Can the two attention reads hoist?** If their queries do NOT
   depend on earlier select outputs, read the past unconditionally in
   parallel and select afterward — each hoisted read + its downstream
   re-flatten is worth multiple layers. If a query is genuinely gated
   on a prior select's value, that segment is irreducible as designed.
2. **Which select stretches are priority ladders vs true data
   chains?** Priority ladders flatten by the D1 recipe (exclusive
   masks + `type_switch`); data chains (cond derived from the previous
   select's output) do not.
3. Re-run `scripts/critical_chain.py` after each cut — the spine will
   reshape, and the next stretch may be different from the map above.

No committed estimate — this is research; even 5 of the 23 layers
would be the largest single depth win available.

## Milestones

- M1: D1 landed and certified; production compile at 47.
- M2: D2 + D3 landed; oracle re-measured (expect 47, possibly 46 via
  D3); lane total drops ~18k.
- M3: D4 landed; expect ~43–44.
- M4: D5 findings — either a derivation that cuts the cascade or a
  documented proof it is data-dependent and irreducible.

Each landing: `make test` green, oracle number recorded in the commit
message, runtime gate (`make run COMPARE=1`, lowres first) at every
certification point. Two-committed-configs rule holds; config headers'
certified-numbers lines refresh only at re-certification.

## Consolidation contract (mirrored in `width_d4096_plan.md`)

- **File ownership** — depth track: `pixel_dispatcher.py`,
  `lighting.py`, `assets.py`, `statusbar_renderer.py`,
  `wall_column_renderer.py`, `doom_lighting.py`. Width track:
  torchwright `ops/swiglu/*` (new ops), doom `render_ops.py`
  (`_ray_count` / floors), `emit.py`, `uv_compute.py`,
  `flat_state.py`. Shared, additive-only: `std.py`, small
  `render_ops.py` helpers — keep those diffs minimal and coordinate.
- **Each branch stays independently green** (`make test` + its own
  oracle numbers recorded per landing) so either can land first.
- **Merge order**: whichever certifies first lands to `main`; the
  other rebases and re-runs its oracle (depth wins change the floor
  and the slack tables the width track reads; width wins don't move
  the depth oracle). The umbrella pointer bumps only on `main`
  landings.
- **The joint finish**: production CP-SAT at d=4096 with both tracks
  merged; the wins multiply (e.g. 45 layers × 4096 ≈ 85 GB of KV at
  57.4k positions vs today's 184).

## Execution record

- **2026-07-04, D1 landed** (`b35d250`): oracle 49 → 47 (fused 686 →
  695), exactly the R2 prototype numbers. `make test` green (275
  passed, 0 failed — includes the flat-pixel oracle, the AR rollout,
  and the d=4096 compile gate, which now passes outright). Lowres
  CP-SAT compile solved at 47 layers (floor met).
- **2026-07-04, D2 measured +1 layer — runtime swap NOT landed.**
  Implemented `apply_colormap_row` as
  `table_lookup_2d(cmap_row, raw_palette_index, COLORMAP_ROWS,
  sharpness=1000)`; the exact-match oracle
  (`test_texture_oracle.py::test_apply_colormap_row_matches_reference`)
  passes, and both verification caveats hold (the light row is
  integer-valued at source on both spines — derived embedding column
  on the wall side, constant integer tables on the flat side — and
  reaches the lookup only through attention recovery, so it never
  sits in a half-integer transition band). But the DAG floor measured
  **48, not 47**: `table_lookup_2d` spends one sublayer
  (`_upper_clamp` on the j axis) range-guarding its late-arriving
  input, which the old PWL-then-pick form never paid — old chain from
  `raw_palette_index` is 2 sublayers (PWL → gated pick, sum fuses),
  new is 3 (`clamp_j → col_step → col_gate`). Witness: the
  `stor/R_StoreWallRange` spine, colormap at L39–41 of 48. The
  transposed orientation pays the same +1 on the i axis (`clamp_i →
  row_step → row_deltas` = 3 to the live row vector). The
  depth-neutral premise in the findings doc is falsified; the
  ~17.6k-lane win is real but belongs to a width budget, not this
  track. **Consequence: skip the D2 runtime swap; go straight to D4**
  (composing the colormap into the texel tables deletes the whole
  stage from both spines and leaves `apply_colormap_row` with no
  callers — D2's swap would be churn). D2's verification work
  (integrality at all call sites, per-column/per-span light
  constancy) carries into D4 and is recorded above.
- **2026-07-04, D3 landed** (`5ce6096`): constant-table picks lowered
  through the new `std.pick_const_by_index` → `onehot_lookup`
  selection Linear. Oracle 47 unchanged (the sites sit on the slack
  branch; the v-coordinate chain binds), fused 695 → 766. `make test`
  green (275 passed, 5/5 shards).
- **2026-07-04, D4 is a width-infeasibility NO-GO as specced —
  documented, not landed.** The composed tables build and verify
  (`lit[r, light·W + u] = COLORMAP[light][tex[r, u]]`, spot-checked
  against the WAD banks), but the real bank inventory kills the
  schedule: 8 wall banks, and `table_lookup_2d` materializes a live
  row vector of width = table columns on the residual stream between
  its row and column stages. Composing light onto the column axis
  makes that vector `32·W` wide — bank7 (W=256) alone is 8192 = the
  ENTIRE residual stream at d=8192, and the per-bank vectors sum to
  ~10.7k simultaneously-live columns in the zero-slack 3-layer window
  before the bank pick (all 8 banks feed it eagerly; no slack to
  stagger). The row-axis orientation is worse (`32·R` staircase-step
  transients, ~12k for bank6). The findings doc's "balanced axes ≈
  7.5k lanes" assumed a floor/mod index split, which is exactly the
  spine depth D4 was supposed to buy. Conclusion: the composition
  cannot schedule at the DAG floor at d=8192 (hopeless at d=4096);
  it would only revive with a new torchwright op (e.g. a fused
  two-stage lookup that never materializes the full row vector) —
  out of this track's doom-only contract. Colormap depth on the
  spine today is 3 layers (witness L39–41: `colormap_row` PWL →
  `broadcast_select` → `dynamic_extract_sum`); that prize stays on
  the table for an op-level design.
- **2026-07-04, production compile at D1 (`optimize: 2`) reached 52,
  not 47** — the CP-SAT floor probe (horizon 48) went UNKNOWN at its
  150s budget and warm-start descent stalled at 52; the lowres graph
  solved to its 47 floor in 152s. Retry with a /tmp `optimize: 3`
  variant (300s) launched; if the production solve still cannot
  approach the floor, the D1 landing needs a width look (the flat
  `type_switch` holds 4 gated emit-head copies live at once where
  the ladder held at most 2).
