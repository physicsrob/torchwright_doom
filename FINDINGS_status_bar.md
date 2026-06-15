# Findings: status-bar (HUD) rendering — Phase 3

Working notes for the status-bar implementation (`plan_status_bar.md` Phase 3,
the faithful per-asset `V_DrawPatch` design). Branch `phase3-status-bar`
(worktree), atop the weapon Phase 2 commits.

**Status in one line:** the bar bakes, composites in-graph as a draw-list of
`V_DrawPatch` calls, and the **lowres render-cert is green** — 100% coverage,
100% within-option (the bar now in the reference), `stopped=terminal`, accept
back to 87.5%; the production 320x200 cert is the remaining gate.

**Lowres render-cert (configs/e1m1_lowres.yaml, both close-out fixes, run
`hud_fix4`):** 17332 tokens in 247s (was 602s), 7.90 tok/pass (was 3.23),
**accept 87.5%** (was 69.2%), draft efficiency 87.6% (was 28.1%),
`stopped=terminal`; image compare 160x100: **coverage 100%, only-in-generated 0
(was 2560 = the whole bar), within-option 15854/15854 = 100%**, exact-color
93.2% (the half-res decimation of the small numbers — inherent; the 320x200 path
is crisp). 51 layers.

---

## Design: the model draws the bar (no pre-compositing)

Rejected the original "bake the whole bar into one image" design — that does
`V_DrawPatch`'s compositing job outside the model. Instead (per the user, and
faithful to `st_stuff.c`):

- **Each HUD lump is embedded as its own masked patch** (`hud_assets.py` ->
  `AssetBanks.hud_table_2d`, one stacked sentinel table banked by `patch_id`),
  exactly as flats/textures are embedded — never merged.
- **A static draw-list** of `(patch_id, origin_x, origin_y, w, h)` (one entry per
  `V_DrawPatch`) records DOOM's painter-order sequence for the hardcoded E1M1
  pistol-start state — 32 items.
- **The graph composites them one at a time** under last-write-wins, so widgets
  overwrite the plate beneath them.

**Verified NOT the weapon's path:** the bar is `V_DrawPatch` (raw, unscaled,
unlit, 1:1 blit); the weapon is the scaled/lit psprite/masked-column drawer.
Kept as two phases (`plan_status_bar.md` §8d).

## What landed (all on the worktree, uncommitted)

- `wad_assets.py` — public `patch()` loader; `PatchImage` carries the DOOM
  offsets `V_DrawPatch` needs (the face `STFST01` is `(-5,-2)`).
- `hud_assets.py` — patch bank, draw-list, the faithful `V_DrawPatch` reference
  blit, the per-item draw-list tables. Scale-aware (1 / 2).
- `assets.py` — `HudAssets.color_or_transparent(patch_id, u, v)` (banked, one
  sentinel value for transparency — the weapon's proven pattern, not the plan's
  two-accessor sketch).
- `vocab.py` — `HUD_BEGIN` (slotless) + `HUD_ITEM(item)` tokens.
- `flat_state.py` — `HudPassState` (mirrors the weapon's `hud_seen` + cursor
  recovery, plus the one new piece, `hud_item`).
- `statusbar_renderer.py` — the spine: the weapon's column walk wrapped in a
  draw-item loop.
- `pixel_dispatcher.py` / `render_main.py` / `psprite_renderer.py` — the
  `hud_seen` forks (gated on `HUD_ENABLED`), the dispatch branches, and the
  weapon->`HUD_BEGIN` splice.
- `pydoom/drafter.py` — `_statusbar_plan_tail` (the token oracle); the
  weapon/bar scaffold types added to `_FLAT_SCAN_TYPES` so the drafter stays in
  sync (close-out #1).
- `inference/compare.py` — `_hud_reference_cells`, now wired into
  `reference_pixels` / `reference_options` (close-out #2).
- `tests/scene/test_drafter_resync.py` — HUD-on drafter re-sync regression
  (terminates + bar tail matches); `scripts/probe_drafter_desync.py` — the
  per-phase accept-rate measurement that found close-out #1.

## Gates

- **HUD-off compile: bit-identical** (after gating the HUD machinery on
  `HUD_ENABLED` — the first cut built the heavy `decision()` in the shared
  branches even HUD-off and blew the d=4096 cramming point with "No progress").
- **HUD-on compiles to a valid transformer at +2 layers.** Local heuristic
  structural check: 76 layers; the **Modal CP-SAT (`optimize: 2`) compile lands at
  50 layers** at both scale 1 and scale 2 — vs the 48-layer HUD-off baseline (with
  the linear-fusion gate), so **the bar adds only +2 layers**: the residual width
  absorbed the spine. The earlier "+19" fear was a heuristic-compile artifact.
  I1–I4 invariants pass; d_embed=1409, vocab_rows=133096.
- **`test_hud_assets.py` (5) + `test_hud_oracle.py` (3): green** — the bake,
  right-justification, and the graph color lookup (vs the bake).
- **Drafter token stream well-formed**; hand-verified drafter == spine token
  sequences (spec_decode will accept).

## Decisions / corrections

- **KV-cache: no change.** `HUD_ITEM` must be **permanent** (read across the whole
  plate, ~10k tokens) — the plan's §6 idea to make it expiring would corrupt mid
  plate. Cursor recoveries reuse the already-expiring `setCursorX/Y` at <= column
  height (~32) back, shorter than the weapon's certified reads.
- **Plan §3b table was incomplete** — DOOM also draws the four BULL/SHEL/RCKT/CELL
  current+max counts (`w_ammo`/`w_maxammo`, small yellow shortnum, reordered by y
  so am_cell/am_misl land under CELL/RCKT), and the ARMS panel is six numbers 2-7.
  Both corrected in the plan + the bake.

## The real bug: a `gt_screen` units error (found by MEASURING, not inferring)

**Root cause (one sentence):** the bar's "is this the last column?" test was
`gt_screen(local_col, width-1.5)`, but `gt_screen(a,b)` is `compare(a-b, 0.5)` —
the integer `a > b`, which already bakes in the 0.5 — so my `-1.5` made it compute
`local_col >= width`, which the cursor (max `width-1`) can never reach; the widget
never advanced and the spine overran. `last_item` had the identical `-1.5` error,
so DONE could never fire either. **Fix:** `width-1.5 -> width-2.0` (and
`n_items-1.5 -> n_items-2.0`), i.e. `gt_screen(index, count-2)` == `index >= count-1`.

This is a pure LOGIC error, visible in `reference_eval` (exact math) — catchable
without any render. The honest process failure: I inferred a "value decode ±1"
story from the render symptom and chased it through two wrong fixes (a parity
channel, then a position-counter design) before MEASURING. The measurement
(`scripts/probe_cursor_decode.py`) showed the decode is **exact** at every value
including the screen edge (setCursorX is a 1-digit IntSlot — no floor layer), and
a 5-line `reference_eval` of `gt_screen` showed the threshold was the bug. Lesson:
test the suspected logic directly in exact math before theorizing about compiled
numerics. Both wrong fixes were reverted; the spine is back to vanilla
`inp.cursor_x`.

## Close-out bug #1: the drafter desynced through the whole weapon+bar (the 69% accept)

**Root cause (one sentence):** `ARDrafter.consume` only advances the flat-pass
plan pointer for token types listed in `_FLAT_SCAN_TYPES`, and the weapon/bar
scaffold tokens (`DRAW_PSPRITES_BEGIN`, the weapon's re-asserted
`SET_CURSOR_DIRECTION_Y`, `HUD_BEGIN`, and every `HUD_ITEM`) were missing from
that set — so the plan pointer never advanced past them and the drafter's
proposal fell one step further behind for each one, mispredicting almost every
token from the weapon onward.

The render is correct regardless (spec-decode falls back to the model's own
token on a mispredict), so this was purely a **speed** regression — but a large
one: the proposal is stale, so spec-decode rejects it and pays an extra forward
pass. A CPU simulation of the runtime loop (`scripts/probe_drafter_desync.py`,
ground truth = the plan flattened, drafter = a fresh `_FlatScanState` walked
with `consume`'s exact routing) measured the per-phase accept rate:

| phase  | before | after |
|--------|-------:|------:|
| flat (3D) | 100.0% | 100.0% |
| weapon | 4.9% | 100.0% |
| bar | 17.1% | 100.0% |

So the weapon has silently drafted at ~5% accept since Phase 2 — its
correctness cert is a token-match oracle that never exercised the consume loop.
The bar made it visible (the render's 69% overall, the draft count exploding at
the bar start). **Fix:** add the four scaffold types to `_FLAT_SCAN_TYPES`. The
frame's first `SET_CURSOR_DIRECTION_Y` is still caught earlier in `consume`
(before the flat-scan router), so only the weapon's re-assertion reaches the new
branch. With the fix the self-feedback oracle (`expected_ar_tokens`) terminates
HUD-on (9499 tokens) and its bar tail matches `_statusbar_plan_tail`
token-for-token; without it that loop spins forever (proposes the same stale
step).

## Close-out bug #2: the bar was missing from the image-compare reference

`_hud_reference_cells()` (the bar's V_DrawPatch reference blit) was written but
never wired into `reference_pixels` / `reference_options` — both only added the
weapon cells. So the compare reported the entire bar as `only in generated`
(2560 cells at scale 2) and never validated it. Not an env-timing issue (the
render's `HUD_ENABLED` is on, since the screen dims were right) — a plain missed
call. **Fix:** add the `_hud_reference_cells()` loop to both functions
(last-write-wins, like the weapon). Verified locally: `reference_pixels` now
returns the 2560 bar pixels at y>=84.

## (Earlier, incomplete) full-width-patch framing

The first lowres render-cert **failed**: the compiled spine overran (never hit
DONE, ground to the position cap). The `token_dump.json` showed only **2 HUD
tokens** — `HUD_BEGIN` + one `HUD_ITEM(0)` — and the cursor walking `0,1,…,159`
then **stuck at 159**. Root cause: the plate fills the full screen width (160
cols at scale 2 / 320 at scale 1), and the original item-done check advanced the
cursor to `width` (160) before testing — but `setCursorX`'s slot range is
`[0, SCREEN_WIDTH)`, so `setCursorX(160)` **clamps to 159**, sticking the walk on
the last column; the item counter never advanced. The weapon never hit this (its
bbox isn't full-width). The teacher-forced oracle couldn't catch it either — only
a free-running render exercises the position encoding (the §0/§5 lesson again).

**Fix:** the item-done transition now fires **on** the last column
(`local_col >= width-1`) inside `decision()`, emitting `HUD_ITEM(i+1)`/`DONE`
directly — the cursor never advances to `width`, so it stays in range at both
scales. The drafter's last column likewise emits no trailing `setCursorX`.

## Open

- Full Modal render-cert at production 320x200 (the render-verify gate) — the
  lowres cert above is green; 320x200 is crisper (the half-res small-number
  exact-color gap should close).
- Layer count read 51 here vs the earlier "50" note — within CP-SAT search
  variance (the graph is unchanged by these host-side fixes); worth a one-line
  solver-status check before trusting any layer-count delta.
- Half-res (scale 2) small numbers are texture-quality (decimated) — inherent;
  the 320x200 path is crisp.
