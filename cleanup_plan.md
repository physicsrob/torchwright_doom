# torchwright_doom cleanup plan

Origin: the 2026-06-12 tech-debt audit (105 raw findings across 9
dimensions; 93 survived two-lens adversarial verification). Full
per-finding detail with evidence and line numbers is in
`out/tech_debt_review_2026-06-12.md` (gitignored). Finding numbers
below (`#N`) index that file.

**Batch 1 landed** — `torchwright_doom` `4865a92`, umbrella pointer bump
`3bf2162`: doc truth, dependency closure, the `ruff check --select F`
lint gate, and fail-loud error paths. This file tracks what remains.

---

## 2026-06-13 — comprehension + density pass (Batch 5)

A second audit, scoped to **newcomer comprehension of the computational
graph** and **density that increases clarity** (not generic tech debt).
13 lens-scoped finders + two-lens adversarial verification (accuracy +
"does this actually help a newcomer") + a completeness critic: 56 findings
confirmed, 68 rejected (4 as duplicates of the 2026-06-12 settled set).

**Landed (uncommitted, working tree).** All docs-only + host-code findings:

- *Orientation docs.* `README.md` rewritten (3 lines -> what-it-is, the
  dumb-host principle, the entry point, the read-side->write-side reading
  path, the WAD->prefill chain). `GLOSSARY.md` gained ~4 sections /
  ~20 entries (read side / write side, the `after_<token>` convention, the
  import-time-node rule, the Phase-letter legend, owner split, handle /
  validity / presence / subcontext, scene fact / header context, `span`,
  marker's 3 senses, assets+lighting cluster, prefill cluster, runtime
  cluster). `CLAUDE.md` "Rough layout" gained the reading path + prefill
  chain and re-homed `pwl_banks` (pixel-pass, not lighting).
- *Module docstrings/comments (50 edits, comment-only — zero graph nodes
  added, so the compiled graph is bit-identical).* Highlights: `scene_facts`
  channel-contract correction (it described the OPPOSITE of how presence
  lookups key); `ProtocolEntry` field docs; `ProtocolTokenView` category map
  + banners; `WallColumnState.publish` section skeleton; `visplane_*` branch
  maps + layout; the flat-pass control-flow map; the windowed-cache
  read-distance invariant noted at `past`/`attention_handles`; the
  spec-decode reuse machinery explained; `render_ops` section list; the
  `digit_quad_row` dangling reference; `value_ranges` R0..R9 purposes.
- *Host-code (lint-green; behavior validated locally — see below).*
  `dispatch_next_token` dead `input_vec` param dropped; `PastHandleScope`'s
  four pass-through delegators deleted (rely on `__getattr__`, -34);
  `assets.py` dead `_WALL_*`/`_FLAT_*` aliases + misleading comment deleted;
  `PlaneTables.flat_names` (computed, never read) deleted; `build_prompt`
  marker->value boilerplate folded into a `_marked` helper + the BBOX block
  into a scannable table-loop (marker/range stay explicit); raw-`Seg` vs
  baked-`Segment` disambiguated; dead `is not None` plane-id guards removed;
  `diagnostic.teacher_forced_scan`'s `compiled` param typed `OnnxDebugSession`
  (its `empty_past()` legitimately takes no `max_len`).

**Certification.** `make lint` green. Behavior-relevant host-code changes
validated locally via `make test-local`: `test_prompt_equivalence`
(build_prompt byte-identical to sandbox `get_prefill`), `test_graph_past`
(delegator removal), `test_dispatch_dedup` + `test_forward_compiles` (the
full forward graph still compiles to ONNX, render_main param). The docs-only
edits add no graph node, so the compiled schedule is unchanged by
construction. **The formal gate — the full `make test` Modal suite (~22 min)
— has NOT been run** (it exceeds the foreground tool cap, and the project
rule is "never run tests in the background"); run it to finalize before
relying on these in production.

**Deferred from this pass (not done):**

- `_xtoviewangle_rad` recomputes `_TAN_FOV_HALF` inline (a third copy the
  4c cross-module dedup missed) — **graph-touching**, needs production-scale
  token-identical certification; bundle into a future certified graph batch.
- A PLAYPAL-consistency test (`asset_config.PLAYPAL` == `asset_banks.PLAYPAL`)
  — skipped because importing `asset_banks` triggers a WAD load whose
  presence in the Modal test image isn't certain; add only with a graceful
  skip or once WAD-in-image is confirmed.
- A few LOW items: committing `render_protocol_table()` output somewhere
  discoverable; `_U_MOD_BY_BANK` is a dead duplicate of
  `WallAssets.__post_init__` but is cited as a pattern exemplar (left in,
  flagged). Full per-finding detail: `out/audit_confirmed.json` (gitignored).

---

**Progress (2026-06-12, this cleanup session):**

- **Loose end CLOSED.** Batch 1 certified at production scale: fresh-
  process pure-AR full frame on B200, `compare_token_dumps` vs the
  tier-1 certified baseline (`out/tier1_prod_b`) — TOKEN-IDENTICAL
  (prefill 3,613 / predictions 25,350 / rollout 25,350, zero diffs).
  NB an in-process `mode=both` run at this config trips the
  propagated-tie budget (27 pixel flips) — that is the documented
  leg-2 contamination (plan_tier1_expiry.md "Gate findings"), not a
  divergence; production certification is fresh-process pure-vs-
  baseline via `scripts/compare_token_dumps.py`.
- **Batch 3 + 4a LANDED** (`52c20a7`): all 23 oracle/bridge gates run
  under `make test` (262 passed, 0 skipped). Re-enabling schema-sync
  exposed real contract drift (4ca305c's id_lifted_key on
  node.child0/child1 never mirrored) — fixed sandbox-side
  (doom_sandbox `5e7196c`), baseline regenerated. #62 and #64 also
  done.
- **Batch 2 LANDED** (`5f63c75`). Suite 270 passed, 0 skipped.
  Verification render: bare `make run` (zero variables) resolved every
  knob from the YAML — full frame 25,350 tokens, stopped=terminal,
  spec-decode 7.61 tok/pass at 87% per-token accept, 100% of pixels
  within the reference option set.
- **Batch 4b LANDED** (`08ae6d9`). Suite 274 passed, 0 skipped.
- **Batch 4c LANDED** (`3b1e581`): constants + byte-identical helpers
  (ANGLE_BAM/_TAN_FOV_HALF, _PROJ_RATIO, the screen-column radix
  scheme, scene_facts `_keyed_value_lookup`, decode `_walk_pixels`).
  Gates: suite 274 passed / 0 skipped; production pure-AR render
  TOKEN-IDENTICAL vs the certified baseline (out/batch4c_cert).
- **Batch 4d LANDED** (`9b2706a`): `rw_distance` dedup.  Gates: suite
  274 passed / 0 skipped; production pure-AR render TOKEN-IDENTICAL
  vs the certified baseline (out/batch4d_cert).  **#33
  (radix-successor scaffolding unification) examined and DECLINED** —
  inspection shows the solid_intervals and visplane_state publishers
  are parametrically different, not copy-paste: different radix
  domains (screen columns vs plane ids), different table semantics
  (`lo >= k` inclusive vs `lo > t` strict), different table widths
  (the plane `above_lo` table is B+1 wide with a leading "above -1"
  slot for R_DrawPlanes' find-first query), different H2 sentinel
  wiring and payload compositions.  A shared builder would need ~8
  parameters and would bury exactly the semantic differences that are
  load-bearing — the "parallel-but-different siblings are the design"
  verdict the audit's verifiers applied elsewhere.  Reopen only with a
  design that keeps the per-domain tables explicit.

**Session finding — the mode=both propagated-tie budget is
mis-calibrated for the production frame.** Cross-process spec-vs-pure
at the production config differs at 200/25,350 positions: 173 `value`
scratch ties (141 at ±1 bin; 3 at 2048 bins on degenerate
`scalestep.den = -1` sentinel rows), 27 pixel colormap/±-texel ties —
all type-aligned, both legs 100% within the reference option set,
zero structural divergence.  The same 27 pixel diffs appear in-process
(deterministic batched-vs-single-row kernel shapes, NOT run-to-run
noise), so `cli._PROPAGATED_TIE_BUDGET = 8` (calibrated at
d4096/9,739 tokens: 2 propagated) fails any production-scale
`mode=both` run on arrival.  Decision deferred to the user: either
scale the budget to frame length (~30 for 160×100) or document
`mode=both` as a sub-production-scale tool only.

---

## Working discipline (applies to every batch)

- **One config.** There is exactly one committed config,
  `configs/e1m1.yaml`. A variant copies it to `/tmp` and runs with
  `--config /tmp/<name>.yaml`. Never commit a second YAML under
  `configs/`.
- **Dumb host.** All rendering logic lives in the graph; the host only
  feeds tokens and blits pixels. No batch may move computation host-side.
- **Two verification tiers.**
  - *Non-graph changes* (docs, deps, runtime/host code, tests): `make
    lint` green + the full `make test` Modal suite (262+ passed,
    0 skipped since batch 3).
  - *Graph-touching changes* (anything that alters node construction in
    the `torchwright_doom/*.py` renderer modules): the above PLUS
    production-scale certification: a fresh-process `make run
    RENDER_MODE=pure_ar` full frame, compared against the prior
    certified dump with `scripts/compare_token_dumps.py`
    (`--allow-value-ties N` for cross-artifact compares).  NB
    in-process `mode=both` CANNOT gate at the production config — its
    leg-2 kernel-shape ties trip the propagated budget by themselves
    (see plan_tier1_expiry.md "Gate findings" and the session findings
    below).  `make test` only certifies at test scale (60×50, small
    `d`).
- **The node-id-shift rule.** Adding or removing *any* graph node —
  even an unreachable one — shifts `global_node_id` for every node
  built after it, which moves set/dict iteration order in the compiler,
  which can move residual-column assignments and drift compiled values
  within noise. So "I only deleted dead code" is not a safe claim for
  graph modules. Every graph-touching change recompiles (the git-sha
  cache key forces it) and must be certified token-identical at
  production scale, not assumed.

---

## Loose end — certify batch 1 at production scale

Batch 1 deleted one dead graph node (`zero_value = constant(0.0)` in
`flat_state.py`). The reachable graph is identical, but per the
node-id-shift rule the compiled schedule is not *guaranteed* identical.
The Modal suite certified it at test scale only.

**Action:** the next `make run` recompiles the e1m1 artifact (git-sha
key changed) and its spec/pure equivalence gate certifies it
token-identical. No separate work — just don't treat batch 1 as
production-certified until a render has passed that gate once.

---

## Batch 2 — render defaults into the YAML

**Goal.** Single-source the render-job defaults that currently drift
across three layers (Makefile, `inference/cli.py`, `modal_render.py`).
The drift is the failure class CLAUDE.md names ("change them in
lockstep or the stale copy bites"): direct invocations truncate the
frame mid-render and silently disable speculative decoding.

**Closes:** #5, #19, #21, #23 (defaults drift three ways), #77
(`DEFAULT_PREFILL_CHUNK_SIZE=1024` vs the 128 every entry point uses),
#29 / #47 / #59 / #69 / #87 (default pose hardcoded in five places),
#72 (partial — `run_config`'s kwarg mirror in `modal_render.py`), and
the GPU-default disagreement #18 / #86 as a companion fix.

**Design (already agreed in conversation).**

- Add an **optional `run:` section** to `configs/e1m1.yaml` carrying
  `max_positions`, `draft_window`, `mode`, `prefill_chunk_size`, and
  `pose` (`x`, `y`, `angle`, `viewz`). These belong in the config
  because they are *coupled to fields already there*: `max_positions`
  is sized from the frame length, which depends on `model.scale`;
  `prefill_chunk_size`'s safe value is derived from
  `cache_stride`/`max_seq_len`; the default pose is map data, the same
  kind of fact as `region:`.
- The section is **optional with code defaults** so existing `/tmp`
  variants and tests that build `RenderConfig()` directly keep working.
- **`RENDER_GPU` stays an env var** — `modal_render.py` reads it at
  module-import time to parameterize the `@app.function(gpu=...)`
  decorator, before `--config` is known. Cannot move without
  restructuring the Modal app. While here, reconcile the disagreement
  (#18/#86): Makefile says `b200`, `modal_render.py` falls back to
  `a100-80gb` with a comment describing wiring that no longer exists —
  make the fallback + docstring + comment match the Makefile.
- **Stays in Makefile/env** (infra + cosmetics, no coupling to render
  correctness): `OUT_DIR`, `PNG`/`COMPARE`/`PROFILE`, `png_zoom`,
  `progress_every`.

**Wiring.**

- `cli.py` argparse defaults and `modal_render.py` entrypoint args
  become `None`; after loading the config, resolve
  `cli_value if cli_value is not None else config.run.<field>`. CLI
  flags remain working overrides.
- Delete the Makefile's `RENDER_MAX_POSITIONS ?=` / `RENDER_DRAFT_WINDOW
  ?=` / pose defaults rather than updating them — the Makefile only
  passes a flag when the user explicitly sets the variable. This
  removes the Makefile from the defaults business entirely, which is
  what makes the drift class *unrepresentable* rather than just
  currently-fixed.
- `default_pose_world(config)` currently ignores its `config` param and
  returns a hardcoded tuple — wire it to read `config.run.pose`, giving
  that param its purpose and collapsing the five pose copies into one.
- Split `DEFAULT_PREFILL_CHUNK_SIZE`'s two roles: it is both the API
  default in `generation.py` AND the windowed staging-tail budget in
  `onnx_runtime.py:256`. Give the staging-tail use its own named
  constant so the run default can move without touching the budget.

**New test (load-bearing).** Assert that `canonical_compile_payload`'s
key set is unchanged when the `run:` knobs change — they are runtime
knobs and must NOT enter the compile-cache key. This pins the current
behavior (the payload selects explicit fields, not `asdict(config)`) so
a future refactor can't silently start recompiling on a runtime-knob
edit. Mirror the existing `expiring_types`-not-in-key guarantee.

**Verification.** `make lint` + `make test`, then **a render** —
batch 2 changes the effective behavior of direct invocations (frame
now covered, speculation on), so confirm on `make run` that the
spec/pure gate passes and the frame completes (not "cap").

**Risk.** Low-to-medium. No graph construction changes, but it alters
what a bare `modal run modal_render.py` / `python -m
torchwright_doom.inference run` actually does. The danger is a
resolution-order bug (config value silently winning over an explicit
CLI flag, or vice versa) — cover both directions in the test.

---

## Batch 3 — oracle gates run in CI  *(the audit's one HIGH finding)*

**Goal.** Make the renderer's correctness gates actually run under
`make test`. Today they all skip, so the green CI signal says nothing
about the geometry/raster write side.

**Closes:** #0 (HIGH). Also folds the umbrella `make test` mislabel
(#62) and the dormant-sharding docstrings (#50/#61/#79, docstring
already fixed in batch 1 — re-enabling sharding stays optional).

**The problem (finding #0).** `modal_test.py:19` builds containers from
`IMAGE`, which does not include the `doom_sandbox` sibling
(`modal_image.py` `add_local_python_source` lists `torchwright`,
`torchwright_doom`, `tests`, `scripts`, `modal_image` — no
`doom_sandbox`). And the tests' `_umbrella()` = `parents[3]` resolves
to `/` inside the container. So every test guarded by
`pytest.skip("doom_sandbox sibling not present (standalone checkout)")`
skips on the project's only sanctioned full-suite entry point. The
latest log: `238 passed, 23 skipped` — the 23 are exactly the
oracle/bridge gates:

```
test_traversal_oracle, test_bbox_oracle, test_projection_oracle,
test_wall_column_oracle, test_wall_pixel_oracle,
test_flat_pixel_oracle (×5), test_flat_span_equivalence,
test_octant_angle (e1m1), test_schema_sync, test_prompt_equivalence,
test_w_embed cross-check, test_tokens_bridge vocab-mirror/round-trip
```

`test_forward_compiles` explicitly defers math validation to these
gates ("validated separately by the reference_eval oracle gates") — and
those gates never run.

**Fix.**

1. **Get `doom_sandbox` into the test image.** `modal_image.py`'s
   `ASSETS_IMAGE` already adds it (mounts at `/root/doom_sandbox`) —
   reuse that `add_local_dir`/`add_local_python_source` for the test
   `IMAGE`, or build the test image from it. *First investigate how the
   tests import it* (lazy `import doom_sandbox...`) and what makes it
   importable in-container: it needs to be on `sys.path` or installed,
   not just present on disk.
2. **Make umbrella/sandbox discovery container-aware.** The skip logic
   keys off `_umbrella()` finding the sibling. Inside the container
   that path is wrong. Options: a `TWDOOM_UMBRELLA` env var set in the
   Modal test run, or probe `/root/doom_sandbox`. The skip is correct
   for a genuine standalone checkout; it should fire there and *not*
   under `make test`.
3. **Add a loud skip-guard.** A suite-level test that FAILS (not skips)
   when the oracle gates are skipped in an environment where they
   should run — gated on an env marker set in the Modal test run. This
   is what prevents a silent regression back to today's state. Without
   it, the gates can quietly fall out of CI again on any image refactor.

**Verification.** Run `make test` and confirm the 23 gates now execute
and pass (they presumably pass locally where `doom_sandbox` is present —
verify they pass under Modal too, where the reference renderer now runs
in-container). Confirm the loud guard fails when you deliberately break
sandbox discovery.

**Risk.** Does not change render behavior. Makes `make test` heavier
(the sandbox reference renderer runs in-container) and longer. The real
unknown is image plumbing — `doom_sandbox` import resolution and the
1.26.0-pinned onnxruntime interaction. Budget Modal iteration.

**Alternative (only if local-only oracle runs are the intended
policy).** Document that explicitly in CLAUDE.md's Testing section and
keep just the loud guard. The audit's position — and mine — is that
running them is the right fix: the gates exist to be the CI safety net
for the write side.

---

## Batch 4 — consolidation *(graph-touching; needs compile gates)*

**Goal.** Collapse the duplication and single-source the constants the
audit found, and add the one missing runtime assert. Split into
sub-items by risk; do them as separate certified changes, not one
sweep.

### 4a — oracle-test scaffolding into a shared conftest  *(non-graph)*

Pairs naturally with batch 3 (both touch the oracle test harness) — can
fold together. Absorb the copy-paste the audit catalogued:

- `_umbrella()` defined ×10 (#10); the NUMBA-disable + `sys.path` +
  "doom_sandbox sibling not present" skip preamble ×9 (#37); the
  Assert-silencing monkeypatch (`Assert._check = lambda self, x: None`)
  ×12 — 5 tests + 7 scripts (#38, #80).
- `_carrier_delta` copied into 5 oracle tests with **hardcoded vocab
  constants** (`_VALUE_ROWS = 65536`, wrap constants) — the exact
  hardcoding the canonical `inference/diagnostic.py` version warns
  against (#36). Replace the copies with an import of the canonical
  helper.

Target: a `tests/scene/conftest.py` (and a small shared scripts helper
for the monkeypatch). **Non-graph** — `make test` certifies it.

### 4b — windowed-cache read-distance assert  *(additive; no graph change)*

Finding #66, and **more valuable now**: the tier-1 expiry commit grew
the expiring set from 1 type (pixel) to 10, with a 619-position
certified read scope. CLAUDE.md documents the invariant ("no attention
read may target an expiring-type row at long range") but nothing
asserts it; the cache only fails loud at full saturation.

- Record each slot's last-write committed position (one int list on
  `WindowedState`, updated in the write/alloc paths in `kv_cache.py`).
- In `alloc_slot`, on recycle, assert `current_position -
  slot_write_position` exceeds the certified resident read offset for
  the expiring set. Make the bound a named constant, not the bare `3`
  that was correct only for pixel-only.

Runtime/host code, not graph construction — `make test`'s windowed-cache
unit tests (`tests/inference/test_windowed_cache.py`) plus a new case
certify it. No production recompile needed.

### 4c — graph-side constant single-sourcing  *(graph-touching, lower risk)*

Same *values*, one definition — but it's graph code, so the node-id
rule applies and each needs the spec/pure gate. Lowest-risk graph
sub-item; do it before the math dedups.

- `ANGLE_BAM` (= 8192) and `_TAN_FOV_HALF` defined in both `vocab.py`
  and `render_ops.py`; `render_ops` already imports from `vocab` (#73,
  #90).
- `_PROJ_RATIO` duplicated in `value_ranges.py:53` and
  `render_ops.py:433` with load-bearing range coupling (#74).
- screen-column radix base/buckets + `_radix_col_key` in
  `solid_intervals` / `visplane_state` / `wall_column_state` under two
  naming schemes (`_RADIX_BASE`/`_N_BUCKETS` vs
  `_CLIP_RADIX_BASE`/`_CLIP_N_BUCKETS`) (#34, #75).

### 4d — load-bearing graph-math dedup  *(graph-touching, highest risk)*

Identical math, but it's the rasterizer's load-bearing numerics — do
last, one at a time, each certified token-identical at production
scale.

- `rw_distance` (perpendicular view-to-seg distance) built twice:
  `wall_range_builder.rw_distance` and inline in
  `wall_range_state.SegLevelFacts.publish` (#35).
- The three-head radix-successor publish/scan scaffolding duplicated
  between `solid_intervals` and `visplane_state` (#33).
- `decode.py` cursor-walk: `decode_rows_to_pixels` and
  `decode_xy_by_position` share an identical ~20-line cursor state
  machine (#45, #70). NB the teacher-forced diagnostic uses
  `decode_xy_by_position`, so the two must not drift — this is the
  argument *for* sharing. This one is host-side decode, not graph
  construction, so it's actually lower risk than the two above; group
  it with 4c if preferred.
- `scene_facts._node_value_lookup` / `_seg_value_lookup` byte-identical
  bodies (#40).

**Risk framing for batch 4.** 4a/4b are safe (non-graph / additive).
4c is same-value refactor — low risk but still needs the production
gate per the node-id rule. 4d touches the math the renderer's
correctness rests on — highest risk; certify each independently and be
ready to revert a single sub-item without unwinding the others.

---

## Folded, deferred, or won't-do

- **`config.py` hand-rolled YAML parser (#44):** the verifier found
  it's deliberate dependency-avoidance (pyyaml isn't in the lockfile)
  and block-lists fail loud, so the only debt is that the decision is
  undocumented. One comment stating "no-pyyaml on purpose; accepted
  grammar is …". Fold into batch 2 (config work).
- **Abandoned `plan_i` worktree + 5 fully-merged local branches
  (#64):** housekeeping — `git worktree remove` + branch deletes.
  Trivial, standalone, do anytime.
- **Re-enabling `modal_test.py` sharding (#50/#61/#79):** the stale
  docstrings were fixed in batch 1. Actually splitting the suite into
  per-container shards is a behavior/cost decision, not debt — deferred
  until the suite is slow enough to need it.
- **`cli.run_config` is ~250 lines / 18 kwargs mirrored in
  `modal_render.py` (#72):** batch 2 removes the defaults half of the
  mirror; the structural refactor (extracting the seven jobs
  run_config does) is a larger change with no correctness payoff —
  deferred unless it gets in the way.
- **Refuted findings (13):** mostly verifiers correctly rejecting
  "parallel-but-different sibling modules" as debt (that's the design)
  and overstated severity. No action — recorded in the audit file.

---

## Suggested order

1. **Loose end** — happens on the next `make run`; do it before relying
   on batch 1 in production.
2. **Batch 3** — the one HIGH; restores the CI safety net that every
   later graph change (batch 4) will lean on. Doing it first means
   batch 4's graph edits land against gates that actually run.
3. **Batch 2** — well-scoped, design agreed; fixes the direct-invocation
   traps.
4. **Batch 4** — last, in sub-item risk order (4a → 4b → 4c → 4d),
   each certified independently.
