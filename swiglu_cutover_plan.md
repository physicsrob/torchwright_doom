# SwiGLU flagship cutover — execution plan (swiglu Phase D)

The flagship (this repo's DOOM graph) moves from torchwright's ReLU op
machine to the SwiGLU machine, and the compiled artifact flips to
`bias=False` (biases folded into weight matrices against the pinned
constant-1 column). Throughout this doc, a **machine** is which of the
two op libraries — and hence which activation family, all-ReLU or
all-swish — a graph is built from: `torchwright.ops.relu` and
`torchwright.ops.swiglu` mirror each other module-for-module, the
import path *is* the machine choice, and a compiled graph is uniformly
one machine (mixed graphs are a compile error). "Machine-neutral"
below means code belonging to neither library (the shared linear and
attention ops at `ops/` top level).

This is Phase D of torchwright's `docs/swiglu_step2_plan.md`; the
op-by-op design spec is torchwright's `docs/ops_plain_english.md`.
Drafted 2026-07-04 from a full call-site inventory of this repo at
torchwright bb3af2e (scale=128, Phi-3 converter landed).

**Status: in execution (2026-07-04).** Torchwright Phases A–C are
complete. **D0 landed torchwright-side** (punch list
`torchwright/docs/swiglu_d0_handoff.md`; relocation `eefb34f`,
obligation resolution + lane pins `6bf8af9`, umbrella bumped to
`c6f097a`). D0b resolved beyond the recommended option: the obligation
was retracted as a category error — the noise harness's numbers
reflect *neither* bias mode (its "compiled" leg is `node.compute()`,
where a bias is an exact addition), so no noise re-measurement gates
D3. **D1 landed** (this repo): the ORT-CUDA probe imports the machine
constants and pins the no-bias lane. **D2 landed** (`4f83840` + the
emit hi-byte snap `28bd1f8`), all fast gates green; three cutover
finds fixed en route (torchwright `c5db214` broadcast_select
both-zero collapse, torchwright `2fb6bd6` floor_int residual range
pins — which also un-wedged the d=4096 compile gate, strict xfail
removed on XPASS — and the doom digit-quad hi-snap). **D3 landed.**

**D3 gate results (2026-07-04).** Full doom suite green over the D3
tree. Debug gate (`scripts/d3_debug_gate.py`, on a /tmp-workflow
80×50 `bias=False` variant — the production artifact's 3.3 GB
`embed_table` exceeds ORT's 2 GB embedded-initializer limit):
residual self-consistency PASS, `probe_compiled` vs the exact-math
oracle PASS (both bias modes, atol=500, no divergent node).
**Lowres walkthrough CERTIFIED**: 100.0% coverage, 100.0%
within-option color, 91.6% exact (17,336 tokens to terminal DONE).

**Two open items, neither a cutover numerics failure:**

1. *Debug-surface anomaly (torchwright follow-up).* Under
   `OnnxDebugSession.step()` chunked feeding, a `bias=False` artifact
   reads the BOS global-position recovery at its PWL plateau (148352
   at every depth), while the identical positions are oracle-clean
   under the probe's single-pass `build_prefill` feed on the SAME
   artifact, and `bias=True` is clean under both feeds. The
   production HF runtime is unaffected (the certified lowres
   walkthrough exercises recency over 20k+ live positions). Isolated
   to the step()-feed × bias=False interaction; root cause not yet
   pinned.

2. *Production 320×200 certification BLOCKED on KV memory.* The
   stock-Phi3 conversion pads per-layer trimmed heads back to uniform
   (`num_key_value_heads = max over layers` = 64 of 64 — one layer
   uses every slot, so the ONNX head pruning's 2181/3136 savings
   vanish from the cache). At 49 layers × 64 heads × fp32 that is
   ~3.3 MB/position; the full frame (~57.4k positions) needs ~188 GB
   of KV + 33 GB weights ≈ 221 GB — over a B200's 178 GiB even
   unfragmented (observed OOM at ~15.6k positions with DynamicCache
   torch.cat fragmentation ~2.8×). The swiglu compile itself IMPROVED
   (49 layers vs the relu 51; CP-SAT schedule-cache hit). Options
   parked for Rob: multi-GPU cert run (device_map sharding), busiest-
   layer head packing (drops the uniform pad), windowed-cache re-add
   (the recorded big-ticket item), or accept lowres certification for
   the cutover and treat full-res as the pre-existing ceiling item.

**Execution deviation — the walkthrough stages merged.** Settled
decision 1's "certification runs twice (2 configs × 2 stages)" turned
out impossible at the D2 stage: torchwright's HF conversion routes
swish artifacts exclusively to the stock `Phi3ForCausalLM` target,
which requires `bias=False` (`compiler/hf/convert.py`
`_conversion_target`, deliberate per `docs/phi3_conversion_plan.md`
P1) — a `bias=True` swish artifact has no production runtime path.
The machine flip is instead certified at `bias=True` by the
non-runtime gates (both suites, both oracles, the compiled AR
rollout, the d=4096 gate — none route through HF conversion), and
the walkthroughs certify once, at swish + `bias=False`, both
configs. The A/B fallback if a walkthrough regresses:
`OnnxDebugSession` + `probe_compiled` on a /tmp `bias: true` config
variant isolates machine-vs-fold without HF conversion.

**What this unblocks:** Phi-3 conversion P3 (torchwright
`docs/phi3_conversion_plan.md`) — exporting e1m1 as a stock
`Phi3ForCausalLM` checkpoint requires the swish + `bias=False`
artifact this plan produces.

## The two flips and how each one threads (ground truth)

The cutover is two independent flags with completely different
plumbing:

1. **The op machine (relu → swiglu) is chosen by import path.**
   There is no `activation` kwarg on `compile_to_onnx` — the compiler
   selects the physical MLP kind from the FFN nodes the graph
   contains. Flipping the machine means rewriting the
   `torchwright.ops.relu.*` imports in **11 package files + 1 test
   file** (inventory below; the set was verified exhaustive by
   grepping all three import forms — `ops.relu` submodule imports,
   `from torchwright.ops import ...`, `import torchwright.ops` — the
   latter two have zero hits in this repo). The compile cache stays
   correct through the git-SHA component of the cache key
   (`config.py:302-305` keys on both repos' HEADs, each extended with
   a working-tree digest of `git diff HEAD` + `git status
   --porcelain`, so even uncommitted edits re-key) — no config change
   is needed or wanted for this flip.

2. **`bias=False` is a compiler kwarg** (`compile_to_onnx(...,
   bias=...)`, torchwright `export.py:1354`) that doom never passes
   today. It must become a **new `ModelConfig` field** rather than a
   hard-coded default inside `compile_to_onnx_path`. The reason is
   not the one-time flip — any committed source edit re-keys the
   cache via the git-SHA component. The reason is call-time
   variation: a hard-coded signature default can be overridden by any
   caller at runtime (a script argument, an A/B experiment) with no
   source diff, and then two different artifacts map to the same
   cache key and the stale one is served. The existing
   `rms_norm_const_exp=63` hard-code (`compiled_model.py:84`)
   survives only because no caller ever varies it. A config field
   makes the knob key-visible (the key hashes `asdict(config.model)`,
   `config.py:329`) and expressible in the /tmp-config-variant
   workflow (experiments copy a config to /tmp and edit a field —
   the CLAUDE.md rule against third committed configs) — which the
   two-stage landing below relies on for its bias=True-vs-False A/B.

## Settled decisions (proposed; confirm before D2)

1. **Two flips, landed separately: machine first (at `bias=True`),
   then `bias=False`.** The flags are orthogonal by construction
   (either machine compiles with or without biases), and landing them
   separately preserves the A/B diagnostic at each step — "is this
   divergence the machine or the bias fold?" is one config edit or
   one checkout away. Cost: the full walkthrough certification runs
   twice (~30 min/frame × 2 configs × 2 stages on Modal).

2. **The pre-pick clamps stay in place at the flip; removal is a
   separate audited follow-up (D4).** These clamps exist to bound the
   relu `broadcast_select`'s additive offset `M` (defined in the
   tolerance-audit section below); the swiglu op has no `M`, so they
   lose their stated reason. But they are idempotent, cheap relative
   to the graph, and removing them changes value-range bookkeeping —
   keep the flip commits minimal and audit removal separately. This
   deferral is the disposition torchwright's Phase D handoff itself
   records ("dead pre-clamp choreography ... keep or remove on their
   own merits" — an optional post-cutover opportunity, not flip
   work). The slot clamps inside `make_token` / `make_token_head` are
   **not** part of this: they bound garbage rows for the digit-quad
   payload, a reason that survives the cutover (`std.py:335-351`).

3. **The doom-side kernel pins land first and import their constants
   from torchwright** (`from torchwright.ops.const import scale,
   bias_lane_gate, bias_lane_up`), so they can never go stale the way
   the current hard-coded `SCALE = 100.0` did (D1 below).

4. **Acceptance is behavioral, not byte-level.** Renders are not
   expected byte-identical to the relu baseline — the bit-exactness
   profile changes in both directions (saturated indicator chains
   become bit-exact at scale=128; grid-based products and pick
   emissions move to ~1-2 ulp relative). The gates in "Acceptance"
   below are the criterion.

## Migration surface

Files importing `torchwright.ops.relu.*`, with what each flip needs:

| File | Imports | Work |
|---|---|---|
| `std.py:42-53` | `cond_gate, bool_all_true, bool_any_true` (logic_ops); `in_range, broadcast_select, dynamic_extract, select, switch, table_lookup_2d` (map_select); `clamp, piecewise_linear` (arithmetic_ops) | Import flip + **drop `approximate=False`** at `std.py:269` (kwarg deleted; see "pick_by_one_hot" below) + rewrite the comment at `std.py:263-267` |
| `render_ops.py:40-51` | `ceil_int, clamp, compare, floor_int, mod_const, multiply_2d, piecewise_linear, thermometer_floor_div` (arithmetic_ops); `linear_relu_linear`; `bool_all_true, bool_any_true, bool_not` (logic_ops) | Import flip + **two real rewrites** (below): `multiply_2d` → `multiply` (4 sites), `_ray_count` re-authored on `swiglu_ffn` |
| `past.py:22` | `attend_most_recent_globally, global_position_from_bos` (global_recency) | `global_position_from_bos` → swiglu (identical signature); `attend_most_recent_globally` → its new machine-neutral home (D0a) |
| `solid_intervals.py:28-32` | `compare, mod_const, thermometer_floor_div` | Import flip only (identical signatures) |
| `visplane_state.py:43-48` | `compare, mod_const, piecewise_linear, thermometer_floor_div` | Import flip only |
| `seg_projection.py:57` | `clamp` | Import flip only |
| `flat_state.py:38`, `statusbar_renderer.py:53`, `psprite_renderer.py:45` | `compare` | Import flip only |
| `extract.py:50-52` | `compare`; `cond_gate` | Import flip only |
| `emit.py:91` | `floor_int` | Import flip only |
| `tests/scene/test_radix_digit_extraction.py:32` | `mod_const, thermometer_floor_div` | Import flip only |

The signature claim behind every "import flip only" verdict was
checked mechanically (ast-level diff of positional args, defaults,
and keyword-only args) for all 18 ops doom imports: `compare, clamp,
piecewise_linear, mod_const, thermometer_floor_div, floor_int,
ceil_int, in_range, select, switch, dynamic_extract, table_lookup_2d,
cond_gate, bool_all_true, bool_any_true, bool_not,
global_position_from_bos, broadcast_select`. 16 of 18 are
byte-identical; the only two diffs — both handled above — are
`broadcast_select` (loses `approximate`) and
`piecewise_linear`, whose `d_max` default changed from the literal
`1024` to the symbol `min_d_hidden` — currently equal to 1024, so
caller-invisible. The relu-only ops doom does **not** use (verified
by grep): `relu_add`, `multiply_integers`, `per_column_offsets` /
`scalar_M`, `square`'s `max_value`/`step`, `output_sequence`. No doom
code passes `c_tol`, `assert_bool`, `assert_01`, or
`assert_score_gap_at_least` explicitly — every tolerance coupling is
inside the torchwright ops or in comment-level sizing arguments
(audit list below).

### The two real rewrites in `render_ops.py`

**`multiply_2d` → `multiply` (exactly 4 call sites, verified by
grep: `mul_side` :171, `MUL_SCREEN` :328, `MUL_CROSS` :446,
`_mul_grid`/`mul_normal_coord` :497).** The swiglu `multiply(a, b)`
computes `Swish(a)·b + Swish(−a)·(−b)`, which cancels the sigmoid
factors algebraically — no grid, no range limit, no extrapolation
behavior. Every grid parameter (`max_abs*`, `step*`, `breakpoints*`)
disappears, and `_mul_grid` itself dies. Its measured noise (op
docstring footer) is 2.24e-7 relative — about 2 ulp at the product's
magnitude. Per site, against the error each consumer was sized for:

- `mul_side`: |product| ≤ 512·2400 ≈ 1.2e6 → new error ≈ 0.28 abs,
  vs the ~step₁·step₂/4 = 75 grid noise the sign test tolerates
  today. Sign-only consumer; ~270× better.
- `MUL_CROSS`: ≤ 1.5e5 → ≈ 0.03 vs 37.5; sign-only consumer.
- `mul_normal_coord`: ≤ 1200 → ≈ 3e-4 vs ~1e-3 abs on the 257-bp
  grid, and the extrapolation trap is gone.
- `MUL_SCREEN` is the one site that *regresses* in kind while staying
  far inside margin: the relu grid was exactly 0 at integer grid
  points, while the swiglu ± pair leaves both lanes unsaturated for
  |a| < 17, so integer screen columns now carry ~2 ulp (≤ ~1e-3 abs
  at products ≤ 4356). Its consumers compare against half-integer
  thresholds with the default `compare` deadband of 0.1
  (`render_ops.py:375-378`) — a ≥100× margin.

The extrapolation-trap commentary (`render_ops.py:157-167, 320-325,
482-493`) describes deleted behavior — remove it with the call-site
changes.

**`_ray_count` re-authoring (`linear_relu_linear` at :286 and
:297).** No `linear_relu_linear` exists in swiglu; the builder is
`swiglu_ffn(input_node, gate_proj, gate_bias, output_proj,
output_bias, *, up_proj=None, up_bias=None)`, and per the spec's
convention the author folds `scale` into gate rows and `1/scale`
into the output projection explicitly (no helper does it for you).
The construction ports hinge-for-hinge — the swish hinge
`Swish(scale·z)/scale` equals `relu(z)` exactly once `|scale·z| ≥ 17`
(the fp32 sigmoid saturation point; 18 on the CPU-onnxruntime kernel
only), so:

- Stage 1, lane i: gate row `scale·s·v_i` (s = `_RAY_SHARPNESS` =
  32000), degenerate up lane, out `1/scale` → `a_i = hinge(s·v_i)`.
  Exact 0/identity outside `|v| ≥ 17/(scale·s) ≈ 4.2e-6`; the
  smallest nonzero fixture ray (~1.5e-4, `render_ops.py:277`) clears
  by ~36× — a wider margin than the relu ramp gave.
- Stage 2, lane i: gate row `scale·(1 − a_i)`, out `−1/scale` summed
  into `count = n − Σ hinge(1 − a_i)`. The cancellation-free property
  that motivated the original min-form carries over: for a saturated
  ray `a_i` is huge, the gate argument is hugely negative, and the
  hinge is exactly 0 — no subtraction of two large near-equal
  numbers anywhere. `hinge(1) = Swish(128)/128 = 1.0` exactly
  (σ(128) = 1.0 on every deployed kernel; ·128 and /128 are exact),
  so each step is exactly 1 or 0 and the count — a sum of ≤1024
  exact 0/1 terms, an integer far below 2²⁴ — is exact in fp32.

Every number in the two bullets above is a derivation anchored on the
pinned kernel constants, **not yet a measurement** — the D6
obligation is a unit test at this layer (a `_ray_count`-level fixture
pinning exact integer counts at the fixture's extreme ray magnitudes,
including the smallest nonzero ray) landing in the same commit as the
rewrite, and that test is what converts the derivation into evidence.

### `pick_by_one_hot` and the emission-equality weakening

`std.pick_by_one_hot` (23 call sites, counted by grep: 10 in
visplane_state, 3 each in solid_intervals and render_main, 2 each in
assets, flat_state, wall_range_builder, 1 in uv_compute) lowers to
`broadcast_select`, today with `approximate=False` — the relu-era
exact mode. The swiglu op has one form and no flag; the properties
doom relied on map as follows:

- **Junk masks stay safe.** The renderer builds every branch's pick
  eagerly and discards rows whose masks are fractional. The swiglu op
  makes this a documented contract (no ±1 assert; a fractional mask
  blends within the hull of zero and the branch ranges) — and doom's
  zero-literal false branch satisfies the contract's one condition
  (zero must lie in the branch-range union) by construction. Bonus:
  a literal-zero false branch drops its lanes at build time, so the
  Phase D "lane halving" opportunity is automatic, not extra work.
- **The winning row weakens from byte-identical to ~1e-7 relative.**
  `_collapse_scalar_emits` (`render_main.py:514`) claims byte-
  identical head emission in its docstring; after the flip the picked
  carrier is equal within ~1 ulp relative instead. (A **carrier** is
  a token bearing a numeric slot value — `VALUE` / `ANGLE_VALUE` —
  decoded through the value encoding; a **marker** is a protocol
  token whose identity alone is the payload.) The sign-off argument
  (spec, broadcast_select entry): picked scalars feed
  clamp-and-quantize decode paths whose inputs already carry
  ~1e-3-class recovered-state noise, so ~1e-7·|value| is invisible.
  **Sign-off is not an argument, it's a gate:** the flat-pixel oracle
  holds carriers to `_VALUE_ENC_TOL = 5e-3` / ±2 BAM / ±1 wallColU
  and markers to exact rows — those staying green *is* the sign-off.
  Update the `_collapse_scalar_emits` docstring in the flip commit.

### Tolerance/budget audit (D2 checklist item)

The swiglu machine changes what a mask/cond tolerance *means*. The
relu select-family ops cannot multiply a gate by a value directly;
they gate by adding a large constant first — forms like
`0.5·[ReLU(offset·mask + v) − ReLU(offset·mask − v)]`, where the
offset `M` must exceed the largest absolute value the branches can
take (it is derived from the union of the branches' declared value
ranges — hence the finite-range requirement, and hence the
caller-side clamps that exist to keep `M` small). A mask off by δ
leaks error proportional to `M`, so relu-era budgets were `δ·M`
absolute widenings. The swiglu ops multiply gate×value directly:
`M` and everything downstream of it is deleted, and a δ mask error
costs `δ·|actual value|`. The mask-quality contract on select-like
ops is `_MASK_TOL = 4·swish_dip/scale ≈ 0.0087`, where `swish_dip =
0.2784645` is the depth of the swish curve's negative dip (a pinned
constant in torchwright `ops/const.py`) — numerically larger than
the relu-era 0.005 cond budget, but landing proportional to the
value, not to `M`. Doom passes no tolerances explicitly, so the
audit is over sizing arguments and assert-consuming heads:

1. `render_constants.py:38-52` — `RECENCY_GAIN = 8` is sized so the
   recency cond ≈ 0.9993 clears the relu-era "cond > 0.995"
   (`c_tol=0.005`) budget. Against the swiglu mask contract the
   threshold moves to ~0.9913 — the sizing *gains* margin. Verify
   which assert actually consumes it post-flip and update the
   comment's budget reference.
2. `std.one_hot` (`std.py:174-185`) — doom's only `in_range` call.
   swiglu `in_range` carries a ±1 slack of `4·swish_dip/scale ≈
   0.0087` in its value-range assert; `bool_to_01` halves deviations.
   Enumerate the one_hot consumers (attention keys, pick masks) and
   confirm each tolerates 0/1 values off by ≤ ~0.004.
3. Mask producers feeding `pick_by_one_hot` — enumerate the
   `indicator_to_bool` sources (recovered one-hot dots, protocol
   masks) and check the winning-row mask sits within `_MASK_TOL` of
   +1; a mask off by δ mis-scales the picked value by exactly
   δ·|value|.
4. Comment sweep: prose references to the dead apparatus
   (`approximate` in `render_main.py:454,532`, `flat_state.py:534`,
   `past.py:61`; the `c_tol` references in `graph_debug.py:21`,
   `std.py:263`) get rewritten in the flip.

## Phases

### D0 — torchwright prerequisites (torchwright repo)

- **D0a — relocate `attend_most_recent_globally` to the
  machine-neutral level.** It is pure attention hardware — verified
  by reading its full body (`relu/global_recency.py:185-383`): it
  builds an `Attn` / `rotary_content_head` from graph-core and
  machine-neutral pieces only, and the module's one machine-op import
  (`piecewise_linear`) is consumed solely by
  `global_position_from_bos`, which stays. It was stranded in
  `ops/relu/` only because it shares that module. It has no entry in
  `op_noise_data.json` (not a measured piecewise op), so the move is
  numerically inert and the frozen relu baseline is untouched. Move
  it (and `_RECENCY_SCALE`) to `ops/attention_ops.py`, matching the
  2026-07-03 machine-neutral relocation; update all callers
  (torchwright `tests/ops/test_global_recency.py`,
  `tests/compile/forward/test_rope_global_recency.py`, doom `past.py`
  at D2) — no aliases, no re-exports.
- **D0b — resolve the noise-under-`bias=False` obligation. D3 is
  blocked on this resolution.** `numerical_noise_findings.md`
  records: before the flagship flips `bias=False`, re-run `make
  measure-noise` under the flag. **The drafting inventory found that
  knob does not exist**: `measure_op_isolated`
  (torchwright/debug/noise.py:88-151) evaluates graphs in-process via
  `node.compute()` and contains no compile call (verified by reading
  the measurement body, not just a failed grep), while the bias fold
  is an export-time transform (`export.py` emit_bias) — the flag is
  structurally invisible to the pipeline as built. Two options,
  Rob's call:
  - *(recommended — amend the obligation)* The fold's end-to-end cost
    is already measured (`tests/debug/test_no_bias_onnx.py`: logits
    move ≤ ~4e-4 abs at ~700 magnitude, worst ~8e-5 relative on small
    cancelling logits — same error class, shifted accumulation
    order); doom holds no per-op budgets that folded per-op numbers
    could re-derive; and D3's gates (debug=True asserts + oracle +
    walkthrough on the actual `bias=False` artifact) test the real
    question directly. Record the amendment in
    `numerical_noise_findings.md` when D3 lands. If a D3 gate then
    surfaces a divergence the end-to-end bound doesn't explain, stop
    (foundation rule) and fall back to the second option before
    proceeding.
  - *(heavyweight — honor the letter)* Extend the harness to
    optionally measure through `compile_headless(..., bias=False)`.
    Under this option the harness extension lands and runs **before**
    D3, and D3's gate list gains a review of the folded per-op
    numbers plus re-derivation of any budget they move.

### D1 — kernel/constant pins (doom repo, lands before the flip)

`tests/inference/test_ort_cuda_saturation.py` is the ORT-CUDA member
of the kernel-pin family (torch-CUDA and ORT-CPU live torchwright-side
in `tests/docs/`). It is **stale**: it pins `SCALE = 100.0`
(`:44`) against the machine constant that moved to 128 on 2026-07-04.
One commit:

- Import `scale`, `bias_lane_gate` (32.0), `bias_lane_up` (0.03125)
  from `torchwright.ops.const` instead of local literals.
- Mirror torchwright's upgraded probe points (Swish(128)/±64 classes;
  saturation from 17 unchanged).
- Add the no-bias constant-lane pin (the follow-up
  `no_bias_plan.md:270-273` assigns to this file): `σ(32) == 1.0`
  bit-exact and `32 · σ(32) · (1/32) == 1.0` with no rounding, under
  the deployed ORT-CUDA kernel — the arithmetic every folded bias
  rides in the `bias=False` artifact.

### D2 — machine flip (doom repo, `bias=True`)

**The flip lands atomically** — one commit (or a short local stacked
series merged as one unit). There is no incremental landing path: a
partially-flipped tree assembles a graph containing both machines'
FFN nodes, and the compiler's uniformity check rejects mixed graphs
with a `ValueError`, so every full-graph compiled test is broken
between a first and last flip commit. The atomic unit contains all of:

1. Mechanical import flips — 8 package files (solid_intervals,
   visplane_state, seg_projection, flat_state, statusbar_renderer,
   psprite_renderer, extract, emit) + the test file
   (tests/scene/test_radix_digit_extraction.py). With std.py,
   render_ops.py, and past.py below, that reconciles to the full
   11 + 1 surface.
2. `std.py`: import flip + drop `approximate=False` + comment
   rewrite.
3. `render_ops.py`: import flip; `multiply_2d` → `multiply` (delete
   `_mul_grid` and the grid/extrapolation commentary); `_ray_count`
   on `swiglu_ffn` + its D6 ray-count unit test.
4. `past.py`: swiglu `global_position_from_bos` + relocated
   `attend_most_recent_globally`.
5. Docstring/comment sweep: `_collapse_scalar_emits`' byte-identical
   claim, the audit list's item 4.

The tolerance/budget audit (previous section) runs during
development and its findings land in the same unit or as an
immediate follow-up commit. Development cadence: while the series is
being built, per-commit targeted tests are module-level (subgraph
fixtures that are uniformly swiglu) plus exact-math oracle checks —
`reference_eval` runs `node.compute` per node and never invokes the
compile-time uniformity check, so it works on a partially-flipped
tree. All compiled gates run at the atomic boundary, which is also
the tiered cadence's full-suite batch boundary.

Gates at the boundary, in order:

- `make test` (doom) and torchwright's `make test` both green.
- `tests/scene/test_flat_pixel_oracle.py` — exercises the swiglu
  exact math end-to-end via `reference_eval`: markers (protocol
  tokens whose identity is the payload) decode to exact rows;
  carriers (numeric slot values) hold to `_VALUE_ENC_TOL = 5e-3`,
  ±2 BAM on angles, ±1 on wallColU.
- `tests/scene/test_forward_ar_rollout.py` — compiled-vs-reference
  free-run, token-exact, on `compile_headless` (first compiled-swish
  doom graph).
- `make run COMPARE=1` at `configs/e1m1_lowres.yaml`, then
  `configs/e1m1.yaml` — scored against pydoom. Baseline to hold, from
  the certified numbers recorded in `configs/e1m1.yaml:10-16`: 100%
  pixel coverage, ~99.99% within-option color at 320×200. The
  exact-color rate (96.5% there) may move either way; coverage or
  within-option regressions are stop-and-investigate
  (`probe_compiled` first).

Expected side effects, not failures: compiled layer count moves off
51 (the count in the config header is a compile output; `multiply`
lanes replace grid banks), and the compile cache re-keys via the git
SHA.

### D3 — `bias=False` flip (doom repo; blocked on D0b's resolution)

1. Add `bias: bool = True` to `ModelConfig` (`inference/config.py`),
   threaded `compile_cached` → `compile_to_onnx_path` →
   `compile_to_onnx`. It rides the cache key via
   `asdict(config.model)` automatically — and busts every cached
   artifact once, like the config comments note for past fields.
2. Flip `bias: false` in `configs/e1m1.yaml` **and**
   `configs/e1m1_lowres.yaml` in lockstep (the two-config rule).
3. Gates: same sequence as D2 (full suites, both oracles, both
   walkthroughs). Run one `debug=True` step + `probe_compiled` via
   the artifact's `OnnxDebugSession` as the direct check on the
   folded arithmetic. Under D0b's heavyweight option, additionally
   review the folded per-op noise numbers before this phase starts
   (see D0b).

### D4 — cleanup and opportunities (post-green, each optional)

- **Pre-pick clamp audit.** Decide `clamp_to_slot`
  (`render_main.py:566`) and the ±3072 dispatch clamps
  (`render_main.py:489,492`) on their own merits now that `M` is
  gone. The code comment at `render_main.py:482-487` states both the
  M-bounding purpose and that `signed_world_angle` re-clamps to the
  same ±3072 internally via `_abs_coord` — so the dispatch clamps may
  be fully redundant; verify value-range bookkeeping before deleting.
- **`table_lookup_2d` axis rebalance.** The swiglu op costs 3 lanes
  per boundary on *both* axes (`3(A+B)`), so balanced axes minimize
  lanes: the spec's example flat bank goes ~6.5k lanes at 2048×128 to
  ~3.1k at 512×512, paying ~100 lanes of divide-and-remainder index
  arithmetic and ~2 sublayers of depth. Doom's four lookup sites
  (`assets.py:140,185,206,232`) get a per-table Pareto (wall banks /
  flats / weapon / HUD have different shapes via `asset_banks.py`).
  Lane budget lives in these big-N constructions, so this is where
  d_hidden headroom is won or lost — but it is an optimization, not
  cutover-blocking.
- **Doom-side script sweep.** `scripts/audit_relu.py` imports the
  graph-level `torchwright.graph.relu.ReLU` (not `ops.relu`, so the
  flip doesn't break it) but builds the full doom graph via
  `render_main.forward` — post-flip it audits a swiglu graph under a
  relu-era name and purpose. Rename, repurpose, or delete it here,
  plus anything else relu-named the flip obsoletes.

## Risks / parked

- **CPU-onnxruntime saturates later than the other kernels.** On
  torch-CPU/CUDA and ORT-CUDA, fp32 sigmoid is exactly 1.0 from input
  17; on the CPU-onnxruntime kernel only from 18 (it can sit ~1.8e-7
  below 1.0 on [17, 18)). Production inference is the HF torch path
  and the walkthrough runs on GPU, so this only bites exact-equality
  checks against the CPU-ORT `load_onnx` oracle. If such a test
  lands, hinge arguments need ≥ 18 past the bend, not 17.
- **Screen-env import-order trap.** Any ad-hoc verification of the
  new artifact must set the screen env (via `apply_screen_env` /
  `run_config`) before importing graph modules, or it silently builds
  the 60×50 hud-off graph. The debug-session path fails loud
  (`compile_cache.py:150-159`); the production path trusts the cache
  key.
- **`optimize=2` schedule cache across the flip.** The CP-SAT
  scheduler is machine-blind — it never reads the machine choice,
  only lane counts — so a schedule-cache hit across a machine flip
  of an unchanged topology is correct by design. Topology *does*
  change here (multiply lanes, ray-count banks), so expect fresh
  CP-SAT solves, not cache corruption.
- **Walkthrough wall-time**: 4 certified renders (~30 min each on
  Modal) across D2+D3. Lowres-first ordering front-loads failure
  detection.

## Execution record (2026-07-04) — the reference

Everything below happened in one execution day. This section is the
consolidated record; the status block at the top is its summary. The
torchwright-side prerequisite work has its own record in torchwright's
`docs/swiglu_d0_handoff.md` (the D0 punch list, executed by the
torchwright agent).

### Commit trail

Doom (`torchwright_doom`), in landing order:

| SHA | What |
|---|---|
| `97f60bd` | This plan lands (D0 already complete torchwright-side) |
| `a02ed57` | D1 — ORT-CUDA kernel pins import the machine constants; no-bias lane pin (5/5 on the deployed A100 + onnxruntime-gpu pair) |
| `28bd1f8` | Emit hardening — digit-quad high byte snapped to an integer (find #3 below) |
| `4f83840` | **D2 — the atomic machine flip** (11 files + 1 test; `multiply_2d`→`multiply`, `_ray_count` on `swiglu_ffn` + `tests/scene/test_ray_count.py`, comment/audit sweep, d=4096 xfail removed on XPASS) |
| `aa0b13a` | **D3 — `bias=False`** (`ModelConfig.bias`, threaded + key-visible; both configs in lockstep) |
| `2838cf2` | `load_debug_session` mirrors compile-side linear fusion (find #4); `scripts/d3_debug_gate.py` |
| `0d9be09` | D3 gate results + the two open items recorded |

Torchwright: `eefb34f` (D0a relocation), `6bf8af9` (D0b obligation
retraction + lane pins), `c6f097a` (freeze note), `c5db214` (find #1),
`2fb6bd6` (find #2). Umbrella pointer bumped after each torchwright
landing.

### What the gates caught (five finds, all root-caused)

1. **`broadcast_select` with BOTH branches all-zero literals built a
   zero-lane FFN** that crashed the affine rules (`torch.tensor([])`
   is float32, `torch.where` rejects it). Doom hits it via
   `assets._snap_index` over a missing-texture bank's zero rows.
   Fixed torchwright-side (`c5db214`): the op collapses to the zero
   literal — exact semantics, plus a dtype hardening. Caught by: the
   first D2 `make test`.
2. **`floor_int`'s residual-resident intermediates carried the affine
   relaxation's ~1e16 declared ranges**, blowing the RMS-norm energy
   certifier past the fp32-feasible q=63 budget. Fixed torchwright-side
   (`2fb6bd6`): both stages pinned to their universal hinge bounds
   (sound for any input) plus the fp32 ulp class W absorbs. Side
   effect: the d=4096 compile gate's strict xfail XPASSed — the flip's
   lane reduction had dissolved the static-schedule deadlock — and the
   marker is removed. Caught by: `test_forward_compiles_to_onnx`.
3. **A carrier value straddling a byte boundary produced a fractional
   digit-quad high byte**, and the low-byte recovery amplified it
   ×BASE (192 emitted rows ≈ 5.9e-3, over the 5e-3 carrier bar).
   Machine-independent, pre-existing; the flip's sub-noise shifts
   re-rolled the dice onto it. Fixed doom-side (`28bd1f8`): the high
   byte is snapped to the nearest integer before the low byte reads
   it, making the documented ±1-step contract actually true (residual
   hazard ~1e-9). Pinned by
   `test_two_digit_boundary_sliver_snaps_to_one_step`. Caught by: the
   flat-pixel oracle (`segDcTmidMid`, pos 7780).
4. **`load_debug_session` rebuilt the graph without the
   `fuse_consecutive_linears` pre-pass** the compile always applies —
   fingerprint mismatch on every post-fusion artifact (656 fused pairs
   at production). Fixed doom-side (`2838cf2`). Caught by: the D3
   debug gate's first run.
5. **A `bias=True` swish artifact has no production runtime path** —
   torchwright's HF conversion routes swish exclusively to stock
   `Phi3ForCausalLM`, which requires the biasless normed emission
   (deliberate, phi3 plan P1). Resolution: the walkthrough stages
   merged (the deviation block above); the machine flip is certified
   at `bias=True` by the non-runtime gates instead.

### Final gate state

- `make test`: green, both repos, over the final tree.
- Flat-pixel oracle + compiled AR rollout (token-exact) + d=4096
  compile gate: green.
- D3 debug gate (80×50 `bias=False` variant; the production artifact's
  3.3 GB `embed_table` exceeds ORT's 2 GB embedded-initializer limit):
  self-consistency PASS, `probe_compiled` vs oracle PASS on BOTH bias
  modes (atol=500, no divergent node).
- **Lowres walkthrough (160×100, swish+`bias=False`): CERTIFIED —
  100.0% coverage, 100.0% within-option, 91.6% exact, 17,336 tokens
  to terminal DONE.**
- Production walkthrough (320×200): compile green (49 layers — better
  than the relu 51), render **blocked by the KV ceiling** (open item
  2 in the status block).
- Layer count: 51 → 49; the config headers' certified-numbers lines
  refresh when the production walkthrough re-certifies.

### Open items (also in the status block)

1. `OnnxDebugSession.step()` × `bias=False` global-position plateau —
   torchwright debug-surface follow-up; production unaffected.
   Instruments: `scripts/d3_debug_gate.py` (`--debug-first-chunk`,
   the global-position dump).
2. Production 320×200 certification blocked on stock-Phi3 uniform KV
   (~221 GB unbounded-cache demand vs 178 GiB B200) — options parked
   in the status block, Rob's call.
3. D4 (optional, unstarted): pre-pick clamp audit, `table_lookup_2d`
   axis rebalance, `scripts/audit_relu.py` disposition.
