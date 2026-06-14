# Depth/width cost investigation — running log

## ★★ DISPATCH/EMIT RESTRUCTURE — design panel + co-residency gate (2026-06-02)
**Design panel (11 agents) converged unanimously**: the one structural win is
**"route-the-scalar-first, then emit ONE shared digit-quad per carrier type"** —
collapse the eager per-branch digit-quads to 2. Code-verified corrections to the
panel's own framing:
- The eager quads are **15 VALUE + 9 ANGLE = ~24** distinct top-level dispatch
  branches (wall_range_builder 11 + bbox_pruning 4 VALUE; seg_scanner/bbox/
  wall_range 9 ANGLE), each keyed on its OWN marker predicate. NOT inside the
  after_value/after_angle_value select-towers (those emit MARKERS only).
- **ANGLE_VALUE is NOT cheap**: ANGLE_VALUE.angle (IntSlot card 8192) shares slot
  position 0 with VALUE.v (FloatSlot levels 65536) → shared_position_cardinality
  =65536 → BOTH emit the full 2-digit 255-wide floor_int. ANGLE collapse saves the
  same per-site as VALUE. So target = 24→2, not 16→1.
- Mechanism = flat **pick_by_one_hot** (std.py:215-246, broadcast_select
  approximate=False) over clamped 1-wide candidate scalars, masked by the branch's
  own registry predicates. _distinct_head_pairs groups by id(head) + OR-s
  predicates → mapping all 15 VALUE keys to one shared head auto-collapses. NOT a
  nested select chain (deep-serial trap). Clamp EACH candidate to its slot range
  BEFORE the fold so the union M into broadcast_select stays ~2 (fp32 sharp-step).

**CO-RESIDENCY GATE (the panel's #1 "measure-first", schedule-only, memory-safe):**
Peak-layer live-set by subsystem, fanout=8:
| subsystem      | d=6400 (COMPILE width) | d=12000 (unconstrained) |
|----------------|------------------------|--------------------------|
| **digit-quad** | **2523 (39% of 6400)** | 2818                     |
| affine glue    | 1180                   | 2235                     |
| geometry PWL   | 1094                   | **4463**                 |
| emit other     | 850                    | 945                      |
| E8 type-code   | 400                    | 456                      |
| select/dispatch| 330                    | 362                      |

### IMPLEMENTED + MEASURED (committed) — output-identical, the win is in SERIALIZABILITY not peak
The collapse landed (ScalarEmit + value_scalar/angle_scalar in emit.py; the 24
VALUE/ANGLE owner `after_*` methods return their 1-wide scalar; render_main
`_collapse_scalar_emits` picks the active scalar by predicate via the float-exact
`pick_by_one_hot` and emits ONE shared digit-quad per carrier). Distinct dispatch
`type_switch` terms: **45 → 23** (15 VALUE branches share one head, 9 ANGLE share
one). All gates green: 4 oracle byte-identity gates, dispatch-dedup, compile_to_onnx,
compiled AR free-run — **byte-identical**.

**The headline prediction (peak drop) did NOT materialize; the real win did.**
Measured (schedule-only, fanout=8), BEFORE → AFTER:
| d (residual) | BEFORE layers | AFTER layers |  note |
|--------------|---------------|--------------|-------|
| 12000 (unconstr.) | 37 | 34 | natural peak 11337 → **11273 (≈unchanged)** |
| 6400 (compile)    | 44 | 42 | both saturated at d |
| 4800              | 56 | 51 | |
| 3600              | 91 | **69** | −24% layers |
| 3000              | **DEADLOCK** | **102 (fits)** | min-d crossed here |
| min compilable d  | **~3500** (ok 3600, deadlock 3400) | **~2800** (ok 3000, deadlock 2600) |

- **Natural (unconstrained) peak is ≈unchanged** (11337→11273). The digit-quad
  bucket DID fall (2818→1869 at d=12000) but the machinery to converge 24 scalars
  at one pick (concat masks + clamps + broadcast_select + sum) raised "affine glue"
  2235→3245 — a near wash on the *parallel* width. So the d=6400 "39% digit-quad"
  co-residency was the scheduler's CHOICE under saturation, not a removable peak.
- **The win is SERIALIZABILITY → lower min-d → quadratic memory.** 24 thin scalars +
  1 quad thread through a narrow residual where 24 wide 255-col quads jammed: min
  compilable d dropped ~3500 → ~2800 (−20%), and layers fall at *every* d (more so
  the tighter d is: 91→69 at d=3600). Since streaming-compile peak RSS and the
  densified-ONNX size both scale ~4·d²·(resident), a lower min-d is a quadratic
  memory cut at the floor — and d≈2800 vs 6400 is ~4.7× less densified weight,
  which is the lever that could make the >26GB local inference (Phase K) feasible.
- Honest: this is NOT the 10–15× the proposers hyped, and NOT a peak-width cut. It
  is an output-identical depth + min-d improvement in exactly the dimensions that
  matter for the memory wall. The fp32 mandate held: each candidate must be clamped
  to its slot range BEFORE the pick (std.clamp_to_slot) or broadcast_select's M
  offset blows the 1e6 sanity bound (a raw angle's value_type is ±2.6M).

PRE-EXISTING (NOT caused by this change; fails identically on HEAD via git-stash):
`tests/embedding/test_emit_near_miss.py` 3 byte-boundary tests (e.g. ANGLE_VALUE
angle=-3840.4 → argmax 65791 vs expected 65792) — an off-by-one in the digit-quad
quantizer at a byte boundary, orthogonal to the dispatch. Flagged for a separate
numerical follow-up.

## ★ SYNTHESIS (validated, overnight 2026-06-02)
1. **Depth was the dispatch fold, not geometry.** `type_switch(max_fanout=2)` builds a
   ~35-layer SERIAL Linear→Concat accumulator. Raising max_fanout: 67→44 (=4), →37 (=8),
   →32 (flat). **True min depth = 32 layers, BELOW the original's 51.** Lever: raise
   max_fanout in render_main.dispatch_next_token (graph-side, ~free: +7% width). The
   compiler CANNOT collapse it today: fuse_consecutive_linears isn't wired into any compile
   path (tests-only), skips Concatenate inputs, and DEADLOCKS this graph when run manually.
   The fanout change is OUTPUT-IDENTICAL (fanout=2 vs flat: 0/20 next-token mismatches via
   reference_eval — scripts/check_fanout_equivalence.py), so it's a safe 1-line change.
2. **Width: graph wants ~11-12k cols** (geometry PWL ~4463 = atan ray-counts + multiply_2d;
   digit-quad emit ~2500; affine ~2300; E8 negligible). Heuristic crams into d=4800 (77
   layers) by serializing → width and depth trade off.
3. **MEMORY (the big one): compile_headless holds ALL ~77 layers' DENSE weights resident
   → ~29-42GB and OOMs.** Attention = `n_heads=d//d_head` (40 heads!) × Q/K/V/O = 4·d²/layer
   (the hog; independent of d_hidden). Only ~22% of params are nonzero (doom uses 1-6 of 40
   heads/layer). **FIX = use `compile_to_onnx` (streaming + sparse): 2.18GB peak, 55MB ONNX,
   all 66 layers** — and it's the token-I/O (token_ids→logits, KV-cached) artifact for Phase K.
   It was ONE ordering bug away: the streaming callback nulls each layer's weights, then the
   post-loop trim_unused_heads/slots sliced None. FIXED in torchwright compile.py (skip the
   in-place trims when on_layer_compiled is set) + regression test
   tests/compile/forward/test_streaming_trim.py (no onnxruntime needed; the compile_to_onnx
   tests are onnxruntime-skipped, which is why this was never caught). Local tests pass;
   non-streaming path unaffected (test_d_hidden_decoupling green). NOT committed — needs the
   full Modal `make test` before landing a compiler change.

### CROSS: {fanout} × {optimize} via compile_to_onnx (d=6400, streaming, token-I/O)
| fanout | optimize     | layers | compile | peak RSS | ONNX  |
|--------|--------------|--------|---------|----------|-------|
| 2      | 0 heuristic  | 66     | 76s     | 2.19GB   | 55.2MB|
| 2      | 2 CP-SAT     | **66** | 265s    | 4.35GB   | 55.2MB|
| 8      | 0 heuristic  | 44     | 54s     | 2.19GB   | 55.1MB|
| 8      | 2 CP-SAT     | **44** | 244s    | 4.16GB   | 55.1MB|

**CP-SAT (optimize=2) gives ZERO depth win at either fanout** — it only confirms the
heuristic warm-start count (66, 44), at 3.5-4.5x compile time + ~2x memory. The
HEURISTIC IS ALREADY OPTIMAL-DEPTH for this graph (CP-SAT bounded to warm-start+1 can't
beat it). The only depth lever is the GRAPH-SIDE max_fanout (66→44); CP-SAT can't
reassociate the dispatch sum. Memory is solved by streaming regardless (2.2GB heuristic /
4.2-4.4GB CP-SAT — the +2GB is the CP-SAT model). ONNX ~55MB regardless. Threaded
`optimize` through compile_to_onnx (export.py) to run this; it's API-completeness, NOT a
doom win. Earlier inconclusive optimize results (compile_headless opt=2 @d4800=77;
deadlock @d3200) now resolved: CP-SAT == heuristic; the heuristic is optimal.

### d_hidden × fanout sweep (schedule-only) — d_hidden saturates at d_hidden=d
Fixed d, varying d_hidden (MLP intermediate width) × fanout. layers / peak_width:

d=6400:                                  d=4096:
| fanout | d_hidden | layers | peak |    | fanout | d_hidden | layers | peak |
|--------|----------|--------|------|    |--------|----------|--------|------|
| 2      | 2048     | 116    | 4955 |    | 2      | 2048     | 119    | 4096 |
| 2      | 4096     | 76     | 6192 |    | 2      | 4096=d   | 90     | 4096 |
| 2      | 6400=d   | 66     | 6400 |    | 2      | 6400     | 89     | 4096 |
| 2      | 12000    | 66     | 6400 |    | 2      | 12000    | 89     | 4096 |
| 8      | 2048     | 96     | 5387 |    | 8      | 2048     | 98     | 4096 |
| 8      | 4096     | 53     | 6051 |    | 8      | 4096=d   | 69     | 4096 |
| 8      | 6400=d   | 44     | 6400 |    | 8      | 6400     | 69     | 4096 |
| 8      | 12000    | 43     | 6400 |    | 8      | 12000    | 69     | 4096 |

Conclusions:
- **d_hidden cuts layers BELOW d_hidden=d, saturates AT d_hidden=d, useless ABOVE.** The
  knee tracks d (6400 @ d=6400; 4096 @ d=4096). Mechanism: a layer's MLP outputs must land
  in the d residual columns, so once d_hidden ≥ d the residual width d caps per-layer
  parallelism, not the hidden pool.
- **Smaller d ⇒ more layers, uncompensable by d_hidden.** At d_hidden=d: d=6400→66/44 vs
  d=4096→90/69 (fanout 2/8). Cranking d_hidden→12000 at d=4096 doesn't recover it (89/69).
- **So larger d_hidden does NOT enable a smaller d — the opposite:** smaller d_hidden →
  smaller residual peak (via serialization, +layers); larger d_hidden → wider peak. d and
  d_hidden are coupled (useful range d_hidden ≤ d); d_hidden=d is depth-optimal.
- Net levers for this graph: **d (residual width ↔ depth) + max_fanout (the fold)**.
  d_hidden is not an independent lever; below d it only trades layers for less MLP-weight
  memory (moot under streaming/sparse compile_to_onnx).

---


Tool: `scripts/analyze_forward_cost.py` (env: D, D_HEAD, OPT, FANOUT, OPT_GRAPH, DH).
VM: 30GB RAM, NO swap. d=8000 full compile OOM'd (crashed VM). Keep d≤6400, ONE
compile at a time. d_hidden (DH) caps MLP weight memory (weights = d·d_hidden·4·n_layers);
default DH=d. DH must be ≥ widest single-layer MLP op (ray_count is 1024-wide).

## Established facts (from code)
- Compiled token-I/O forward (heuristic, optimize=0): **77 layers @ d=4800, 66 @ d=6400**
  → depth is WIDTH-LIMITED (more room → less serialization).
- Critical-path tail (L30→65) = regular `Linear→Concatenate→Linear` serial accumulator
  = the dispatch `type_switch(*_distinct_head_pairs, max_fanout=2)` fold over ~19 gated heads.
- WIDTH peak ~50% digit-quad VALUE emit (255-wide floor_int + 129-wide in_range × ~16 make_value).
  E8 type-code only 480 cols — NOT a bottleneck.
- `fuse_consecutive_linears` / `optimize_graph` (graph/optimize.py): **NEVER called in any
  compile path — only in tests.** So the doom compile runs with ZERO linear fusion.
- `fuse_consecutive_linears` EXPLICITLY SKIPS Concatenate inputs (line 62-63, "would need
  special handling") → even if run, it can't fuse the dispatch fold's `Linear(Concat(...))`.
- CP-SAT objective = layers + attn_heads + mlp_bypass (default pure layer-min); width is a
  CONSTRAINT not objective. At d=4800 CP-SAT=heuristic=77 layers. Inconclusive at tight d.

## Hypotheses to test
H1. Running `optimize_graph(nt)` before compile cuts layers (fuses non-concat serial linears).
H2. The dispatch fold survives optimize_graph (concat-blocked) → still ~35 layers.
H3. A flat-sum / tree dispatch (max_fanout=None/large) OR a concat-aware fusion collapses the
    fold → depth ≈ deepest-term-chain (~31), removing ~35 layers. (Width cost: materializes
    more digit-quads at once.)
H4. The TWO real levers: (a) wire optimize_graph + add concat-aware fusion (compiler-side);
    (b) shrink the digit-quad emit width (graph-side) so a flat dispatch fits.

## Results (schedule-only, memory-safe — `SCHED_ONLY=1`, validated: matches compiled 66@d6400)

### DEPTH — it's the dispatch fold, fixable nearly free via max_fanout
Unconstrained (d=12000, no width pressure):
| dispatch              | layers | peak width |
|-----------------------|--------|-----------|
| max_fanout=2 (current)| **67** | 11065     |
| max_fanout=4          | 44     | 11083     |
| max_fanout=8          | **37** | 11337     |
| max_fanout=None (flat)| **32** | 11854     |

- The `type_switch(max_fanout=2)` serial fold is ~35 REMOVABLE layers (67→32).
- Flat/tree dispatch cuts depth 52% for +7% width — my earlier "flat balloons width"
  fear was WRONG; width is geometry+digit-quad (live regardless of fanout), heads are thin.
- **True min depth = 32 layers — BELOW the original's 51.** Confirms the user: the bloat
  was the dispatch fold, not geometry. After the fold, the 32-layer critical path is
  select/dispatch (cond_gate) + affine + digit-quad (inherent renderer logic).
- H1 (optimize_graph) NOT TESTABLE as-is: running fuse_consecutive_linears manually
  DEADLOCKS the heuristic at d=6400 — the existing fusion pass isn't safely applicable
  to this graph (and skips concat anyway). The clean lever is the dispatch fanout.

### WIDTH — graph wants ~11-12k cols; heuristic crams into 4800-6400 by serializing
Peak live-set by subsystem (d=12000, fanout=2): geometry PWL **4463** (atan ray-counts +
multiply_2d grids) > digit-quad emit **2564** > affine glue **2255** > select/dispatch.
E8 negligible. The heuristic fits this into d=4800 (77 layers) / 6400 (66) by serializing
(width-limited depth). So d and depth trade off; the natural residual is ~12k.

### LEVERS (validated)
1. [graph-side, 1-line] Raise dispatch `max_fanout` 2→8 in render_main.dispatch_next_token:
   67→37 layers (~free, +3% width). #1 depth lever. (Need: re-verify gates + compile still fit.)
2. [graph-side] Shrink WIDTH: geometry PWL (atan ray-count is 2×1024 per signed_world_angle
   call site; multiply_2d grids) is the biggest width chunk, then digit-quad emit.
3. [compiler-side] Wire optimize_graph into the pipeline + make fusion concat-aware + fix the
   deadlock it currently causes. Equivalent depth win to the fanout change.

## MEMORY (the detour) — compile peak ~29GB is dense ATTENTION weights
- schedule-only = 1.3GB; full compile @d=4800 OOM'd @29GB → the ~28GB is WEIGHTS.
- `AttnLayerComponent.__init__` (compiler/components/attn.py:42-49): `n_heads = d // d_head`,
  then Q/K/V/O = `torch.zeros(n_heads, d, d_head)` → n_heads·d·d_head = **d² per matrix,
  4·d² per layer** (DENSE, independent of d_hidden — why capping DH didn't help).
  @d=4800: 4·4800²·4B = 369MB/layer × 77 = **28GB attention alone.**
- MLP (mlp_sublayer / components/linear.py): linear1 `zeros(d, d_hidden)` + linear2
  `zeros(d_hidden, d)` = 2·d·d_hidden DENSE. trim_heads trims d_hidden to used hidden
  (helps final size) but the PEAK during build is pre-trim.
- The doom graph uses only ~2-5 of the 30 allocated heads/layer (≈148 attn nodes / 66 layers)
  → ~85-90% of the attention matrices are structurally ZERO. Dense storage wastes it.
- **Biggest memory lever: allocate per-layer n_heads = actual head demand, not d//d_head
  (and/or stream+free or trim per layer immediately, and/or store sparse).** ~10-15x cut.
  d_head=160 vs smaller doesn't change TOTAL attention weight (always 4·d²); the over-
  allocation of heads to full capacity is the waste. [compiler-side] (memory workflow
  wf_058c99e2 verifying sparse/streaming/trim details).
- IMPLICATION: the memory fix is on the critical path — validating the fanout depth fix via
  a FULL compile (and Phase H/I/J, and the Phase-K free-running frame) needs d≈11k × layers
  which is infeasible dense today. Fix attention-weight allocation first.
