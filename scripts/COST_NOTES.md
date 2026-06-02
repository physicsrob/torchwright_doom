# Depth/width cost investigation — running log

## ★ SYNTHESIS (validated, overnight 2026-06-02)
1. **Depth was the dispatch fold, not geometry.** `type_switch(max_fanout=2)` builds a
   ~35-layer SERIAL Linear→Concat accumulator. Raising max_fanout: 67→44 (=4), →37 (=8),
   →32 (flat). **True min depth = 32 layers, BELOW the sandbox's 51.** Lever: raise
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
- **True min depth = 32 layers — BELOW the sandbox's 51.** Confirms the user: the bloat
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
