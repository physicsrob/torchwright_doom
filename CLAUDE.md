# Communication

## Use plain English; reintroduce terms on every use

When explaining technical concepts, describe the mechanics in plain
English rather than introducing named abstractions. Say "the input
range crosses zero" not "straddling." Say "the slope of the line
connecting the endpoints" not "the chord relaxation." If a term
doesn't already exist in the codebase, prefer the description — the
user will name it if it needs a name.

When a named term genuinely earns its keep (you'll reference the
concept many times and the name saves confusion), define it inline on
every use until the user starts using it themselves — that's the
signal they've adopted it. "The PL-drift (the gap between the
piecewise-linear approximation and the exact function) compounds
through..." not just "The PL-drift compounds through..."

Never stack coined terms. "The chord relaxation of the straddling ReLU
in the forward-mode LiRPA" is four layers of undefined vocabulary.
Each layer of jargon you build on top of another layer compounds
confusion. If you need multiple concepts, introduce them one at a time
with plain-English definitions between them.

The user manages multiple projects and does not have your earlier
definitions loaded. Write every explanation so it can be understood
cold.

## Admit uncertainty; don't fill gaps with plausible stories

When you aren't sure whether two things are really the same, whether a
mechanism works the way you think, or whether a number is right — say
so. "I think these might be the same thing but I'm not sure" is always
better than treating them as interchangeable and building an
explanation on top. Check the code before building an explanation on
any factual claim (a constant's value, what a function reads, how a
data structure is used). Never construct a narrative that "sounds
right" without tracing the actual code path — the most dangerous
explanations are the ones that are internally consistent but don't
match reality.

## Flag complexity before building it

Before introducing a new abstraction, indirection layer, or
deferred-execution pattern, flag it to the user: what it is, why you
think you need it, and what the simpler alternative would be. "I'm
about to add a placeholder system because the basis needs to be fully
known before computing bounds — the simpler alternative is making the
basis mutable. The placeholder approach is more complex but avoids
changing Basis. Which do you prefer?" Don't commit to elaborate
machinery without explicit agreement. If the user says "that feels
gross," trust that instinct and look for the direct path.

## State constraints alongside proposals

When proposing a mechanism, state all its constraints and assumptions
upfront. Don't wait for the user to discover them through follow-up
questions. "This requires X and constrains Y" is always better than
explaining X only when asked. If a design requires a fixed layout,
say so. If it introduces a dependency, name it. If it limits future
flexibility, flag it. Minimizing complexity in the explanation doesn't
reduce the complexity of the mechanism — it just hides it, and the
user will find it later in a more frustrating way.

# Current state of this submodule

This submodule is the active DOOM renderer: a computation graph that
`torchwright` compiles into a transformer which renders DOOM
autoregressively. `torchwright_doom/render_main.py` is the entry point —
`forward(input_vec, past, pos)` builds the read-side scene and protocol
views, publishes the runtime protocol owners, builds each dispatch
branch's next-token, and selects one by the current input token's type.

Rough layout of `torchwright_doom/`:

- **Read side** — `vocab` / `tokens` / `value_ranges` (the token
  vocabulary and value encodings), `embedding` / `emit` / `extract`
  (token ↔ residual encode/decode), `scene_*` and `protocol_*` (static
  map facts and the autoregressive protocol / dispatch table).
- **Geometry write side** — `bsp_traversal` / `bbox_pruning` /
  `traversal_edges` (BSP walk + visibility pruning), `seg_projection` /
  `seg_scanner` / `seg_cycle` (wall-segment projection), `render_ops`
  (shared math: atan2, distance, clamps).
- **Rasterization** — `wall_range_*` / `wall_column_*` (wall columns),
  `visplane_*` / `flat_*` (floor/ceiling planes), `solid_intervals`
  (occlusion), `pixel_dispatcher` / `uv_compute` (final pixels).
- **Assets + lighting** — `wad_assets` / `assets` / `asset_*` (textures
  and flats compiled to lookup tables), `doom_lighting` / `lighting` /
  `pwl_banks` (light levels and colormaps).
- **Graph infrastructure** — `past` / `attention_handles` (reading
  previously-emitted tokens) and `std` (the helper-op shim).

Each renderer module is ported from a `doom_sandbox` counterpart; many
docstrings note that provenance and the sandbox path.

See `GLOSSARY.md` for the coined vocabulary (carrier, head, lifted key,
radix, digit-quad, flat, visplane, …).

# Dumb host principle

The DOOM renderer is an exercise in extreme constraints. The goal is
not to render DOOM efficiently — it's to render DOOM in a way that is
entirely analogous to autoregressive LLM inference. Each step has a
discrete input token and predicts a discrete output token. The host
copies the output token to the next input. This is normal LLM
autoregression — the same loop that drives any chat model. The only
addition is that output tokens include pixel information which the
host copies to the screen, analogous to a mixed-modal model predicting
image patches.

That's it. The host feeds tokens and blits pixels. It does zero
computation — no sorting, no geometry, no visibility culling, no
arithmetic. The host may skip already-filled pixels when writing
(front-to-back compositing), but that is a conditional write, not
computation. All rendering logic — wall selection, visibility,
distance, texture lookup, compositing decisions — lives inside the
transformer.

**Never violate this principle.** If a proposed design, optimization,
or bug fix moves computation from the transformer to the host, it is
wrong regardless of how much simpler it would make things. If you
discover existing code that violates this principle — host-side logic
that does anything beyond token I/O and pixel blitting — flag it to
the user immediately and stop other work until resolved.

# Testing

Tests live under `tests/` (`tests/embedding/`, `tests/scene/`,
`tests/past/`, …) and run via the `make test` / `make test-local`
machinery (with sharding in `modal_test.py`).

Rules for running them:

- ALWAYS use `make test` to run them; NEVER invoke pytest directly.
- NEVER run tests in the background; always foreground.
- NEVER run tests in parallel (no pytest-xdist, no `&`).
- NEVER re-run tests just because you piped output through `| tail`
  and lost it — the full output is in the log file (printed at start
  and end; `/tmp/torchwright_doom-test.log` symlinks to the latest).
  Grep that file instead.

# Numerical noise

Every approximate op in torchwright is measured against its exact-math
reference. Per-op noise budgets live in torchwright's
torchwright's `docs/op_noise_data.json` (canonical) with commentary in
`docs/numerical_noise_findings.md`. The `doom_*` distributions are
calibrated against DOOM call sites — they're the inputs the framework
measured each op against to produce the per-op bound that doom code
relies on.

When you investigate a tolerance issue, the relevant per-op bound is
what your graph's chain noise has to fit under. The full
measure/maintain workflow for op noise lives in torchwright's
`CLAUDE.md`.

# Running scripts on GPU

**If a script needs a GPU, run it on Modal via `make modal-run`.**
Never write a new `modal_*.py` file at the repo root just to run a
script remotely.

    # Run a committed module (preferred)
    make modal-run MODULE=<dotted.path>

    # Pass args through
    make modal-run MODULE=<dotted.path> ARGS="--input bar"

    # Run an arbitrary file
    make modal-run SCRIPT=path/to/one_shot.py

    # CPU-only shard (no GPU reservation)
    make modal-run MODULE=<dotted.path> CPU_ONLY=1

## When NOT to use modal-run

- **Tests** — use `make test`. Its sharding + log-file plumbing is
  not reproduced by `modal-run`.
- **Scripts that produce local artifacts** (GIFs, JSON files under
  `docs/`, etc.). `modal-run` captures stdout/stderr only; anything
  the script writes to disk stays on the Modal worker. If your
  script needs artifact sync-back, that is the *only* acceptable
  reason to add a new purpose-built `modal_*.py` entrypoint — and
  when you do, import the image from `modal_image.py` rather than
  duplicating it.

## Critical rules

- NEVER write a new `modal_*.py` at the repo root just to run a
  one-off investigation. Put the script under a `scripts/` directory
  (or `tests/` if it's really a test) and run it via `make modal-run`.
- NEVER duplicate the Modal image definition. Import `IMAGE` from
  `modal_image.py`.
- NEVER re-run `make modal-run` just because you piped output
  through `| tail` and lost it. The full output is in the log file
  printed at the start and end; `/tmp/torchwright_doom-modal-run.log`
  symlinks to the latest. Grep that file instead.

# Debugging compiled graphs

When a compiled graph produces wrong output, the cause is almost
always in the user graph (wrong op, wrong wiring, numerical noise
accumulation) — not in the compiler.  The compiler has four
runtime-enforced invariants (I1–I4, documented in torchwright's
`CLAUDE.md`) that catch the structural bugs (column aliasing,
truncated writes, wrong Q/K/V widths, premature frees) that would
look like "the compiler broke my values."  If those invariants pass
during compilation, the compiler produced a structurally correct
transformer.  The remaining question is whether the user graph's
math, when evaluated through piecewise-linear approximations, stays
within its noise budget — and the tools below answer that question
directly.

Before suspecting a compiler bug, **run the tools in this section**.
They are ordered from cheapest to most expensive.  If all of them
come back clean, the problem is in the graph's numerical design
(op choice, gain settings, breakpoint placement, tolerance budgets),
not in the compiler.

## debug=True forward pass

The cheapest check.  Pass `debug=True` to `__call__` or `step`:

    output = compiled(inputs, debug=True)
    output, new_past = compiled.step(inputs, past, debug=True)

This runs the full forward pass with per-sublayer residual-stream
capture, then performs three checks:

1. **Self-consistency**: for every graph node that appears in
   multiple sublayer snapshots, verifies the value at its assigned
   columns is identical across all of them.  A node's columns sit
   in the residual stream untouched until freed; if they differ
   between snapshots, something overwrote those columns while the
   node was still live.  This is a compiler or scheduling bug (it
   would mean I1 or I4 failed to catch an allocation error at
   compile time).  Raises `RuntimeError` on failure.

2. **Assert predicates**: every `Assert` node in the graph has its
   predicate run against the compiled value.  Raises `AssertionError`
   with annotation context on failure.

3. **DebugWatch predicates**: every `DebugWatch` node in the graph
   has its predicate run against the compiled value.  Prints on
   trigger, does not raise.

**If `debug=True` passes with no errors or warnings**, the compiled
transformer's residual stream is internally self-consistent and
every asserted invariant holds on compiled values.  That rules out
the compiler as the source of the problem — whatever's wrong lives
in the graph logic or numerical tolerances.

## debug_value(node)

After a `debug=True` forward, extract any graph node's compiled
value:

    compiled(inputs, debug=True)
    val = compiled.debug_value(some_intermediate_node)

Returns an `(n_pos, node.d_output)` tensor, or `None` if the node
has no residual assignment.  Unwraps Assert/DebugWatch wrappers
automatically.  Useful for spot-checking a specific node without
setting up the full probe machinery.

Raises `RuntimeError` if no `debug=True` forward has been run.

## probe_compiled — full oracle comparison

Runs the compiled transformer side-by-side with a recursive graph
evaluation (the oracle — `node.compute` on every node) and reports
every node whose compiled value disagrees beyond a tolerance:

    from torchwright.debug.probe import probe_compiled

    report = probe_compiled(compiled, output_node, input_values, n_pos, atol=1e-3)
    print(report.format_short())

`report.first_divergent` is the earliest node in topological order
that exceeds `atol`.  `report.per_node` has the full error record
for every checked node.  If `first_divergent is None`, the compiled
transformer matches the oracle everywhere — the graph math is the
math you designed, and the only error is the piecewise-linear
approximation noise measured in torchwright's `docs/op_noise_data.json`.

`probe_graph` is a convenience wrapper that compiles and probes in
one call:

    from torchwright.debug.probe import probe_graph

    report = probe_graph(output_node, pos_encoding, input_values, n_pos,
                         d=2048, d_head=32, atol=500.0)

**Interpreting `atol`.**  The tolerance must account for accumulated
piecewise-linear approximation error through the graph.  For a
shallow graph (few ops, small value range), `atol=1e-3` is
appropriate.  For the DOOM renderer (deep op chains, values in the
10^4 range), `atol=500` is the empirical floor.

## probe_residual — layer-by-layer node values

Extract a specific node's compiled value at every post-MLP layer
where it's materialized:

    from torchwright.debug.probe import probe_residual

    rp = probe_residual(compiled, prefill_tensor, node)
    for layer_i in rp.layers:
        print(f"layer {layer_i}: {rp.at(layer_i)}")

Restrict to specific positions with `rp.positions([0, 3, 7])` or
a single layer with `at_layer=5`.

## probe_attention — softmax weight inspection

Capture the explicit softmax weights and pre-softmax logits at a
specific attention node and query position:

    from torchwright.debug.probe import probe_attention

    ap = probe_attention(compiled, prefill, attn_node, query_pos=2)
    print(ap.top(k=5, head=0))  # top-5 keys by weight

`ap.weights` is `(n_heads, n_keys)` and `ap.logits` is the same
shape.  Useful for diagnosing softmax concentration failures —
the attention isn't picking a single key, so it blends values
instead.  The fix is upstream: widen the score gap (raise the
gain on the matching term) or sharpen the softmax temperature.

## probe_layer_diff — drift tracking

Track how a node's value evolves across layers, compared to a
known reference:

    from torchwright.debug.probe import probe_layer_diff

    report = probe_layer_diff(compiled, prefill, node,
                              reference=oracle_value,
                              positions=[0, 1, 2],
                              drift_threshold=1e-3)
    if report.first_drift_layer is not None:
        print(f"drift starts at layer {report.first_drift_layer}")

Can also detect sentinel values (e.g. `sentinel=-1000.0`) that
should never appear in a healthy forward pass.

## Assert and DebugWatch nodes

Graph-level invariants are encoded as `Assert` and `DebugWatch`
nodes that wrap intermediate values.  Both are stripped at compile
time (the compiled transformer is identical with or without them)
and re-checked during `debug=True` forward passes.

Helpers in `torchwright/graph/asserts.py`:

- `assert_in_range(node, lo, hi)` — value bounds
- `assert_integer(node)` — near-integer values
- `assert_bool(node)` — values near +1 or -1
- `assert_01(node)` — values near 0 or 1
- `assert_onehot(node)` — one-hot rows (pre-attention only)
- `assert_unique_values(node)` — pairwise-distinct components
- `assert_distinct_across(value, where)` — cross-position uniqueness
- `assert_score_gap_at_least(score, where)` — softmax resolvability
- `assert_picked_from(result, values, keys)` — attention picked
  exactly one key
- `assert_strictly_less(a, b)` — elementwise ordering
- `debug_watch(node, predicate, message)` — observational (print,
  not raise)

These run on exact-math values during `reference_eval` and on
compiled values during `debug=True`.  An assert that passes in
reference eval but fails in the compiled forward pinpoints a node
where piecewise-linear approximation error exceeds the tolerance —
that's a noise-budget problem in the graph, not a compiler bug.

## FP nondeterminism at tolerance boundaries

GPU matmul is non-deterministic across runs — cuBLAS algorithm
selection, TF32, and atomics ordering produce run-to-run variation
on the order of `1e-5` to `1e-6` on float32 accumulation.  Same
code, same inputs, same GPU.  This is below every per-op mean-error
budget in torchwright's `docs/op_noise_data.json`, but it can flip a borderline
cond across a `c_tol` boundary: a cond landing at `|cond|=0.995`
under a `c_tol=0.005` budget fires the postcondition assert on
some runs and passes on others.

**Symptom.** `debug=True` asserts fire intermittently on identical
inputs; re-running `step` sometimes passes, sometimes fails.  Not a
bug in the op, not a scheduling issue — a tolerance sitting too
close to the actual cond magnitude.

**Rule.** `c_tol` and assertion tolerances need margin above *both*
the op's measured noise and GPU FP variation.  If a cond lives at
its budget, that's brittle — either widen the tolerance (cheap,
biases the cond "on") or tighten the upstream compute so the cond
lands further from zero (principled, often requires graph-level
changes).

## Triage sequence for wrong output

1. **`compiled(inputs, debug=True)`** — does the self-consistency
   check pass?  Do any asserts or watches fire?  If the consistency
   check fails, that's a real compiler/scheduling bug (report per
   D1).  If an assert fires, the failure message names the node
   and the invariant that broke — investigate that node's inputs.
   If the same `debug=True` call passes on some runs and fails on
   others with identical inputs, see *FP nondeterminism at tolerance
   boundaries* above before investigating the op.

2. **`probe_compiled`** — does the oracle agree with compiled?
   If `first_divergent` is `None`, the compiled transformer matches
   the graph's exact math within `atol`.  The problem is upstream
   (graph logic, scene setup, input encoding).  If there is a
   divergent node, it names the first place where compiled values
   drift from exact math — investigate the op at that node and its
   noise budget.

3. **`debug_value(node)`** or **`probe_residual`** — spot-check
   specific intermediate nodes.  Compare against hand-computed
   expected values or oracle values from `reference_eval`.

4. **`probe_attention`** — if the divergence is downstream of an
   attention layer, check whether the softmax is concentrating
   correctly.  A spread-out weight distribution (no single key
   above 0.99) means the attention is blending values instead of
   picking one — the gain or score gap is too small.

5. **Per-op noise bounds** — check torchwright's `docs/op_noise_data.json` for
   the op producing the divergent node.  If the measured worst-case
   error for that op (at the relevant input range) exceeds the
   tolerance the graph needs, the fix is in the op's breakpoint
   grid or the graph's tolerance budget, not the compiler.

If all five come back clean, the compiled transformer is correct
and the bug is in the test expectation, input setup, or reference
implementation.

# Doctrine

These rules target Claude's coding tendencies that persist regardless
of project structure: guessing dressed as understanding, deferring
numerical problems, leaving anomalies for later.

The diagnostic that applies across all of them: if you can't explain
a behavior's root cause in one sentence without hedged conjunctions,
you don't understand it yet. Research until the sentence compresses;
if the doc that would have let someone else write the sentence is
missing, add it. The warning sign is the *and-of-maybes* pattern:
"likely X near Y under Z." When only an and-of-maybes will fit, the
sentence is hiding ignorance.

## D1 — Trust the compiler; escalate if a guarantee seems broken

Assume the compiler is robust. It enforces four invariants at compile
time, asserted at runtime:

- **I1**: no residual-column aliasing among simultaneously-live nodes
- **I2**: literal and bias writes match their source tensor's size
  (no silent truncation)
- **I3**: attention Q/K/V/O row widths match the node's declared
  shapes
- **I4**: a node's residual columns stay allocated until every
  effective consumer has run

If those assertions pass during compilation, the compiled transformer
is structurally correct. Full descriptions and the canonical list live
in torchwright's `CLAUDE.md`.

When the compiled output looks wrong, the cause is almost always in
the user graph (op choice, gain settings, breakpoint placement,
tolerance budget) — not in the compiler. The triage sequence under
*Debugging compiled graphs* above works through that hypothesis space
in order.

**If you suspect a compiler guarantee is actually broken**, escalate
to the user immediately with the specific trigger that fired (which
invariant, on which test). Don't reshape graph code to route around a
suspected compiler bug — workarounds leave landmines, and history says
most "I packed it differently and the bug went away" fixes were
masking a real bug elsewhere.

## D2 — Never defer numerical problems

When a value is off by an unexpected amount, two answers are
acceptable: the bit-level reason for the divergence, or "I don't
know yet, investigating." A plausible-sounding guess is not — the
next reader takes it as truth, stops looking, and the real cause
grows another layer of camouflage.

Numerical problems compound. A guess shipped today is what
tomorrow's debugging starts from.

**First tool**: `torchwright.debug.probe.probe_compiled` runs the
compiled module side-by-side with the graph oracle and reports the
first divergent node. Per-op noise bounds live in torchwright's
torchwright's `docs/op_noise_data.json`.

## D3 — Foundation rule

Don't move on if the foundation isn't 100% solid. An un-investigated
anomaly in phase N is the first task of phase N+1, not a footnote.
Every layer added on top of an anomaly multiplies the cost of going
back: downstream changes pile up against the unfixed thing and turn
what would have been a local fix into a re-architecture.

## D4 — xfail hygiene

No `xfail` without a precisely documented root cause. Two acceptable
forms:

1. `xfail(reason="precise root cause: X; will be fixed by Y",
   strict=True)` — cause known, fix deferred for a stated reason.
2. `xfail(reason="unknown, investigating, linked to issue N",
   strict=True)` — cause not yet known, but a tracked follow-up
   exists.

Unacceptable: `xfail(reason="likely due to <guess>")`. A guessed
reason isn't a TODO, it's a trap — the next contributor reads it,
takes it as explanation, and stops looking.

`torchwright.debug.probe.probe_compiled` is the tool that converts
form 2 into form 1.
