# Smooth Recency Rank Plan

## Goal

Replace the noisy raw global-position tiebreak in long-range
`pick_most_recent` reads with a separate, smoother `recency_rank` signal.

`global_position` remains the approximate absolute position used by pixel and
span arithmetic. `recency_rank` is used only to order matching historical rows.
It does not need to equal the exact token position; it needs to be globally
increasing and to have a reliably large adjacent step.

The plan evaluates two versions:

1. **All-history mean:** average the recovered global position over every
   causally visible row, then multiply the result by two.
2. **Fixed-`K` mean:** average the recovered global position over the current
   row and the previous `K - 1` rows.

Both versions are built once per token position and shared by every global
recency lookup. Neither version averages the value returned by
`pick_most_recent`; only its key-side recency rank is smoothed.

## Problem statement and assumptions

Use the following names throughout this document:

```text
t              true zero-based token position
raw[t]         global_position recovered at position t
error[t]       raw[t] - t
rank[t]        candidate recency_rank at position t
K              fixed-window length
E              observed bound on absolute recovery error; initially 0.45
gain           coefficient applied to rank in the selection logit; currently 8
N              maximum sequence length; currently 65,536
```

The initial working model is:

```text
raw[t] = t + error[t]
-E <= error[t] <= E
E = 0.45
```

The stronger empirical observation is that `raw` has long-run slope very close
to one even where individual adjacent differences are noisy.

The current failure mode is insufficient selection hardness. For example:

```text
raw[new] - raw[old] = 0.49
gain = 8
new-versus-old logit margin = 8 * 0.49 = 3.92
```

With two matching candidates, a margin of 3.92 leaves about 2% of the attention
weight on the older row. Segment indices, lifted keys, and boolean state cannot
tolerate that much blending.

The proposed rank must satisfy all of the following:

- It is strictly increasing at every position where a global recency read is
  consumed.
- Its adjacent step is large enough to meet a specified softmax-hardness target.
- Its total range remains approximately `0..N`, so content match still dominates
  recency over the full cache.
- It has no finite recency window and no long-distance RoPE aliasing.
- It behaves identically during prefill and cached decoding.
- It is computed once and reused by all `pick_most_recent` calls.

## Shared architecture

### Add a distinct cached rank

Extend `GraphPast` with a cached recency-rank node, separate from the cached
global-position node:

```python
class GraphPast:
    def global_position(self) -> Node:
        ...

    def recency_rank(self) -> Node:
        ...
```

`global_position()` must remain byte-identical for its existing arithmetic
consumers. Only `GraphPast.pick_most_recent()` changes from:

```text
global_position key-side tiebreak
```

to:

```text
recency_rank key-side tiebreak
```

Keep the raw strategy available during development:

```text
raw               rank[t] = raw[t]
all_history_mean  Version A below
last_k_mean       Version B below
```

The strategy should be a graph-build option or temporary experiment constant,
not a runtime branch embedded in the model. Keep `raw` as the default until a
candidate passes the acceptance gates.

### Preserve the current selection structure

The global selection score remains conceptually:

```text
selection score for row i =
    content_match_score[i]
    + gain * rank[i] * slow_plane_attenuation(query_position - i)
```

Production partial RoPE places content in the NoPE tail and the rank tiebreak on
the slowest rotated plane. The slow-plane attenuation is positive and close to
one over the full cache. This plan changes the `rank[i]` input, not the head's
content matching or value path.

After choosing a rank, recompute the content-dominance bound using the measured
rank range:

```text
match_gain * minimum_content_gap
    > gain * (maximum_rank - minimum_rank)
      + numerical headroom
```

Do not raise `gain` without raising or re-proving the relevant match gains.

## Version A: average over all causal positions

### Definition

At position `t`, compute:

```text
mean[t] = average(raw[0], raw[1], ..., raw[t])
rank[t] = 2 * mean[t]
```

For perfect positions:

```text
average(0, 1, ..., t) = t / 2
rank[t] = t
```

Thus the ideal rank has exactly unit slope and no lag.

### Noise behavior

When `raw[t] = t + error[t]`:

```text
rank[t] = t + 2 * average(error[0], ..., error[t])
```

Its adjacent step is:

```text
rank[t] - rank[t-1]
    = 1
      + 2 * (error[t] - previous_average_error) / (t + 1)
```

If every error is bounded by `E`, then:

```text
minimum adjacent step at t = 1 - 4E / (t + 1)
maximum adjacent step at t = 1 + 4E / (t + 1)
```

For `E = 0.45`:

```text
t = 100       minimum step about 0.982
t = 1,000     minimum step about 0.9982
t = 10,000    minimum step about 0.99982
```

This bound is deterministic. It does not require independent or zero-mean
errors. A constant recovery bias becomes a constant rank offset and has no
effect on ordering.

### One-head construction

Add or reuse a causal uniform-mean attention operation:

```text
query/key logits: equal for every causally visible row
value:             raw global position
output:            uniform causal mean of raw
```

The clearest generic torchwright primitive would be:

```python
attend_causal_mean(rope, value)
```

It should construct exactly equal logits, preferably with zero Q/K projections,
so RoPE cannot make the mean position-dependent. This is one logical attention
head with `d_v = 1`.

Do not assume the current `attend_mean_where` is exact under every RoPE layout.
Under production partial RoPE its validity content rides the NoPE tail and is
position-independent; under full RoPE it rides a slow plane and retains a small
distance-dependent attenuation. Either:

1. add the dedicated zero-logit `attend_causal_mean`, or
2. prove and test that a chosen existing construction produces equal logits on
   every supported runtime surface and configuration.

Multiply the mean by `2.0`, attach a justified value-range assertion, and cache
the result as the all-history `recency_rank`.

### Range and numerical considerations

Plain interval propagation may infer `2 * mean(raw)` to have range `0..2N`, even
though its semantic range is approximately `0..N`. Establish a tighter type or
assertion from the position/error proof; otherwise the compiler and
content-dominance analysis will see an unnecessarily large range.

Measure all of the following at production length:

- fp32 uniform-softmax behavior with about 65,536 equal logits;
- fp32 accumulation error while averaging values as large as 65,536;
- oracle, compiled prefill, and cached-decode agreement;
- whether the running mean responds acceptably to any slow systematic change in
  the raw position recovery's slope;
- head and layer cost after scheduling.

### Advantages

- One shared attention head.
- Natural behavior from the first token; no invalid negative offsets.
- Under bounded error, adjacent noise shrinks as the sequence grows.
- No fixed `K` to tune.
- No local-window RoPE lobe and therefore no recency aliasing.

### Risks

- A full-history average responds slowly to a genuine late change in the raw
  signal's slope.
- Equal-logit softmax and long fp32 value accumulation need compiled full-length
  validation, not only oracle validation.
- A dedicated exact causal-mean primitive may be needed in torchwright.
- The inferred value range needs care after multiplying the mean by two.

## Version B: average over the previous `K` positions

### Definition

At position `t`, after the window is full, compute:

```text
window_mean[t] = average(
    raw[t],
    raw[t-1],
    ...,
    raw[t-K+1],
)

rank[t] = window_mean[t] + (K - 1) / 2
```

The final constant recenters the ideal mean on `t`:

```text
average(t, t-1, ..., t-K+1) = t - (K - 1) / 2
rank[t] = t
```

The constant is not required for selection because it cancels between rows, but
it makes diagnostics and range reasoning easier.

### Why the adjacent step is stable

Consecutive windows share `K - 1` values. When the two means are subtracted,
those shared values cancel:

```text
rank[t] - rank[t-1]
    = (raw[t] - raw[t-K]) / K
```

The adjacent rank step is therefore based on the observed position change over
`K` tokens, divided by `K`. This directly converts the assumed reliable
long-term slope into a reliable adjacent slope.

Under the bounded-error model:

```text
minimum adjacent step = 1 - 2E / K
maximum adjacent step = 1 + 2E / K
```

For `E = 0.45`:

```text
K = 2     step range 0.5500 .. 1.4500
K = 4     step range 0.7750 .. 1.2250
K = 8     step range 0.8875 .. 1.1125
K = 16    step range 0.94375 .. 1.05625
```

At `K = 8` and `gain = 8`, the worst-case adjacent logit margin from this model
is:

```text
8 * 0.8875 = 7.1
```

With two equal-content candidates, that leaves about 0.08% weight on the older
row.

### Important implementation distinction

Fetching only `raw[t-K]` with one fixed-offset head computes the stable slope:

```text
(raw[t] - raw[t-K]) / K
```

That value stays near one and is not itself an increasing rank. It cannot be
used directly as the key-side recency tiebreak.

The rolling update:

```text
rank[t] = rank[t-1] + (raw[t] - raw[t-K]) / K
```

would need the previous rank as recurrent state as well as `raw[t-K]`. The
current `GraphPast.publish` construction does not naturally support a channel
whose graph node depends on its own previous-position handle. Do not assume a
one-head rolling implementation without first designing and proving that state
mechanism.

### Reference implementation: parallel fixed-offset reads

Build the first correct version using existing fixed-offset reads:

```text
current raw position                       no head
raw at offsets -1 through -(K - 1)         K - 1 offset heads
sum and multiply by 1/K                    affine reduction
add (K - 1)/2                              affine bias
```

The offset heads are logically independent and should be schedulable in
parallel, but their physical head, residual-width, layer, and KV costs must be
measured.

Start with `K` in `{4, 8, 16}`. `K = 8` is the initial preferred candidate from
the error-bound calculation.

### Possible optimized implementation

If the reference version wins on correctness but costs too many heads, consider
a dedicated primitive:

```python
attend_mean_last_k(rope, value, k=K)
```

It must give equal weight to exactly the last `K` causal rows, truncate safely
at BOS, and exclude all older rows without a periodic RoPE alias. Implementing
this may require compiler/runtime support for a local causal mask; do not replace
the reference implementation with an approximate rotary lobe.

A self-recurrent rolling average is another possible optimization, but only
after the graph and cache semantics of reading `rank[t-1]` while publishing
`rank[t]` are explicitly designed.

### Startup behavior

Fixed negative offsets are undefined before enough causal rows exist. Choose and
test one explicit policy:

1. Prove that no Doom global-recency result is consumed before position
   `K - 1`, and document that invariant.
2. Use an all-history prefix mean for the first `K - 1` positions, then switch to
   the fixed window. This adds the all-history mean head unless it is otherwise
   shared.
3. Add a true `attend_mean_last_k` primitive whose causal window naturally
   truncates at BOS.

Do not rely on out-of-range `attend_to_offset` results or silently average
garbage rows.

### Advantages

- Directly exploits the measured `K`-step slope.
- Deterministic adjacent-gap bound independent of absolute position.
- Responds faster than the all-history mean to a real change in recovery slope.
- Uses existing fixed-offset machinery in the reference implementation.

### Risks

- The straightforward implementation costs `K - 1` attention heads.
- Startup requires an explicit policy.
- `K` must be selected from measured slope/error data rather than convenience.
- A one-head optimized window mean is not currently an established primitive.
- Fixed-offset oracle/compiled/cached-decode parity must hold for every selected
  offset.

## Measurement before implementation

First capture the actual raw-position behavior that motivated this work. The
measurement should include the failing render, not only synthetic noise.

For every position in a representative full rollout, record:

```text
true integer position
raw recovered global position
raw adjacent step
raw K-step slopes for K = 2, 4, 8, 16, 32
token type
whether the row participates in a global recency lookup
```

For every consumed global recency selection, also record:

```text
query position
matching candidate positions
newest and second-newest candidate positions
raw rank of those candidates
content logits
recency logits
winning softmax weight or hardness
returned value and expected value
```

Report at least:

- minimum, percentile, and maximum raw adjacent step;
- minimum `K`-step slope for each proposed `K`;
- error correlation length and evidence of slow drift;
- earliest position at which a global recency result is consumed;
- minimum winner-versus-runner logit margin;
- worst softmax contamination of the selected value;
- any true rank inversions, as distinct from correct-but-soft selections.

These measurements determine whether `E = 0.45` is a defensible bound and
whether an all-history mean would hide a slow slope change.

## Prototype and test plan

### Phase 1: pure numerical replay

Implement both rank formulas in a CPU replay over:

1. the captured production `raw[t]` sequence;
2. bounded synthetic errors with `E = 0.45`:
   - alternating `+E, -E`;
   - isolated positive and negative spikes;
   - long constant-bias runs;
   - a bias step midway through the rollout;
   - slow sinusoidal drift;
   - worst-case endpoint errors for each `K`;
3. the existing recency candidate trace, extended to substitute each rank.

For each strategy, compute:

```text
minimum adjacent rank step
maximum adjacent rank step
rank range
minimum selection logit margin
minimum winning softmax weight
argmax flips
content-dominance violations
```

Sweep `K` over `{4, 8, 16}`. Keep `gain = 8` initially so the comparison isolates
the rank. Sweep `gain` only after measuring rank quality and rechecking match
gain.

### Phase 2: graph-oracle unit tests

Add focused tests for the rank builders before changing `pick_most_recent`:

- Perfect input positions produce unit-slope rank.
- Constant raw bias does not change adjacent steps.
- Bounded adversarial errors meet the formulas above.
- All-history mean behaves correctly at the first several positions.
- Fixed-`K` startup follows the selected explicit policy.
- Rank is strictly monotone over a long oracle sequence.
- Value-type/range claims match computed values.

Then test recency selection with:

- adjacent matching markers carrying very different values;
- two adjacent markers queried immediately and thousands of positions later;
- many matching markers;
- a single old match plus recent nonmatching rows;
- marker values containing scalars, lifted keys, and `+1/-1` booleans;
- `exclude_self=True` where supported.

Assert both the selected value and attention hardness. A value-only tolerance can
hide a meaningful blend when the two candidate values happen to be close.

### Phase 3: compiled primitive tests

At production `d_head = 128`, `d_rot = 64`, and sequence cap 65,536, verify:

- graph oracle versus compiled prefill;
- full prefill versus token-by-token cached decode;
- all-history equal-logit mean accuracy at increasing sequence lengths;
- every fixed offset used by the chosen `K`;
- minimum compiled adjacent rank step;
- minimum compiled recency hardness;
- physical head count, layer count, residual width, and KV cost.

Run at short lengths in ordinary CI and add a full-cap or representative
long-context validation on the render GPU. Analytic full-cap validation is not
sufficient for the fp32 accumulation issue this change is intended to fix.

### Phase 4: Doom integration behind a build-time strategy

Wire the selected candidate into `GraphPast.pick_most_recent` behind the
build-time strategy while keeping `raw` available for A/B comparisons.

Run:

- existing `tests/past/test_graph_past.py` recency tests;
- clip-memory recency tests;
- scene/oracle tests involving recent marker and recent key handles;
- the original failing render;
- a full reference render comparison;
- prefill/cached rollout parity;
- compile cost and schedule regression checks.

Update comments in `past.py`, `attention_handles.py`, `render_constants.py`, and
torchwright's global-recency operation so they describe `recency_rank`, its
range, and its measured minimum step rather than assuming raw adjacent position
differences equal one.

## Acceptance gates

A candidate may replace the raw rank only if all gates pass:

1. **Monotonicity:** zero non-positive adjacent rank steps over the full measured
   rollout and all bounded-error fixtures.
2. **Selection correctness:** zero most-recent argmax flips in the full candidate
   replay and render tests.
3. **Hardness:** minimum winning weight at least `0.999`, or a stricter threshold
   derived from the most sensitive `+1/-1`, integer-id, and lifted-key consumers.
4. **Content isolation:** matching rows beat all nonmatching rows at every
   position under the measured rank range.
5. **Runtime parity:** compiled prefill and cached decode agree within the stated
   numerical budget.
6. **Render parity:** the failing render is fixed and no reference-render
   regression appears.
7. **Cost:** the production graph stays inside head, layer, residual-width, KV,
   compile-time, and render-memory budgets.
8. **Documentation:** the error assumption, measured envelope, startup behavior,
   and content-dominance calculation are recorded next to the implementation.

## Decision criteria

Prefer **all-history mean** if:

- the dedicated uniform causal mean is numerically stable at full length;
- raw recovery has no meaningful late slope regime change;
- one extra shared head materially beats the fixed-window cost;
- its early-position behavior meets the hardness gate.

Prefer **fixed-`K` mean** if:

- measured `K`-step slopes have a strong lower bound;
- the raw recovery has slow drift that the all-history mean follows too slowly;
- the additional offset heads fit comfortably, or an exact local-window mean
  primitive is implemented;
- startup can be handled without an unsafe out-of-range read.

If both pass, compare minimum compiled hardness and total scheduled cost. Do not
choose based only on the simpler formula.

## Initial recommendation

Prototype the all-history mean first because it offers a genuine one-head shared
rank and has natural startup behavior. In parallel, keep `K = 8` as the fixed
window reference: with a validated `E = 0.45` bound, it guarantees an adjacent
rank step of at least `0.8875` before compiled numerical error.

Do not yet change the default recency path. The first deliverable is a replay
report showing, for raw rank, all-history mean, and `K` in `{4, 8, 16}`, the
minimum adjacent rank step, minimum selection margin, minimum winning weight,
argmax flips, and estimated graph cost.
