# Evaluating a replacement position encoding

This document is for an agent tasked with **changing torchwright's position
encoding** and proving the change is safe for the DOOM renderer. It explains
exactly what the current encoding is used for, what a replacement must preserve,
and how to validate a candidate against real renderer behavior using the
instrumentation artifact in this directory.

Read this top to bottom before proposing a scheme. The constraints in
[§3](#3-hard-requirements-a-replacement-must-satisfy) are load-bearing — most
"obvious" swaps (RoPE-as-is, NoPE, low-resolution / log-bucketed recency,
`1/pos`-style decays) violate at least one and silently corrupt the render.

---

## 1. What position is actually used for

The graph is compiled to a transformer; position enters attention in exactly
**three** ops (everything else in `torchwright/ops/attention_ops.py` selects on
content only and never reads position). Source: `torchwright/graph/pos_encoding.py`.

The position encoding (`PosEncoding`) has **two independent parts**, used by
disjoint mechanisms:

| Part | Columns | Built by | Consumed by |
|---|---|---|---|
| **Sinusoidal trig block** | `0 .. d_pos-2` (even count) | `get_pos_encoding` (sin/cos pairs, base 10000) | `attend_to_offset` only |
| **Raw integer counter** | last column (`counter_col`) | `pe[:, counter_col] = arange(n_pos)` | recency tiebreak in `attend_most_recent_matching`, `get_prev_value` |

The two halves are never mixed. A replacement can redesign them independently,
but must satisfy **both** sets of requirements below.

### 1a. Recency selection — `attend_most_recent_matching` (a.k.a. `pick_most_recent`)
`torchwright/ops/attention_ops.py` (~line 1383).

- **Selection:** among causal positions, pick the one with the largest
  *content* dot product `query·key`; **break ties toward the most recent**
  (largest counter).
- **Position's role:** one extra logit column = `_QUERY_GAIN · counter`
  (`_QUERY_GAIN = 8.0`). Content lives in *separate* qk columns, so
  `logit = content_logit + 8·counter` exactly — content and position are
  additive and separable.
- **Caller invariant** (asserted at build): `match_gain·content_margin >
  _QUERY_GAIN·n_pos`. With `MATCH_GAIN_* = 300_000`, content dominates up to
  ~37.5k positions; recency only ever disambiguates *content-equal* candidates.
- **Degenerate fallback:** if no causal key content-matches (the marker/key
  isn't present yet), content is uniform and the op returns the most recent
  position. This is common early in a frame and is **not** a real selection —
  the validation harness filters these out (see §4).

### 1b. Exact-offset read — `attend_to_offset(value, delta_pos)`
`torchwright/graph/pos_encoding.py` (~line 95).

- **Selection:** the token at *exactly* `current_pos + delta_pos`.
- **Position's role:** the entire score is the trig block. The key side is
  multiplied by `trig_shift_matrix(-delta_pos)` (a per-frequency rotation), so
  the logit peaks where `key_pos == query_pos + delta_pos`. There is **no
  content** — this is pure position.
- **Usage in DOOM:** `delta_pos ∈ {-1, -2, -3}` (pipeline/register-spill reads:
  previous token type, scale sidecars, SET_CURSOR_X data 3 back, etc.).

### 1c. `get_prev_value` — condition-gated recency
Same counter-based recency pattern as 1a; **not currently used in DOOM** but
part of the public surface. If you change the counter semantics, keep this op
working too.

---

## 2. The instrumentation artifact

`scripts/position_attention_log.py` hooks the one chokepoint every position-using
op compiles through (`Attn.compute`), teacher-forces a real frame through the
exact-math oracle (`reference_eval`), and logs **every position-using selection**,
splitting each into the frozen *content* score and the replaceable *position*
score.

### Running it

```bash
# Full low-res frame (~11.6k positions, through DONE) on a B200, JSONL to the
# mounted Volume, then pull it locally:
MODAL_RUN_GPU=B200 MODAL_RUN_TIMEOUT=7200 MODAL_RUN_GPU_MEMORY=65536 \
  make modal-run MODULE=scripts.position_attention_log \
  ARGS="--frame --config configs/e1m1_lowres.yaml --validate --device cuda \
        --max-queries 5000 \
        --out /root/.cache/torchwright_doom/compiled/position_attention_log_full.jsonl"
modal volume get torchwright-doom-render-cache \
  position_attention_log_full.jsonl scripts/
```

Notes / gotchas (all learned the hard way — don't rediscover them):
- **Must run on `run_gpu`** (the assets image). `run_cpu` has no `configs/`/WAD.
- **`reference_eval` has no device knob.** The script moves the graph tensors to
  `--device cuda`; on CPU it is O(n²) and will time out on a real frame.
- **Don't reintroduce per-position `.item()` in the hook** — it syncs the GPU
  every iteration and was the original bottleneck. Keep stats vectorized.
- **Modal caps captured stdout (~hundreds of lines).** Large dumps truncate;
  always write the JSONL to the Volume and `modal volume get` it.
- **`--span N`** caps the AR body for fast iteration; **omit it for the full
  frame** (default is None = through DONE). `--config configs/e1m1.yaml` is the
  full 320×200 frame (~42k positions — heavier; size the GPU/timeout up).
- Ephemeral Modal apps die when the laptop sleeps — use `modal run --detach` for
  long runs.

### JSONL record schema

One record per `(head_uid, query_pos)`. Common fields:
`uid`, `kind` (`"recency_match"` | `"offset"`), `query_pos`, `n_pos_at_capture`,
`selected` (key position chosen under the **current** scheme = ground truth),
`target_delta = query_pos - selected`.

`recency_match` adds:
- `candidates`: `[[key_pos, content_logit], ...]` — the keys the new scheme must
  rank correctly (top-K by full logit ∪ content-tied keys, capped at
  `_MAX_CANDIDATES = 64`, keeping the most-recent ties). **Content is frozen
  here**; your scheme only changes the position term added on top. (The cap
  matters: in the no-match fallback every causal key is "tied"; without it a
  full-frame log balloons to multiple GB. Those rows are excluded from
  validation by `is_real` anyway.)
- `selected_content_logit`, `selected_pos_logit`
- `content_argmax`, `recency_decisive` (did content alone pick a *different* key
  than content+recency? — i.e. is position load-bearing for this selection)
- `content_margin` (best − 2nd-best distinct content), `n_content_ties`
  (#causal keys within ε of the top content tier).

`offset` adds:
- `delta` (= `selected - query_pos`), `peak_logit`, `runner_logit`,
  `peak_margin` (logit at the target minus the best off-target — the sharpness
  your encoding must reproduce).

---

## 3. Hard requirements a replacement must satisfy

A candidate position encoding is **only** safe if it provides all of these. State
which mechanism each part of your design serves before implementing.

**R1 — Exact monotone recency (counter half).** `attend_most_recent_matching`
needs a per-position scalar that is *strictly increasing* with position and
resolves **adjacent** positions. On the full frame, recency must split
content-tied candidates as little as **1 token apart**, and at large absolute
positions the relevant gaps are a tiny *fraction* of the position. Any scheme
whose recency resolution degrades with position (log-buckets, `1/pos`, fixed
relative precision, coarse quantization) **fails**: the coarse ">10% relative
distance" probe flips **224,852 / 276,831 (81%)** of real selections on the full
low-res frame.

**R2 — Bounded recency magnitude vs content.** The recency term must stay small
relative to the content match gain (current invariant
`match_gain·content_margin > _QUERY_GAIN·n_pos`). A recency signal that grows
unbounded or becomes comparable to content lets off-target keys win. If your
scheme changes the counter's scale, re-derive this bound and keep the assert.

**R3 — Exact small-offset shift (trig half).** `attend_to_offset` must resolve
`query_pos + delta` for `delta ∈ {-1,-2,-3}` with a sharp, unique softmax peak
(current peak margin ≈ 51 logits, scene-independent). Your encoding needs an
equivalent **closed-form, position-unique** shift-by-k operator. RoPE-style
rotations qualify in principle; NoPE/learned-absolute do **not** give an exact
analytic shift and would force rewriting all ~12 `attend_to_offset` call sites as
counter-arithmetic instead.

**R4 — `get_prev_value` parity.** Same counter contract as R1 (§1c).

---

## 4. How to validate a candidate (replay)

Express your scheme's position contribution as a Python function
`p_new(query_pos, key_pos, n_pos) -> float` (the logit it adds for attending from
`query_pos` to `key_pos`). Then replay it against the logged real selections:

```python
import json, math
recs = [json.loads(l) for l in open("scripts/position_attention_log_full.jsonl")]

def is_real(r):                      # exclude the no-match recency fallback
    nk = r["query_pos"] + 1
    return nk >= 2 and r["n_content_ties"] < nk

def replay(p_new):
    flips = 0; checked = 0
    for r in recs:
        if r["kind"] != "recency_match" or not is_real(r):
            continue
        checked += 1
        best = max(r["candidates"],
                   key=lambda c: c[1] + p_new(r["query_pos"], c[0], r["n_pos_at_capture"]))[0]
        if best != r["selected"]:
            flips += 1
    return checked, flips

# Sanity: the current scheme reproduces every selection.
print(replay(lambda q, k, n: 8.0 * k))            # -> (276831, 0)
```

**Pass condition for the recency half:** `flips == 0` on the full-frame log
(`is_real` selections). Any flip is a pixel the renderer would get wrong.
Bucket flips by `query_pos` to see *where* a scheme breaks.

The **offset half (R3) is not replayable** — `attend_to_offset` has no content
to freeze. Validate it analytically: confirm the new encoding yields a unique
maximum at `query_pos + delta` for `delta ∈ {-1,-2,-3}` across the full position
range, with margin comparable to the current ~51 logits. The `offset` records
are the coverage list of `(query_pos, delta)` pairs that occur.

**Beyond replay:** once a scheme passes replay, the real gate is the
graph-level oracle tests (`tests/scene/test_flat_pixel_oracle.py`,
`test_forward_ar_rollout.py`) and then the full render-vs-pydoom comparison
(`make run COMPARE=1`). Replay is a fast pre-filter, not a substitute.

---

## 5. Current empirical findings (full low-res frame, 11,624 positions)

- 99 recency heads, 12 offset heads.
- **276,831** real content-driven recency selections. Current `8·counter`:
  **0 flips**. Coarse ">10% relative": **224,852 flips (81%)** — fine,
  position-proportional resolution is broadly load-bearing, and *more* so at
  large positions (the failure rate rose from 68% at a 6.6k-position window to
  81% on the full frame).
- A subset of recency heads must split candidates exactly **1 token apart**
  (tie gap = 1); most sit at a few-token tie gap, which is still well under any
  fixed *relative* tolerance once absolute positions are large.
- Offset heads: `delta ∈ {-1,-2,-3}`, peak margin ≈ 51 logits, independent of
  scene/position.

The data file is **generated on demand**, not committed (a dense full-frame log
is large). Regenerate with the command in §2; `--max-queries` sets the
per-head query subsample (raise for denser coverage, lower for a quick file).
With the `_MAX_CANDIDATES` cap in place a full low-res frame is a manageable
size; pull it from the Volume as shown.
