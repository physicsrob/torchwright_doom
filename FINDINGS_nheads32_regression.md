# Findings: the n_heads=32 low-res render regression

Investigation of the 2026-07-14/15 low-res within-option regression
(100.0% → 99.5%, +13 emitted rows). Each finding carries a confidence:
**High** = directly measured, usually twice or with a control; **Medium** =
measured once or inferred with strong support; **Low** = hypothesis
consistent with the data but not isolated.

Artifacts referenced throughout: clean-old = bundle `4fe366…` (35L,
64-head, pre-routing torchwright), bad = `e4c410…` (38L, 32-head, schedule
fingerprint `406cd5cd…`), de-confounder = `760c2672…` (35L, 64-head,
current code, fingerprint `cdc5ac19…`). Reference stream =
`e1m1_lowres__x1056_y-3616_a64__1784073549`'s retained `output.ids.json`.
Tooling: `scripts/teacher_force_margins.py`,
`scripts/schedule_regression_probe.py` (two-schedule replay + node diff).

---

## 1. The regression is real, systematic, and value-typed — not structural decay

**High.** Teacher-forcing the bad bundle along the clean stream (drift-free:
every position is an independent measurement) shows 756/17,336 argmax
disagreements. 96% are same-token-type value picks one quantization bin
off; the earliest are `angleValue` flips at exact radix boundaries
(−3072 vs −3073, −1024 vs −1025) recurring deterministically from emitted
index 103. ~32 disagreements are protocol-boundary flips
(pixel↔setCursor), which account for the +13-row drift. The machine's
structure (token types, protocol shape) is intact.

## 2. The A/B that surfaced the regression was confounded, and the code alone is exonerated at 64 heads

**High.** Four torchwright commits implementing MLP-Add routing
(7a5bf57, a24f21c, 217e452, 660e5e7) landed 18:32–23:06 on Jul 14 —
after the clean 64-head compiles (16:42) and before the regressed 32-head
compiles (23:18). De-confounder: a 64-head low-res bundle compiled under
current code renders **100.0% within-option, 100.0% coverage, and exactly
17,336 rows** — indistinguishable from the clean baseline at the gate
level. The regression therefore requires the 32-head condition; "new code
alone broke it" is refuted at this screen/geometry.

Caveat (Medium): "n_heads=32 under *old* code" was never run (the old
compiler is gone from the tree), so "head scarcity alone" vs "head
scarcity × new code" is not separated by the gate experiments. Finding 7
makes this distinction less load-bearing than it appears.

## 3. The 32-head schedule inverts the cancel-mechanism mix and lightly reshuffles routing

**High** (read directly from the two cached CP-SAT snapshots).
Clean-old 35L: 3,697 attention cancels / 1,023 MLP-mechanism cancels.
Bad 38L: 969 / 3,751. Op routing barely moves (89 Linears attn→mlp, 44
back; the 89 are mostly `is_type_dot_*` ±1 predicates). Whether the cancel
inversion is *causal* for the regression is *not* established (see 7);
it is the largest structural difference between the schedules.

## 4. The `R_RenderSegLoop` staircase "corruption" is a pre-existing, schedule-independent claim violation — a red herring

**High** (triple-verified). The `floor_int_step` nodes under
`proj/paint/R_RenderSegLoop` (claimed range [−0.502, 2.502]) read 4.0–32.0
at ~229 protocol-inactive positions: (a) in the bad schedule, (b) in the
clean schedule (identical bad-entry count), and (c) **in exact-math
reference evaluation of the graph on the production prompt**. Any
`debug=True` forward on this graph+prompt fires this assert first,
masking all later checks. It says nothing about any schedule.
Follow-up worth filing: either the claim is wrong or the guard intent is
"active positions only" — as-is it makes assert sweeps unusable on real
prompts.

## 5. The inter-schedule differences at emitted positions are mostly ordinary noise plus a handful of discrete flips

**High.** Node-by-node diff of the two compiled artifacts on the same
stream (918 nodes differ in the emitted window): the magnitude histogram
bulks at 1e-3–1e-1 (normal cross-schedule PL/fp variation) with ~46 nodes
≥ 1e0 clustered around a few events (emit 83, 94, 101, 102, 103, 104).
Calibration point: the clean-old and de-confounder schedules also differ
from each other (exact-color 91.6% vs 90.8% against pydoom, both 100%
within-option) — *some* boundary-sitting picks flip between ANY two
schedules; the regression is that the 32-head schedule's flips escape
option sets and protocol margins.

## 6. The first discrete flip is a position-keyed attention read fetching the wrong row, and its seed is traced end-to-end

**High for this cascade.** At emit[83], an attention read in the
wall-range storage machinery (`proj/stor` / R_StoreWallRange) outputs
−23.93 (bad) vs −0.65 (clean). Its query and value inputs are
bit-identical; only the **key** differs. The key is 2 lanes:
a ±1 predicate (identical) and **the row's absolute position scalar**
(differs by 0.55 at row 3530). Upstream trace, all at position 3530:

    BOS-attention weight:        0.94929504 vs 0.94930267   (Δ 7.6e-6)
      └─ bos_weight_to_position_0_1024 (PWL inversion, @scene)
    recovered position:          3529.377   vs 3529.929     (Δ 0.55, ×72,000)
      └─ stored as the key's position lane for row 3530
    attention pick flips at emit[83] → wrong wall-range payload fetched
      └─ cascade: wallColU digit wrap (Δ512), ray_scaled (Δ7.8),
         bbox angle (Δ1.296 → the −3072/−3073 emits), colormap rows

**Medium** that this single seed explains *all* 756 flips: the position
scalar feeds keys at every row, so one wobbling scalar plausibly perturbs
many reads, but only the emit[83] cascade was traced end-to-end.

## 7. The amplifier is `global_position_from_bos`, and its correctness has been resting on bit-reproducibility, not numerical margin

**High for the mechanism; Medium for the "all along" framing.** The
readout inverts a BOS-attention weight through a PWL to recover the
absolute position; at P≈3,530 the measured gain is ~7.3e4 (0.55 / 7.6e-6).
fp32 attention does not hold a weight stable to better than ~1e-6..1e-7
across *differently-realized* compiles (different shapes → different
kernels/reduction orders; different folded constants; different upstream
accumulation). Every prior artifact pair computed this weight
bit-identically (ONNX/HF parity is bit-exact because shapes match);
n_heads=32 was the first production compile of this graph at a different
shape. Under this model, the Add-routing commits and head scarcity chose
*which side of the margin* the noise landed on rather than creating the
fragility. Note the clean schedule recovered 3529.93 against a true 3530 —
error 0.07 — so even the good artifact consumes margin here.

## 8. The consumer budget is a score gap, not a rounding boundary

**High.** The recovered position is consumed *continuously* as an
attention-key lane: a position error of e shifts a candidate row's score
by e × (query position-gain). The tolerable key error is
(designed score gap) / (position gain). Earlier "wrong side of the .5
rounding boundary" framing was incorrect and is retracted; nothing rounds
on this path.

## 9. Why the BOS weight differs between schedules — RESOLVED: generated inside the head's own computation

**High** (zero-threshold input comparison, 2026-07-15). All three inputs
to the boosted-BOS head are **bit-identical** between the two artifacts:
the query (literal 1.0), and the key and value (both `is_type_01_bos`).
The 7.6e-6 weight difference is therefore created entirely *inside* the
attention computation — the two artifacts pack this head into
differently-shaped fused QKV tensors (32- vs 64-head layers), so
onnxruntime selects different kernels with different fp32 summation
orders, and the folded RoPE/head constants round differently. The
upstream doors (residual-accumulation differences, new bypass-Add /
MLP-cancel residue) are refuted for this seed: nothing upstream of the
head differs at all.

The noise signature confirms it: the weight differs at 3,055 of 3,724
positions with mean |Δ| ≈ 1.0e-6 — broadband realization noise across
nearly every position, not a localized anomaly. This also fully
exonerates the Add-routing commits as a *numerical* contributor: their
only role was helping produce a different-shaped artifact (finding 2's
caveat is now moot).

## 10. The originally-hypothesized mechanisms (swish-bypass residue H1, cancel-residue-on-reuse H2) are NOT the direct seed of the traced cascade

**High for emit[83]; Medium generally.** The measured seed is 7.6e-6 —
about eight orders of magnitude above 2⁻⁴¹-scale bypass residue — and the
corrupted-column reuse story was directly refuted for the staircase case
(the sub's output column is clean before and after its tenancy; the
staircase values match exact math). Residue may still contribute through
door 9(d), but no observation requires it.

## 11. The inversion's amplification is intrinsic to the encoding; a more accurate inversion cannot fix this

**High** (read from `torchwright/ops/swiglu/global_recency.py` + measured
numbers). The forward encoding is `w(m) = C/(C+m)` with
`C = 65,536^cos(m·θ_slow) ≈ 65,536`; the recovery gain is the reciprocal
slope `(C+m)²/C ≈ 72,800` at m=3530, matching the measured 0.5515/7.6e-6
= 72,300. Any inversion — including a perfect one — has exactly this
gain, because it is the derivative of the true inverse. The error
decomposes: the PWL table's own static bias is −0.07 (clean recovers
3529.93 vs true 3530); the cross-compile 0.55 is entirely input noise ×
intrinsic gain. Equivalently: at m≈3530 the weight signal separating
adjacent positions is 1.4e-5 (~230 fp32 ulps) and the measured
cross-realization wobble is 7.6e-6 (~128 ulps) — over half a
position-step. The gain grows quadratically with position: ~97k at the
production frame's ~14k rows, ~247k at the design max 61,440 — full-res
sits deeper in this regime than low-res.

## 12. The blended read is `store_range_row.pick` in `RecentDrawsegState.read_from_recent_rows`, operating in its no-match regime

**High** (five independent signature matches: annotation `proj/stor`,
4-wide value split [1,3], ±1 marker key with constant query, gain
600,000, recency-only resolution). `wall_range_state.py` — the read
recovers the drawseg store-progress state (store_i + 3-lane store key)
from the most recent R_StoreWallRange row. At qpos 3697 **no marker row
had fired yet** (every candidate's match lane is −1; winner logit =
−600,000 + 8×3697 exactly): the read is in its "sought row doesn't
exist yet" regime, relying on the value handle being zero-gated on
inactive rows. The bad schedule's 1.5% blend (winner weight 0.985 vs
0.9996; the 0.48-position relative key error costs 8×0.48 ≈ 3.9 logits
of the designed 8-logit adjacent-row gap) leaks the neighbor row's
published lane (≈ −1,560, world-scale) into the payload → −23.93 vs
−0.65, live at least several nodes downstream. Not yet verified whether
this specific read (vs a matched-regime sibling) is what ultimately
moves the emitted tokens.

## 13. Schedule-cache / fingerprint mechanics (investigation infrastructure facts)

**High** (bisected empirically). `graph_fingerprint` embeds
`compiler_code` (hash of every torchwright package .py), so schedules
solved under older code are permanently unreplayable after any torchwright
edit — diff against a baseline recompiled under current code instead.
`n_heads` equal to d//d_head is omitted from the payload (explicit 64
hashes like None); `reserve_residual=2` at d=8192 / rms_const_exp=63.
Local fingerprint reproduction: nh32+reserve2 → `406cd5cd…` (bad),
nh64+reserve2 → `cdc5ac19…` (de-confounder).

## 14. Measurement hygiene findings (how this investigation misled itself, recorded for next time)

**High.** (a) The first 5-seed n_heads=32 sweep omitted `d_hidden: 16384`
and measured a half-width-MLP machine — its "47 layers, CP-SAT UNKNOWN ×5"
conclusion was an artifact; real production compiles solve to 38L.
(b) The staircase assert (finding 4) was believed for several runs because
no clean-schedule control had been run. (c) Prompt-position lanes are
largely protocol-inactive; node diffs must be restricted to emitted
positions (or active lanes) before ranking.

---

## Open questions / queued experiments

1. ~~Position-recovery error curve~~ **done**: clean spread 0.32, bad
   0.94, one row past 0.5 absolute (but the consumer budget is relative —
   finding 8; the binding pair is rows 3697/3696).
2. ~~Seed-walk below 1e-4 into the BOS head's inputs~~ **done** —
   resolved finding 9 (inputs bit-identical; noise generated in-head).
3. **Trace a second cascade** (e.g. the emit[103] angle family) to test
   the single-seed theory (finding 6's Medium claim).
4. **Unmask `assert_picked_from` / score-gap asserts** fully: requires
   stripping the near-tolerance integer claims that fire on both
   schedules before the pick asserts run.
5. **Design-budget check**: what noise budget did the
   `global_position_from_bos` design assume for the BOS weight, and what
   score gaps do the position-keyed handles reserve? Finding 11's gain
   arithmetic says the answer at current gains is "≈1.5 position-steps of
   weight noise at m=3530, shrinking quadratically with m" — the design
   question is whether RECENCY_GAIN=8 vs the 600k match gain leaves any
   realization-noise margin at production stream lengths.
6. **Same-artifact run-to-run stability**: run one artifact's debug
   forward twice and compare the BOS weight bitwise — torchwright's docs
   warn fp32 GPU accumulation can vary 1e-6..1e-5 run-to-run, which would
   put even a frozen artifact within noise of the flip margin.
7. **Downstream gating of the blended no-match read** (finding 12): does
   the contaminated fallback payload pass through a `present`-style
   validity gate, or is a matched-regime sibling the actual carrier?
