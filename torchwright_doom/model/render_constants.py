"""Forward/render constants.

This module runs once at compile time: it builds part of the computation
graph that torchwright lowers into the transformer's weights. Nothing here
executes during inference — at render time, only the compiled transformer
runs. Coined terms: see GLOSSARY.md.

Render-domain magnitudes, kept separate from :mod:`.constants` (which holds
the vocab-scale ``SCREEN_WIDTH`` / ``SCREEN_HEIGHT`` and must stay
dependency-free); each value here is wrapped by :func:`.std.constant` at its
call site (see the note below), not at import.
"""

from __future__ import annotations

# Used on long ``pick_most_recent`` spans where a false match must lose
# hard to the one real 0/1 marker. Sized so a content-matched key beats an
# unmatched newer key at any position:
# ``match_gain * content_gap > RECENCY_GAIN * max_positions``. The marker
# content gap is 1 (0/1 marker), so at ``RECENCY_GAIN = 8`` and the
# ``max_positions = 65536`` RoPE cap this needs ``match_gain > 524_288``;
# 600_000 clears it with 12.6% headroom. (Was 300_000 under the pre-RoPE
# ``SCORE_GAIN`` when the dominance bound was the ~32768 rollout span, not
# the position cap.) Safe in fp32 (ULP at 600_000 ≈ 0.07, far below the
# gain-8 recency step). The
# op/facade default stays 200.0; long-span callers thread this explicitly.
MATCH_GAIN_LONG = 600_000.0

# Wall-column per-column clip-array recovery (``ClipMemory.pick_most_recent``).
# The radix column key's match dot is ``bucket_match + digit_match`` (one-hot
# products): a full match (correct column) scores 2, but a PARTIAL match (a
# different column sharing the query's bucket OR digit — common, since ~13
# buckets cover 160 columns) scores 1. So the binding content gap that recency
# must not overturn is full-vs-partial = 1 * match_gain, NOT 2 * match_gain: a
# more-recent partial-match column must lose to the correct full match. With
# RECENCY_GAIN = 8 that needs ``match_gain > RECENCY_GAIN * max_positions``
# = 8 * 65536 = 524_288, so 600_000 (the old 300_000 cleared only the 2*gain
# vs total-non-match case, which left a sibling-bucket column updated >37500
# positions later able to win at full-res ~42k). The radix key has no fp32
# cancellation, so any sufficiently large gain works.
MATCH_GAIN_CLIP = 600_000.0

# Per-position recency gain for ``pick_most_recent`` — the position tiebreak in
# torchwright's ``attend_most_recent_globally``. Adjacent positions differ by
# RECENCY_GAIN in the attention logit, so among content-matching keys the most
# recent wins with softmax weight ``exp(RECENCY_GAIN)/(exp(RECENCY_GAIN)+1)``;
# at 8 that is ≈ 0.99966 (cond ≈ 0.9993) — sharp enough for the ±1 boolean
# marker reads. The binding consumer is the swiglu gated ops' shared ±1
# assert (``_assert_cond_pm1``, c_tol=0.005: cond > 0.995; a cond off by δ
# mis-scales the gated value by δ·|value|); the select-family mask contract
# is looser (``_MASK_TOL`` ≈ 0.0087, cond > ~0.9913). This mirrors
# the pre-RoPE scheme's ``SCORE_GAIN = 8`` exactly: old recency was ``8*counter``
# (a global absolute position × gain 8); the new mechanism recovers the same
# global absolute position from the BOS softmax weight (smoothed — an exact
# causal mean of the recovery, so its adjacent step stays ≥ 0.965 over a
# full-cap rollout where the raw recovery dipped to 0.53) and applies the same
# gain, so it is the same mechanism, not a softer one. torchwright's op default is 1.0
# (exp(1) ≈ 0.73 — far too soft for a boolean); DOOM threads this 8.0 through the
# facade. Content dominance (a matched older key must beat an unmatched newer
# key) requires ``match_gain * content_gap > RECENCY_GAIN * max_positions``; the
# MATCH_GAIN_* values are sized against that at the rollout span.
RECENCY_GAIN = 8.0

# Renderer protocol enums and sentinels, shared by the wall-column and visplane
# owners. All are plain floats wrapped by ``std.constant`` at their call sites
# (see the note below).

# R_CHECK_PLANE / PLANE_MARK ``kind`` slot: which plane a mark refers to.
PLANE_KIND_CEILING = 0.0
PLANE_KIND_FLOOR = 1.0

# Wall-part index sentinel: no upper/mid/lower part to draw (valid parts 0/1/2).
PART_NONE = 3.0

# Open (unclipped) ceiling bound for a column's clip array — one row above the
# top of the screen, so any real ceiling clips below it. The matching open floor
# bound is the view bottom ``VIEW_HEIGHT`` (== ``SCREEN_HEIGHT`` with no bar).
OPEN_CLIP_CEILING = -1.0

# A recovered one-hot dot scores ~1 when its row is present and ~0 when absent;
# this threshold separates the two in the radix-successor presence checks.
PRESENT_THRESHOLD = 0.9

# NOTE (the import-time-node rule; see GLOSSARY.md): a ``constant`` is a graph
# ``Node`` with a global auto-incrementing id; creating one at import time
# gives it a fixed low id that collides with test-built nodes after the test
# harness resets the id counter (``tests/conftest.py``), aliasing them under
# ``reference_eval`` / ``probe_compiled`` memoization. So — matching
# ``extract.py``'s convention — the 1.0 query constant is created *inside* the
# call sites (via ``std.constant(1.0)``) rather than here.
