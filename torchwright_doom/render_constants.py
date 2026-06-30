"""Forward/render constants.

Real-side mirror of the read-side render-constant subset. Kept separate from
:mod:`.constants` (which holds the vocab-scale ``SCREEN_WIDTH`` /
``SCREEN_HEIGHT`` and must stay dependency-free): the values here are
render-domain magnitudes, wrapped by :func:`.std.constant` at their call
sites (see the note below), not at import.
"""

from __future__ import annotations

# Used on long ``pick_most_recent`` spans where a false match must lose
# hard to the one real 0/1 marker. Sized so a content-matched key beats an
# unmatched newer key at any position:
# ``match_gain * content_gap > RECENCY_GAIN * max_positions``. The marker
# content gap is 1 (0/1 marker), so at ``RECENCY_GAIN = 8`` and
# ``max_positions = 61440`` this needs ``match_gain > 491_520``; 600_000 clears
# it with headroom. (Was 300_000 under the pre-RoPE ``SCORE_GAIN`` when the
# dominance bound was the ~32768 rollout span, not the 61440 cache cap.) Safe
# in fp32 (ULP at 600_000 ≈ 0.06, far below the gain-8 recency step). The
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
# = 8 * 61440 = 491_520, so 600_000 (the old 300_000 cleared only the 2*gain
# vs total-non-match case, which left a sibling-bucket column updated >37500
# positions later able to win at full-res ~42k). The radix key has no fp32
# cancellation, so any sufficiently large gain works.
MATCH_GAIN_CLIP = 600_000.0

# Per-position recency gain for ``pick_most_recent`` — the position tiebreak in
# torchwright's ``attend_most_recent_globally``. Adjacent positions differ by
# RECENCY_GAIN in the attention logit, so among content-matching keys the most
# recent wins with softmax weight ``exp(RECENCY_GAIN)/(exp(RECENCY_GAIN)+1)``;
# at 8 that is ≈ 0.99966 (cond ≈ 0.9993) — sharp enough for the ±1 boolean
# marker reads (``assert_bool``'s c_tol=0.005 needs cond > 0.995). This mirrors
# the pre-RoPE scheme's ``SCORE_GAIN = 8`` exactly: old recency was ``8*counter``
# (a global absolute position × gain 8); the new mechanism recovers the same
# global absolute position from the BOS softmax weight and applies the same gain,
# so it is the same mechanism, not a softer one. torchwright's op default is 1.0
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

# NOTE: the original keeps ``ONE``/``ZERO``/``FALSE`` as module-level
# ``constant`` Vecs. On the real side a ``constant`` is a graph ``Node`` with a
# global auto-incrementing id; creating one at import time gives it a fixed low
# id that collides with test-built nodes after the test harness resets the id
# counter (``tests/conftest.py``), aliasing them under ``reference_eval`` /
# ``probe_compiled`` memoization. So — matching ``extract.py``'s convention —
# the 1.0 query constant is created *inside* the call sites (via
# ``std.constant(1.0)``) rather than here.
