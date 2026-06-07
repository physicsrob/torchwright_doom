"""Forward/render constants.

Real-side mirror of the read-side subset of
``doom_sandbox/implementation/forward/constants.py``. Kept separate from
:mod:`.constants` (which holds the vocab-scale ``SCREEN_WIDTH`` /
``SCREEN_HEIGHT`` and must stay dependency-free): the values here are
render-domain magnitudes, wrapped by :func:`.std.constant` at their call
sites (see the note below), not at import.
"""

from __future__ import annotations

# Used on long ``pick_most_recent`` spans where a false match must lose
# hard to the one real 0/1 marker. Sized so
# ``match_gain * content_gap > SCORE_GAIN * max_recency_span``; 300_000
# gives headroom over the longest rollout. This is safe
# (fp32 on all paths). The op/facade default stays 200.0; long-span
# callers thread this explicitly.
MATCH_GAIN_LONG = 300_000.0

# Wall-column per-column clip-array recovery (``ClipMemory.pick_most_recent``).
# The radix column key's match dot is ``bucket_match + digit_match``
# (one-hot products, no cancellation), so the gained matched logit is an exact
# ``2 * match_gain``; this must dominate the recency span, exceeding
# ``SCORE_GAIN (8) * max_recency_span`` (8500-pos rollout, 32768-pos regression).
# 300_000 > 8 * 32768 with headroom. (An earlier lifted-key form needed a power
# of two to keep its ``match_gain*c^2`` cancellation fp32-exact; the radix key
# has no such cancellation, so any sufficiently large gain works.)
MATCH_GAIN_CLIP = 300_000.0

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
# bound is ``SCREEN_HEIGHT``.
OPEN_CLIP_CEILING = -1.0

# A recovered one-hot dot scores ~1 when its row is present and ~0 when absent;
# this threshold separates the two in the radix-successor presence checks.
PRESENT_THRESHOLD = 0.9

# NOTE: the sandbox keeps ``ONE``/``ZERO``/``FALSE`` as module-level
# ``constant`` Vecs. On the real side a ``constant`` is a graph ``Node`` with a
# global auto-incrementing id; creating one at import time gives it a fixed low
# id that collides with test-built nodes after the test harness resets the id
# counter (``tests/conftest.py``), aliasing them under ``reference_eval`` /
# ``probe_compiled`` memoization. So — matching ``extract.py``'s convention —
# the 1.0 query constant is created *inside* the call sites (via
# ``std.constant(1.0)``) rather than here.
