"""Forward/render constants (Plan D / D0).

Real-side mirror of the read-side subset of
``doom_sandbox/implementation/forward/constants.py``. Kept separate from
:mod:`.constants` (which holds the dependency-free vocab-scale
``SCREEN_WIDTH`` / ``SCREEN_HEIGHT`` and must not gain a graph
dependency): the literal nodes below need :func:`.std.constant`, a graph
op.
"""

from __future__ import annotations

# Used on long ``pick_most_recent`` spans where a false match must lose
# hard to the one real 0/1 marker. Sized so
# ``match_gain * content_gap > SCORE_GAIN * max_recency_span``; 300_000
# gives headroom over the longest rollout. Plan C confirmed this is safe
# (fp32 on all paths). The op/facade default stays 200.0; long-span
# callers thread this explicitly.
MATCH_GAIN_LONG = 300_000.0

# Wall-column per-column clip-array recovery (``ClipMemory.pick_most_recent``,
# Phase H). Same magnitude as ``MATCH_GAIN_LONG`` — the clip key is one unit of
# content gap (a screen-x one-hot match) and must dominate the recency span over
# a full rollout. Plain float (no graph node), import-safe.
MATCH_GAIN_CLIP = 300_000.0

# Weight on the sentinel column of the ClipMemory query, so a column with no
# prior clip-update falls through to the ``DEFAULT_CLIP_KEY`` sentinel record
# (the initial ``(-1, SCREEN_HEIGHT)`` open clip) rather than a garbage blend.
CLIP_SENTINEL_QUERY_WEIGHT = 0.5

# NOTE: the sandbox keeps ``ONE``/``ZERO``/``FALSE`` as module-level
# ``constant`` Vecs. On the real side a ``constant`` is a graph ``Node`` with a
# global auto-incrementing id; creating one at import time gives it a fixed low
# id that collides with test-built nodes after the test harness resets the id
# counter (``tests/conftest.py``), aliasing them under ``reference_eval`` /
# ``probe_compiled`` memoization. So — matching ``extract.py``'s convention —
# the 1.0 query constant is created *inside* the call sites (via
# ``std.constant(1.0)``) rather than here.
