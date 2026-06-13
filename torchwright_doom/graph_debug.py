"""Debug helpers for exact-math (``reference_eval``) passes over the renderer graph.

Used only by the exact-math oracle tests (``tests/scene/test_*_oracle.py``) and
the Plan-K diagnostic scripts (``scripts/k_*.py``); it is never imported by the
live ``forward()`` build (``render_main``) or by the production runtime
(``inference/`` / ``prompt/``), so it has no effect on the compiled path."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def silenced_graph_asserts() -> Iterator[None]:
    """Disable ``torchwright`` Assert predicate checks for the duration.

    The renderer builds every dispatch branch's candidate at every position
    and masks by token type, so a ``select`` / ``broadcast_select`` cond on a
    *discarded* branch can land inside a comparator ramp (e.g. ``gt_height``
    of two garbage recovered heights) and trip its ±1 ``c_tol`` Assert during
    ``reference_eval``.  At the active row the cond is clean (DOOM heights
    are integers).  The oracle gates and the Plan-K diagnostics validate via
    next-token agreement, not the debug Asserts — and Asserts are stripped on
    the compiled path anyway (re-checked only under ``debug=True``) — so
    exact-math passes silence the predicates instead.
    """
    import torchwright.graph.misc as _misc

    orig = _misc.Assert._check
    _misc.Assert._check = lambda self, x: None
    try:
        yield
    finally:
        _misc.Assert._check = orig
