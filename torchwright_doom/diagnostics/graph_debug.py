"""Debug helpers for exact-math (``reference_eval``) passes over the renderer graph.

Used only by the exact-math oracle tests (``tests/scene/test_*_oracle.py``);
it is never imported by the live ``forward()`` build
(``model/render_main.py``) or by anything on the production path, which is
why it lives in ``diagnostics/``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def silenced_graph_asserts() -> Iterator[None]:
    """Disable ``torchwright`` attached-check predicates for the duration.

    The renderer builds every dispatch branch's candidate at every position
    and masks by token type, so a ``select`` / ``cond_gate`` cond on a
    *discarded* branch can land inside a comparator ramp (e.g. ``gt_height``
    of two garbage recovered heights) and trip the gated ops' shared ±1
    assert during ``reference_eval``.  At the active row the cond is clean
    (DOOM heights are integers).  The oracle gate tests and the render
    comparison (``interpret/compare``) validate via output agreement —
    next-token and image-level — not the debug asserts, so exact-math
    passes silence the predicates instead.

    Checks are node metadata since the assert-metadata migration
    (torchwright docs/assert_metadata_plan.md — no wrapper nodes), so this
    delegates to torchwright's own ``suppress_checks`` switch.
    """
    from torchwright.graph.node import suppress_checks

    with suppress_checks():
        yield
