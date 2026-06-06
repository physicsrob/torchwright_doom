"""Phase J2 — R_MakeSpans open/close equivalence (DIVERGENCE #1).

``flat_state.py`` ports the **sandbox** R_MakeSpans formulation, which uses the
RAW ``t1/b1/t2/b2`` in all four close/open sub-steps and compensates with two
``*_non_empty`` guards — *not* the reference's ``t1_after``/``b1_after`` chaining
(``reference._make_spans``). That is an algebraic equivalence claim that holds
only when the four row-ranges are disjoint. An off-by-one in the
``min``/``max``/``±1`` boundary arithmetic silently drops or doubles a scanline.

This test pins that claim directly: ``_make_spans_raw`` below is the pure-Python
mirror of ``flat_state.FlatPassState.publish``'s open/close arithmetic (the graph
builds the same expressions as ``Node``s). It must produce the *same* spans as the
reference ``_after``-threaded ``_make_spans`` over adversarial coverage tables
(nested, disjoint top+bottom, single-row, empty columns, boundary rows). The
fixture-driven teacher-forced gate (``test_flat_pixel_oracle``) exercises the
graph end-to-end on one map's coverage; this covers the table shapes that map
does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from torchwright_doom.constants import SCREEN_HEIGHT


def _umbrella() -> Path:
    return Path(__file__).resolve().parents[3]


def _make_spans_raw(
    table: list[tuple[int, int]], minx: int, maxx: int
) -> list[tuple[int, int, int]]:
    """Pure-Python mirror of ``flat_state``'s graph open/close arithmetic.

    Uses the RAW ``t1/b1/t2/b2`` in all four sub-steps with the
    ``cur_non_empty`` / ``prev_non_empty`` guards (the sandbox form), the
    ``slot0 = close_top`` / ``slot1 = close_bottom`` packing, ``x_close = x - 1``,
    and ``x_open = x`` with per-row open-x recovery (the ``pick_most_recent`` x1
    recovery)."""
    open_x_by_y: dict[int, int] = {}
    spans: list[tuple[int, int, int]] = []

    def close_rows(lo: int, hi: int, x_close: int) -> None:
        if lo > hi:
            return
        for y in range(lo, hi + 1):
            x1 = open_x_by_y.pop(y, minx)
            spans.append((y, x1, x_close))

    def open_rows(lo: int, hi: int, x_open: int) -> None:
        if lo > hi:
            return
        for y in range(lo, hi + 1):
            open_x_by_y[y] = x_open

    for x in range(minx, maxx + 2):
        t1, b1 = (SCREEN_HEIGHT, -1) if x == minx else table[x - 1]
        t2, b2 = (SCREEN_HEIGHT, -1) if x == maxx + 1 else table[x]

        prev_non_empty = t1 <= b1
        cur_non_empty = t2 <= b2

        close_top_lo = t1
        close_top_hi = min(t2 - 1, b1)
        close_top_valid = close_top_lo <= close_top_hi

        close_bottom_lo = max(b2 + 1, t1)
        close_bottom_hi = b1
        close_bottom_valid = (close_bottom_lo <= close_bottom_hi) and cur_non_empty

        open_top_lo = t2
        open_top_hi = min(t1 - 1, b2)
        open_top_valid = open_top_lo <= open_top_hi

        open_bottom_lo = max(b1 + 1, t2)
        open_bottom_hi = b2
        open_bottom_valid = (open_bottom_lo <= open_bottom_hi) and prev_non_empty

        # Close first (slot0 = close_top, then slot1 = close_bottom), then open.
        if close_top_valid:
            close_rows(close_top_lo, close_top_hi, x - 1)
        if close_bottom_valid:
            close_rows(close_bottom_lo, close_bottom_hi, x - 1)
        if open_top_valid:
            open_rows(open_top_lo, open_top_hi, x)
        if open_bottom_valid:
            open_rows(open_bottom_lo, open_bottom_hi, x)

    return spans


_EMPTY = (SCREEN_HEIGHT, -1)  # an empty column (top > bottom)


def _adversarial_tables() -> list[tuple[list[tuple[int, int]], int, int]]:
    """Hand-crafted coverage tables exercising nested / disjoint / boundary
    shapes, expressed as (table_by_column, minx, maxx)."""
    cases: list[tuple[list[tuple[int, int]], int, int]] = []

    def col(table: list[tuple[int, int]]):
        cases.append((table, 0, len(table) - 1))

    # Constant block.
    col([(10, 20), (10, 20), (10, 20)])
    # Growing then shrinking (open at top+bottom, then close at top+bottom).
    col([(20, 30), (10, 40), (20, 30)])
    # Nested: a wide span containing a narrower one re-widening.
    col([(5, 45), (15, 35), (5, 45)])
    # Disjoint top vs bottom motion: top rises while bottom falls.
    col([(20, 25), (10, 35), (5, 45), (10, 35), (20, 25)])
    # Single-row spans.
    col([(25, 25), (24, 26), (25, 25)])
    # Empty columns interleaved (spans start/stop).
    col([_EMPTY, (10, 20), _EMPTY, (10, 20), _EMPTY])
    # Boundary rows (top at 0, bottom at SCREEN_HEIGHT-1).
    col([(0, SCREEN_HEIGHT - 1), (5, SCREEN_HEIGHT - 6), (0, SCREEN_HEIGHT - 1)])
    # Step shifts (top and bottom both move by 1 each column).
    col([(10, 30), (11, 29), (12, 28), (11, 29), (10, 30)])
    # All empty.
    col([_EMPTY, _EMPTY, _EMPTY])
    # Open-only then close-only.
    col([_EMPTY, (10, 20), (10, 20), _EMPTY])
    return cases


def _random_tables(n: int, seed: int):
    import random

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        width = rng.randint(1, 8)
        table = []
        for _x in range(width):
            if rng.random() < 0.2:
                table.append(_EMPTY)
            else:
                a = rng.randint(0, SCREEN_HEIGHT - 1)
                b = rng.randint(0, SCREEN_HEIGHT - 1)
                table.append((min(a, b), max(a, b)))
        out.append((table, 0, width - 1))
    return out


def test_make_spans_raw_matches_reference() -> None:
    """The raw-``t1/b1`` + ``*_non_empty`` formulation equals the reference's
    ``_after``-threaded ``_make_spans`` on every adversarial / random table."""
    umbrella = _umbrella()
    if not (umbrella / "doom_sandbox").is_dir():
        pytest.skip("doom_sandbox sibling not present (standalone checkout)")
    if str(umbrella) not in sys.path:
        sys.path.insert(0, str(umbrella))
    reference = pytest.importorskip("doom_sandbox.implementation.reference")
    # Sentinel/height parity between the two implementations.
    assert reference.SCREEN_HEIGHT == SCREEN_HEIGHT

    tables = _adversarial_tables() + _random_tables(500, seed=1234)
    mismatches = []
    for table, minx, maxx in tables:
        raw = sorted(_make_spans_raw(table, minx, maxx))
        ref = sorted(reference._make_spans(table, minx, maxx))
        if raw != ref:
            mismatches.append((table, raw, ref))
    assert not mismatches, (
        f"{len(mismatches)} R_MakeSpans equivalence mismatch(es); first:\n"
        f"  table={mismatches[0][0]}\n"
        f"  raw  ={mismatches[0][1]}\n"
        f"  ref  ={mismatches[0][2]}"
    )
