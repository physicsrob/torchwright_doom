"""Focused gate for the lifted per-column ClipMemory recovery (Phase H, step #3).

``ClipMemory`` replaces a width-(SCREEN_WIDTH+1) one-hot column key with a
width-3 lifted scalar-id equality so the R_RenderSegLoop clip read stops being a
``d_head`` floor (d_qk 62 -> 4). A lifted key has no orthogonal "no match" — it
returns the NEAREST column — so the "no prior update for this column -> default
open clip (-1, SCREEN_HEIGHT)" semantics are recovered explicitly: each clip row
carries its own column scalar, the recovered scalar is compared to the query
column (``same_int``), and a mismatch selects the default.

This pins the recovered (ceiling, floor) at the boundaries the lift introduces:
an updated column returns its values; an un-updated column returns the default;
a column updated twice returns the LATEST (the ``pick_most_recent`` recency
tiebreak); adjacent updates ``x-1`` / ``x+1`` do not blend to ``x`` (the recency
tiebreak separates distinct positions). The compiled case drives the recency
tiebreak at a high column index — the fp32 path the power-of-two ``MATCH_GAIN_CLIP``
keeps exact, which ``reference_eval`` (float64) cannot exercise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.attention_handles import RecentMarkerHandle
from torchwright_doom.constants import SCREEN_HEIGHT
from torchwright_doom.past import GraphPast
from torchwright_doom.std import concat, constant, linear
from torchwright_doom.wall_column_state import ClipMemory

_DEFAULT = (-1.0, float(SCREEN_HEIGHT))


def _build_graph(updates, query_col):
    """updates: list of (col, y1, y2) clip-update rows in causal order (later =
    more recent). Final row is the query. Returns (out_node, inputs, n_pos)."""
    n_pos = len(updates) + 1
    rows = [[1.0, float(c), float(y1), float(y2)] for (c, y1, y2) in updates]
    rows.append([0.0, float(query_col), 0.0, 0.0])  # query row (not an update)
    data = create_input("clip", 4)

    iv = create_input("iv", 4)
    past = GraphPast(input_vec=iv, pos_encoding=create_pos_encoding())

    is_update_01 = linear(data, [[1.0], [0.0], [0.0], [0.0]])
    range_active = linear(concat(is_update_01, constant(1.0)), [[2.0], [-1.0]])
    col = linear(data, [[0.0], [1.0], [0.0], [0.0]])
    y1 = linear(data, [[0.0], [0.0], [1.0], [0.0]])
    y2 = linear(data, [[0.0], [0.0], [0.0], [1.0]])

    cursor_x_scalar_pub = past.publish("cursor_x_scalar_value", col)
    clip_update_row = RecentMarkerHandle.publish(past, "clip_update", range_active)

    inp = SimpleNamespace(
        screen_range_after_clip_update=range_active,
        screen_range_y1=y1,
        screen_range_y2=y2,
    )
    # current_x_scalar is the per-row column ``col`` (an input-derived node, so a
    # non-degenerate affine bound — a literal constant trips a too-tight
    # to_interval snap inside the query-column floor_int). The query row encodes
    # query_col in that column, so at the read position it is the query column.
    clip = ClipMemory.publish(
        past,
        inp,
        col,
        clip_update_row,
        cursor_x_scalar_pub,
    )
    out = concat(clip.ceiling, clip.floor)
    return out, {"clip": torch.tensor(rows)}, n_pos


def _eval(updates, query_col):
    out, inputs, n_pos = _build_graph(updates, query_col)
    vals = reference_eval(out, inputs, n_pos)[out][n_pos - 1]
    return vals[0].item(), vals[1].item()


def _expected(updates, query_col):
    latest = None
    for c, y1, y2 in updates:
        if c == query_col:
            latest = (float(y1), float(y2))  # later overwrites earlier (recency)
    return latest if latest is not None else _DEFAULT


# (name, updates, query_col)
_CASES = [
    ("single_hit", [(10, 5, 40)], 10),
    ("single_miss", [(10, 5, 40)], 20),
    ("recency_two_updates", [(10, 5, 40), (10, 8, 35)], 10),  # latest (8,35)
    ("recency_three_updates", [(10, 5, 40), (10, 8, 35), (10, 12, 30)], 10),
    ("adjacent_gap_defaults", [(9, 1, 49), (11, 2, 48)], 10),  # x-1/x+1, query x
    ("col_zero_hit", [(0, 3, 44)], 0),
    ("col_zero_miss", [(5, 3, 44)], 0),  # ABSENT must not match query 0
    ("many_cols_pick_right", [(8, 1, 9), (10, 2, 8), (12, 3, 7)], 10),
    ("interleaved_recency", [(10, 5, 40), (11, 0, 49), (10, 9, 31)], 10),  # (9,31)
    ("high_col_hit", [(59, 6, 33)], 59),
    ("high_col_recency", [(59, 6, 33), (59, 10, 22)], 59),  # latest (10,22)
]


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_clip_memory_lifted(case):
    name, updates, query_col = case
    exp = _expected(updates, query_col)
    got = _eval(updates, query_col)
    # The recency tiebreak is a softmax (exp(8) per position gap), so a column
    # updated at adjacent positions leaks ~1e-2 of the prior update — the same
    # softness the one-hot ClipMemory had. Far-apart updates (the real renderer)
    # are far harder. 0.05 admits the worst-case adjacent-update softness.
    assert got == pytest.approx(exp, abs=0.05), (
        f"{name}: updates={updates} query={query_col} expected {exp} got {got}"
    )


# The recency tiebreak among repeated updates to ONE column must survive compiled
# fp32 at a high column index (c~59 -> c^2~3481, scaled by MATCH_GAIN_CLIP). The
# power-of-two gain keeps the lifted dot bit-exact across the two same-column
# rows, so the latest update wins instead of blending with the earlier one.
_COMPILED_CASES = [
    ("compiled_high_col_recency", [(59, 6, 33), (59, 12, 21)], 59),
    ("compiled_high_col_miss", [(59, 6, 33)], 58),
]


@pytest.mark.parametrize(
    "case", _COMPILED_CASES, ids=[c[0] for c in _COMPILED_CASES]
)
def test_clip_memory_compiled(case):
    name, updates, query_col = case
    assert _eval(updates, query_col) == pytest.approx(
        _expected(updates, query_col), abs=0.05
    ), f"{name}: oracle disagrees"
    out, inputs, n_pos = _build_graph(updates, query_col)
    pe = create_pos_encoding()
    report = probe_graph(out, pe, inputs, n_pos, d=2048, d_head=32, atol=0.05)
    assert report.first_divergent is None, report.format_short()
