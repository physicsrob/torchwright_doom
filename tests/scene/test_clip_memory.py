"""Focused gate for lifted per-column ``ClipMemory`` recovery.

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
from torchwright.ops.inout_nodes import create_input, create_rope_config

from torchwright_doom.model.attention_handles import RecentMarkerHandle
from torchwright_doom.model.constants import SCREEN_HEIGHT
from torchwright_doom.model.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.model.past import GraphPast
from torchwright_doom.model.std import concat, constant, linear
from torchwright_doom.model.vocab import BOS, NO_OP
from torchwright_doom.model.raster.wall_column_state import ClipMemory

from ..prefill_fixture import row_index

_DEFAULT = (-1.0, float(SCREEN_HEIGHT))
_D_EMBED = TOKEN_VOCAB.layout.d_embed


def _iv_rows(n_pos: int) -> torch.Tensor:
    """A real d_embed input_vec with the inert BOS at position 0 and a non-BOS
    filler elsewhere.  ``pick_most_recent`` is now the global mechanism, which reads
    each token's absolute position from the BOS-weight readout
    (``GraphPast.global_position`` → ``is_type(input_vec, BOS)``), so a focused
    recency test must supply the genuine embedding, not a narrow placeholder."""
    bos = W_EMBED[row_index(BOS, {})]
    filler = W_EMBED[row_index(NO_OP, {})]
    return torch.stack([bos if i == 0 else filler for i in range(n_pos)])


# Non-update filler rows inserted between consecutive clip writes.  pick_most_recent
# is now the GLOBAL mechanism: the recency tiebreak among same-column writes is
# resolved by absolute position with sharpness exp(recency_scale·Δpos), so two
# writes Δ=1 apart blend (~73 % / 27 %), while the real renderer's same-column
# writes are many tokens apart and resolve cleanly.  Separating the test's writes
# by this gap reproduces that regime (Δ=9 → exp(9) ≈ 8000, >99.98 % concentration).
# Filler rows are non-updates (is_update=0): ClipMemory gates their key to zero, so
# they are excluded from the read and only advance the absolute position.
_RECENCY_GAP = 8


def _build_graph(updates, query_col):
    """updates: list of (col, y1, y2) clip-update rows in causal order (later =
    more recent). Final row is the query. Returns (out_node, inputs, n_pos)."""
    rows: list[list[float]] = []
    for i, (c, y1, y2) in enumerate(updates):
        if i > 0:
            rows.extend([[0.0, 0.0, 0.0, 0.0]] * _RECENCY_GAP)
        rows.append([1.0, float(c), float(y1), float(y2)])
    rows.append([0.0, float(query_col), 0.0, 0.0])  # query row (not an update)
    n_pos = len(rows)
    data = create_input("clip", 4)

    iv = create_input("iv", _D_EMBED)
    past = GraphPast(
        input_vec=iv,
        rope=create_rope_config(d_head=32, max_positions=65536, d_rot=16),
    )

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
    return out, {"clip": torch.tensor(rows), "iv": _iv_rows(n_pos)}, n_pos


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
    # The recency tiebreak is the GLOBAL mechanism (exp(recency_scale·Δpos) per
    # position gap); with _RECENCY_GAP separation the same-column writes are Δ=9
    # apart (exp(9) ≈ 8000, >99.98 % concentration), so the latest update wins
    # cleanly. 0.05 admits the residual softmax leak.
    assert got == pytest.approx(
        exp, abs=0.05
    ), f"{name}: updates={updates} query={query_col} expected {exp} got {got}"


# The recency tiebreak among repeated updates to ONE column must survive compiled
# fp32 at a high column index (c~59 -> c^2~3481, scaled by MATCH_GAIN_CLIP), so the
# latest update wins instead of blending with the earlier one.
_COMPILED_CASES = [
    ("compiled_high_col_recency", [(59, 6, 33), (59, 12, 21)], 59),
    ("compiled_high_col_miss", [(59, 6, 33)], 58),
]


@pytest.mark.parametrize("case", _COMPILED_CASES, ids=[c[0] for c in _COMPILED_CASES])
def test_clip_memory_compiled(case):
    name, updates, query_col = case
    assert _eval(updates, query_col) == pytest.approx(
        _expected(updates, query_col), abs=0.05
    ), f"{name}: oracle disagrees"
    out, inputs, n_pos = _build_graph(updates, query_col)
    # The oracle check above pins the FINAL (query-position) value at abs=0.05.  The
    # probe checks every node at every position, including intermediate filler rows
    # where the global recency read is a near-Δ soft blend that fp32 and the float64
    # oracle resolve slightly differently through the global_position PWL (~0.15
    # inversion error).  atol=1.0 matches torchwright's global-recency probe
    # convention and still dwarfs nothing real: a wrong clip selection differs by
    # ≥ several units (recovered values range to 59).
    report = probe_graph(out, inputs, n_pos, d=2048, d_head=32, atol=1.0)
    assert report.first_divergent is None, report.format_short()
