"""Phase F / F3 focused gate: the SolidIntervals coverage key (the new numeric).

``SolidIntervals`` is filled at ``R_STORE_WALL_RANGE`` and queried at
``FIND_RUN`` to decide horizontal occlusion. The genuinely new math is
``_interval_key``: it encodes a fragment ``[x1, x2]`` as a width-3 key
``[-2, 2(a+b), -2ab]`` (with ``a=x1-1, b=x2+1``) so the query ``[col², col, 1]``
scores ``-2(col-a)(col-b)`` — positive inside the padded interval, and the
flat-1 sentinel key wins outside it. This checks that the ``MUL_SCREEN`` grid
(the new ``multiply_2d``) plus the affine assembly resolve coverage correctly;
the full publish/query against the golden stream is covered by the F gate.
"""

from __future__ import annotations

import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.graph import Node
from torchwright.graph.attn import Attn
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.constants import SCREEN_WIDTH
from torchwright_doom.past import GraphPast, PastHandleScope
from torchwright_doom.solid_intervals import _interval_key
from torchwright_doom.solid_intervals import (
    _N_BUCKETS,
    _RADIX_BASE,
    _publish_successor_fields,
    SolidIntervals,
)
from torchwright_doom.std import concat

from .test_radix_successor_oracle import brute_next_start_after


def _scores(cases: list[tuple[int, int, int]]) -> list[float]:
    """For each (x1, x2, col), reference_eval the query·key coverage score."""
    x1 = create_input("x1", 1)
    x2 = create_input("x2", 1)
    col = create_input("col", 1)
    key = _interval_key(x1, x2)
    # query = [col², col, 1] (col² fed as its own input so this stays a pure
    # key-vs-query check without an extra product op in the path).
    query = concat(
        create_input("col_sq", 1),
        col,
        create_input("one", 1),
    )
    cache = reference_eval(
        concat(key, query),
        {
            "x1": torch.tensor([[float(c[0])] for c in cases]),
            "x2": torch.tensor([[float(c[1])] for c in cases]),
            "col": torch.tensor([[float(c[2])] for c in cases]),
            "col_sq": torch.tensor([[float(c[2] * c[2])] for c in cases]),
            "one": torch.ones((len(cases), 1)),
        },
        len(cases),
    )
    key_v = cache[key]
    query_v = cache[query]
    return [(key_v[i] * query_v[i]).sum().item() for i in range(len(cases))]


def test_coverage_score_discriminates_inside_vs_outside() -> None:
    # (x1, x2, col): inside the fragment, padded edge, and clearly outside.
    intervals = [(0, 5), (10, 20), (40, 58), (3, 3), (2, 59)]
    inside_cases = []
    outside_cases = []
    for x1, x2 in intervals:
        for col in range(x1, x2 + 1):  # strictly inside [x1, x2] -> covered
            inside_cases.append((x1, x2, col))
        # clearly outside the padded [x1-1, x2+1] interval -> sentinel wins
        if x1 - 2 >= 0:
            outside_cases.append((x1, x2, x1 - 2))
        outside_cases.append((x1, x2, x2 + 2))

    # The sentinel key scores exactly 1.0; an interval covering its column must
    # beat it. The minimum interior score is 2·(width+1) ≥ 4 at the endpoints.
    inside_scores = _scores(inside_cases)
    assert all(s > 1.0 + 0.5 for s in inside_scores), (
        f"some covered column scored <= sentinel: "
        f"{[(c, round(s, 2)) for c, s in zip(inside_cases, inside_scores) if s <= 1.5]}"
    )

    outside_scores = _scores(outside_cases)
    assert all(s < 1.0 - 0.5 for s in outside_scores), (
        f"some uncovered column scored >= sentinel: "
        f"{[(c, round(s, 2)) for c, s in zip(outside_cases, outside_scores) if s >= 0.5]}"
    )


def _successor_graph(prefix: str = "succ") -> tuple[Node, Node]:
    pos = create_pos_encoding()
    past = GraphPast(
        input_vec=create_input(f"{prefix}_iv", 1),
        pos_encoding=pos,
    )
    scope = PastHandleScope(past)
    start = create_input(f"{prefix}_start", 1)
    solid = create_input(f"{prefix}_solid", 1)
    handles = _publish_successor_fields(scope, start, solid)
    solids = SolidIntervals(
        past=scope,
        key=scope.publish(f"{prefix}_unused_key", create_input(f"{prefix}_key", 3)),
        interval_state=scope.publish(
            f"{prefix}_unused_state",
            create_input(f"{prefix}_state", 2),
        ),
        solid_emit=handles.solid_emit,
        start_s=handles.start_s,
        start_hi=handles.start_hi,
        start_lo=handles.start_lo,
        start_bucket_onehot=handles.start_bucket_onehot,
        start_above_lo=handles.start_above_lo,
        start_hi_for_h2=handles.start_hi_for_h2,
        start_hi_above_for_h2=handles.start_hi_above_for_h2,
        start_above_all=handles.start_above_all,
        same_payload=handles.same_payload,
        carry_payload=handles.carry_payload,
    )
    column = create_input(f"{prefix}_column", 1)
    return solids.next_start_after(column), pos


def _successor_inputs(
    starts: list[int],
    column: int,
    prefix: str = "succ",
) -> dict[str, torch.Tensor]:
    rows = list(starts) + [SCREEN_WIDTH]
    solid = [1.0] * len(starts) + [-1.0]
    columns = [column] * len(rows)
    return {
        f"{prefix}_start": torch.tensor([[float(v)] for v in rows]),
        f"{prefix}_solid": torch.tensor([[v] for v in solid]),
        f"{prefix}_column": torch.tensor([[float(v)] for v in columns]),
    }


@pytest.mark.parametrize(
    "starts,column",
    [
        ([2, 5], 3),  # same-bucket hit
        ([3, 16], 5),  # same-bucket row exists, but is below threshold
        ([0, 2 * _RADIX_BASE], _RADIX_BASE + 2),  # scalar-bucket-average hazard
        ([58], 50),  # real top-bucket start plus invalid sentinel rows
        ([12, 12, 20], 10),  # duplicate starts
        ([], 7),  # no published solid rows
        ([0, _RADIX_BASE, _RADIX_BASE + 1, SCREEN_WIDTH], _RADIX_BASE - 1),
        ([0, _RADIX_BASE, _RADIX_BASE + 1, SCREEN_WIDTH], _RADIX_BASE),
        ([SCREEN_WIDTH], SCREEN_WIDTH - 1),
        ([0, 12, 58], SCREEN_WIDTH),
    ],
)
def test_radix_next_start_after_reference_cases(starts, column) -> None:
    out, _pos = _successor_graph()
    inputs = _successor_inputs(starts, column)
    n_pos = len(starts) + 1
    got = reference_eval(out, inputs, n_pos)[out][-1, 0].item()
    expected = brute_next_start_after(starts, column, SCREEN_WIDTH)
    assert got == pytest.approx(expected, abs=5e-2)


def _attn_nodes(root: Node) -> list[Attn]:
    found: list[Attn] = []
    seen: set[int] = set()

    def walk(node: Node) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, Attn):
            found.append(node)
        for child in getattr(node, "inputs", []):
            walk(child)

    walk(root)
    return found


def test_radix_successor_attention_widths_and_identity_value_paths() -> None:
    out, _pos = _successor_graph()
    by_dqk = {attn.d_qk: attn for attn in _attn_nodes(out)}

    h1 = by_dqk[2 + _N_BUCKETS + _RADIX_BASE]
    h2 = by_dqk[1 + _N_BUCKETS + 1]
    h3 = by_dqk[2 + _N_BUCKETS + 1]

    assert h1.d_qk == 18
    assert h2.d_qk == 10
    assert h3.d_qk == 11

    for attn in (h1, h3):
        eye = torch.eye(attn.d_v)
        assert torch.allclose(attn.value_matrix, eye)
        assert torch.allclose(attn.output_matrix, eye)

    assert h1.d_v == 1 + 1 + _N_BUCKETS + _RADIX_BASE
    assert h3.d_v == 1 + 1 + _N_BUCKETS


def test_radix_next_start_after_compiled_probe() -> None:
    out, pos = _successor_graph()
    starts = [0, 2 * _RADIX_BASE, 12, 58]
    rows = [
        (0, 1.0, 0),
        (2 * _RADIX_BASE, 1.0, 0),
        (SCREEN_WIDTH, -1.0, _RADIX_BASE + 2),  # carried-bucket path
        (12, 1.0, 0),
        (SCREEN_WIDTH, -1.0, 10),  # same-bucket path
        (58, 1.0, 0),
        (SCREEN_WIDTH, -1.0, SCREEN_WIDTH - 1),  # no later start
    ]
    inputs = {
        "succ_start": torch.tensor([[float(start)] for start, _solid, _col in rows]),
        "succ_solid": torch.tensor([[solid] for _start, solid, _col in rows]),
        "succ_column": torch.tensor([[float(col)] for _start, _solid, col in rows]),
    }
    n_pos = len(rows)
    ref = reference_eval(out, inputs, n_pos)[out]
    assert ref[2, 0].item() == pytest.approx(
        brute_next_start_after(starts[:2], _RADIX_BASE + 2, SCREEN_WIDTH),
        abs=5e-2,
    )
    assert ref[4, 0].item() == pytest.approx(
        brute_next_start_after(starts[:3], 10, SCREEN_WIDTH),
        abs=5e-2,
    )
    assert ref[6, 0].item() == pytest.approx(SCREEN_WIDTH, abs=5e-2)

    report = probe_graph(
        out,
        pos_encoding=pos,
        input_values=inputs,
        n_pos=n_pos,
        d=1008,
        d_head=18,
        atol=0.1,
    )
    assert report.first_divergent is None, report.format_short()
