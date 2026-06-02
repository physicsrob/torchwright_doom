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

import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input

from torchwright_doom.solid_intervals import _interval_key
from torchwright_doom.std import concat


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
