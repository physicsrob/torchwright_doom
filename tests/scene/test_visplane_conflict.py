"""Focused crafted-input gate for the R_CheckPlane overlap deadband (Phase H).

``e1m1_subset`` never produces a visplane overlap/merge (verified: every
R_CheckPlane run is length 1, every result vp=0, across all 8 poses), so the
``_range_score_bits`` overlap test — the chunk's most delicate attention key,
riding a ~0.001-wide deadband around the 1.5 threshold — is built but never
exercised by the e1m1 frame gate. This test drives it directly: it publishes a
visplane's occupied columns, then calls ``check_conflict`` with a candidate
column range that overlaps (must report a conflict, +1) and one that is disjoint
(must not, -1), plus a fresh-instance query against an empty visplane (-1).

``reference_eval`` (exact math, CPU, no compile) — the smallest layer that
reproduces the overlap decision (torchwright doctrine D6).
"""

from __future__ import annotations

import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input

from torchwright_doom.std import concat, constant, gate, linear, one_hot
from torchwright_doom.visplane_state import (
    RuntimeVisplaneState,
    _INSTANCE_IDX_LINEAR,
    _OCC_PLANE_SCALE,
    _OCC_VP_SCALE,
)
from torchwright_doom.vocab import (
    N_PLANES_MAX,
    N_VISPLANE_MAX,
    N_VP_PER_PLANE_MAX,
)
from torchwright_doom.constants import SCREEN_WIDTH
from torchwright_doom.past import GraphPast
from torchwright.ops.inout_nodes import create_pos_encoding

# Crafted occupancy: visplane instance (plane=3, vp=0) covers screen column 15.
# A single occupied column isolates the overlap deadband itself: ``pick_argmax``
# concentrates cleanly on that column (no tie), so ``picked.x_oh`` is a clean
# one-hot and ``picked_x_score`` is exactly the thermometer coverage count at
# that column — 2 when the candidate range covers it, 1 when it does not — and
# the ~0.001-wide ``_range_overlap`` compare at 1.5 is the thing under test.
# (Multi-column disjoint coverage instead depends on pick *concentration*: tied
# uncovered columns blend and ``pick_by_one_hot`` sums the blend, a separate
# pre-existing property of the sandbox design that e1m1 never exercises.)
_PLANE = 3.0
_VP = 0.0
_OCC_COLS = [15.0]


def _build(plane_id, candidate_vp, x1, x2, occ_cols):
    """Build a graph that publishes ``occ_cols`` for instance (_PLANE,_VP), then
    evaluates ``check_conflict`` for the given candidate at the final row.
    Returns the ±1 conflict value at the query row."""
    n_pub = len(occ_cols)
    n_pos = n_pub + 1  # publish rows + one query row
    # Per-position input: [active, p, vp, x] — occupancy on the publish rows, a
    # gated-off zero row for the query.
    rows = [[1.0, _PLANE, _VP, c] for c in occ_cols] + [[0.0, 0.0, 0.0, 0.0]]
    occ = create_input("occ", 4)

    iv = create_input("iv", 4)  # unused embedding placeholder for GraphPast
    past = GraphPast(input_vec=iv, pos_encoding=create_pos_encoding())

    # active in {0,1} -> ±1 boolean (2*active - 1)
    two_active = linear(occ, [[2.0], [0.0], [0.0], [0.0]])
    occupied_active = linear(concat(two_active, constant(1.0)), [[1.0], [-1.0]])
    p_val = linear(occ, [[0.0], [1.0], [0.0], [0.0]])
    vp_val = linear(occ, [[0.0], [0.0], [1.0], [0.0]])
    x_val = linear(occ, [[0.0], [0.0], [0.0], [1.0]])
    x_oh = one_hot(x_val, SCREEN_WIDTH)

    occupied_key_value = concat(
        linear(one_hot(p_val, N_PLANES_MAX), _OCC_PLANE_SCALE),
        linear(one_hot(vp_val, N_VP_PER_PLANE_MAX), _OCC_VP_SCALE),
        x_oh,
    )
    occupied_key = past.publish(
        "occupied_key", gate(occupied_active, occupied_key_value)
    )
    instance_idx = linear(concat(p_val, vp_val), _INSTANCE_IDX_LINEAR)
    instance_oh = one_hot(instance_idx, N_VISPLANE_MAX)
    occupied_state = past.publish(
        "occupied_state", concat(occupied_active, x_oh, instance_oh)
    )

    dummy = past.publish("dummy", constant(0.0))
    rvs = RuntimeVisplaneState(
        occupied_key=occupied_key,
        occupied_x=dummy,
        occupied_state=occupied_state,
        bounds_min_key=dummy,
        bounds_max_key=dummy,
        col_key=dummy,
        col_range=dummy,
        used_plane_score=dummy,
        used_plane_above=dummy,
        used_plane_value=dummy,
        used_vp_key=dummy,
        used_vp_value=dummy,
    )
    conflict = rvs.check_conflict(
        past,
        plane_id=constant(float(plane_id)),
        candidate_vp=constant(float(candidate_vp)),
        x1=constant(float(x1)),
        x2=constant(float(x2)),
    )
    cache = reference_eval(conflict, {"occ": torch.tensor(rows)}, n_pos)
    return cache[conflict][n_pos - 1].item()


def test_overlap_reports_conflict():
    """A candidate range covering an occupied column (x=15 in [12,18]) reports a
    conflict: the picked column's thermometer coverage score is 2 (> the 1.5
    deadband)."""
    val = _build(_PLANE, _VP, 12, 18, _OCC_COLS)
    assert val > 0.5, f"overlap must report conflict (+1), got {val}"


def test_disjoint_reports_no_conflict():
    """A candidate range covering no occupied column ([20,25] vs col 15)
    reports no conflict: the picked column's coverage score is 1 (< 1.5)."""
    val = _build(_PLANE, _VP, 20, 25, _OCC_COLS)
    assert val < -0.5, f"disjoint range must report no conflict (-1), got {val}"


def test_empty_visplane_no_conflict():
    """A query against a visplane with no occupied columns (all keys gated to
    zero) reports no conflict — the all-masked pick must not false-positive."""
    val = _build(_PLANE, _VP, 12, 18, [])
    assert val < -0.5, f"empty visplane must report no conflict (-1), got {val}"
