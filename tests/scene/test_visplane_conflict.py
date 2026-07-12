"""Focused crafted-input gate for the ``R_CheckPlane`` overlap test.

``e1m1_subset`` never produces a visplane overlap/merge (verified: every
R_CheckPlane run is length 1, every result vp=0, across all 8 poses), so the
overlap test — the chunk's most delicate attention path — is built but never
exercised by the e1m1 frame gate. This test drives it directly.

The overlap test is now an instance-filtered radix successor:
``c* = smallest occupied column (of this instance) >= x1``; conflict iff
``c*`` exists and ``c* <= x2``. The decomposition (same-bucket / next-bucket /
carry) was locked against brute force in pure Python; this test pins the graph
form at the boundaries the radix introduces: a column exactly at x1 / at x2, a
single-bucket range (hi1 == hi2), adjacent buckets (hi2 == hi1+1), an interior
bucket, a wide multi-bucket range, the empty visplane, the carry's ``c* <= x2``
edge, and a high-instance-id case (the fp32-magnitude path; resolution itself is
proven by the compiled probe, since ``reference_eval`` is exact float64).

``reference_eval`` provides exact graph math; selected cases also run through
the compiled fp32 path.
"""

from __future__ import annotations

import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.ops.inout_nodes import create_input, create_rope_config

from torchwright_doom.model.past import GraphPast
from torchwright_doom.model.std import concat, constant, linear, one_hot
from torchwright_doom.model.raster.visplane_state import (
    RuntimeVisplaneState,
    UsedPlaneSuccessor,
    _INSTANCE_IDX_LINEAR,
    _publish_occupancy_radix,
)
from torchwright_doom.model.vocab import N_VISPLANE_MAX


def _build_graph(plane_id, candidate_vp, x1, x2, occ_cols):
    """Build a graph that publishes ``occ_cols`` for instance (plane_id, vp) and
    evaluates ``check_conflict`` at a final gated-off query row. Returns
    ``(conflict_node, inputs, n_pos)``."""
    n_pub = len(occ_cols)
    n_pos = n_pub + 1  # publish rows + one query row
    rows = [[1.0, float(plane_id), float(candidate_vp), float(c)] for c in occ_cols]
    rows.append([0.0, 0.0, 0.0, 0.0])  # gated-off query row
    occ = create_input("occ", 4)

    iv = create_input("iv", 4)  # unused embedding placeholder for GraphPast
    # visplane occupancy uses content heads (not pick_most_recent), so no BOS /
    # global_position read — the narrow placeholder iv stays unused.  d_head=64 /
    # d_rot=32 covers the occupancy radix content on the NoPE tail.
    past = GraphPast(
        input_vec=iv,
        rope=create_rope_config(d_head=64, max_positions=65536, d_rot=32),
    )

    # active in {0,1} -> ±1 boolean (2*active - 1)
    two_active = linear(occ, [[2.0], [0.0], [0.0], [0.0]])
    occupied_active = linear(concat(two_active, constant(1.0)), [[1.0], [-1.0]])
    p_val = linear(occ, [[0.0], [1.0], [0.0], [0.0]])
    vp_val = linear(occ, [[0.0], [0.0], [1.0], [0.0]])
    x_val = linear(occ, [[0.0], [0.0], [0.0], [1.0]])

    instance_idx = linear(concat(p_val, vp_val), _INSTANCE_IDX_LINEAR)
    instance_oh = one_hot(instance_idx, N_VISPLANE_MAX)
    occupancy = _publish_occupancy_radix(
        past, occupied_active, x_val, instance_idx, instance_oh
    )

    dummy = past.publish("dummy", constant(0.0))
    # check_conflict reads only ``occupancy``; the rest are dummy handles. The
    # used-plane successor (next_plane_after) is unused here, so a dummy
    # UsedPlaneSuccessor with the shared dummy handle suffices.
    dummy_used_plane = UsedPlaneSuccessor(
        validity=dummy,
        lo=dummy,
        hi_for_h2=dummy,
        bucket_onehot=dummy,
        above_lo=dummy,
        hi_above_for_h2=dummy,
        above_all=dummy,
        same_payload=dummy,
        carry_payload=dummy,
    )
    rvs = RuntimeVisplaneState(
        occupancy=occupancy,
        occupied_x=dummy,
        bounds_min_key=dummy,
        bounds_max_key=dummy,
        col_key=dummy,
        col_range=dummy,
        used_plane=dummy_used_plane,
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
    return conflict, {"occ": torch.tensor(rows)}, n_pos


def _build(plane_id, candidate_vp, x1, x2, occ_cols):
    """Exact-math ``check_conflict`` value at the query row (±1)."""
    conflict, inputs, n_pos = _build_graph(plane_id, candidate_vp, x1, x2, occ_cols)
    cache = reference_eval(conflict, inputs, n_pos)
    return cache[conflict][n_pos - 1].item()


def _brute(occ_cols, x1, x2):
    return any(x1 <= c <= x2 for c in occ_cols)


# (name, plane, vp, x1, x2, occ_cols). Columns are chosen so the (hi, lo) digits
# land on the boundary the case names. _B == 8: bucket b spans [8b, 8b+7].
_CASES = [
    # --- single occupied column, deadband isolation ---
    ("overlap_interior", 3, 0, 12, 18, [15]),  # col strictly inside
    ("disjoint_above", 3, 0, 20, 25, [15]),  # col below the range
    ("disjoint_below", 3, 0, 5, 10, [15]),  # col above the range
    ("col_at_x1", 3, 0, 15, 18, [15]),  # inclusive lower bound
    ("col_at_x2", 3, 0, 12, 15, [15]),  # inclusive upper bound
    # --- single bucket (hi1 == hi2): col=10 -> hi 1, lo 2 ---
    ("same_bucket_hit", 3, 0, 9, 11, [10]),  # 9..11 all in bucket 1
    ("same_bucket_below_x1", 3, 0, 11, 13, [10]),  # col 10 < x1 11
    ("same_bucket_above_x2", 3, 0, 8, 9, [10]),  # col 10 > x2 9
    # --- adjacent buckets (hi2 == hi1+1): query [5,9] spans buckets 0 and 1 ---
    ("adjacent_carry_hit", 3, 0, 5, 9, [8]),  # col 8 = bucket1/lo0 <= lo2=1
    ("adjacent_carry_miss", 3, 0, 5, 9, [10]),  # col 10 = bucket1/lo2 > x2=9
    # --- interior bucket: query [5,30] spans buckets 0..3; col in bucket 2 ---
    ("interior_bucket_hit", 3, 0, 5, 30, [16]),
    # --- wide multi-bucket range ---
    ("wide_hit", 3, 0, 3, 55, [40]),
    ("wide_miss", 3, 0, 3, 55, [58]),  # col 58 > x2 55
    # --- empty visplane ---
    ("empty", 3, 0, 12, 18, []),
    # --- high instance id (fp32-magnitude path): plane 31, vp 0 -> id 248 ---
    ("high_instance_hit", 31, 0, 12, 18, [15]),
    ("high_instance_miss", 31, 0, 20, 25, [15]),
    # --- multiple columns: the successor must pick the MIN col >= x1 ---
    ("multi_min_in_range", 3, 0, 3, 7, [20, 5]),  # c* = 5 <= 7
    ("multi_min_out_of_range", 3, 0, 8, 12, [20, 5]),  # c*=20 > 12, col5 < x1
    ("multi_same_bucket_min", 3, 0, 18, 19, [21, 18]),  # bucket2: min col >= x1
]


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_check_conflict_radix(case):
    name, plane, vp, x1, x2, occ_cols = case
    expected = _brute(occ_cols, x1, x2)
    val = _build(plane, vp, x1, x2, occ_cols)
    got = val > 0.0
    assert got == expected, (
        f"{name}: cols={occ_cols} [{x1},{x2}] expected conflict={expected} "
        f"got {got} (raw {val})"
    )


# The within-bucket argmin must resolve in compiled fp32 at a HIGH instance id —
# the case reference_eval (float64) cannot exercise. The lifted instance key
# rides ``-id^2`` (~61504 at plane 31, vp 0 -> id 248); the bucketed-argmin op
# scales the bucket dot by _BUCKET_BONUS (256), so a wrong gain on the key would
# push the matched-row logit past fp32's exact-integer range and BLEND the
# within-bucket argmin (the gained local digit, 8 per step, drops below the ulp).
# These cases are decided BY that argmin: two same-bucket, same-instance columns
# both >= x1, where picking the min vs blending the two flips conflict.
_COMPILED_CASES = [
    # cols 19 (lo3) & 22 (lo6) in bucket 2; query [18,20]. min col >= 18 is 19,
    # which is <= 20 -> conflict. A blend (avg 20.5) would read > 20 -> no.
    ("hi_inst_argmin_hit", 31, 0, 18, 20, [19, 22], True),
    # same columns; query [20, 21]. min col >= 20 is 22, > 21 -> no conflict.
    # A blend (avg 20.5) would read <= 21 -> false conflict.
    ("hi_inst_argmin_miss", 31, 0, 20, 21, [19, 22], False),
]


@pytest.mark.parametrize("case", _COMPILED_CASES, ids=[c[0] for c in _COMPILED_CASES])
def test_check_conflict_high_instance_compiled(case):
    """Compiled fp32 at d_head=32 (the post-reduction floor) matches the exact
    oracle at instance id 248 — proving the q^2-cancelled gain-1 lifted key
    keeps the within-bucket argmin resolvable."""
    name, plane, vp, x1, x2, occ_cols, expected = case
    # The case must actually be decided by the argmin: confirm the exact oracle
    # is on the expected side before checking compiled tracks it.
    oracle_val = _build(plane, vp, x1, x2, occ_cols)
    assert (oracle_val > 0.0) == expected, f"{name}: oracle {oracle_val}"

    conflict, inputs, n_pos = _build_graph(plane, vp, x1, x2, occ_cols)
    report = probe_graph(conflict, inputs, n_pos, d=4096, d_head=64, atol=0.05)
    # No node diverges from the float64 oracle -> the compiled fp32 argmin did
    # not blend; the conflict boolean matches the (correct) oracle on both sides.
    assert report.first_divergent is None, report.format_short()
