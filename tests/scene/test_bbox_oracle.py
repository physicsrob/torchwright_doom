"""Plan G mandatory gate: R_CheckBBox visibility pruning teacher-forced
next-token agreement.

Drives the real ``forward()`` graph teacher-forced on the sandbox golden token
stream for ``e1m1_subset`` pose 0, and asserts the graph emits the exact next
token at every bbox-owned position — the ``TRAVERSE_BETWEEN`` entry
(``bspCheckBack``), the boxpos region classifier, the two extreme corner marks
(``bbox.x1`` / ``bbox.y1`` / ``bbox.x2`` / ``bbox.y2``), their world/theta angle
marks (``bbox.angle1`` / ``bbox.theta1`` / ``bbox.angle2`` / ``bbox.theta2``), the
occlusion scan (``bboxClipScan``), and the ``value`` / ``angleValue`` carriers
emitted after those markers.

Every bbox terminal routes to a *real* head — descend (``R_Subsector`` /
``bspFront``) or prune (``bspReturn``) — so unlike the projection gate there is no
Phase-H/J boundary to exclude; the full bbox sub-protocol is compared.

The headline is that the prune decision is **load-bearing on the now-populated
occlusion state**. Two regimes are exercised:

* The first two bbox checks (golden idx ~1147 / ~1200) run *before* the first
  ``R_StoreWallRange`` fill (idx ~1234), so they query an empty
  ``SolidIntervals`` (the always-visible regime) and descend.
* Later checks query a *populated* ``SolidIntervals``. A column covered by a
  solid wall makes ``after_scan`` skip to the covering interval's end + 1 — a
  second ``bboxClipScan`` at a jumped ``x`` (golden idx ~2470 / ~2543 / ~2631).
  That skip cannot happen against an empty interval set, so it is the bbox
  occlusion-load-bearing proof.

One bbox check prunes (golden idx ~2572 -> ``bspReturn`` at ~2590). On
``e1m1_subset`` pose 0 that prune is the **zero-width** path (the bbox projects
to no screen columns, so ``after_bbox_angle_value`` emits ``TRAVERSE_RETURN``
instead of ``BBOX_SCAN``), not an interval-driven ``beyond``-last prune — pose 0
has no interval-driven prune. The gate therefore asserts the prune path *and*,
separately, the interval-driven skip; together they cover "the bbox sub-protocol
prunes correctly" and "the occlusion scan reads the populated intervals."

``reference_eval``-only (exact math, CPU, no compile), the Plan E/D/F pattern.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.graph_debug import silenced_graph_asserts
from torchwright_doom.inference.diagnostic import carrier_delta as _carrier_delta
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward
from torchwright_doom.vocab import VOCAB_TYPES

from ..prefill_fixture import row_index, tokens_to_input
from ..sandbox_support import import_sandbox, require_doom_sandbox

# Bbox protocol markers (sandbox token names). A carrier (value/angleValue) is
# compared only when the token *before* it is a bbox corner/angle marker (below),
# so the projection / scene carriers that share the same VALUE/ANGLE_VALUE type
# are excluded.
_G_MARKERS = {
    "bspCheckBack",  # TRAVERSE_BETWEEN
    "boxpos",
    "bbox.x1",
    "bbox.y1",
    "bbox.x2",
    "bbox.y2",
    "bbox.angle1",
    "bbox.theta1",
    "bbox.angle2",
    "bbox.theta2",
    "bboxClipScan",
}
_CARRIERS = {"value", "angleValue"}
# Bbox markers that emit a numeric carrier next. ``bbox.theta2``'s angleValue
# carrier is included: its *next* token is the prune/scan decision
# (``bspReturn`` / ``bboxClipScan``), both real heads — so it is compared, not
# excluded (contrast the projection gate's ``drawseg.uPhase``, whose carrier ran
# into a Phase-H NO_OP stub).
_CARRIER_MARKERS = {
    "bbox.x1",
    "bbox.y1",
    "bbox.x2",
    "bbox.y2",
    "bbox.angle1",
    "bbox.theta1",
    "bbox.angle2",
    "bbox.theta2",
}

# Carrier-value tolerances (same basis as the projection gate). Angles: octant
# count vs round(atan2) can differ by ±1 BAM at a tie. Corner values: the R0
# digit-quad floor carries the multiply/encode noise, a few × 1e-3 in the
# encoded [-1, 1] space.
_ANGLE_BAM_TOL = 2
_VALUE_ENC_TOL = 3.0e-3


# Span past the prefill to teacher-force. The bbox checks span golden idx
# ~1147 (first, empty-interval descend) through ~2633 (last cycle). Reaching the
# interval-driven skips (~2470 / ~2543 / ~2631) and the single prune (~2590)
# needs a span that covers the first ~1.5 subsectors' worth of stubbed render in
# between. 1585 reaches ~idx 2641 (prefill ~1057), covering every bbox cycle.
# ``reference_eval`` cost is O(n_pos^2) over the graph's attention nodes, so this
# is the heaviest gate in the suite; it is still seconds-to-low-minutes on CPU.
_AR_SPAN = 1585


def _is_marker(full, i: int) -> bool:
    return full[i].type.name in _G_MARKERS


def _is_carrier(full, i: int) -> bool:
    return (
        full[i].type.name in _CARRIERS
        and i > 0
        and full[i - 1].type.name in _CARRIER_MARKERS
    )


def _compared(full, i: int) -> bool:
    return _is_marker(full, i) or _is_carrier(full, i)


@pytest.fixture(scope="module")
def bbox_eval():
    """Build the ``forward()`` graph once, teacher-force it on the sandbox golden
    stream, and ``reference_eval`` a single pass."""
    require_doom_sandbox()

    fixtures = import_sandbox("doom_sandbox.fixtures")
    sb_prefill = import_sandbox("doom_sandbox.implementation.prefill")
    drafter = import_sandbox("doom_sandbox.implementation.reference_drafter")

    name_to_real = {t.name: t for t in VOCAB_TYPES}

    scene = fixtures.load_fixture("e1m1_subset")
    pose = scene.test_poses[0]
    prefill = list(sb_prefill.get_prefill(scene, pose))
    golden = list(drafter.expected_ar_tokens(scene, pose))
    full = prefill + golden
    begin = len(prefill) - 1  # the BEGIN row seeds the AR loop

    n_pos = min(begin + _AR_SPAN, len(full) - 1) + 1
    real_pairs = [(name_to_real[t.type.name], dict(t.values)) for t in full]
    inputs = {"iv": tokens_to_input(real_pairs[:n_pos])}

    d_embed = TOKEN_VOCAB.layout.d_embed
    iv = create_input("iv", d_embed)
    past = GraphPast(input_vec=iv, pos_encoding=create_pos_encoding())
    next_token = forward(iv, past, create_pos_encoding())

    with silenced_graph_asserts():
        cache = reference_eval(next_token, inputs, n_pos)
    return {
        "emitted": cache[next_token],
        "full": full,
        "real_pairs": real_pairs,
        "begin": begin,
        "n_pos": n_pos,
    }


def _scan_next_tokens(bbox_eval) -> dict:
    """Teacher-forced next-token agreement scan over the bbox-owned span.

    Returns marker mismatches (exact-row), carrier mismatches (value-tolerance),
    token-type coverage, the descend/prune/skip tallies, and the first
    ``R_StoreWallRange`` index — shared by the marker and carrier tests so the
    module-scoped ``reference_eval`` runs once.
    """
    emitted = bbox_eval["emitted"]
    full = bbox_eval["full"]
    real_pairs = bbox_eval["real_pairs"]
    begin = bbox_eval["begin"]
    n_pos = bbox_eval["n_pos"]
    w_embed_t = W_EMBED.t()

    coverage: Counter[str] = Counter()
    marker_mismatches = []
    carrier_mismatches = []
    n_descend = n_prune = n_interval_skip = 0
    first_store_idx = None
    for i in range(begin, n_pos - 1):
        if full[i].type.name == "R_StoreWallRange" and first_store_idx is None:
            first_store_idx = i
        if not _compared(full, i):
            continue
        coverage[full[i].type.name] += 1
        next_name = full[i + 1].type.name
        # Bbox terminal-decision tallies, read off the golden stream (the
        # asserted agreement below confirms the graph reproduces them).
        if full[i].type.name == "bboxClipScan":
            if next_name in ("R_Subsector", "bspFront"):
                n_descend += 1
            elif next_name == "bboxClipScan":
                n_interval_skip += (
                    1  # covered -> skip to next_x (needs populated intervals)
                )
        if next_name == "bspReturn":
            n_prune += 1

        predicted_row = int(torch.argmax(emitted[i] @ w_embed_t).item())
        expected_row = row_index(*real_pairs[i + 1])
        desc = (
            f"pos {i} (in {full[i].type.name} {dict(full[i].values)}): emitted row "
            f"{predicted_row} != sandbox {next_name} "
            f"{dict(full[i + 1].values)} (row {expected_row})"
        )
        if next_name in _CARRIERS:
            tol = _ANGLE_BAM_TOL if next_name == "angleValue" else _VALUE_ENC_TOL
            delta = _carrier_delta(next_name, predicted_row, expected_row)
            if delta is None or delta > tol:
                carrier_mismatches.append(f"{desc}  [delta={delta}]")
        else:
            if predicted_row != expected_row:
                marker_mismatches.append(desc)
    return {
        "marker_mismatches": marker_mismatches,
        "carrier_mismatches": carrier_mismatches,
        "coverage": coverage,
        "n_descend": n_descend,
        "n_prune": n_prune,
        "n_interval_skip": n_interval_skip,
        "first_store_idx": first_store_idx,
    }


def test_bbox_carriers_within_tolerance(bbox_eval) -> None:
    """The bbox corner ``value`` (R0) and world/theta ``angleValue`` carriers are
    within tolerance (octant ±BAM; the R0 digit-quad floor's encode noise)."""
    scan = _scan_next_tokens(bbox_eval)
    assert not scan[
        "carrier_mismatches"
    ], "bbox CARRIER value mismatches:\n" + "\n".join(scan["carrier_mismatches"][:25])


def test_bbox_markers_exact(bbox_eval) -> None:
    scan = _scan_next_tokens(bbox_eval)
    marker_mismatches = scan["marker_mismatches"]
    coverage = scan["coverage"]

    # The protocol routing — region classify, corner select, occlusion
    # descend/skip/prune — is the hard correctness claim; it must be exact.
    assert not marker_mismatches, "bbox MARKER next-token mismatches:\n" + (
        "\n".join(marker_mismatches[:25])
    )

    # Coverage floors — a capped span must actually exercise the phase.
    assert coverage["boxpos"] >= 2, f"too few bbox cycles (boxpos): {coverage}"
    assert coverage["bbox.theta2"] >= 2, f"too few full bbox corner cycles: {coverage}"
    assert coverage["bboxClipScan"] >= 2, f"too few occlusion scans: {coverage}"

    # Decision coverage: descend and prune both compared, and the occlusion scan
    # read a populated SolidIntervals (a covered-column skip).
    assert scan["n_descend"] >= 1, "no bbox descend (bboxClipScan -> child) compared"
    assert scan["n_prune"] >= 1, (
        "no bbox prune (-> bspReturn) compared: extend _AR_SPAN to reach the "
        "zero-width prune at golden idx ~2590"
    )
    assert scan["n_interval_skip"] >= 1, (
        "occlusion not exercised: no bboxClipScan covered-column skip compared "
        "(the bbox scan never read a populated SolidIntervals); extend _AR_SPAN"
    )
    # The first bbox checks precede the first interval fill (empty-interval
    # descend regime), so the populated-interval skips are genuinely later.
    assert scan["first_store_idx"] is not None
