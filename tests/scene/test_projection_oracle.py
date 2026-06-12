"""Plan F mandatory gate: seg-projection + occlusion + drawseg teacher-forced
next-token agreement.

Drives the real ``forward()`` graph teacher-forced on the sandbox golden token
stream for ``e1m1_subset`` pose 0, and asserts the graph emits the exact next
token the sandbox emits at every Phase-F-owned position — the seg-scan loop
(``R_Subsector`` / ``R_AddLine`` / endpoint angles / ``clipScan`` / ``drawseg.x2``
/ ``R_StoreWallRange`` / ``nextSeg``), the drawseg-scalar chain (``segKpart`` …
``drawseg.uPhase``), and the ``value`` / ``angleValue`` carriers emitted after
those markers.

Deferred owners are teacher-forced-and-skipped: the bbox sub-protocol
(``bspCheckBack`` / ``boxpos`` / ``bbox.*`` — Phase G) and the wall-column /
visplane / flat / pixel tokens (``setCursorX`` / ``R_CheckPlane`` / ``screenY``
/ ``planeMark`` / ``screenRange`` / ``clipUpdate`` — Phase H/J). Their carriers
are skipped by the "carrier preceded by an F marker" rule. The cross-position
channels (``SolidIntervals``, the recent-marker rows) are published at every
position regardless, so a later seg's ``clipScan`` query resolves through the
stubbed span — which is how the occlusion-discrimination assert below holds.

``reference_eval``-only (exact math, CPU, no compile), the Plan E/D pattern.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward
from torchwright_doom.vocab import VOCAB_TYPES

from ..prefill_fixture import row_index, tokens_to_input

# Phase-F protocol markers (sandbox token names). A carrier (value/angleValue)
# is compared only when the token *before* it is one of these — that excludes
# the deferred bbox/scene carriers, which follow bbox/scene markers.
_F_MARKERS = {
    "R_Subsector",
    "R_AddLine",
    "angle1",
    "theta1",
    "angle2",
    "theta2",
    "nextSeg",
    "clipScan",
    "drawseg.x2",
    "R_StoreWallRange",
    "segKpart",
    "segDcTmidMid",
    "segDcTmidUpper",
    "segDcTmidLower",
    "drawseg.meta",
    "drawseg.scale1.den",
    "drawseg.scale1",
    "drawseg.scale2.den",
    "drawseg.scale2",
    "drawseg.scalestep.den",
    "drawseg.scalestep",
    "drawseg.bsilheight",
    "drawseg.tsilheight",
    "drawseg.uPhase",
}
_CARRIERS = {"value", "angleValue"}
# A carrier is compared only when preceded by one of these markers (the markers
# whose carriers stay inside Phase F). ``drawseg.uPhase`` is excluded: its
# angleValue carrier's *next* token is ``R_CheckPlane`` (Phase H), which the
# reduced build NO_OP-stubs — that position is the F→H boundary, not an F
# comparison. (``drawseg.uPhase`` itself is still compared as a marker.)
_CARRIER_MARKERS = {
    "angle1",
    "theta1",
    "angle2",
    "theta2",
    "segDcTmidMid",
    "segDcTmidUpper",
    "segDcTmidLower",
    "drawseg.scale1.den",
    "drawseg.scale1",
    "drawseg.scale2.den",
    "drawseg.scale2",
    "drawseg.scalestep.den",
    "drawseg.scalestep",
    "drawseg.bsilheight",
    "drawseg.tsilheight",
}

# Carrier-value tolerances. Angles: the octant count vs the drafter's
# round(atan2) can differ by ±1 BAM at a tie. Values: the PWL-multiply scale
# denominators are inherently ~1-2 abs off (the sandbox forward has the same
# error vs the exact-trig drafter), ≈ a few × 1e-3 in the encoded [-1, 1] space;
# this tolerance passes that noise while still flagging gross divergence.
_ANGLE_BAM_TOL = 2
_VALUE_ENC_TOL = 3.0e-3

_VALUE_ROWS = 65536  # VALUE block [0, 65536), one row per quantization level
_ANGLE_ROW0 = 65536  # ANGLE_VALUE block start
_ANGLE_LO = -4096  # ANGLE_VALUE IntSlot lo

# Span past the prefill to teacher-force. ``reference_eval`` cost is O(n_pos²)
# over the F graph's many attention nodes, and the prefill is ~1057 positions,
# so n_pos (and memory) is dominated by the prefill floor — keep the AR span as
# small as the coverage needs. 215 covers the first subsector's full cycle
# through ``drawseg.uPhase`` (golden idx 202) plus its ``R_StoreWallRange`` fill.
# The cross-seg occlusion floor needs a later ``clipScan`` (golden idx ~578),
# i.e. span ~580 — gated by ``_OCCLUSION_AR_SPAN`` and skipped when the span is
# shorter (so the projection gate stays light).
_AR_SPAN = 215
_OCCLUSION_AR_SPAN = 580


def _umbrella() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_marker(full, i: int) -> bool:
    return full[i].type.name in _F_MARKERS


def _is_carrier(full, i: int) -> bool:
    return (
        full[i].type.name in _CARRIERS
        and i > 0
        and full[i - 1].type.name in _CARRIER_MARKERS
    )


def _is_compared(full, i: int) -> bool:
    return _is_marker(full, i) or _is_carrier(full, i)


def _carrier_delta(name: str, predicted_row: int, expected_row: int) -> float | None:
    """Value-space distance between predicted and expected carrier rows, or
    ``None`` if the predicted token is the wrong *type* (outside the carrier's
    block)."""
    if name == "angleValue":
        if not (_ANGLE_ROW0 <= predicted_row < _ANGLE_ROW0 + 8192):
            return None
        pa = predicted_row - _ANGLE_ROW0 + _ANGLE_LO
        ea = expected_row - _ANGLE_ROW0 + _ANGLE_LO
        return abs(((pa - ea + 4096) % 8192) - 4096)  # wrap-aware BAM distance
    # value
    if not (0 <= predicted_row < _VALUE_ROWS):
        return None
    return abs(predicted_row - expected_row) * 2.0 / (_VALUE_ROWS - 1)


@pytest.fixture(scope="module")
def projection_eval():
    """Build the reduced-F ``forward()`` graph once, teacher-force it on the
    sandbox golden stream, and ``reference_eval`` a single pass."""
    umbrella = _umbrella()
    if not (umbrella / "doom_sandbox").is_dir():
        pytest.skip("doom_sandbox sibling not present (standalone checkout)")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    if str(umbrella) not in sys.path:
        sys.path.insert(0, str(umbrella))

    fixtures = pytest.importorskip("doom_sandbox.fixtures")
    sb_prefill = pytest.importorskip("doom_sandbox.implementation.prefill")
    drafter = pytest.importorskip("doom_sandbox.implementation.reference_drafter")

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

    # The renderer builds every branch candidate at every position and masks by
    # token type in the dispatch, so a ``select`` / ``broadcast_select`` cond on
    # a *discarded* branch can land in a comparator ramp (e.g. ``gt_height`` of
    # two garbage recovered heights) and trip its ±1 ``c_tol`` Assert. At the
    # active row the cond is clean (DOOM heights are integers). The gate
    # validates via next-token agreement, not the debug Asserts, so silence the
    # Assert predicates for this oracle pass (they are stripped on the compiled
    # path anyway and only re-checked under ``debug=True``).
    import torchwright.graph.misc as _misc

    _orig_check = _misc.Assert._check
    _misc.Assert._check = lambda self, x: None
    try:
        cache = reference_eval(next_token, inputs, n_pos)
    finally:
        _misc.Assert._check = _orig_check
    return {
        "emitted": cache[next_token],
        "full": full,
        "real_pairs": real_pairs,
        "begin": begin,
        "n_pos": n_pos,
    }


def _scan_next_tokens(projection_eval) -> dict:
    """Teacher-forced next-token agreement scan over the compared span.

    Returns the marker mismatches (exact-row), carrier mismatches
    (value-tolerance), token-type coverage, and the first
    ``R_StoreWallRange`` index — shared by the marker and carrier tests so
    the module-scoped ``reference_eval`` runs once.
    """
    emitted = projection_eval["emitted"]
    full = projection_eval["full"]
    real_pairs = projection_eval["real_pairs"]
    begin = projection_eval["begin"]
    n_pos = projection_eval["n_pos"]
    w_embed_t = W_EMBED.t()

    coverage: Counter[str] = Counter()
    marker_mismatches = []
    carrier_mismatches = []
    first_store_idx = None
    for i in range(begin, n_pos - 1):
        if full[i].type.name == "R_StoreWallRange" and first_store_idx is None:
            first_store_idx = i
        if not _is_compared(full, i):
            continue
        coverage[full[i].type.name] += 1
        predicted_row = int(torch.argmax(emitted[i] @ w_embed_t).item())
        expected_row = row_index(*real_pairs[i + 1])
        next_name = full[i + 1].type.name
        desc = (
            f"pos {i} (in {full[i].type.name} {dict(full[i].values)}): emitted row "
            f"{predicted_row} != sandbox {next_name} "
            f"{dict(full[i + 1].values)} (row {expected_row})"
        )
        # Tolerance is set by the *next* token's type: a continuous VALUE /
        # ANGLE_VALUE carrier gets a value-space tolerance; everything else (a
        # discrete marker / control token) must match the exact row.
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
        "first_store_idx": first_store_idx,
    }


def test_projection_carriers_within_tolerance(projection_eval) -> None:
    """Numeric VALUE / ANGLE_VALUE carriers within tolerance (octant ±BAM,
    multiply noise). The 2-byte VALUE-carrier digit-quad high byte now floors a
    continuous 16-bit q with the cancellation-free ``floor_int`` (was the
    integer-only ``thermometer_floor_div``, which interpolated junk)."""
    scan = _scan_next_tokens(projection_eval)
    assert not scan[
        "carrier_mismatches"
    ], "projection CARRIER value mismatches:\n" + "\n".join(
        scan["carrier_mismatches"][:25]
    )


def test_projection_markers_exact(projection_eval) -> None:
    full = projection_eval["full"]
    n_pos = projection_eval["n_pos"]
    scan = _scan_next_tokens(projection_eval)
    marker_mismatches = scan["marker_mismatches"]
    coverage = scan["coverage"]
    first_store_idx = scan["first_store_idx"]

    # The protocol routing is the hard correctness claim — it must be exact.
    assert not marker_mismatches, "projection MARKER next-token mismatches:\n" + (
        "\n".join(marker_mismatches[:25])
    )

    # Coverage floors — a capped span must actually exercise the phase.
    assert coverage["R_AddLine"] >= 3, f"too few R_AddLine compared: {coverage}"
    assert coverage["angleValue"] >= 8, f"too few angle carriers: {coverage}"
    assert (
        coverage["R_StoreWallRange"] >= 1
    ), f"no R_StoreWallRange (interval fill) exercised: {coverage}"
    assert coverage["drawseg.uPhase"] >= 1, f"no full drawseg-scalar chain: {coverage}"

    # Occlusion-discrimination floor: at least one clipScan (FIND_RUN) compared
    # *after* the first R_StoreWallRange fill, so its covered_and_end query ran
    # against a populated SolidIntervals and still produced the exact next token.
    # Only enforced when the (heavier) occlusion span is run.
    assert first_store_idx is not None
    later_clipscans = sum(
        1
        for i in range(first_store_idx + 1, n_pos - 1)
        if full[i].type.name == "clipScan" and _is_compared(full, i)
    )
    if _AR_SPAN >= _OCCLUSION_AR_SPAN:
        assert later_clipscans >= 1, (
            "occlusion not exercised: no clipScan compared after the first "
            "R_StoreWallRange (extend _AR_SPAN)"
        )
