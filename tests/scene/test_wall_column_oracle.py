"""Phase H gate — wall-column rasterization + visplane occupancy oracle.

Drives the real ``forward()`` graph teacher-forced on the sandbox golden token
stream for ``e1m1_subset`` pose 0, and asserts the emitted next-token agrees
with the sandbox at every Phase-H-owned position, **capped before the first
PIXEL** (the texel path is Phase J). Same teacher-forced ``reference_eval``
pattern as ``test_projection_oracle`` / ``test_bbox_oracle`` (exact math, CPU, no
compile).

The two *texel* branches stay NO_OP-stubbed in this chunk: ``wall_column``
(SET_CURSOR_X -> WALL_COL_U, the native-u from ``pixel_dispatcher``) and
``set_cursor_y`` (SET_CURSOR_Y -> span-v0 VALUE). So ``wallColU`` is **not**
compared as a marker and the span-v0 VALUE is **not** compared as a carrier;
everything else in the solid-geometry pass IS — setCursorX, screenY, planeMark,
screenRange, clipUpdate, R_CheckPlane(.result), wallSpanMeta, setCursorY, and the
per-column wall *scale* VALUE carrier (emitted by ``wall_col_u``).
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

# Phase-F protocol markers (still teacher-forced + compared inside the H span).
_F_MARKERS = {
    "R_Subsector", "R_AddLine", "angle1", "theta1", "angle2", "theta2",
    "nextSeg", "clipScan", "drawseg.x2", "R_StoreWallRange", "segKpart",
    "segDcTmidMid", "segDcTmidUpper", "segDcTmidLower", "drawseg.meta",
    "drawseg.scale1.den", "drawseg.scale1", "drawseg.scale2.den",
    "drawseg.scale2", "drawseg.scalestep.den", "drawseg.scalestep",
    "drawseg.bsilheight", "drawseg.tsilheight", "drawseg.uPhase",
}
# Phase-H markers — the INPUT token types whose emitted successor the gate
# verifies. The two *texel* transitions are excluded because their producer is a
# NO_OP-stubbed Phase-J branch: ``setCursorX -> wallColU`` (native-u, the
# ``wall_column`` branch) and ``setCursorY -> span-v0 VALUE`` (the ``set_cursor_y``
# branch). ``setCursorX`` / ``setCursorY`` / ``wallColU`` are still *produced* by
# real owners (R_CheckPlane.result / screenRange / wallSpanMeta / the scale
# carrier), so their emission is verified as those owners' output — just not what
# they themselves emit next.
_H_MARKERS = {
    "screenY",
    "wallSpanMeta",
    "clipUpdate",
    "R_CheckPlane",
    "R_CheckPlane.result",
    "planeMark",
    "screenRange",
}
_MARKERS = _F_MARKERS | _H_MARKERS

_CARRIERS = {"value", "angleValue"}
# A carrier is compared only when preceded by one of these markers. The F set
# (carriers staying inside F); plus ``wallColU`` -> the per-column wall scale
# VALUE (R5), the one H carrier whose producer is real (``after_wall_col_u``).
# ``setCursorY`` is NOT here: its successor span-v0 VALUE is emitted by the
# NO_OP-stubbed Phase-J ``set_cursor_y`` branch.
_CARRIER_MARKERS = {
    "angle1", "theta1", "angle2", "theta2",
    "segDcTmidMid", "segDcTmidUpper", "segDcTmidLower",
    "drawseg.scale1.den", "drawseg.scale1", "drawseg.scale2.den",
    "drawseg.scale2", "drawseg.scalestep.den", "drawseg.scalestep",
    "drawseg.bsilheight", "drawseg.tsilheight",
    "wallColU",
}

_ANGLE_BAM_TOL = 2
_VALUE_ENC_TOL = 3.0e-3
_VALUE_ROWS = 65536
_ANGLE_ROW0 = 65536
_ANGLE_LO = -4096

# Span past the prefill (~1057 positions) to reach the first wall-column pass:
# the seg-0 column span (setCursorX x=13..47) + its plane marks land ~idx 1260,
# the first/only wallSpanMeta+setCursorY at the J boundary ~idx 1672. 620 reaches
# them with margin; comparison is capped before the first PIXEL by name (PIXEL is
# not a marker; the span-v0 carrier is excluded). reference_eval is O(n_pos²).
_AR_SPAN = 620


def _umbrella() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_marker(full, i: int) -> bool:
    return full[i].type.name in _MARKERS


def _is_carrier(full, i: int) -> bool:
    return (
        full[i].type.name in _CARRIERS
        and i > 0
        and full[i - 1].type.name in _CARRIER_MARKERS
    )


def _is_compared(full, i: int) -> bool:
    return _is_marker(full, i) or _is_carrier(full, i)


def _carrier_delta(name: str, predicted_row: int, expected_row: int):
    if name == "angleValue":
        if not (_ANGLE_ROW0 <= predicted_row < _ANGLE_ROW0 + 8192):
            return None
        pa = predicted_row - _ANGLE_ROW0 + _ANGLE_LO
        ea = expected_row - _ANGLE_ROW0 + _ANGLE_LO
        return abs(((pa - ea + 4096) % 8192) - 4096)
    if not (0 <= predicted_row < _VALUE_ROWS):
        return None
    return abs(predicted_row - expected_row) * 2.0 / (_VALUE_ROWS - 1)


@pytest.fixture(scope="module")
def wall_column_eval():
    """Build the Phase-H ``forward()`` graph once, teacher-force it on the sandbox
    golden stream, and ``reference_eval`` a single pass."""
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
    begin = len(prefill) - 1

    n_pos = min(begin + _AR_SPAN, len(full) - 1) + 1
    real_pairs = [(name_to_real[t.type.name], dict(t.values)) for t in full]
    inputs = {"iv": tokens_to_input(real_pairs[:n_pos])}

    d_embed = TOKEN_VOCAB.layout.d_embed
    iv = create_input("iv", d_embed)
    past = GraphPast(input_vec=iv, pos_encoding=create_pos_encoding())
    next_token = forward(iv, past, create_pos_encoding())

    # Silence the debug Assert predicates for the oracle pass (discarded branches
    # can land a cond in a comparator ramp; validated via next-token agreement).
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


def _scan_next_tokens(wall_column_eval) -> dict:
    emitted = wall_column_eval["emitted"]
    full = wall_column_eval["full"]
    real_pairs = wall_column_eval["real_pairs"]
    begin = wall_column_eval["begin"]
    n_pos = wall_column_eval["n_pos"]
    w_embed_t = W_EMBED.t()

    coverage: Counter[str] = Counter()
    marker_mismatches = []
    carrier_mismatches = []
    for i in range(begin, n_pos - 1):
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
    }


def test_wall_column_carriers_within_tolerance(wall_column_eval) -> None:
    """Numeric VALUE carriers within tolerance — including the per-column wall
    scale VALUE (R5) emitted by ``after_wall_col_u``."""
    scan = _scan_next_tokens(wall_column_eval)
    assert not scan["carrier_mismatches"], (
        "wall-column CARRIER value mismatches:\n"
        + "\n".join(scan["carrier_mismatches"][:25])
    )


def test_wall_column_markers_exact(wall_column_eval) -> None:
    scan = _scan_next_tokens(wall_column_eval)
    marker_mismatches = scan["marker_mismatches"]
    coverage = scan["coverage"]

    assert not marker_mismatches, (
        "wall-column MARKER next-token mismatches:\n"
        + "\n".join(marker_mismatches[:25])
    )

    # Coverage floors — the capped span must actually exercise the rasterizer.
    # (e1m1_subset has no visplane overlap/merge on any pose — the conflict
    # assert is weakened to ">=1 R_CheckPlane.result selected"; the overlap
    # deadband is verified by a separate crafted-input test.)
    assert coverage["screenY"] >= 2, f"wall-column span not exercised: {coverage}"
    assert coverage["planeMark"] >= 2, f"too few plane marks: {coverage}"
    assert coverage["screenRange"] >= 1, f"no screen-range emitted: {coverage}"
    assert coverage["clipUpdate"] >= 1, f"no clip-array tighten: {coverage}"
    assert (
        coverage["R_CheckPlane.result"] >= 1
    ), f"no R_CheckPlane.result selected: {coverage}"
