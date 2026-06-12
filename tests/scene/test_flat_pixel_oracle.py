"""Phase J2 gate — flat floor/ceiling span pass oracle (through DONE).

Drives the real ``forward()`` graph teacher-forced on the sandbox golden token
stream for ``e1m1_start_room_textured`` over a window that reaches the flat pass
(``R_DrawPlanes`` … the per-visplane ``R_MakeSpans`` columns, ``SPAN_ROW``s, the
flat ``SET_CURSOR_X`` arm, flat pixels) and the frame tail to ``DONE``. Asserts:

* every non-PIXEL control token (wall + flat) agrees at the exact ``W_EMBED``
  row — including the flat-pass spine ``R_DrawPlanes`` / ``setCursorDirectionX`` /
  ``R_DrawPlanes.nextPlane`` / ``R_DrawPlanes.nextVp`` / ``visplaneBegin`` /
  ``R_MakeSpans.col`` / ``R_MakeSpans.closeSlot`` / ``R_MapPlane.row`` and the
  terminal ``DONE``;
* numeric carriers within the value-encoding tolerance;
* every PIXEL color (wall *and* flat) is within the sandbox's option set (the
  flat option set additionally allows ±1 colormap row).

Same teacher-forced ``reference_eval`` (exact math, CPU, no compile). The window
spans the whole frame by default; set ``TWDOOM_J2_SPAN=<n>`` to cap the AR span
to ``n`` positions past the seed for faster iteration. ``reference_eval`` is
O(n_pos²); the full-frame run is the heaviest gate in the suite.
"""

from __future__ import annotations

import os
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
from torchwright_doom.vocab import PIXEL, VOCAB_TYPES

from ..prefill_fixture import row_index, tokens_to_input
from ..sandbox_support import import_sandbox, require_doom_sandbox

_CARRIERS = {"value", "angleValue"}
_ANGLE_BAM_TOL = 2
# Projection VALUE carriers (dc_texturemid R3 / scale-denominators R6) carry up
# to ~4e-3 encoded float32 PWL noise on the textured fixture's segs (the H/F
# projection chain; the geometry-only fixture stays under 3e-3). Downstream-
# absorbed by the ±4 UV pixel option set. 5e-3 covers it with margin; the
# J-owned R3 v0 carrier sits far below.
_VALUE_ENC_TOL = 5.0e-3
# wallColU.u_idx is floor(-(rw_distance·tan)) via a float32 PWL product; near an
# integer boundary it differs ±1 from the float64 reference's int(). The texel
# takes it mod texture width and the ±4 UV option set absorbs ±1 (the sandbox
# forward uses the identical multiply+floor). So wallColU is a ±1-tolerant marker.
_WALLCOL_U_NAME = "wallColU"
_WALLCOL_U_TOL = 1

_PIXEL_NAME = PIXEL.name


def _decode_pixel_xy(full) -> dict[int, tuple[int, int]]:
    """Host pixel decode (mirrors sandbox ``extract.extract_pixel_pass``): walk the
    token stream tracking cursor + direction, recording each PIXEL's ``(x, y)``.
    Walls advance Y (default); flats advance X (after SET_CURSOR_DIRECTION_X)."""
    cursor_dx, cursor_dy = 0, 1
    cursor_x: int | None = None
    cursor_y: int | None = None
    xy: dict[int, tuple[int, int]] = {}
    for i, tok in enumerate(full):
        name = tok.type.name
        if name == "setCursorDirectionX":
            cursor_dx, cursor_dy = 1, 0
        elif name == "setCursorDirectionY":
            cursor_dx, cursor_dy = 0, 1
        elif name == "setCursorX":
            cursor_x = int(tok.values["x"])
        elif name == "setCursorY":
            cursor_y = int(tok.values["y"])
        elif name == _PIXEL_NAME:
            assert cursor_x is not None and cursor_y is not None
            xy[i] = (cursor_x, cursor_y)
            cursor_x += cursor_dx
            cursor_y += cursor_dy
    return xy


@pytest.fixture(scope="module")
def flat_pixel_eval():
    require_doom_sandbox()

    fixtures = import_sandbox("doom_sandbox.fixtures")
    sb_prefill = import_sandbox("doom_sandbox.implementation.prefill")
    drafter = import_sandbox("doom_sandbox.implementation.reference_drafter")
    reference = import_sandbox("doom_sandbox.implementation.reference")
    asset_banks = import_sandbox("doom_sandbox.implementation.asset_banks")

    name_to_real = {t.name: t for t in VOCAB_TYPES}

    scene = fixtures.load_fixture("e1m1_start_room_textured")
    pose = scene.test_poses[0]
    prefill = list(sb_prefill.get_prefill(scene, pose))
    golden = list(drafter.expected_ar_tokens(scene, pose))
    full = prefill + golden
    begin = len(prefill) - 1

    span_env = os.environ.get("TWDOOM_J2_SPAN")
    if span_env is not None:
        n_pos = min(begin + int(span_env), len(full) - 1) + 1
    else:
        n_pos = len(full)  # full frame, through DONE
    real_pairs = [(name_to_real[t.type.name], dict(t.values)) for t in full]
    inputs = {"iv": tokens_to_input(real_pairs[:n_pos])}

    options = {}
    for opt in reference.expected_pixel_color_options(scene, pose):
        options.setdefault((opt.x, opt.y), set()).update(opt.colors)
    playpal = [tuple(int(c) for c in rgb) for rgb in asset_banks.PLAYPAL]

    d_embed = TOKEN_VOCAB.layout.d_embed
    iv = create_input("iv", d_embed)
    past = GraphPast(input_vec=iv, pos_encoding=create_pos_encoding())
    next_token = forward(iv, past, create_pos_encoding())

    with silenced_graph_asserts():
        cache = reference_eval(next_token, inputs, n_pos)

    pixel_start, _ = TOKEN_VOCAB.type_to_row_range[name_to_real[_PIXEL_NAME]]
    return {
        "emitted": cache[next_token],
        "full": full,
        "real_pairs": real_pairs,
        "begin": begin,
        "n_pos": n_pos,
        "pixel_xy": _decode_pixel_xy(full),
        "options": options,
        "playpal": playpal,
        "pixel_start": pixel_start,
    }


def _scan(flat_pixel_eval) -> dict:
    emitted = flat_pixel_eval["emitted"]
    full = flat_pixel_eval["full"]
    real_pairs = flat_pixel_eval["real_pairs"]
    begin = flat_pixel_eval["begin"]
    n_pos = flat_pixel_eval["n_pos"]
    pixel_xy = flat_pixel_eval["pixel_xy"]
    options = flat_pixel_eval["options"]
    playpal = flat_pixel_eval["playpal"]
    pixel_start = flat_pixel_eval["pixel_start"]
    w_embed_t = W_EMBED.t()

    coverage: Counter[str] = Counter()
    marker_mismatches = []
    carrier_mismatches = []
    pixel_mismatches = []
    for i in range(begin, n_pos - 1):
        next_name = full[i + 1].type.name
        coverage[next_name] += 1
        predicted_row = int(torch.argmax(emitted[i] @ w_embed_t).item())
        expected_row = row_index(*real_pairs[i + 1])
        desc = (
            f"pos {i} (in {full[i].type.name} {dict(full[i].values)}): emitted row "
            f"{predicted_row} != sandbox {next_name} "
            f"{dict(full[i + 1].values)} (row {expected_row})"
        )
        if next_name == _PIXEL_NAME:
            x, y = pixel_xy[i + 1]
            color = predicted_row - pixel_start
            if not (0 <= color < 256):
                pixel_mismatches.append(f"{desc}  [not a PIXEL row]")
                continue
            rgb = playpal[color]
            allowed = options.get((x, y))
            if allowed is None or rgb not in allowed:
                pixel_mismatches.append(
                    f"pos {i}: pixel ({x},{y}) color {color} -> {rgb} not in "
                    f"option set (size {0 if allowed is None else len(allowed)})"
                )
        elif next_name in _CARRIERS:
            tol = _ANGLE_BAM_TOL if next_name == "angleValue" else _VALUE_ENC_TOL
            delta = _carrier_delta(next_name, predicted_row, expected_row)
            if delta is None or delta > tol:
                carrier_mismatches.append(f"{desc}  [delta={delta}]")
        elif next_name == _WALLCOL_U_NAME:
            if abs(predicted_row - expected_row) > _WALLCOL_U_TOL:
                marker_mismatches.append(desc + "  [wallColU off by >1]")
        else:
            if predicted_row != expected_row:
                marker_mismatches.append(desc)
    return {
        "marker_mismatches": marker_mismatches,
        "carrier_mismatches": carrier_mismatches,
        "pixel_mismatches": pixel_mismatches,
        "coverage": coverage,
    }


def test_flat_markers_exact(flat_pixel_eval) -> None:
    scan = _scan(flat_pixel_eval)
    assert not scan[
        "marker_mismatches"
    ], "flat-pass MARKER next-token mismatches:\n" + "\n".join(
        scan["marker_mismatches"][:30]
    )


def test_flat_carriers_within_tolerance(flat_pixel_eval) -> None:
    scan = _scan(flat_pixel_eval)
    assert not scan[
        "carrier_mismatches"
    ], "flat-pass CARRIER value mismatches:\n" + "\n".join(
        scan["carrier_mismatches"][:25]
    )


def test_flat_pixel_colors_in_option_set(flat_pixel_eval) -> None:
    scan = _scan(flat_pixel_eval)
    assert not scan[
        "pixel_mismatches"
    ], "flat-pass PIXEL color option-set mismatches:\n" + "\n".join(
        scan["pixel_mismatches"][:30]
    )


def test_flat_coverage_floors(flat_pixel_eval) -> None:
    scan = _scan(flat_pixel_eval)
    coverage = scan["coverage"]
    # The flat-pass spine + at least one open/close span + flat pixels.
    assert coverage["R_DrawPlanes"] >= 1, f"flat pass not reached: {coverage}"
    assert coverage["visplaneBegin"] >= 1, f"no visplane begun: {coverage}"
    assert coverage["R_MakeSpans.col"] >= 1, f"no R_MakeSpans column: {coverage}"
    assert coverage["R_MakeSpans.closeSlot"] >= 1, f"no span close: {coverage}"
    assert coverage["R_MapPlane.row"] >= 1, f"no span row: {coverage}"
    assert coverage[_PIXEL_NAME] >= 1, f"no pixels: {coverage}"


def test_flat_reaches_done(flat_pixel_eval) -> None:
    """When the window spans the full frame, the terminal DONE transition is
    teacher-forced and compared (asserted exact by ``test_flat_markers_exact``)."""
    if os.environ.get("TWDOOM_J2_SPAN") is not None:
        pytest.skip("capped AR span (TWDOOM_J2_SPAN set) does not reach DONE")
    scan = _scan(flat_pixel_eval)
    assert scan["coverage"]["done"] >= 1, "frame tail did not reach DONE"
