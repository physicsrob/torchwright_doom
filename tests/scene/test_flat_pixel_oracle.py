"""Phase J2 gate — flat floor/ceiling span pass oracle (through DONE).

Drives the real ``forward()`` graph teacher-forced on the in-tree pydoom
drafter's golden token stream for the e1m1 production frame (loaded from
``configs/e1m1.yaml`` + ``doom1.wad``) over a window that reaches the flat pass
(``R_DrawPlanes`` … the per-visplane ``R_MakeSpans`` columns, ``SPAN_ROW``s, the
flat ``SET_CURSOR_X`` arm, flat pixels) and the frame tail to ``DONE``. Asserts:

* every non-PIXEL control token (wall + flat) agrees at the exact ``W_EMBED``
  row — including the flat-pass spine ``R_DrawPlanes`` / ``setCursorDirectionX`` /
  ``R_DrawPlanes.nextPlane`` / ``R_DrawPlanes.nextVp`` / ``visplaneBegin`` /
  ``R_MakeSpans.col`` / ``R_MakeSpans.closeSlot`` / ``R_MapPlane.row`` and the
  terminal ``DONE``;
* numeric carriers within the value-encoding tolerance;
* every PIXEL color (wall *and* flat) is within the Python renderer's option set
  (the flat option set additionally allows ±1 colormap row).

Same teacher-forced ``reference_eval`` (exact math, CPU, no compile). The window
spans the whole frame by default; set ``TWDOOM_J2_SPAN=<n>`` to cap the AR span
to ``n`` positions past the seed for faster iteration. ``reference_eval`` is
O(n_pos²); the full-frame run is the heaviest gate in the suite.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest
import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.graph_debug import silenced_graph_asserts
from torchwright_doom.inference.diagnostic import carrier_delta as _carrier_delta
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward
from torchwright_doom.inference.tokens_bridge import row_index, tokens_to_input
from torchwright_doom.vocab import PIXEL

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
# takes it mod texture width and the ±4 UV option set absorbs ±1 (the original
# uses the identical multiply+floor). So wallColU is a ±1-tolerant marker.
_WALLCOL_U_NAME = "wallColU"
_WALLCOL_U_TOL = 1

_PIXEL_NAME = PIXEL.name
# Color shares the pixel row with the width slot: row = start + color * w_levels
# + w_idx. Recover color by flooring the row offset by the number of widths.
_PIXEL_W_LEVELS = PIXEL.slots["w"].hi - PIXEL.slots["w"].lo


def _predicted_rows(emitted_slice: torch.Tensor) -> list[int]:
    """Decode each emitted residual to its ``W_EMBED`` row via
    ``argmax(emitted @ W_EMBEDᵀ)``. This is the full-frame hot spot: a
    ``(count, vocab)`` GEMM. ``reference_eval`` runs on CPU, where that GEMM is
    memory-bound and slow, so run the decode on the GPU when one is present
    (argmax is exact — identical integer rows either way). Chunked to bound the
    intermediate's footprint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w_embed_t = W_EMBED.t().to(device)
    rows: list[int] = []
    for start in range(0, emitted_slice.shape[0], 8192):
        block = emitted_slice[start : start + 8192].to(device)
        rows.extend(torch.argmax(block @ w_embed_t, dim=1).tolist())
    return rows


def _decode_pixel_xy(full) -> dict[int, tuple[int, int]]:
    """Host pixel decode (mirrors the production host ``decode.py::_walk_pixels``):
    walk the token stream tracking cursor + direction, recording each PIXEL's base
    ``(x, y)``. Walls advance Y (default); flats advance X (after
    SET_CURSOR_DIRECTION_X). Width-aware: each PIXEL paints ``w`` cells, so in a
    flat span (moving in X) the cursor resumes ``w`` past the run it painted; in a
    wall column (moving in Y) the run is horizontal, orthogonal to the advance, so
    the cursor steps one row down. At ``w == 1`` this is identical to a unit step."""
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
            w = int(tok.values["w"])
            xy[i] = (cursor_x, cursor_y)
            if cursor_dx:
                cursor_x += w
            else:
                cursor_y += cursor_dy
    return xy


@pytest.fixture(scope="module")
def flat_pixel_eval():
    import torchwright_doom.asset_banks as asset_banks
    import torchwright_doom.pydoom as pydoom
    from torchwright_doom.inference.config import load_render_config
    from torchwright_doom.inference.wad_scene import (
        load_render_scene,
        pose_from_world,
        pydoom_scene_for,
    )
    from torchwright_doom.prompt.build import build_prompt

    submodule_root = Path(__file__).resolve().parents[2]
    config_path = os.environ.get(
        "TWDOOM_ORACLE_CONFIG", str(submodule_root / "configs" / "e1m1.yaml")
    )
    config = load_render_config(config_path)
    scene = load_render_scene(config, base_dir=submodule_root)
    pose = pose_from_world(scene)
    py_scene = pydoom_scene_for(scene, pose)
    py_pose = py_scene.test_poses[0]

    prefill = list(build_prompt(scene.map_data, pose, asset_config=scene.asset_config))
    golden = list(pydoom.expected_ar_tokens(py_scene, py_pose))
    full = prefill + golden
    begin = len(prefill) - 1

    span_env = os.environ.get("TWDOOM_J2_SPAN")
    if span_env is not None:
        n_pos = min(begin + int(span_env), len(full) - 1) + 1
    else:
        n_pos = len(full)  # full frame, through DONE
    real_pairs = [(t.type, dict(t.values)) for t in full]
    inputs = {"iv": tokens_to_input(real_pairs[:n_pos])}

    options = {}
    for opt in pydoom.expected_pixel_color_options(py_scene, py_pose):
        options.setdefault((opt.x, opt.y), set()).update(opt.colors)
    playpal = [tuple(int(c) for c in rgb) for rgb in asset_banks.PLAYPAL]

    d_embed = TOKEN_VOCAB.layout.d_embed
    iv = create_input("iv", d_embed)
    past = GraphPast(input_vec=iv, pos_encoding=create_pos_encoding())
    next_token = forward(iv, past, create_pos_encoding())

    with silenced_graph_asserts():
        cache = reference_eval(next_token, inputs, n_pos)

    pixel_start, _ = TOKEN_VOCAB.type_to_row_range[PIXEL]
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


@pytest.fixture(scope="module")
def flat_pixel_scan(flat_pixel_eval) -> dict:
    """Compare every teacher-forced transition once, shared by all the assertion
    tests below. The per-position ``argmax(emitted[i] @ W_EMBEDᵀ)`` is the hot
    cost over the full frame, so it runs as a single batched matmul here instead
    of once per test."""
    emitted = flat_pixel_eval["emitted"]
    full = flat_pixel_eval["full"]
    real_pairs = flat_pixel_eval["real_pairs"]
    begin = flat_pixel_eval["begin"]
    n_pos = flat_pixel_eval["n_pos"]
    pixel_xy = flat_pixel_eval["pixel_xy"]
    options = flat_pixel_eval["options"]
    playpal = flat_pixel_eval["playpal"]
    pixel_start = flat_pixel_eval["pixel_start"]

    # One batched argmax over all teacher-forced rows (GPU when available),
    # rather than a per-position matvec inside the Python loop below.
    predicted_rows = _predicted_rows(emitted[begin : n_pos - 1])

    coverage: Counter[str] = Counter()
    marker_mismatches = []
    carrier_mismatches = []
    pixel_mismatches = []
    for i in range(begin, n_pos - 1):
        next_name = full[i + 1].type.name
        coverage[next_name] += 1
        predicted_row = predicted_rows[i - begin]
        expected_row = row_index(*real_pairs[i + 1])
        desc = (
            f"pos {i} (in {full[i].type.name} {dict(full[i].values)}): emitted row "
            f"{predicted_row} != reference {next_name} "
            f"{dict(full[i + 1].values)} (row {expected_row})"
        )
        if next_name == _PIXEL_NAME:
            x, y = pixel_xy[i + 1]
            color = (predicted_row - pixel_start) // _PIXEL_W_LEVELS
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


def test_flat_markers_exact(flat_pixel_scan) -> None:
    assert not flat_pixel_scan[
        "marker_mismatches"
    ], "flat-pass MARKER next-token mismatches:\n" + "\n".join(
        flat_pixel_scan["marker_mismatches"][:30]
    )


def test_flat_carriers_within_tolerance(flat_pixel_scan) -> None:
    assert not flat_pixel_scan[
        "carrier_mismatches"
    ], "flat-pass CARRIER value mismatches:\n" + "\n".join(
        flat_pixel_scan["carrier_mismatches"][:25]
    )


def test_flat_pixel_colors_in_option_set(flat_pixel_scan) -> None:
    assert not flat_pixel_scan[
        "pixel_mismatches"
    ], "flat-pass PIXEL color option-set mismatches:\n" + "\n".join(
        flat_pixel_scan["pixel_mismatches"][:30]
    )


def test_flat_coverage_floors(flat_pixel_scan) -> None:
    coverage = flat_pixel_scan["coverage"]
    # The flat-pass spine + at least one open/close span + flat pixels.
    assert coverage["R_DrawPlanes"] >= 1, f"flat pass not reached: {coverage}"
    assert coverage["visplaneBegin"] >= 1, f"no visplane begun: {coverage}"
    assert coverage["R_MakeSpans.col"] >= 1, f"no R_MakeSpans column: {coverage}"
    assert coverage["R_MakeSpans.closeSlot"] >= 1, f"no span close: {coverage}"
    assert coverage["R_MapPlane.row"] >= 1, f"no span row: {coverage}"
    assert coverage[_PIXEL_NAME] >= 1, f"no pixels: {coverage}"


def test_flat_reaches_done(flat_pixel_scan) -> None:
    """When the window spans the full frame, the terminal DONE transition is
    teacher-forced and compared (asserted exact by ``test_flat_markers_exact``)."""
    if os.environ.get("TWDOOM_J2_SPAN") is not None:
        pytest.skip("capped AR span (TWDOOM_J2_SPAN set) does not reach DONE")
    assert flat_pixel_scan["coverage"]["done"] >= 1, "frame tail did not reach DONE"
