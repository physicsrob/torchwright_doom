"""Phase J1 gate — wall texel pass oracle.

Drives the real ``forward()`` graph teacher-forced on the sandbox golden token
stream for ``e1m1_start_room_textured`` (the textured fixture carrying the WAD
texture slice) and asserts, over a bounded window covering the first few wall
columns *with pixels*:

* every **non-PIXEL** J-owned / H-owned control token agrees with the sandbox at
  the exact ``W_EMBED`` row (``setCursorX``, ``wallColU``, ``setCursorY``,
  ``screenY``, ``wallSpanMeta``, ``clipUpdate``, ``screenRange``, ``planeMark``,
  ``R_CheckPlane(.result)`` …);
* numeric ``VALUE`` / ``angleValue`` carriers agree within the value-encoding
  tolerance (including the per-span **R3 v0** carrier emitted by the new
  ``set_cursor_y`` branch and the per-column R5 scale carrier);
* every **PIXEL** color is *within the sandbox's option set* — the 9×9 UV
  neighborhood of ``expected_pixel_color_options`` — rather than exact-row-equal,
  matching the sandbox's own pixel gate (benign texture-boundary rounding).

The window caps **before** ``R_DrawPlanes`` (the flat pass — Phase J2), so the
flat-pass branches stay ``no_op`` and ``flat_span_seen`` is structurally false.
Same teacher-forced ``reference_eval`` pattern as ``test_wall_column_oracle``
(exact math, CPU, no compile). ``reference_eval`` is O(n_pos²); the window is
sized to reach the first textured columns with margin and no further.
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
from torchwright_doom.vocab import PIXEL, VOCAB_TYPES

from ..prefill_fixture import row_index, tokens_to_input

_CARRIERS = {"value", "angleValue"}
_ANGLE_BAM_TOL = 2
# Projection VALUE carriers carry up to ~4e-3 encoded float32 PWL noise on the
# textured fixture's segs (the H/F projection chain); downstream-absorbed by the
# ±4 UV pixel option set. 5e-3 covers it; the J-owned R3 v0 carrier sits below.
_VALUE_ENC_TOL = 5.0e-3
_VALUE_ROWS = 65536
# wallColU.u_idx is floor(-(rw_distance·tan)) via a float32 PWL product; near an
# integer boundary it differs ±1 from the float64 reference's int(). The texel
# takes it mod texture width and the ±4 UV option set absorbs ±1. ±1-tolerant.
_WALLCOL_U_NAME = "wallColU"
_WALLCOL_U_TOL = 1
_ANGLE_ROW0 = 65536
_ANGLE_LO = -4096

# Prefill on the textured fixture is ~3613 tokens; the first wall columns *with
# visible texture spans* (the first PIXEL run) land ~idx 4283, and the third such
# column finishes ~idx 4332 just before the next R_AddLine. 725 positions past the
# AR seed reaches three full pixel columns with margin and caps well before
# R_DrawPlanes (~idx 8320, the Phase-J2 flat pass). reference_eval is O(n_pos²).
_AR_SPAN = 725

_PIXEL_NAME = PIXEL.name


def _umbrella() -> Path:
    return Path(__file__).resolve().parents[3]


def _decode_pixel_xy(full) -> dict[int, tuple[int, int]]:
    """Host pixel decode (mirrors sandbox ``extract.extract_pixel_pass``): walk the
    token stream tracking the cursor + direction and record each PIXEL token's
    ``(x, y)``. The default direction is Y (vertical wall columns)."""
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
def wall_pixel_eval():
    """Build the J1 ``forward()`` graph once, teacher-force it on the textured
    fixture's golden stream, and ``reference_eval`` a single pass."""
    umbrella = _umbrella()
    if not (umbrella / "doom_sandbox").is_dir():
        pytest.skip("doom_sandbox sibling not present (standalone checkout)")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    if str(umbrella) not in sys.path:
        sys.path.insert(0, str(umbrella))

    fixtures = pytest.importorskip("doom_sandbox.fixtures")
    sb_prefill = pytest.importorskip("doom_sandbox.implementation.prefill")
    drafter = pytest.importorskip("doom_sandbox.implementation.reference_drafter")
    reference = pytest.importorskip("doom_sandbox.implementation.reference")
    asset_banks = pytest.importorskip("doom_sandbox.implementation.asset_banks")

    name_to_real = {t.name: t for t in VOCAB_TYPES}

    scene = fixtures.load_fixture("e1m1_start_room_textured")
    pose = scene.test_poses[0]
    prefill = list(sb_prefill.get_prefill(scene, pose))
    golden = list(drafter.expected_ar_tokens(scene, pose))
    full = prefill + golden
    begin = len(prefill) - 1

    n_pos = min(begin + _AR_SPAN, len(full) - 1) + 1
    real_pairs = [(name_to_real[t.type.name], dict(t.values)) for t in full]
    inputs = {"iv": tokens_to_input(real_pairs[:n_pos])}

    # Option sets (9x9 UV neighborhood) keyed by (x, y), mapped emitted-color ->
    # RGB through the *same* PLAYPAL the reference built the options from.
    options = {}
    for opt in reference.expected_pixel_color_options(scene, pose):
        options.setdefault((opt.x, opt.y), set()).update(opt.colors)
    playpal = [tuple(int(c) for c in rgb) for rgb in asset_banks.PLAYPAL]

    d_embed = TOKEN_VOCAB.layout.d_embed
    iv = create_input("iv", d_embed)
    past = GraphPast(input_vec=iv, pos_encoding=create_pos_encoding())
    next_token = forward(iv, past, create_pos_encoding())

    # Silence debug Assert predicates for the oracle pass (discarded branches can
    # land a cond in a comparator ramp; validated via next-token agreement).
    import torchwright.graph.misc as _misc

    _orig_check = _misc.Assert._check
    _misc.Assert._check = lambda self, x: None
    try:
        cache = reference_eval(next_token, inputs, n_pos)
    finally:
        _misc.Assert._check = _orig_check

    # Inverse of the PIXEL color slot: color is the only slot, stride 1.
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


def _scan(wall_pixel_eval) -> dict:
    emitted = wall_pixel_eval["emitted"]
    full = wall_pixel_eval["full"]
    real_pairs = wall_pixel_eval["real_pairs"]
    begin = wall_pixel_eval["begin"]
    n_pos = wall_pixel_eval["n_pos"]
    pixel_xy = wall_pixel_eval["pixel_xy"]
    options = wall_pixel_eval["options"]
    playpal = wall_pixel_eval["playpal"]
    pixel_start = wall_pixel_eval["pixel_start"]
    w_embed_t = W_EMBED.t()

    coverage: Counter[str] = Counter()
    marker_mismatches = []
    carrier_mismatches = []
    pixel_mismatches = []
    pixel_columns: set[int] = set()
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
            pixel_columns.add(x)
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
        "pixel_columns": pixel_columns,
    }


def test_wall_pixel_markers_exact(wall_pixel_eval) -> None:
    scan = _scan(wall_pixel_eval)
    assert not scan["marker_mismatches"], (
        "wall-pixel MARKER next-token mismatches:\n"
        + "\n".join(scan["marker_mismatches"][:25])
    )


def test_wall_pixel_carriers_within_tolerance(wall_pixel_eval) -> None:
    scan = _scan(wall_pixel_eval)
    assert not scan["carrier_mismatches"], (
        "wall-pixel CARRIER value mismatches:\n"
        + "\n".join(scan["carrier_mismatches"][:25])
    )


def test_wall_pixel_colors_in_option_set(wall_pixel_eval) -> None:
    scan = _scan(wall_pixel_eval)
    coverage = scan["coverage"]
    assert not scan["pixel_mismatches"], (
        "wall-pixel COLOR option-set mismatches:\n"
        + "\n".join(scan["pixel_mismatches"][:25])
    )
    # Coverage floors — the window must actually rasterize textured wall columns.
    assert coverage[_PIXEL_NAME] >= 10, f"too few wall pixels: {coverage}"
    assert len(scan["pixel_columns"]) >= 1, "no full wall column rasterized"
