"""Teacher-forced divergence localizer for the compiled free-run (Plan K diagnostic).

When the generated frame is wrong, switch from autoregression to a **teacher-forced**
pass: feed the *reference* token stream (so positions align by construction) through
the compiled model in one wide forward, and check each position's argmax against the
reference under the J2 bars (markers exact, carriers within value tolerance, pixels
within the option set). The first hard divergence is the smallest reproducer — the
op-level noise-budget bug to fix (probe_compiled localizes the node from there).

This is the legitimate use of the parallel pass: a debugger, not the proof.

The J2 bars and carrier-delta logic mirror
``tests/scene/test_flat_pixel_oracle.py`` (the exact-math oracle), but applied to
**compiled** argmaxes instead of the reference_eval cache.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..embedding import TOKEN_VOCAB
from ..vocab import ANGLE_VALUE, PIXEL, VALUE
from .compiled_model import argmax_rows
from .decode import decode_xy_by_position
from .tokens_bridge import rows_to_input

_CARRIERS = {"value", "angleValue"}
_VALUE_ENC_TOL = 5.0e-3
_ANGLE_BAM_TOL = 2
_WALLCOL_U_NAME = "wallColU"
_WALLCOL_U_TOL = 1
_PIXEL_NAME = PIXEL.name

# Carrier row ranges, derived from the vocab (not hardcoded) so a layout change
# can't silently break the delta math.
_VALUE_START, _VALUE_END = TOKEN_VOCAB.type_to_row_range[VALUE]
_VALUE_LEVELS = _VALUE_END - _VALUE_START
_ANGLE_START, _ANGLE_END = TOKEN_VOCAB.type_to_row_range[ANGLE_VALUE]
_ANGLE_LEVELS = _ANGLE_END - _ANGLE_START
_ANGLE_LO = -_ANGLE_LEVELS // 2  # angle slot is centered (IntSlot(-N/2, N/2))
_PIXEL_START, _ = TOKEN_VOCAB.type_to_row_range[PIXEL]


def _carrier_delta(name: str, predicted_row: int, expected_row: int):
    """Encoded-value distance for a carrier; None if the prediction isn't even the
    right carrier type (a hard divergence)."""
    if name == "angleValue":
        if not (_ANGLE_START <= predicted_row < _ANGLE_END):
            return None
        pa = predicted_row - _ANGLE_START + _ANGLE_LO
        ea = expected_row - _ANGLE_START + _ANGLE_LO
        half = _ANGLE_LEVELS // 2
        return abs(((pa - ea + half) % _ANGLE_LEVELS) - half)
    if not (_VALUE_START <= predicted_row < _VALUE_END):
        return None
    return abs(predicted_row - expected_row) * 2.0 / (_VALUE_LEVELS - 1)


@dataclass
class Divergence:
    pos: int  # stream position of the *input* token (prediction is for pos+1)
    expected_type: str
    expected_row: int
    predicted_type: str
    predicted_row: int
    kind: str  # "marker" | "carrier" | "wallColU" | "pixel"
    detail: str


def teacher_forced_scan(
    compiled,
    full_rows: list[int],
    begin: int,
    options: dict[tuple[int, int], set[tuple[int, int, int]]],
    *,
    window: int | None = None,
) -> list[Divergence]:
    """Teacher-force ``full_rows`` through the compiled model in one wide forward and
    return every hard divergence (under the J2 bars) at AR positions ``>= begin``.

    ``window`` caps the number of leading positions fed (memory control for the
    O(n_pos²) single forward); ``None`` feeds the whole stream.
    """
    n = len(full_rows) if window is None else min(window, len(full_rows))
    out, _ = compiled.step(rows_to_input(full_rows[:n]), compiled.empty_past(), past_len=0)
    predicted = argmax_rows(out)  # one argmax per fed position
    pixel_xy = decode_xy_by_position(full_rows[:n])

    divergences: list[Divergence] = []
    for i in range(begin, n - 1):
        pred_row = predicted[i]
        exp_row = full_rows[i + 1]
        exp_type = TOKEN_VOCAB.row_to_token[exp_row][0].name
        pred_type = TOKEN_VOCAB.row_to_token[pred_row][0].name

        if exp_type == _PIXEL_NAME:
            xy = pixel_xy.get(i + 1)
            color = pred_row - _PIXEL_START
            ok = (
                xy is not None
                and 0 <= color < 256
                and pred_type == _PIXEL_NAME
            )
            if ok:
                from ..asset_banks import PLAYPAL

                rgb = PLAYPAL[color]
                allowed = options.get(xy)
                ok = allowed is not None and rgb in allowed
            if not ok:
                divergences.append(Divergence(
                    i, exp_type, exp_row, pred_type, pred_row, "pixel",
                    f"pixel at {xy}: color {color} not in option set",
                ))
        elif exp_type in _CARRIERS:
            tol = _ANGLE_BAM_TOL if exp_type == "angleValue" else _VALUE_ENC_TOL
            delta = _carrier_delta(exp_type, pred_row, exp_row)
            if delta is None or delta > tol:
                divergences.append(Divergence(
                    i, exp_type, exp_row, pred_type, pred_row, "carrier",
                    f"delta={delta} > tol={tol}",
                ))
        elif exp_type == _WALLCOL_U_NAME:
            if abs(pred_row - exp_row) > _WALLCOL_U_TOL:
                divergences.append(Divergence(
                    i, exp_type, exp_row, pred_type, pred_row, "wallColU",
                    f"|delta|={abs(pred_row - exp_row)} > {_WALLCOL_U_TOL}",
                ))
        else:  # marker — exact row required
            if pred_row != exp_row:
                divergences.append(Divergence(
                    i, exp_type, exp_row, pred_type, pred_row, "marker",
                    "marker row mismatch",
                ))
    return divergences


def first_divergence(divergences: list[Divergence]) -> Divergence | None:
    return divergences[0] if divergences else None
