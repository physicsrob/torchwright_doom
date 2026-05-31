"""End-to-end checks on the spec09-shaped TokenVocab and W_EMBED.

Mirrors the four checks called out in the embedding-port plan:

1. Total cardinality fits the 2^17 budget.
2. Shape sanity: every row's E8 category code is non-zero, slot raw
   columns sit in [0, 1], and the digit-quad block on every slot row
   matches ``digit_quad_row(slot, slot_value)``.
3. Derived-column round-trip: for a sample of tokens spanning each
   type that declares ``derived`` columns, the values in W_EMBED match
   the declared functions evaluated on the same slot values.
4. Cross-check against ``doom_sandbox`` embed: for a parallel ANGLE_VALUE
   built through the sandbox API, the derived columns the sandbox writes
   match what W_EMBED holds for the same angle.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchwright_doom.embedding import (
    BASE,
    CENTER,
    D_CATEGORY,
    DEFAULT_MAX_CARDINALITY,
    TOKEN_VOCAB,
    W_EMBED,
    digit_quad_query_columns_for,
    digit_quad_row,
)
from torchwright_doom.tokens import FloatSlot, IntSlot
from torchwright_doom.vocab import (
    ANGLE_BAM,
    ANGLE_VALUE,
    EMIT_X1,
    NODE,
    SCREEN_WIDTH,
    SEG,
    VALUE,
)


def test_cardinality_fits_budget() -> None:
    assert TOKEN_VOCAB.n_rows <= DEFAULT_MAX_CARDINALITY, (
        f"vocab cardinality {TOKEN_VOCAB.n_rows:,} exceeds budget "
        f"{DEFAULT_MAX_CARDINALITY:,}"
    )
    print(
        f"n_types={len(TOKEN_VOCAB.types)} "
        f"n_rows={TOKEN_VOCAB.n_rows:,} "
        f"d_embed={TOKEN_VOCAB.layout.d_embed}"
    )


def test_shape_sanity() -> None:
    layout = TOKEN_VOCAB.layout
    assert W_EMBED.shape == (TOKEN_VOCAB.n_rows, layout.d_embed)
    # Every row has a non-zero category code: E8 codes are scaled (10·)
    # unit vectors, so the per-row norm of cols [0:8] is bounded away
    # from 0.
    cat_norms = W_EMBED[:, 0:D_CATEGORY].norm(dim=1)
    assert (cat_norms > 1.0).all(), (
        f"some rows have near-zero E8 category code (min norm "
        f"{cat_norms.min().item()})"
    )


def test_raw_slot_columns_in_unit_interval() -> None:
    layout = TOKEN_VOCAB.layout
    for (type_name, _slot_name), col in layout.slot_columns.items():
        values = W_EMBED[:, col]
        # Rows from other types contribute 0; rows from the declaring
        # type write a normalized value in [0, 1].
        assert values.min().item() >= 0.0
        assert values.max().item() <= 1.0 + 1e-6, (
            f"raw col for {type_name} exceeds 1.0: {values.max().item()}"
        )


def test_digit_quad_block_widths_match_cardinality() -> None:
    """Every (type, slot) block width is 2 (cardinality ≤ 256) or 4
    (cardinality 257..65536), matching ``digit_quad_query_columns_for``."""
    layout = TOKEN_VOCAB.layout
    for (type_name, slot_name), (_, n_cols) in layout.digit_quad_columns.items():
        t = layout.types_by_name[type_name]
        slot = t.slots[slot_name]
        digits, expected_n = digit_quad_query_columns_for(slot)
        assert n_cols == expected_n, (
            f"{type_name}.{slot_name}: layout width {n_cols} != "
            f"digit_quad_query_columns_for {expected_n}"
        )
        cardinality = (
            slot.hi - slot.lo if isinstance(slot, IntSlot) else slot.levels
        )
        if cardinality <= 256:
            assert n_cols == 2
        else:
            assert n_cols == 4
        # Other types' rows should leave this block at 0.
        start, _ = TOKEN_VOCAB.type_to_row_range[t]
        col_start, _ = layout.digit_quad_columns[(type_name, slot_name)]
        # Pick a non-empty foreign type to spot-check zeros (use the
        # first type whose name differs).
        for foreign in TOKEN_VOCAB.types:
            if foreign.name == type_name:
                continue
            f_start, f_end = TOKEN_VOCAB.type_to_row_range[foreign]
            if f_end == f_start:
                continue
            f_rows = W_EMBED[
                f_start:f_end,
                col_start : col_start + n_cols,
            ]
            assert torch.all(f_rows == 0), (
                f"foreign type {foreign.name} has non-zero values in "
                f"{type_name}.{slot_name} digit-quad block"
            )
            break


def _digit_quad_block_for(
    type_name: str, slot_name: str, row: torch.Tensor
) -> torch.Tensor:
    layout = TOKEN_VOCAB.layout
    start, n = layout.digit_quad_columns[(type_name, slot_name)]
    return row[start : start + n]


def test_digit_quad_payload_value_slot() -> None:
    """VALUE.v is a 65,536-level FloatSlot — every row's digit-quad
    block is the 4-wide ``digit_quad_row(slot, slot_value)`` payload
    for that row's quantized value."""
    layout = TOKEN_VOCAB.layout
    value_start, _ = TOKEN_VOCAB.type_to_row_range[VALUE]
    slot = VALUE.slots["v"]
    span = slot.hi - slot.lo
    for k in [0, 1, 100, 32767, 32768, 65534, 65535]:
        quantized = slot.lo + (k / (slot.levels - 1)) * span
        row = W_EMBED[value_start + k]
        actual = _digit_quad_block_for("value", "v", row)
        expected = digit_quad_row(slot, quantized)
        assert torch.allclose(actual, expected, atol=1e-3), (
            f"VALUE.v digit-quad mismatch at k={k}: actual={actual} "
            f"expected={expected}"
        )


def test_digit_quad_payload_angle_value_slot() -> None:
    """ANGLE_VALUE.angle is a 8,192-level IntSlot — 4-wide digit-quad
    block on every row."""
    layout = TOKEN_VOCAB.layout
    av_start, _ = TOKEN_VOCAB.type_to_row_range[ANGLE_VALUE]
    slot = ANGLE_VALUE.slots["angle"]
    levels = slot.hi - slot.lo
    for k in [0, 1, 256, 1024, 4095, levels - 1]:
        angle_value = slot.lo + k
        row = W_EMBED[av_start + k]
        actual = _digit_quad_block_for("angleValue", "angle", row)
        expected = digit_quad_row(slot, angle_value)
        assert torch.allclose(actual, expected, atol=0.0), (
            f"ANGLE_VALUE.angle digit-quad mismatch at k={k}: "
            f"actual={actual} expected={expected}"
        )


def test_digit_quad_payload_small_int_slot() -> None:
    """SEG has two IntSlots both with cardinality ≤ 256 → 2-wide
    blocks."""
    layout = TOKEN_VOCAB.layout
    seg_start, _ = TOKEN_VOCAB.type_to_row_range[SEG]
    n_i = SEG.slots["i"].hi - SEG.slots["i"].lo
    for i in [0, 1, 5, n_i - 1]:
        for flag in [0, 1]:
            row_idx = seg_start + i * 2 + flag
            row = W_EMBED[row_idx]

            actual_i = _digit_quad_block_for("seg", "i", row)
            expected_i = digit_quad_row(SEG.slots["i"], i)
            assert torch.allclose(actual_i, expected_i)

            actual_flag = _digit_quad_block_for("seg", "is_first_of_ss", row)
            expected_flag = digit_quad_row(
                SEG.slots["is_first_of_ss"], flag
            )
            assert torch.allclose(actual_flag, expected_flag)


def _angle_row(angle: int) -> torch.Tensor:
    """Pick the W_EMBED row representing ``ANGLE_VALUE(angle=...)``."""
    av_start, _ = TOKEN_VOCAB.type_to_row_range[ANGLE_VALUE]
    idx = angle - ANGLE_VALUE.slots["angle"].lo
    return W_EMBED[av_start + idx]


def _value_row_for(v: float) -> tuple[torch.Tensor, float]:
    """Pick the W_EMBED row representing the quantized neighbor of ``v``.

    Returns the row and the quantized value (since FloatSlot snaps to a
    65,536-step grid, callers compare against the snapped value).
    """
    v_start, _ = TOKEN_VOCAB.type_to_row_range[VALUE]
    slot = VALUE.slots["v"]
    span = slot.hi - slot.lo
    idx = round((v - slot.lo) / span * (slot.levels - 1))
    quantized = slot.lo + (idx / (slot.levels - 1)) * span
    return W_EMBED[v_start + idx], quantized


def test_derived_column_round_trip_angle_value() -> None:
    """Every ANGLE_VALUE derived column equals fn(angle) on the row's angle."""
    layout = TOKEN_VOCAB.layout
    angle_slot = ANGLE_VALUE.slots["angle"]
    test_angles = [-2048, -1024, 0, 256, 1024, 2048]
    for angle in test_angles:
        row = _angle_row(angle)
        for derived_name, d in angle_slot.derived.items():
            start, _w = layout.derived_columns[
                (ANGLE_VALUE.name, "angle", derived_name)
            ]
            expected = float(d.fn(angle))
            actual = row[start].item()
            assert math.isclose(actual, expected, abs_tol=1e-6), (
                f"angle={angle} derived={derived_name}: "
                f"actual={actual} expected={expected}"
            )


def test_derived_column_round_trip_value() -> None:
    """Every VALUE derived column equals fn(v) on the quantized v."""
    layout = TOKEN_VOCAB.layout
    v_slot = VALUE.slots["v"]
    test_values = [-1024.0, -1.0, 0.0, 1.0, 100.0, 1024.0]
    for v in test_values:
        row, quantized = _value_row_for(v)
        for derived_name, d in v_slot.derived.items():
            start, _w = layout.derived_columns[(VALUE.name, "v", derived_name)]
            expected = float(d.fn(quantized))
            actual = row[start].item()
            assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6), (
                f"v={quantized} derived={derived_name}: "
                f"actual={actual} expected={expected}"
            )


def test_derived_column_round_trip_one_hot_emit_x1() -> None:
    """EMIT_X1's ``x_oh_NNN`` columns are 1.0 at the matching x, 0
    elsewhere — the canonical column-addressed token for screen-X."""
    layout = TOKEN_VOCAB.layout
    emit_start, _ = TOKEN_VOCAB.type_to_row_range[EMIT_X1]
    test_xs = [0, 1, SCREEN_WIDTH // 2, SCREEN_WIDTH - 1]
    for x in test_xs:
        row = W_EMBED[emit_start + x]
        for col_name in (f"x_oh_{c:03d}" for c in range(SCREEN_WIDTH)):
            start, _w = layout.derived_columns[(EMIT_X1.name, "x", col_name)]
            target = int(col_name.split("_")[-1])
            expected = 1.0 if x == target else 0.0
            actual = row[start].item()
            assert actual == expected, (
                f"EMIT_X1.x={x} derived={col_name}: {actual} != {expected}"
            )


def test_cross_check_against_sandbox() -> None:
    """Build a parallel ANGLE_VALUE through the sandbox API and confirm
    its derived columns match what W_EMBED holds for the same angle.

    ``doom_sandbox`` is a workspace sibling that uv installs with
    ``package = false``, so it isn't on the venv's site-packages.
    Add the umbrella checkout to ``sys.path`` so the import resolves.
    Skipped only when the umbrella checkout isn't laid out the way the
    workspace expects.
    """
    import os
    import sys

    umbrella = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if os.path.isdir(os.path.join(umbrella, "doom_sandbox")):
        if umbrella not in sys.path:
            sys.path.insert(0, umbrella)
    sb_api = pytest.importorskip("doom_sandbox.api")
    sb_runtime = pytest.importorskip("doom_sandbox.runtime.embedding")

    Derived = sb_api.Derived
    sb_angle_value = sb_api.TokenType(
        "tw_doom_angleValue",
        slots={
            "angle": sb_api.IntSlot(
                -ANGLE_BAM // 2,
                ANGLE_BAM // 2,
                derived={
                    "sin": Derived(
                        lambda a: math.sin(a * 2 * math.pi / ANGLE_BAM)
                    ),
                    "cos": Derived(
                        lambda a: math.cos(a * 2 * math.pi / ANGLE_BAM)
                    ),
                },
            ),
        },
    )
    sb_vocab = sb_api.TokenVocab(types=[sb_angle_value])

    layout = TOKEN_VOCAB.layout
    test_angles = [0, 256, 1024, -1024, 2048]
    for angle in test_angles:
        sb_token = sb_api.Token(
            type=sb_angle_value, values={"angle": angle}
        )
        sb_vec = sb_runtime.embed(sb_token, sb_vocab.layout)
        sb_row = sb_vec._data[0]

        our_row = _angle_row(angle)
        for derived_name in ("sin", "cos"):
            sb_col_start, _width = sb_vocab.layout.derived_columns[
                (sb_angle_value.name, "angle", derived_name)
            ]
            sb_value = float(sb_row[sb_col_start])
            our_start, _w = layout.derived_columns[
                (ANGLE_VALUE.name, "angle", derived_name)
            ]
            our_value = our_row[our_start].item()
            assert math.isclose(sb_value, our_value, abs_tol=1e-6), (
                f"angle={angle} derived={derived_name}: "
                f"sandbox={sb_value} ours={our_value}"
            )
