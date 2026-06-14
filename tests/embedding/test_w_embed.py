"""End-to-end checks on the spec09-shaped TokenVocab and W_EMBED.

Mirrors the four checks called out in the embedding-port plan:

1. Total cardinality fits the 2^17 budget.
2. Shape sanity: every row's E8 category code is non-zero, slot raw
   columns sit in [0, 1], and the digit-quad block on every slot row
   matches ``digit_quad_row(slot, slot_value)``.
3. Derived-column round-trip: for a sample of tokens spanning each
   type that declares ``derived`` columns, the values in W_EMBED match
   the declared functions evaluated on the same slot values.
4. Cross-check the derived columns: for an ANGLE_VALUE row, the derived
   columns W_EMBED holds match the declared derived functions evaluated
   on the same angle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import torch

from torchwright_doom.embedding import (
    D_CATEGORY,
    DEFAULT_MAX_CARDINALITY,
    TOKEN_VOCAB,
    W_EMBED,
    _digit_quad_block,
    digit_quad_query_columns_for,
    digit_quad_row,
)
from torchwright_doom.tokens import Derived, FloatSlot, IntSlot, TokenType
from torchwright_doom.vocab import (
    ANGLE_VALUE,
    EMIT_X2,
    SCREEN_WIDTH,
    SEG,
    VALUE,
)


def _float_slot(token_type: TokenType, slot_name: str) -> FloatSlot:
    slot = token_type.slots[slot_name]
    assert isinstance(slot, FloatSlot)
    return slot


def _int_slot(token_type: TokenType, slot_name: str) -> IntSlot:
    slot = token_type.slots[slot_name]
    assert isinstance(slot, IntSlot)
    return slot


def _derived_items(slot: IntSlot | FloatSlot):
    return cast(Mapping[str, Derived], slot.derived).items()


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
        assert (
            values.max().item() <= 1.0 + 1e-6
        ), f"raw col for {type_name} exceeds 1.0: {values.max().item()}"


def test_digit_quad_block_widths_are_shared_by_position() -> None:
    """Each (type, slot) digit-quad block is sized for its slot *position* — the
    widest slot at that position across the vocab. So every block is 2 or 4 cols
    and at least as wide as the slot's own need, and same-position slots of
    different types map to the *same* shared block (the property that keeps the
    output head compact). Columns are deliberately *not* per-type-isolated."""
    layout = TOKEN_VOCAB.layout
    for (type_name, slot_name), (_, n_cols) in layout.digit_quad_columns.items():
        slot = layout.types_by_name[type_name].slots[slot_name]
        _, slot_n = digit_quad_query_columns_for(slot)
        assert n_cols in (2, 4)
        assert (
            n_cols >= slot_n
        ), f"{type_name}.{slot_name}: shared block {n_cols} < slot need {slot_n}"

    # Every type's first slot shares one digit-quad column block.
    first_blocks = {
        layout.digit_quad_columns[(t.name, next(iter(t.slots)))][0]
        for t in TOKEN_VOCAB.types
        if t.slots
    }
    assert len(first_blocks) == 1, f"first-slot blocks not shared: {first_blocks}"


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
    value_start, _ = TOKEN_VOCAB.type_to_row_range[VALUE]
    slot = _float_slot(VALUE, "v")
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
    av_start, _ = TOKEN_VOCAB.type_to_row_range[ANGLE_VALUE]
    slot = _int_slot(ANGLE_VALUE, "angle")
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
    """SEG's two small IntSlots encode into their *shared-position* blocks: each
    value's W_EMBED block matches ``_digit_quad_block`` at the position's width
    (which may exceed the slot's own — the extra high byte is then a constant 0,
    matching how the emit side encodes it)."""
    layout = TOKEN_VOCAB.layout
    seg_start, _ = TOKEN_VOCAB.type_to_row_range[SEG]
    seg_i_slot = _int_slot(SEG, "i")
    n_i = seg_i_slot.hi - seg_i_slot.lo
    _, i_ncols = layout.digit_quad_columns[("seg", "i")]
    _, flag_ncols = layout.digit_quad_columns[("seg", "is_first_of_ss")]
    for i in [0, 1, 5, n_i - 1]:
        for flag in [0, 1]:
            row = W_EMBED[seg_start + i * 2 + flag]

            actual_i = _digit_quad_block_for("seg", "i", row)
            expected_i = torch.from_numpy(_digit_quad_block(np.array([i]), i_ncols)[0])
            assert torch.allclose(actual_i, expected_i)

            actual_flag = _digit_quad_block_for("seg", "is_first_of_ss", row)
            expected_flag = torch.from_numpy(
                _digit_quad_block(np.array([flag]), flag_ncols)[0]
            )
            assert torch.allclose(actual_flag, expected_flag)


def _angle_row(angle: int) -> torch.Tensor:
    """Pick the W_EMBED row representing ``ANGLE_VALUE(angle=...)``."""
    av_start, _ = TOKEN_VOCAB.type_to_row_range[ANGLE_VALUE]
    idx = angle - _int_slot(ANGLE_VALUE, "angle").lo
    return W_EMBED[av_start + idx]


def _value_row_for(v: float) -> tuple[torch.Tensor, float]:
    """Pick the W_EMBED row representing the quantized neighbor of ``v``.

    Returns the row and the quantized value (since FloatSlot snaps to a
    65,536-step grid, callers compare against the snapped value).
    """
    v_start, _ = TOKEN_VOCAB.type_to_row_range[VALUE]
    slot = _float_slot(VALUE, "v")
    span = slot.hi - slot.lo
    idx = round((v - slot.lo) / span * (slot.levels - 1))
    quantized = slot.lo + (idx / (slot.levels - 1)) * span
    return W_EMBED[v_start + idx], quantized


def test_derived_column_round_trip_angle_value() -> None:
    """Every ANGLE_VALUE derived column equals fn(angle) on the row's angle."""
    layout = TOKEN_VOCAB.layout
    angle_slot = _int_slot(ANGLE_VALUE, "angle")
    test_angles = [-2048, -1024, 0, 256, 1024, 2048]
    for angle in test_angles:
        row = _angle_row(angle)
        for derived_name, d in _derived_items(angle_slot):
            start, width = layout.derived_columns[
                (ANGLE_VALUE.name, "angle", derived_name)
            ]
            expected = d.fn(angle)
            if width == 1:
                expected_scalar = cast(float, expected)
                assert math.isclose(
                    row[start].item(), float(expected_scalar), abs_tol=1e-6
                ), (
                    f"angle={angle} derived={derived_name}: "
                    f"actual={row[start].item()} expected={float(expected_scalar)}"
                )
            else:
                exp_list = [float(x) for x in cast(Sequence[float], expected)]
                assert len(exp_list) == width
                for off in range(width):
                    assert math.isclose(
                        row[start + off].item(), exp_list[off], abs_tol=1e-6
                    ), (
                        f"angle={angle} derived={derived_name}[{off}]: "
                        f"actual={row[start + off].item()} exp={exp_list[off]}"
                    )


def test_derived_column_round_trip_value() -> None:
    """Every VALUE derived column equals fn(v) on the quantized v."""
    layout = TOKEN_VOCAB.layout
    v_slot = _float_slot(VALUE, "v")
    test_values = [-1.0, -0.5, -0.125, 0.0, 0.25, 0.5, 1.0]
    for v in test_values:
        row, quantized = _value_row_for(v)
        for derived_name, d in _derived_items(v_slot):
            start, _w = layout.derived_columns[(VALUE.name, "v", derived_name)]
            expected = float(cast(float, d.fn(quantized)))
            actual = row[start].item()
            assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6), (
                f"v={quantized} derived={derived_name}: "
                f"actual={actual} expected={expected}"
            )


def test_derived_column_round_trip_one_hot_emit_x1() -> None:
    """EMIT_X2's ``x_oh_NNN`` columns are 1.0 at the matching x, 0
    elsewhere — the canonical column-addressed token for screen-X."""
    layout = TOKEN_VOCAB.layout
    emit_start, _ = TOKEN_VOCAB.type_to_row_range[EMIT_X2]
    test_xs = [0, 1, SCREEN_WIDTH // 2, SCREEN_WIDTH - 1]
    for x in test_xs:
        row = W_EMBED[emit_start + x]
        for col_name in (f"x_oh_{c:03d}" for c in range(SCREEN_WIDTH)):
            start, _w = layout.derived_columns[(EMIT_X2.name, "x", col_name)]
            target = int(col_name.split("_")[-1])
            expected = 1.0 if x == target else 0.0
            actual = row[start].item()
            assert (
                actual == expected
            ), f"EMIT_X2.x={x} derived={col_name}: {actual} != {expected}"
