"""End-to-end checks on the spec09-shaped TokenVocab and W_EMBED.

Mirrors the four checks called out in the embedding-port plan:

1. Total cardinality fits the 2^17 budget.
2. Shape sanity: every row's E8 category code is non-zero, slot raw
   columns sit in [0, 1], the K column carries small-int slot values
   for types that have one, and the Gray-code block matches
   ``gray_code_16(k, levels)`` on large-cardinality slot rows.
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
    D_CATEGORY,
    D_GRAY_PAYLOAD,
    DEFAULT_MAX_CARDINALITY,
    MAX_INT_K,
    TOKEN_VOCAB,
    W_EMBED,
    gray_code_16,
)
from torchwright_doom.tokens import IntSlot
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
    # Every row has a non-zero category code: E8 codes are unit-scale
    # vectors so the per-row norm of cols [0:8] is bounded away from 0.
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


def test_k_column_for_small_int_types() -> None:
    """SEG's ``i`` slot is the first IntSlot fitting in [0, 255], so
    every SEG row carries ``SEG.i`` in the K column."""
    layout = TOKEN_VOCAB.layout
    seg_start, _seg_end = TOKEN_VOCAB.type_to_row_range[SEG]
    # SEG rows enumerate (i, is_first_of_ss) in declaration order with
    # `is_first_of_ss` varying fastest, so row offset = i * 2 + flag.
    n_i = SEG.slots["i"].hi - SEG.slots["i"].lo
    for i in range(0, n_i):
        for flag in range(2):
            row = seg_start + i * 2 + flag
            k_val = W_EMBED[row, layout.k_col].item()
            assert k_val == float(i), (
                f"SEG i={i} flag={flag}: K col = {k_val}, expected {i}"
            )
    # ANGLE_VALUE has IntSlot but range straddles 0 (lo=-4096), so it
    # falls out of the K-slot eligibility. Its rows leave K at 0.
    av_start, _ = TOKEN_VOCAB.type_to_row_range[ANGLE_VALUE]
    assert W_EMBED[av_start, layout.k_col].item() == 0.0


def test_gray_code_matches_helper_for_value_rows() -> None:
    layout = TOKEN_VOCAB.layout
    value_start, _value_end = TOKEN_VOCAB.type_to_row_range[VALUE]
    levels = VALUE.slots["v"].levels
    for k in [0, 1, 100, 32767, 65535]:
        row = W_EMBED[value_start + k]
        expected = gray_code_16(k, levels)
        actual = row[layout.gray_start : layout.gray_start + D_GRAY_PAYLOAD]
        assert torch.equal(actual, expected), (
            f"Gray code mismatch at VALUE row k={k}"
        )


def test_gray_code_matches_helper_for_angle_value_rows() -> None:
    layout = TOKEN_VOCAB.layout
    av_start, _ = TOKEN_VOCAB.type_to_row_range[ANGLE_VALUE]
    angle_slot = ANGLE_VALUE.slots["angle"]
    levels = angle_slot.hi - angle_slot.lo
    for idx in [0, 1, 1024, levels - 1]:
        row = W_EMBED[av_start + idx]
        expected = gray_code_16(idx, levels)
        actual = row[layout.gray_start : layout.gray_start + D_GRAY_PAYLOAD]
        assert torch.equal(actual, expected), (
            f"Gray code mismatch at ANGLE_VALUE idx={idx}"
        )


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
        for derived_name, fn in angle_slot.derived.items():
            col = layout.derived_columns[derived_name]
            expected = float(fn(angle))
            actual = row[col].item()
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
        for derived_name, fn in v_slot.derived.items():
            col = layout.derived_columns[derived_name]
            expected = float(fn(quantized))
            actual = row[col].item()
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
            col = layout.derived_columns[col_name]
            target = int(col_name.split("_")[-1])
            expected = 1.0 if x == target else 0.0
            actual = row[col].item()
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

    sb_angle_value = sb_api.TokenType(
        "tw_doom_angleValue",
        slots={
            "angle": sb_api.IntSlot(
                -ANGLE_BAM // 2,
                ANGLE_BAM // 2,
                derived={
                    "sin": lambda a: math.sin(a * 2 * math.pi / ANGLE_BAM),
                    "cos": lambda a: math.cos(a * 2 * math.pi / ANGLE_BAM),
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
            sb_col = sb_vocab.layout.derived_columns[
                (sb_angle_value.name, "angle", derived_name)
            ]
            sb_value = float(sb_row[sb_col])
            our_col = layout.derived_columns[derived_name]
            our_value = our_row[our_col].item()
            assert math.isclose(sb_value, our_value, abs_tol=1e-6), (
                f"angle={angle} derived={derived_name}: "
                f"sandbox={sb_value} ours={our_value}"
            )
