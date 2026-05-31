"""Graph-correctness test for the emit helpers.

For each token shape — slotless, single int slot, multi int slot,
single float slot, mixed — build a tiny graph that calls the emit
helper with constant-value :class:`InputNode`\\ s, run
:func:`reference_eval`, project the resulting residual through
``W_EMBED.T``, and confirm host argmax lands on the expected row.

This is the cheapest graph-level check — exact-math evaluation, no
compile. Numerical fidelity of the compiled forward pass is the
subject of ``test_emit_compiled_round_trip.py``.
"""

from __future__ import annotations

import math

import torch

from torchwright.debug.probe import reference_eval
from torchwright.graph import fresh_graph_session
from torchwright.ops.inout_nodes import create_input

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.emit import (
    emit_float_slot_token,
    emit_int_slot_token,
    emit_slotless,
    emit_token,
)
from torchwright_doom.vocab import (
    BEGIN,
    DONE,
    NO_OP,
    NODE,
    DRAWSEG_META,
    SEG,
    VALUE,
)


def _project_and_argmax(emit_value: torch.Tensor) -> int:
    """Project a 1-position emit residual through ``W_EMBED.T`` and
    return the host argmax row index."""
    assert emit_value.shape == (1, TOKEN_VOCAB.layout.d_embed)
    scores = emit_value @ W_EMBED.T
    return int(scores.argmax(dim=1).item())


def _row_for(t, values: dict) -> int:
    """Look up the W_EMBED row index for ``(t, slot_values)``."""
    start, end = TOKEN_VOCAB.type_to_row_range[t]
    for offset, (_t, v) in enumerate(TOKEN_VOCAB.row_to_token[start:end]):
        if v == values:
            return start + offset
    raise KeyError(f"No row for {t.name}{values!r}")


def test_emit_slotless_argmax() -> None:
    """BEGIN, DONE, NO_OP — each is one of-a-kind row in W_EMBED."""
    for t in (BEGIN, DONE, NO_OP):
        with fresh_graph_session():
            out = emit_slotless(t)
            value = out.compute(n_pos=1, input_values={})
        assert value.shape == (1, TOKEN_VOCAB.layout.d_embed)
        argmax = _project_and_argmax(value)
        expected = _row_for(t, {})
        assert argmax == expected, (
            f"slotless {t.name}: argmax {argmax} != expected {expected}"
        )


def test_emit_int_slot_single() -> None:
    """NODE has a single IntSlot j; sweep a handful of values."""
    for j in [0, 1, 5, NODE.slots["j"].hi - 1]:
        with fresh_graph_session():
            j_input = create_input("j", 1, value_range=(-1.0, 256.0))
            out = emit_int_slot_token(NODE, j=j_input)
            cache = reference_eval(
                out,
                input_values={"j": torch.tensor([[float(j)]])},
                n_pos=1,
            )
            value = cache[out]
        argmax = _project_and_argmax(value)
        expected = _row_for(NODE, {"j": j})
        assert argmax == expected, (
            f"NODE(j={j}): argmax {argmax} != expected {expected}"
        )


def test_emit_int_slot_multi() -> None:
    """SEG carries two IntSlots: i (cardinality 128) and is_first_of_ss
    (cardinality 2)."""
    for i, flag in [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (63, 0),
        (127, 1),
    ]:
        with fresh_graph_session():
            i_in = create_input("i", 1, value_range=(-1.0, 256.0))
            f_in = create_input("flag", 1, value_range=(-1.0, 4.0))
            out = emit_int_slot_token(
                SEG, i=i_in, is_first_of_ss=f_in
            )
            cache = reference_eval(
                out,
                input_values={
                    "i": torch.tensor([[float(i)]]),
                    "flag": torch.tensor([[float(flag)]]),
                },
                n_pos=1,
            )
            value = cache[out]
        argmax = _project_and_argmax(value)
        expected = _row_for(SEG, {"i": i, "is_first_of_ss": flag})
        assert argmax == expected, (
            f"SEG(i={i}, flag={flag}): argmax {argmax} != "
            f"expected {expected}"
        )


def test_emit_int_slot_three_slots() -> None:
    """DRAWSEG_META has three IntSlots (mixed cardinality 128/3/4)."""
    for i, wall_kind, silhouette in [
        (0, 0, 0),
        (5, 1, 2),
        (64, 2, 3),
        (127, 2, 1),
    ]:
        with fresh_graph_session():
            i_in = create_input("i", 1, value_range=(-1.0, 130.0))
            wk_in = create_input("wk", 1, value_range=(-1.0, 4.0))
            sil_in = create_input("sil", 1, value_range=(-1.0, 5.0))
            out = emit_int_slot_token(
                DRAWSEG_META, i=i_in, wall_kind=wk_in, silhouette=sil_in
            )
            cache = reference_eval(
                out,
                input_values={
                    "i": torch.tensor([[float(i)]]),
                    "wk": torch.tensor([[float(wall_kind)]]),
                    "sil": torch.tensor([[float(silhouette)]]),
                },
                n_pos=1,
            )
            value = cache[out]
        argmax = _project_and_argmax(value)
        expected = _row_for(
            DRAWSEG_META,
            {"i": i, "wall_kind": wall_kind, "silhouette": silhouette},
        )
        assert argmax == expected, (
            f"DRAWSEG_META(i={i}, wall_kind={wall_kind}, "
            f"silhouette={silhouette}): argmax {argmax} != expected {expected}"
        )


def test_emit_float_slot_at_grid_levels() -> None:
    """VALUE.v is a 65,536-level FloatSlot.

    Exact emit at quantization-grid levels (k integer) must argmax to
    the matching row. Spans the full slot range plus a few interior
    levels.
    """
    slot = VALUE.slots["v"]
    span = slot.hi - slot.lo
    for k in [0, 1, 100, 32767, 32768, 65534, 65535]:
        v = slot.lo + (k / (slot.levels - 1)) * span
        with fresh_graph_session():
            v_in = create_input(
                "v",
                1,
                value_range=(float(slot.lo) - 1.0, float(slot.hi) + 1.0),
            )
            out = emit_float_slot_token(VALUE, v=v_in)
            cache = reference_eval(
                out,
                input_values={"v": torch.tensor([[float(v)]])},
                n_pos=1,
            )
            value = cache[out]
        argmax = _project_and_argmax(value)
        expected = _row_for(VALUE, {"v": v})
        assert argmax == expected, (
            f"VALUE(v={v}, k={k}): argmax {argmax} != expected {expected}"
        )


def test_emit_token_dispatcher() -> None:
    """``emit_token`` picks the right specialized helper for each
    token shape."""
    # Slotless
    with fresh_graph_session():
        out = emit_token(BEGIN)
        value = out.compute(n_pos=1, input_values={})
    assert _project_and_argmax(value) == _row_for(BEGIN, {})

    # IntSlot only
    with fresh_graph_session():
        i_in = create_input("i", 1, value_range=(-1.0, 256.0))
        f_in = create_input("flag", 1, value_range=(-1.0, 4.0))
        out = emit_token(SEG, i=i_in, is_first_of_ss=f_in)
        cache = reference_eval(
            out,
            input_values={
                "i": torch.tensor([[5.0]]),
                "flag": torch.tensor([[1.0]]),
            },
            n_pos=1,
        )
        value = cache[out]
    assert _project_and_argmax(value) == _row_for(
        SEG, {"i": 5, "is_first_of_ss": 1}
    )

    # FloatSlot
    with fresh_graph_session():
        v_in = create_input("v", 1, value_range=(-5000.0, 5000.0))
        out = emit_token(VALUE, v=v_in)
        cache = reference_eval(
            out, input_values={"v": torch.tensor([[0.5]])}, n_pos=1
        )
        value = cache[out]
    slot = VALUE.slots["v"]
    span = slot.hi - slot.lo
    k = round((0.5 - slot.lo) / span * (slot.levels - 1))
    quantized = slot.lo + (k / (slot.levels - 1)) * span
    assert _project_and_argmax(value) == _row_for(
        VALUE, {"v": quantized}
    )
