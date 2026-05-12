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
    PLANE_DEF,
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
    """PLANE_DEF has three IntSlots; verify mixed-cardinality emit."""
    for p, flat_id, is_sky in [
        (0, 0, 0),
        (5, 0, 1),
        (10, 31, 0),
        (31, 31, 1),
    ]:
        with fresh_graph_session():
            p_in = create_input("p", 1, value_range=(-1.0, 64.0))
            f_in = create_input("flat", 1, value_range=(-1.0, 64.0))
            s_in = create_input("sky", 1, value_range=(-1.0, 4.0))
            out = emit_int_slot_token(
                PLANE_DEF, p=p_in, flat_id=f_in, is_sky=s_in
            )
            cache = reference_eval(
                out,
                input_values={
                    "p": torch.tensor([[float(p)]]),
                    "flat": torch.tensor([[float(flat_id)]]),
                    "sky": torch.tensor([[float(is_sky)]]),
                },
                n_pos=1,
            )
            value = cache[out]
        argmax = _project_and_argmax(value)
        expected = _row_for(
            PLANE_DEF, {"p": p, "flat_id": flat_id, "is_sky": is_sky}
        )
        assert argmax == expected, (
            f"PLANE_DEF(p={p}, flat_id={flat_id}, is_sky={is_sky}): "
            f"argmax {argmax} != expected {expected}"
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
            out, input_values={"v": torch.tensor([[42.0]])}, n_pos=1
        )
        value = cache[out]
    slot = VALUE.slots["v"]
    span = slot.hi - slot.lo
    k = round((42.0 - slot.lo) / span * (slot.levels - 1))
    quantized = slot.lo + (k / (slot.levels - 1)) * span
    assert _project_and_argmax(value) == _row_for(
        VALUE, {"v": quantized}
    )
