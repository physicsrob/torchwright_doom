"""Compiled round-trip test: emit → host argmax → deembed.

For each token shape (slotless, single-IntSlot, multi-IntSlot, mixed-
cardinality IntSlots, FloatSlot), compile a minimal transformer
module whose only output is one emit helper's residual; run it on a
handful of representative inputs; project the compiled output
through ``W_EMBED.T`` host-side; argmax; deembed back to
``(token_type, slot_values)``; compare against what was emitted.

The host argmax-and-deembed routine is what
:class:`torchwright.graph.embedding.Unembedding` does in production.
This test catches fp32 / matmul-accumulation noise that
:func:`reference_eval` misses — the compiled forward pass uses a
piecewise-linear approximation of every op and accumulates noise
across layers, where :func:`reference_eval` computes the exact-math
oracle.
"""

from __future__ import annotations

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import build_prefill_from_input_values
from torchwright.graph import fresh_graph_session
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.emit import (
    emit_float_slot_token,
    emit_int_slot_token,
    emit_slotless,
)
from torchwright_doom.vocab import (
    BEGIN,
    NODE,
    DRAWSEG_META,
    SEG,
    VALUE,
)


def _deembed_argmax(compiled_residual: torch.Tensor) -> int:
    """Project a compiled emit residual through ``W_EMBED.T`` and
    return the host argmax row index — mirrors the ``Unembedding``
    runtime path."""
    assert compiled_residual.shape == (1, TOKEN_VOCAB.layout.d_embed)
    scores = compiled_residual @ W_EMBED.T
    return int(scores.argmax(dim=1).item())


def _row_for(t, values: dict) -> int:
    start, end = TOKEN_VOCAB.type_to_row_range[t]
    for offset, (_t, v) in enumerate(TOKEN_VOCAB.row_to_token[start:end]):
        if v == values:
            return start + offset
    raise KeyError(f"No row for {t.name}{values!r}")


def _quantized_v(slot, k: int) -> float:
    span = slot.hi - slot.lo
    return slot.lo + (k / (slot.levels - 1)) * span


def test_compiled_emit_slotless_round_trip() -> None:
    """Slotless emit compiles to a pure-literal residual; argmax →
    that type's single row."""
    with fresh_graph_session():
        pos_enc = create_pos_encoding()
        out = emit_slotless(BEGIN)
        compiled = compile_headless(out, pos_enc, verbose=False)
        # No InputNodes ⇒ empty input specs ⇒ d_input = 0.
        prefill = torch.zeros((1, 0))
        residual = compiled(prefill)
    argmax = _deembed_argmax(residual)
    assert argmax == _row_for(BEGIN, {}), (
        f"BEGIN round-trip: argmax {argmax} != "
        f"expected {_row_for(BEGIN, {})}"
    )


def test_compiled_emit_int_slot_single_round_trip() -> None:
    """NODE.j (cardinality 64) — 1-digit digit-quad block, pure
    affine path."""
    for j in [0, 1, 5, NODE.slots["j"].hi - 1]:
        with fresh_graph_session():
            pos_enc = create_pos_encoding()
            j_in = create_input("j", 1, value_range=(-1.0, 256.0))
            out = emit_int_slot_token(NODE, j=j_in)
            compiled = compile_headless(out, pos_enc, verbose=False)
            prefill = build_prefill_from_input_values(
                compiled, {"j": torch.tensor([[float(j)]])}, n_pos=1
            )
            residual = compiled(prefill)
        argmax = _deembed_argmax(residual)
        expected = _row_for(NODE, {"j": j})
        assert argmax == expected, (
            f"NODE(j={j}) round-trip: argmax {argmax} != expected {expected}"
        )


def test_compiled_emit_int_slot_multi_round_trip() -> None:
    """SEG: two 1-digit slots (i: cardinality 128, is_first_of_ss: 2)."""
    cases = [(0, 0), (1, 1), (63, 0), (127, 1)]
    for i, flag in cases:
        with fresh_graph_session():
            pos_enc = create_pos_encoding()
            i_in = create_input("i", 1, value_range=(-1.0, 256.0))
            f_in = create_input("flag", 1, value_range=(-1.0, 4.0))
            out = emit_int_slot_token(SEG, i=i_in, is_first_of_ss=f_in)
            compiled = compile_headless(out, pos_enc, verbose=False)
            prefill = build_prefill_from_input_values(
                compiled,
                {
                    "i": torch.tensor([[float(i)]]),
                    "flag": torch.tensor([[float(flag)]]),
                },
                n_pos=1,
            )
            residual = compiled(prefill)
        argmax = _deembed_argmax(residual)
        expected = _row_for(SEG, {"i": i, "is_first_of_ss": flag})
        assert argmax == expected, (
            f"SEG(i={i}, flag={flag}) round-trip: argmax {argmax} != "
            f"expected {expected}"
        )


def test_compiled_emit_three_slot_round_trip() -> None:
    """DRAWSEG_META — three IntSlots, mixed cardinality; multi-slot path."""
    for i, wall_kind, silhouette in [(0, 0, 0), (64, 2, 3), (127, 1, 0)]:
        with fresh_graph_session():
            pos_enc = create_pos_encoding()
            i_in = create_input("i", 1, value_range=(-1.0, 130.0))
            wk_in = create_input("wk", 1, value_range=(-1.0, 4.0))
            sil_in = create_input("sil", 1, value_range=(-1.0, 5.0))
            out = emit_int_slot_token(
                DRAWSEG_META, i=i_in, wall_kind=wk_in, silhouette=sil_in
            )
            compiled = compile_headless(out, pos_enc, verbose=False)
            prefill = build_prefill_from_input_values(
                compiled,
                {
                    "i": torch.tensor([[float(i)]]),
                    "wk": torch.tensor([[float(wall_kind)]]),
                    "sil": torch.tensor([[float(silhouette)]]),
                },
                n_pos=1,
            )
            residual = compiled(prefill)
        argmax = _deembed_argmax(residual)
        expected = _row_for(
            DRAWSEG_META,
            {"i": i, "wall_kind": wall_kind, "silhouette": silhouette},
        )
        assert argmax == expected, (
            f"DRAWSEG_META(i={i}, wall_kind={wall_kind}, "
            f"silhouette={silhouette}) round-trip: argmax {argmax} != {expected}"
        )


def test_compiled_emit_float_slot_round_trip() -> None:
    """VALUE.v — the FloatSlot path. 2-digit digit-quad block via
    ``thermometer_floor_div`` (one MLP sublayer)."""
    slot = VALUE.slots["v"]
    span = slot.hi - slot.lo
    # Cover the full FloatSlot range plus byte-boundary neighbors.
    test_ks = [0, 1, 100, 32767, 32768, 65534, 65535]
    for k in test_ks:
        v = slot.lo + (k / (slot.levels - 1)) * span
        with fresh_graph_session():
            pos_enc = create_pos_encoding()
            v_in = create_input(
                "v",
                1,
                value_range=(float(slot.lo) - 1.0, float(slot.hi) + 1.0),
            )
            out = emit_float_slot_token(VALUE, v=v_in)
            compiled = compile_headless(out, pos_enc, verbose=False)
            prefill = build_prefill_from_input_values(
                compiled, {"v": torch.tensor([[float(v)]])}, n_pos=1
            )
            residual = compiled(prefill)
        argmax = _deembed_argmax(residual)
        expected = _row_for(VALUE, {"v": _quantized_v(slot, k)})
        assert argmax == expected, (
            f"VALUE(v={v}, k={k}) round-trip: argmax {argmax} != "
            f"expected {expected}"
        )
