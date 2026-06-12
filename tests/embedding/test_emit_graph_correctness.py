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


import torch

from torchwright.debug.probe import reference_eval
from torchwright.graph import fresh_graph_session
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import BASE, CENTER, TOKEN_VOCAB, W_EMBED
from torchwright_doom.emit import (
    emit_float_slot_token,
    emit_int_slot_token,
    emit_slotless,
    emit_token,
)
from torchwright_doom.past import GraphPast
from torchwright_doom.render_main import forward
from torchwright_doom.tokens import FloatSlot
from torchwright_doom.vocab import (
    BEGIN,
    DONE,
    NO_OP,
    NODE,
    DRAWSEG_META,
    SEG,
    VALUE,
)


def _all_nodes(root):
    """Every distinct node reachable from ``root`` through ``.inputs``."""
    seen: dict[int, object] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen[id(n)] = n
        stack.extend(getattr(n, "inputs", []))
    return list(seen.values())


def _value_slot() -> FloatSlot:
    slot = VALUE.slots["v"]
    assert isinstance(slot, FloatSlot)
    return slot


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
        assert (
            argmax == expected
        ), f"slotless {t.name}: argmax {argmax} != expected {expected}"


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
        assert (
            argmax == expected
        ), f"NODE(j={j}): argmax {argmax} != expected {expected}"


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
            out = emit_int_slot_token(SEG, i=i_in, is_first_of_ss=f_in)
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
            f"SEG(i={i}, flag={flag}): argmax {argmax} != " f"expected {expected}"
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
    slot = _value_slot()
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
        assert (
            argmax == expected
        ), f"VALUE(v={v}, k={k}): argmax {argmax} != expected {expected}"


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
    assert _project_and_argmax(value) == _row_for(SEG, {"i": 5, "is_first_of_ss": 1})

    # FloatSlot
    with fresh_graph_session():
        v_in = create_input("v", 1, value_range=(-5000.0, 5000.0))
        out = emit_token(VALUE, v=v_in)
        cache = reference_eval(out, input_values={"v": torch.tensor([[0.5]])}, n_pos=1)
        value = cache[out]
    slot = _value_slot()
    span = slot.hi - slot.lo
    k = round((0.5 - slot.lo) / span * (slot.levels - 1))
    quantized = slot.lo + (k / (slot.levels - 1)) * span
    assert _project_and_argmax(value) == _row_for(VALUE, {"v": quantized})


def test_narrow_slot_high_byte_is_constant_no_floor() -> None:
    """A narrow IntSlot (cardinality ≤ 256) sitting at a 2-digit position
    encodes its high byte as a literal constant 0 — no ``floor_int`` staircase.

    ``NODE.j`` has cardinality 64 but lands at slot position 0, whose widest
    member (the 65,536-level VALUE slot) makes the shared digit-quad block two
    digits wide. The high byte ``floor(k / 256)`` is then provably 0 for every
    ``k < 64``, so the emit query's high-byte pair is the position-independent
    constant ``[2·(0 − CENTER), 1]`` and the low-byte pair is ``[2·(k − CENTER),
    1]``. This pins the byte-identical layout and the absence of the floor.
    """
    slot = NODE.slots["j"]
    assert (slot.hi - slot.lo) <= BASE  # narrow: high byte is a constant 0
    dq_start, dq_width = TOKEN_VOCAB.layout.digit_quad_columns[(NODE.name, "j")]
    assert dq_width == 4  # two digits — the high byte is the thing being skipped

    for k in [0, 1, 63]:
        with fresh_graph_session():
            j_in = create_input("j", 1, value_range=(-1.0, 256.0))
            out = emit_int_slot_token(NODE, j=j_in)
            # (a) No floor_int op anywhere in the emit subgraph for this slot.
            assert not any(
                getattr(n, "name", "").startswith("floor_int") for n in _all_nodes(out)
            ), f"NODE(j={k}) emit still builds a floor_int staircase"
            cache = reference_eval(
                out, input_values={"j": torch.tensor([[float(k)]])}, n_pos=1
            )
            value = cache[out]

        # (b) The emitted digit-quad block matches the documented layout exactly:
        #     hi byte query = [2·(0 − CENTER), 1], lo byte query = [2·(k − CENTER), 1].
        block = value[0, dq_start : dq_start + dq_width]
        expected = torch.tensor(
            [2.0 * (0.0 - CENTER), 1.0, 2.0 * (float(k) - CENTER), 1.0],
            dtype=torch.float32,
        )
        assert torch.allclose(block, expected, atol=1e-4), (
            f"NODE(j={k}) digit-quad block {block.tolist()} != "
            f"documented {expected.tolist()}"
        )
        # And argmax still lands on the right row — the constant high byte is an
        # additive constant shared across the type's rows, so it cannot bias the
        # pick.
        assert _project_and_argmax(value) == _row_for(NODE, {"j": k})


def test_forward_keeps_only_four_wide_floors() -> None:
    """Forward-level pin: the whole graph builds exactly four 255-wide
    ``floor_int`` staircases — one per genuine two-byte emitted carrier. Every
    other 2-digit emit is a narrow slot whose high byte is a skipped constant 0.

    The four carriers (verified by tracing each floor to its ``emit_dq_*`` head):

    * ``value`` (FloatSlot, cardinality 65,536) — the shared digit-quad VALUE
      head that ``_collapse_scalar_emits`` folds every VALUE ScalarEmit into;
    * ``angleValue`` (IntSlot, cardinality 8,192) — the shared ANGLE_VALUE head;
    * a *second* ``value`` head — Phase J's EAGER R3 v0 carrier emitted in
      ``PixelDispatcher.after_set_cursor_y``'s wall arm. It forks with
      SET_CURSOR_X in a ``select``, so it is a head Node, not a collapsible
      ScalarEmit, and stays a separate wide floor;
    * ``wallColU`` (IntSlot(-1024, 1024), cardinality 2,048) — Phase J's wall
      texel ``u_idx`` emit in ``PixelDispatcher.wall_column_output``.

    H had only the first two (count 2); Phase J's wall texel pass added the
    latter two genuine 2-byte carriers (count 4). Guards the digit-quad
    floor-skip optimization against regression: dropping the narrow-slot
    short-circuit jumps this back to ~40. A count other than 4 means a new wide
    carrier (or a collapse change) — update this list consciously.
    """
    iv = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    pos = create_pos_encoding()
    nt = forward(iv, GraphPast(input_vec=iv, pos_encoding=pos), pos)

    # `floor_int` expands into `floor_int_step_linear2` (one per instance), whose
    # width is the high-byte range. At the 2-digit positions the block is sized
    # by the 65,536-cardinality VALUE slot, so every genuine floor is 255 wide.
    wide_floors = [
        n
        for n in _all_nodes(nt)
        if getattr(n, "name", "") == "floor_int_step_linear2" and len(n) == 255
    ]
    assert len(wide_floors) == 4, (
        f"expected exactly 4 wide floor_int staircases (value x2, angleValue, "
        f"wallColU), found {len(wide_floors)}"
    )
