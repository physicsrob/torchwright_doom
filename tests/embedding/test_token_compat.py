"""Plan A / A7: token-compatibility helpers.

Covers the surface Plan B's ``GraphPast.input_type()`` /
``_input_type_matches`` consume:

* ``TokenType.check`` -> ±1 on/off the active type.
* free ``extract.is_type`` still returns 0/1 (backwards compat).
* ``type_matches(input_type_code(input_vec), T)`` agrees with ``T.check``.
* ``type_matches_any`` over a type set.
* ``indicator_to_bool`` (0/1 -> ±1) inverts the ``is_type`` fold.

Reference-eval (exact math) substrate, mirroring test_extract_correctness.
"""

from __future__ import annotations

import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input

from torchwright_doom import extract
from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.tokens import IntSlot
from torchwright_doom.vocab import DONE, NODE, SEG, VALUE


def _row_for(token_type, slot_values: dict) -> torch.Tensor:
    start, _end = TOKEN_VOCAB.type_to_row_range[token_type]
    if not token_type.slots:
        return W_EMBED[start : start + 1].clone()
    slot_names = list(token_type.slots.keys())
    slot_objs = [token_type.slots[n] for n in slot_names]

    def step_index(slot, value):
        if isinstance(slot, IntSlot):
            return int(value) - slot.lo
        span = slot.hi - slot.lo
        return round((float(value) - slot.lo) / span * (slot.levels - 1))

    sizes = [
        (s.hi - s.lo) if isinstance(s, IntSlot) else s.levels for s in slot_objs
    ]
    indices = [
        step_index(slot_objs[i], slot_values[n]) for i, n in enumerate(slot_names)
    ]
    row = 0
    for i, idx in enumerate(indices):
        stride = 1
        for j in range(i + 1, len(sizes)):
            stride *= sizes[j]
        row += idx * stride
    return W_EMBED[start + row : start + row + 1].clone()


def _eval_one(node, row: torch.Tensor) -> float:
    out = reference_eval(node, {"iv": row}, 1)[node]
    assert out.shape == (1, 1)
    return out.item()


# Representative rows for a handful of distinct types.
_ROWS = {
    "NODE": (NODE, {"j": 5}),
    "SEG": (SEG, {"i": 3, "is_first_of_ss": 0}),
    "VALUE": (VALUE, {"v": 0.25}),
    "DONE": (DONE, {}),
}


def _row(name: str) -> torch.Tensor:
    t, values = _ROWS[name]
    return _row_for(t, values)


def test_check_pm1_on_and_off_type() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    node_check = NODE.check(inp)
    assert _eval_one(node_check, _row("NODE")) == 1.0
    for other in ("SEG", "VALUE", "DONE"):
        assert _eval_one(node_check, _row(other)) == -1.0


def test_free_is_type_still_0_1() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    node_is = extract.is_type(inp, NODE)
    assert _eval_one(node_is, _row("NODE")) == 1.0
    assert _eval_one(node_is, _row("VALUE")) == 0.0


def test_type_matches_via_input_type_code_agrees_with_check() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    code = extract.input_type_code(inp)
    matches = extract.type_matches(code, NODE)
    check = NODE.check(inp)
    for name in ("NODE", "SEG", "VALUE", "DONE"):
        row = _row(name)
        assert _eval_one(matches, row) == _eval_one(check, row)


def test_type_matches_any() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    code = extract.input_type_code(inp)
    any_node_seg = extract.type_matches_any(code, [NODE, SEG])
    assert _eval_one(any_node_seg, _row("NODE")) == 1.0
    assert _eval_one(any_node_seg, _row("SEG")) == 1.0
    for other in ("VALUE", "DONE"):
        assert _eval_one(any_node_seg, _row(other)) == -1.0


def test_indicator_to_bool_inverts_is_type_fold() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    bool_node = extract.indicator_to_bool(extract.is_type(inp, NODE))
    check = NODE.check(inp)
    for name in ("NODE", "VALUE"):
        row = _row(name)
        assert _eval_one(bool_node, row) == _eval_one(check, row)
