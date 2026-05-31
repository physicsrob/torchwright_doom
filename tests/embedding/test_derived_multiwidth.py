"""Plan A: multi-width derived columns round-trip through extract_derived.

``id_lifted_key`` (width 3, on node/seg/SSECTOR/...) and
``u_tan_by_column`` (width SCREEN_WIDTH, on angleValue) are the two
genuinely multi-width derived names. Embed a row carrying the slot,
read the whole span back via ``extract_derived`` (the gather/sum path),
and confirm it matches the declaring function — and that an off-type
row reads zeros.
"""

from __future__ import annotations

import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input

from torchwright_doom import extract
from torchwright_doom.constants import SCREEN_WIDTH
from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.tokens import IntSlot
from torchwright_doom.vocab import ANGLE_VALUE, NODE, VALUE, _u_tan_by_column


def _row_for(token_type, slot_values: dict) -> torch.Tensor:
    start, _end = TOKEN_VOCAB.type_to_row_range[token_type]
    if not token_type.slots:
        return W_EMBED[start : start + 1].clone()
    slot_names = list(token_type.slots.keys())
    slot_objs = [token_type.slots[n] for n in slot_names]
    sizes = [(s.hi - s.lo) if isinstance(s, IntSlot) else s.levels for s in slot_objs]

    def step_index(slot, value):
        if isinstance(slot, IntSlot):
            return int(value) - slot.lo
        span = slot.hi - slot.lo
        return round((float(value) - slot.lo) / span * (slot.levels - 1))

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


def _eval(node, row: torch.Tensor) -> torch.Tensor:
    out = reference_eval(node, {"iv": row}, 1)[node]
    return out.reshape(-1)


def test_id_lifted_key_width3_round_trip() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    node = extract.extract_derived(inp, "id_lifted_key")
    assert len(node) == 3
    for j in (0, 1, 5, 17, 63):
        got = _eval(node, _row_for(NODE, {"j": j}))
        expected = torch.tensor([float(j), float(-(j * j)), 1.0])
        assert torch.allclose(
            got, expected, atol=1e-4
        ), f"NODE.j={j}: id_lifted_key {got.tolist()} != {expected.tolist()}"


def test_id_lifted_key_off_type_zero() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    node = extract.extract_derived(inp, "id_lifted_key")
    # VALUE doesn't declare id_lifted_key -> all-zero span.
    got = _eval(node, _row_for(VALUE, {"v": 0.25}))
    assert torch.allclose(got, torch.zeros(3), atol=1e-4)


def test_u_tan_by_column_width_screen_width_round_trip() -> None:
    inp = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    node = extract.extract_derived(inp, "u_tan_by_column")
    assert len(node) == SCREEN_WIDTH
    for angle in (-1024, 0, 256, 1024):
        got = _eval(node, _row_for(ANGLE_VALUE, {"angle": angle}))
        expected = torch.tensor(_u_tan_by_column(angle), dtype=torch.float32)
        assert torch.allclose(
            got, expected, atol=1e-4
        ), f"angle={angle}: u_tan_by_column mismatch"
