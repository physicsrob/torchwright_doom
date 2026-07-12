"""Shared test helpers for the Plan D renderer-reads port.

The token -> ``W_EMBED``-row helpers live in
``torchwright_doom.tokenizer.rows`` (their public home); this module
re-exports them for the oracle harness plus the tiny shared scene fixture.
"""

from __future__ import annotations

import torch

from torchwright_doom.tokenizer.rows import (  # noqa: F401  (re-exports)
    row_index,
    token_row,
    tokens_to_input,
    value,
)
from torchwright_doom.model.tokens import TokenType
from torchwright_doom.model.value_ranges import ValueRange
from torchwright_doom.model.vocab import (
    BEGIN,
    BOS,
    NODE,
    NODE_BACK_CHILD,
    NODE_DX,
    NODE_DY,
    NODE_FRONT_CHILD,
    NODE_PX,
    NODE_PY,
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    SEG,
    SEG_AX,
    SS,
)

# Smallest scene that exercises the whole renderer spine: one BSP node (root=0,
# both children subsectors) and one subsector/seg. Starts at BOS (the position-0
# anchor) and ends at BEGIN (the AR seed). Shared by the whole-forward compile
# gate and the free-running rollout gate.
TINY_BSP_SCENE: list[tuple[TokenType, dict]] = [
    (BOS, {}),
    (PLAYER_X_MARK, {}),
    value(ValueRange.R1, 100.0),
    (PLAYER_Y_MARK, {}),
    value(ValueRange.R1, -30.0),
    (NODE, {"j": 0}),
    (NODE_PX, {}),
    value(ValueRange.R1, 50.0),
    (NODE_PY, {}),
    value(ValueRange.R1, -20.0),
    (NODE_DX, {}),
    value(ValueRange.R2, 40.0),
    (NODE_DY, {}),
    value(ValueRange.R2, -30.0),
    (NODE_FRONT_CHILD, {"child_u": 64}),
    (NODE_BACK_CHILD, {"child_u": 65}),
    (SS, {"s": 0}),
    (SEG, {"i": 0, "is_first_of_ss": 1}),
    (SEG_AX, {}),
    value(ValueRange.R1, 10.0),
    (BEGIN, {}),
]


def pad_iv(compiled, iv_input: torch.Tensor) -> torch.Tensor:
    """Zero-pad an ``iv`` input tensor to a compiled module's full input width.

    ``CompiledHeadless.__call__`` takes a positional ``(n_pos, d_in)`` tensor and
    re-slices each declared input slot by ``_input_specs`` internally, so a test
    that builds only the ``iv`` slot must place it at its column offset in a
    full-width row. This localizes the one read of the private ``_input_specs``.
    """
    n_pos = iv_input.shape[0]
    specs = compiled._input_specs
    d_in = max(start + width for _, start, width in specs)
    start, width = next((s, w) for nm, s, w in specs if nm == "iv")
    full = torch.zeros(n_pos, d_in, dtype=iv_input.dtype)
    full[:, start : start + width] = iv_input
    return full
