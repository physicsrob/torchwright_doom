"""Compatibility wrapper for speculative decode inference.

Runtime implementation lives in ``render.inference`` so ONNX execution, KV
cache ownership, prefill, and decode loops are in one place.
"""

from __future__ import annotations

from . import inference as _inference
from .tokens_bridge import row_to_sandbox_token, sandbox_token_to_row

DEFAULT_PREFILL_CHUNK_SIZE = _inference.DEFAULT_PREFILL_CHUNK_SIZE
RolloutResult = _inference.RolloutResult
argmax_rows = _inference.argmax_rows


def spec_decode_rollout(
    compiled,
    prefill_ids: list[int],
    drafter,
    max_positions: int,
    terminal_row: int,
    *,
    draft_window: int = 8,
    enable_reuse: bool = True,
    progress_every: int = 0,
    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
) -> tuple[RolloutResult, dict]:
    return _inference.spec_decode_rollout(
        compiled,
        prefill_ids,
        drafter,
        max_positions,
        terminal_row,
        draft_window=draft_window,
        enable_reuse=enable_reuse,
        progress_every=progress_every,
        prefill_chunk_size=prefill_chunk_size,
        argmax_fn=argmax_rows,
        sandbox_token_to_row_fn=sandbox_token_to_row,
        row_to_sandbox_token_fn=row_to_sandbox_token,
    )
