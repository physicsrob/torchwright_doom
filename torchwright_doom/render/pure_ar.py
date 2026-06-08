"""Compatibility wrapper for pure autoregressive inference.

Runtime implementation lives in ``render.inference`` so KV ownership and ONNX
binding behavior are auditable in one file.
"""

from __future__ import annotations

from . import inference as _inference

DEFAULT_PREFILL_CHUNK_SIZE = _inference.DEFAULT_PREFILL_CHUNK_SIZE
RolloutResult = _inference.RolloutResult
argmax_rows = _inference.argmax_rows


def pure_ar_rollout(
    compiled,
    prefill_ids: list[int],
    max_positions: int,
    terminal_row: int,
    *,
    progress_every: int = 0,
    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
) -> RolloutResult:
    return _inference.pure_ar_rollout(
        compiled,
        prefill_ids,
        max_positions,
        terminal_row,
        progress_every=progress_every,
        prefill_chunk_size=prefill_chunk_size,
        argmax_fn=argmax_rows,
    )
