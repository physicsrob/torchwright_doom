"""Pure autoregressive rollout over ``compiled.step`` — Plan K Step 1.

The honest baseline: one token at a time, the model's argmax fed straight back as
the next input, until a terminal row or a position cap. No drafter, no
speculation. The host does only argmax + copy — exactly like decoding any LLM.
This alone proves the thesis (it's just slow: one forward pass per token).

Generalizes ``tests/scene/test_forward_ar_rollout.py::_compiled_rollout`` from a
fixed step budget to a terminal-or-cap stop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .compiled_model import argmax_rows
from .tokens_bridge import rows_to_input


@dataclass
class RolloutResult:
    emitted_rows: list[int]
    stopped: str  # "terminal" | "cap"
    n_forward_passes: int
    seconds: float


def pure_ar_rollout(
    compiled,
    prefill_ids: list[int],
    max_positions: int,
    terminal_row: int,
    *,
    progress_every: int = 0,
) -> RolloutResult:
    """Prefill once, then decode one token at a time until terminal or cap.

    ``emitted_rows[0]`` is the AR seed (the emission at the last prefill position),
    matching the existing rollout test's convention.
    """
    t0 = time.time()
    past = compiled.empty_past()
    out, past = compiled.step(rows_to_input(prefill_ids), past, past_len=0)
    n_passes = 1
    cur = argmax_rows(out[-1:])[0]
    emitted = [cur]
    seq_pos = len(prefill_ids)
    while cur != terminal_row and len(emitted) < max_positions:
        out, past = compiled.step(rows_to_input([cur]), past, past_len=seq_pos)
        n_passes += 1
        seq_pos += 1
        cur = argmax_rows(out[-1:])[0]
        emitted.append(cur)
        if progress_every and len(emitted) % progress_every == 0:
            print(
                f"[pure_ar] {len(emitted)} tokens  ({time.time() - t0:.1f}s, "
                f"{(time.time() - t0) / len(emitted) * 1000:.0f} ms/tok)",
                flush=True,
            )
    return RolloutResult(
        emitted_rows=emitted,
        stopped="terminal" if cur == terminal_row else "cap",
        n_forward_passes=n_passes,
        seconds=time.time() - t0,
    )
