"""Speculative decoding over ``compiled.step`` — Plan K Step 2.

A **strict optimization**: the committed stream is byte-identical to ``pure_ar``
(the accept test is exact ``W_EMBED``-row equality, so every committed row is the
model's own argmax — the drafter only gates *how many* positions one forward
verifies). Deleting the drafter recovers pure AR. Correctness never depends on the
drafter; it's the speed story.

Algorithm ported from ``doom_sandbox/runtime/loop.py`` (``_spec_decode_step`` +
the ``run_loop`` AR phase), adapted from the sandbox ``Past``/``embed_batch``/
``deembed_batch`` to ``compiled.step`` + the row bridge:

* draft accept test  -> integer row compare (``pred_row == sandbox_token_to_row(draft)``)
* ``Past._partial_commit_batch(commit)`` -> slice ``new_K[:, :cache_len + commit]``
* ``drafter.consume`` -> the draft Token (accepted) or ``row_to_sandbox_token(pred)``
  (the model's correction/bonus — the §3.4 re-sync)

Dumb-host note: the drafter is the reference renderer running on the host, but it
only proposes; the model's argmax decides every commit. It is a decode accelerator,
never an answer source.
"""

from __future__ import annotations

import time
from typing import Any

from .compiled_model import argmax_rows
from .pure_ar import RolloutResult
from .tokens_bridge import row_to_sandbox_token, rows_to_input, sandbox_token_to_row

# Adaptive draft-reuse controller (loop.py). Reusing the discarded speculative
# tail as the next batch's leading drafts wins when rejections don't propagate and
# loses when they do; gate it on a saturating health credit that self-disables in
# a propagating regime and re-probes periodically.
_REUSE_HEALTH_INIT = 4
_REUSE_HEALTH_CAP = 8
_REUSE_HEALTH_PENALTY = 2
_REUSE_PROBE_INTERVAL = 64


def _new_stats() -> dict[str, Any]:
    return {
        "batches": 0,
        "forward_passes": 0,
        "attempted_drafts": 0,
        "committed_rows": 0,
        "accepted_drafts": 0,
        "full_accepts": 0,
        "mispredicts": 0,
        "reject_at_0": 0,
        "terminal_truncations": 0,
        "fallback_single_steps": 0,
        "reused_offered": 0,
        "reused_accepted": 0,
        "accept_histogram": {},
    }


def _commit_kv(new_past, target: int):
    new_k, new_v = new_past
    return (
        tuple(k[:, :target] for k in new_k),
        tuple(v[:, :target] for v in new_v),
    )


def _single_step(compiled, past, input_row: int, emitted: list[int], stats: dict[str, Any]):
    out, new_past = compiled.step(rows_to_input([input_row]), past)
    stats["forward_passes"] += 1
    stats["fallback_single_steps"] += 1
    emitted.append(argmax_rows(out[-1:])[0])
    return new_past


def _spec_step(
    compiled,
    past,
    next_input_row: int,
    emitted: list[int],
    drafter,
    max_drafts: int,
    terminal_row: int,
    stats: dict[str, Any],
    reuse_buffer: list,
    reuse_health: int,
    reuse_probe: int,
    enable_reuse: bool,
):
    """One batched draft-verify step. Returns (new_past, reuse_tail, health, probe)."""
    cache_len = past[0][0].shape[1]
    snap = drafter.snapshot()

    drafts: list = []  # sandbox Tokens
    if enable_reuse and reuse_health > 0:
        for r in reuse_buffer:
            if len(drafts) >= max_drafts:
                break
            drafts.append(r)
            drafter.consume(r)  # speculative — rolled back below
    n_reused = len(drafts)
    while len(drafts) < max_drafts:
        d = drafter.next_draft()
        if d is None:
            break
        drafts.append(d)
        drafter.consume(d)  # speculative — rolled back below

    n_drafts = len(drafts)
    if n_drafts == 0:
        drafter.rollback(snap)
        new_past = _single_step(compiled, past, next_input_row, emitted, stats)
        drafter.consume(row_to_sandbox_token(emitted[-1]))
        return new_past, [], reuse_health, reuse_probe

    draft_rows = [sandbox_token_to_row(d) for d in drafts]
    batch_rows = [next_input_row] + draft_rows
    out, new_past = compiled.step(rows_to_input(batch_rows), past)
    stats["forward_passes"] += 1
    pred = argmax_rows(out)  # length n_drafts + 1 (last row is the bonus)

    accept = n_drafts
    for i in range(n_drafts):
        if pred[i] != draft_rows[i]:
            accept = i
            break
    commit = (n_drafts + 1) if accept == n_drafts else (accept + 1)

    # Truncate the commit at the first terminal emission so the stream ends cleanly.
    terminal_hit = False
    for i in range(commit):
        emission_row = draft_rows[i] if i < accept else pred[i]
        if emission_row == terminal_row:
            commit = i + 1
            terminal_hit = True
            stats["terminal_truncations"] += 1
            break

    new_past = _commit_kv(new_past, cache_len + commit)

    stats["batches"] += 1
    stats["attempted_drafts"] += n_drafts
    stats["committed_rows"] += commit
    stats["accepted_drafts"] += min(accept, commit)
    stats["reused_offered"] += n_reused
    stats["reused_accepted"] += min(accept, n_reused)
    if accept == n_drafts:
        stats["full_accepts"] += 1
    else:
        stats["mispredicts"] += 1
        if accept == 0:
            stats["reject_at_0"] += 1
    stats["accept_histogram"][accept] = stats["accept_histogram"].get(accept, 0) + 1

    # Rewind the speculative consumes, then replay exactly what was committed
    # (verified prefix + the model's correction/bonus) so the drafter re-syncs to
    # the model's actual trajectory.
    drafter.rollback(snap)
    for i in range(commit):
        emission_row = draft_rows[i] if i < accept else pred[i]
        emission_tok = drafts[i] if i < accept else row_to_sandbox_token(pred[i])
        drafter.consume(emission_tok)
        emitted.append(emission_row)

    if n_reused > 0:
        if accept > 0:
            reuse_health = min(_REUSE_HEALTH_CAP, reuse_health + 1)
        else:
            reuse_health = max(0, reuse_health - _REUSE_HEALTH_PENALTY)
    if reuse_health == 0:
        reuse_probe += 1
        if reuse_probe >= _REUSE_PROBE_INTERVAL:
            reuse_health = 1  # periodically re-test reuse
            reuse_probe = 0
    else:
        reuse_probe = 0

    if terminal_hit or not enable_reuse or reuse_health == 0:
        return new_past, [], reuse_health, reuse_probe
    reuse_tail = [row_to_sandbox_token(pred[i]) for i in range(commit, n_drafts + 1)]
    return new_past, reuse_tail, reuse_health, reuse_probe


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
) -> tuple[RolloutResult, dict[str, Any]]:
    """Generate the rollout with speculative decoding. Returns (result, stats)."""
    stats = _new_stats()
    t0 = time.time()

    past = compiled.empty_past()
    out, past = compiled.step(rows_to_input(prefill_ids), past)
    stats["forward_passes"] += 1
    seed_row = argmax_rows(out[-1:])[0]
    emitted = [seed_row]

    # Seed skip: the AR seed (setCursorDirectionY) is the drafter's first draft —
    # consume it so the drafter stays aligned from token 1.
    first = drafter.next_draft()
    if first is not None and sandbox_token_to_row(first) == seed_row:
        drafter.consume(first)

    reuse_buffer: list = []
    reuse_health = _REUSE_HEALTH_INIT
    reuse_probe = 0
    drafter_done = drafter.next_draft() is None
    last_print = 1
    stopped = "cap"

    while True:
        next_input_row = emitted[-1]
        if next_input_row == terminal_row:
            stopped = "terminal"
            break
        if len(emitted) >= max_positions:
            stopped = "cap"
            break

        remaining_capacity = max_positions - len(emitted)
        if (
            not drafter_done
            and remaining_capacity >= 2
            and (reuse_buffer or drafter.next_draft() is not None)
        ):
            past, reuse_buffer, reuse_health, reuse_probe = _spec_step(
                compiled,
                past,
                next_input_row,
                emitted,
                drafter,
                min(draft_window, remaining_capacity - 1),
                terminal_row,
                stats,
                reuse_buffer,
                reuse_health,
                reuse_probe,
                enable_reuse,
            )
            if not reuse_buffer and drafter.next_draft() is None:
                drafter_done = True
        else:
            reuse_buffer = []
            past = _single_step(compiled, past, next_input_row, emitted, stats)
            if not drafter_done:
                drafter.consume(row_to_sandbox_token(emitted[-1]))
                if drafter.next_draft() is None:
                    drafter_done = True

        if progress_every and len(emitted) - last_print >= progress_every:
            last_print = len(emitted)
            print(
                f"[spec_decode] {len(emitted)} tokens, {stats['forward_passes']} "
                f"forward passes ({time.time() - t0:.1f}s)",
                flush=True,
            )

    return (
        RolloutResult(
            emitted_rows=emitted,
            stopped=stopped,
            n_forward_passes=stats["forward_passes"],
            seconds=time.time() - t0,
        ),
        stats,
    )
