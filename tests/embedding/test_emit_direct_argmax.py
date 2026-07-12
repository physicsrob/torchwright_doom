"""Direct-argmax cover test: every W_EMBED row's ideal emit residual
projects onto itself under host argmax.

Build the "ideal" emit residual for every ``(token_type, slot_values)``
row in ``W_EMBED`` and verify that projection through ``W_EMBED.T``
argmaxes onto that row. The test runs on the entire vocab
(120,769 rows), checking host argmax on the design itself before any
graph / op noise enters the picture.

The ideal emit residual for a row at step index ``k`` has the same E8
code as the W_EMBED row, the same raw normalized column, the
**digit-quad query** ``[2·hi_c, 1, 2·lo_c, 1]`` (in contrast to the
W_EMBED row's ``[hi_c, -hi_c², lo_c, -lo_c²]`` payload), and zero in
every other column — derived included.
"""

from __future__ import annotations

import numpy as np
import torch

from torchwright_doom.model.embedding import (
    BASE,
    CENTER,
    D_CATEGORY,
    TOKEN_VOCAB,
    W_EMBED,
    _slot_index_table,
    _normalized_slot_column,
)
from torchwright.graph.spherical_codes import index_to_vector


def _digit_quad_query_block(indices: np.ndarray, n_cols: int) -> np.ndarray:
    """Vectorized digit-quadratic *query* block for integer step
    indices.

    Mirrors ``_digit_quad_block`` from embedding.py, but in the
    query layout ``[..., 2·d_c, 1, ...]`` rather than the row layout.
    """
    if n_cols == 2:
        lo_c = indices.astype(np.float32) - np.float32(CENTER)
        return np.stack([2.0 * lo_c, np.ones_like(lo_c)], axis=1).astype(np.float32)
    if n_cols == 4:
        hi = (indices // BASE).astype(np.float32)
        lo = (indices % BASE).astype(np.float32)
        hi_c = hi - np.float32(CENTER)
        lo_c = lo - np.float32(CENTER)
        ones = np.ones_like(hi_c)
        return np.stack([2.0 * hi_c, ones, 2.0 * lo_c, ones], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported digit-quad query width: {n_cols}")


def _build_ideal_emit_rows_for_type(t) -> torch.Tensor:
    """Build the ``(n_rows_for_type, d_embed)`` ideal emit residual
    matrix for ``t`` — one row per ``(slot_values)`` combination,
    matching what :func:`emit_token` produces for that combination."""
    layout = TOKEN_VOCAB.layout
    start, end = TOKEN_VOCAB.type_to_row_range[t]
    n = end - start
    out = np.zeros((n, layout.d_embed), dtype=np.float32)

    # E8 category code
    e8 = index_to_vector(layout.e8_indices[t.name]).numpy()
    out[:, 0:D_CATEGORY] = e8

    if not t.slots:
        return torch.from_numpy(out)

    slot_indices = _slot_index_table(t)

    for s_idx, (slot_name, slot) in enumerate(t.slots.items()):
        idxs = slot_indices[:, s_idx]

        # Raw normalized column — same as W_EMBED on the emit side.
        raw_col = layout.slot_columns[(t.name, slot_name)]
        out[:, raw_col] = _normalized_slot_column(slot, idxs)

        # Digit-quad query payload (NOT the row payload).
        dq_start, dq_n = layout.digit_quad_columns[(t.name, slot_name)]
        out[:, dq_start : dq_start + dq_n] = _digit_quad_query_block(idxs, dq_n)

    # Derived columns stay at zero on the emit side.
    return torch.from_numpy(out)


def test_within_type_argmax_lands_on_row(device) -> None:
    """For every type's rows, projecting the ideal emit residual through
    that type's W_EMBED slice argmaxes onto the matching row."""
    W = W_EMBED.to(device)

    for t in TOKEN_VOCAB.types:
        start, end = TOKEN_VOCAB.type_to_row_range[t]
        if end == start:
            continue
        emit_chunk = _build_ideal_emit_rows_for_type(t).to(device)
        W_chunk = W[start:end]

        n = end - start
        step = 1024
        for cs in range(0, n, step):
            ce = min(cs + step, n)
            scores = emit_chunk[cs:ce] @ W_chunk.T  # (chunk, n_T)
            argmax = scores.argmax(dim=1)
            expected = torch.arange(cs, ce, device=device)
            mismatch = (argmax != expected).nonzero(as_tuple=True)[0]
            if mismatch.numel() > 0:
                first = int(mismatch[0].item())
                bad_row = cs + first
                bad_pick = int(argmax[first].item())
                raise AssertionError(
                    f"Within-type argmax mismatch on {t.name}: row "
                    f"{bad_row} (slot_values="
                    f"{TOKEN_VOCAB.row_to_token[start + bad_row][1]!r}) "
                    f"picked offset {bad_pick} "
                    f"(slot_values="
                    f"{TOKEN_VOCAB.row_to_token[start + bad_pick][1]!r})"
                )


def test_cross_type_e8_discrimination_beats_within_type_floor(device) -> None:
    """Cross-type score between any two distinct types is strictly less
    than the worst-case within-type self-score, so argmax over the full
    W_EMBED.T cannot pick a row of the wrong type.

    Cross-type dot factors through E8 alone: emit's per-(type, slot)
    raw and digit-quad columns are zero outside the active type, and
    every row of a foreign type has zero in the active type's slot
    columns. The only shared, non-zero columns between an emit row of
    type T1 and any W_EMBED row of type T2 are ``[0:8]``. Derived
    columns are zero on the emit side.
    """
    W = W_EMBED.to(device)

    # Per-type within-min: the smallest "self-dot" (emit_i · W[i])
    # across the type's rows. The cross-type bound has to beat this.
    within_min_by_type: dict[str, float] = {}
    e8_codes: dict[str, torch.Tensor] = {}
    for t in TOKEN_VOCAB.types:
        start, end = TOKEN_VOCAB.type_to_row_range[t]
        if end == start:
            continue
        emit_chunk = _build_ideal_emit_rows_for_type(t).to(device)
        self_dots = (emit_chunk * W[start:end]).sum(dim=1)
        within_min_by_type[t.name] = float(self_dots.min().item())
        e8_codes[t.name] = W[start, 0:D_CATEGORY]

    # E8 cross-dot is a constant function of the pair (every row of a
    # foreign type carries the same E8 code; cross slot/digit-quad cols
    # contribute zero).
    for t1 in TOKEN_VOCAB.types:
        if t1.name not in within_min_by_type:
            continue
        floor = within_min_by_type[t1.name]
        for t2 in TOKEN_VOCAB.types:
            if t2 is t1 or t2.name not in within_min_by_type:
                continue
            cross_dot = float((e8_codes[t1.name] @ e8_codes[t2.name]).item())
            assert cross_dot < floor, (
                f"cross-type E8 dot {cross_dot:.3f} between "
                f"{t1.name} → {t2.name} >= within-min self-score "
                f"{floor:.3f} for {t1.name}; argmax across types would "
                f"flip to {t2.name}"
            )
