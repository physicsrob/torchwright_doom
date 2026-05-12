"""DOOM's token vocabulary and embedding table.

A ``TokenVocab`` enumerates every discrete ``(token_type, slot_values)``
combination across the vocab's ``TokenType`` list, producing one
``W_EMBED`` row per combination. The autoregressive loop feeds a 1-wide
``token_ids`` input into a graph ``Embedding`` that looks each ID up
against ``W_EMBED``; on the output side, the matching projection
through ``W_EMBED.T`` recovers the next token via argmax.

Each row carries:

  cols [0 :  8]               — 8-wide E8 category code for the token's
                                 ``TokenType``. All rows of a given type
                                 share the same code; the slot / derived
                                 columns disambiguate within a type.
  per-(type, slot) raw col    — one column per (type, slot). The row's
                                 slot value is written normalized:
                                 ``(v - lo) / (hi - lo)`` for ``IntSlot``,
                                 ``(2k + 1) / (2 * levels)`` for the
                                 ``k``-th ``FloatSlot`` level. Columns
                                 for slots the row's type doesn't carry
                                 stay 0.
  per-(type, slot) digit-quad — 2- or 4-wide block carrying a digit-
                                 quadratic payload of the slot's integer
                                 step index ``k``. The block lets the
                                 producer query ``[2·hi_q_c, 1,
                                 2·lo_q_c, 1]`` (the centered base-256
                                 digits of the predicted ``q``) and have
                                 the dot product against any candidate
                                 row equal
                                 ``-(hi - hi_q)² - (lo - lo_q)² +
                                 const`` — argmax over a type's rows
                                 picks the nearest byte pair to the
                                 query. One digit (2 cols) for slots
                                 with cardinality ≤ 256, two digits
                                 (4 cols) for cardinality 257..65536.
  per-derived-name cols       — one column per distinct ``derived_name``
                                 declared on any slot in the vocab.
                                 Token types are mutually exclusive at
                                 inference, so columns are shared across
                                 types: ``x_oh_007`` is one column whether
                                 it came from ``EMIT_X1.x`` or
                                 ``WALL_COLUMN.x``. The ``Layout`` asserts
                                 that same-named declarations agree on
                                 their function output by sampling at
                                 construction time.

The digit-quad block replaces the older 16-wide Gray-code payload and
the 1-wide K column. The Gray-code design needed a 4096 / 8192 / 16384-
peak triangle-wave PWL chain on the producer side (~3 MLP sublayers)
to lay down a matching pattern from a computed ``q``. The digit-quad
encoder is one vector-PWL staircase (the high byte) plus affine math
(the low byte) — ~2 sublayers, and every slot uses the same recipe.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import torch

from torchwright.graph.embedding import Embedding
from torchwright.graph.spherical_codes import index_to_vector

from .tokens import FloatSlot, IntSlot, TokenType
from .vocab import VOCAB_TYPES


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

D_CATEGORY: int = 8

BASE: int = 256
"""Digit base for the digit-quadratic block. Splits any step index
``k`` into ``hi = k // BASE`` and ``lo = k % BASE`` (both bytes)."""

CENTER: float = (BASE - 1) / 2
"""Center of the byte range [0, BASE). Both row and query coordinates
are stored as ``byte - CENTER`` so the quadratic-difference math is
balanced around zero."""

MAX_INT_K_PER_DIGIT: int = BASE - 1
"""Maximum integer per digit, kept here for callers that want to
assert their precomputed indices fit."""

MAX_CARDINALITY_PER_BLOCK: int = BASE * BASE
"""Largest slot cardinality supported by a 2-digit block. Slots beyond
this would need a 3-digit recursion (not implemented; would only kick
in if a slot grew past 65,536 levels)."""

# Range for E8 category code indices. Any 89 indices in [0, 1024) give
# distinct 8-wide codes; we assign them sequentially per-type. The
# index_to_vector loader caps at 1024 — we'll comfortably stay under
# even with the full prompt + AR vocab (~89 types).
_MAX_E8_INDEX: int = 1024


# ---------------------------------------------------------------------------
# Digit-quadratic encoding helpers
# ---------------------------------------------------------------------------


def _digit_count_for_cardinality(cardinality: int) -> int:
    """Pick 1 or 2 digits based on a slot's cardinality.

    ``cardinality`` is the number of distinct values the slot takes
    (``IntSlot.hi - IntSlot.lo`` or ``FloatSlot.levels``). Slots with
    > ``MAX_CARDINALITY_PER_BLOCK`` levels are not supported.
    """
    if cardinality <= 0:
        raise ValueError(f"cardinality must be > 0: got {cardinality}")
    if cardinality <= BASE:
        return 1
    if cardinality <= MAX_CARDINALITY_PER_BLOCK:
        return 2
    raise ValueError(
        f"cardinality {cardinality} exceeds the 2-digit digit-quadratic "
        f"block limit ({MAX_CARDINALITY_PER_BLOCK}). Add a 3-digit "
        f"recursion if the vocab grows a slot this wide."
    )


def digit_quad_query_columns_for(
    slot: IntSlot | FloatSlot,
) -> tuple[int, int]:
    """Return ``(digit_count, n_cols)`` for a slot's digit-quad block.

    Companion to :func:`digit_quad_row`: producer-side emit helpers use
    this to know how wide a block to write and how to lay out the
    digit columns. ``digit_count`` is 1 or 2; ``n_cols`` is
    ``2 * digit_count`` (each digit contributes a pair
    ``[d_c, -d_c²]``).
    """
    cardinality = _slot_levels(slot)
    digits = _digit_count_for_cardinality(cardinality)
    return digits, 2 * digits


def digit_quad_row(
    slot: IntSlot | FloatSlot, slot_value: int | float
) -> torch.Tensor:
    """Build the digit-quadratic payload for a row representing
    ``slot=slot_value``.

    Splits the integer step index ``k = slot_value_to_step_index(slot,
    slot_value)`` into base-256 digits and emits the centered
    ``[d_c, -d_c²]`` pair for each:

    * 1 digit (cardinality ≤ 256):
      ``[lo_c, -lo_c²]`` where ``lo_c = k - CENTER``.
    * 2 digits (cardinality ≤ 65,536):
      ``[hi_c, -hi_c², lo_c, -lo_c²]`` where ``hi = k // 256``,
      ``lo = k % 256``, and each ``*_c = byte - CENTER``.

    The producer side (see ``emit.py``) computes a continuous query
    ``[2·hi_q_c, 1, 2·lo_q_c, 1]``; the dot product against this row
    equals ``-(hi - hi_q)² - (lo - lo_q)² + const``, so argmax over a
    type's rows picks the nearest byte pair to ``q``.
    """
    cardinality = _slot_levels(slot)
    digits = _digit_count_for_cardinality(cardinality)
    k = _step_index_for(slot, slot_value)
    if not (0 <= k < cardinality):
        raise ValueError(
            f"step index {k} out of range for slot with cardinality "
            f"{cardinality} (slot_value={slot_value!r})"
        )

    if digits == 1:
        lo_c = float(k) - CENTER
        return torch.tensor([lo_c, -lo_c * lo_c], dtype=torch.float32)

    hi = k // BASE
    lo = k % BASE
    hi_c = float(hi) - CENTER
    lo_c = float(lo) - CENTER
    return torch.tensor(
        [hi_c, -hi_c * hi_c, lo_c, -lo_c * lo_c], dtype=torch.float32
    )


def _step_index_for(
    slot: IntSlot | FloatSlot, slot_value: int | float
) -> int:
    """Integer step index for ``slot_value`` against ``slot``'s grid."""
    if isinstance(slot, IntSlot):
        return int(slot_value) - slot.lo
    span = slot.hi - slot.lo
    if span <= 0 or slot.levels <= 1:
        return 0
    return round((float(slot_value) - slot.lo) / span * (slot.levels - 1))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DerivedCheckSample:
    """One (input, expected_output) pair to verify same-named derived
    columns agree across types."""

    slot_value: float
    expected: float


class Layout:
    """Column allocation for ``W_EMBED``.

    Fields:

    * ``d_embed`` — total embedding width.
    * ``e8_indices`` — map ``type_name -> int E8 index`` (the row at
      ``type_name`` writes ``index_to_vector(e8_indices[type_name])``
      into the category block).
    * ``slot_columns`` — map ``(type_name, slot_name) -> int column``
      (the raw normalized slot column).
    * ``digit_quad_columns`` — map
      ``(type_name, slot_name) -> (start_col, n_cols)`` for the
      per-slot digit-quad block (2 or 4 columns).
    * ``derived_columns`` — map ``derived_name -> int column`` (shared
      across types — see docstring at top of file).
    """

    def __init__(self, types: list[TokenType]):
        names = [t.name for t in types]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Vocab has duplicate type names: {sorted(names)}"
            )
        if len(types) > _MAX_E8_INDEX:
            raise ValueError(
                f"Vocab has {len(types)} types; max distinct E8 codes "
                f"available is {_MAX_E8_INDEX}"
            )

        self.types: list[TokenType] = list(types)
        self.types_by_name: dict[str, TokenType] = {t.name: t for t in types}

        # E8: one index per type, assigned sequentially.
        self.e8_indices: dict[str, int] = {
            t.name: i for i, t in enumerate(types)
        }

        col = D_CATEGORY

        # Per-(type, slot) raw columns.
        self.slot_columns: dict[tuple[str, str], int] = {}
        self.slot_kinds: dict[tuple[str, str], IntSlot | FloatSlot] = {}
        for t in types:
            for slot_name, slot in t.slots.items():
                self.slot_columns[(t.name, slot_name)] = col
                self.slot_kinds[(t.name, slot_name)] = slot
                col += 1

        # Per-(type, slot) digit-quadratic blocks. Each slot gets its
        # own block — output-side emit only ever needs one type's
        # columns; other types' digit-quad cols stay 0 in both the
        # query and the rows of other types.
        self.digit_quad_columns: dict[tuple[str, str], tuple[int, int]] = {}
        for t in types:
            for slot_name, slot in t.slots.items():
                _, n_cols = digit_quad_query_columns_for(slot)
                self.digit_quad_columns[(t.name, slot_name)] = (col, n_cols)
                col += n_cols

        # Derived columns: shared by name across types. Verify
        # cross-type agreement by sampling each function.
        self.derived_columns: dict[str, int] = {}
        self._derived_samples: dict[str, _DerivedCheckSample] = {}
        for t in types:
            for slot_name, slot in t.slots.items():
                if not slot.derived:
                    continue
                for derived_name, fn in slot.derived.items():
                    self._register_derived(t, slot_name, slot, derived_name, fn, col)
                    if derived_name not in self.derived_columns:
                        self.derived_columns[derived_name] = col
                        col += 1

        self.d_embed: int = col

    def _register_derived(
        self,
        t: TokenType,
        slot_name: str,
        slot: IntSlot | FloatSlot,
        derived_name: str,
        fn: Callable[[int | float], float],
        col: int,
    ) -> None:
        """Sample ``fn`` at a representative slot value and confirm any
        prior declaration of ``derived_name`` agrees on that value."""
        sample_value: float
        if isinstance(slot, IntSlot):
            sample_value = float(slot.lo)
        else:
            sample_value = float(slot.lo)
        try:
            expected = float(fn(sample_value))
        except Exception as exc:  # pragma: no cover
            raise ValueError(
                f"Derived column {derived_name!r} on {t.name}.{slot_name} "
                f"failed sample evaluation at value={sample_value}: {exc}"
            )
        existing = self._derived_samples.get(derived_name)
        if existing is None:
            self._derived_samples[derived_name] = _DerivedCheckSample(
                slot_value=sample_value, expected=expected
            )
        else:
            if not math.isclose(
                existing.expected, expected, rel_tol=1e-9, abs_tol=1e-9
            ) and not (
                math.isnan(existing.expected) and math.isnan(expected)
            ):
                raise ValueError(
                    f"Derived column {derived_name!r} is shared across "
                    f"types but disagrees on the sample value "
                    f"{sample_value!r}: previously {existing.expected!r}, "
                    f"now {expected!r} (declared on {t.name}.{slot_name})."
                )


# ---------------------------------------------------------------------------
# TokenVocab — owns the ordered row table and the layout, with a
# cardinality budget that mirrors the sandbox's TokenVocab.
# ---------------------------------------------------------------------------


DEFAULT_MAX_CARDINALITY: int = 1 << 17


class TokenVocab:
    """Discrete vocabulary built from a list of ``TokenType``\\ s.

    Enumerates every ``(type, slot_values)`` combination into a flat
    row table. Each ``TokenType`` contributes ``t.cardinality`` rows
    (product of its slot cardinalities, or 1 for a slotless type).
    The total row count must stay under ``max_cardinality``.

    Attributes
    ----------
    types
        Ordered list of token types.
    n_rows
        Total cardinality, summed across types.
    row_to_token
        Index ``[0, n_rows)`` → ``(TokenType, slot_values dict)``.
    type_to_row_range
        Token type → ``(start, end)`` half-open row range.
    layout
        The shared ``Layout`` describing column allocation.
    """

    def __init__(
        self,
        types: list[TokenType],
        max_cardinality: int = DEFAULT_MAX_CARDINALITY,
    ):
        self.types: list[TokenType] = list(types)
        self.max_cardinality: int = max_cardinality
        self.layout: Layout = Layout(self.types)

        n_rows = sum(_type_cardinality(t) for t in self.types)
        if n_rows > max_cardinality:
            offenders = sorted(
                self.types, key=lambda t: _type_cardinality(t), reverse=True
            )[:3]
            details = ", ".join(
                f"{t.name}={_type_cardinality(t):,}" for t in offenders
            )
            raise ValueError(
                f"TokenVocab cardinality {n_rows:,} exceeds budget "
                f"{max_cardinality:,}. Top contributors: {details}."
            )

        self.row_to_token: list[tuple[TokenType, dict[str, int | float]]] = []
        self.type_to_row_range: dict[TokenType, tuple[int, int]] = {}
        for t in self.types:
            start = len(self.row_to_token)
            for combo in _enumerate_slot_combos(t):
                self.row_to_token.append((t, combo))
            self.type_to_row_range[t] = (start, len(self.row_to_token))
        assert len(self.row_to_token) == n_rows
        self.n_rows: int = n_rows


def _type_cardinality(t: TokenType) -> int:
    if not t.slots:
        return 1
    product = 1
    for slot in t.slots.values():
        if isinstance(slot, IntSlot):
            product *= slot.hi - slot.lo
        elif isinstance(slot, FloatSlot):
            product *= slot.levels
        else:  # pragma: no cover — exhaustive over Slot union
            raise TypeError(f"Unknown slot kind: {type(slot).__name__}")
    return product


def _slot_value_for_index(slot: IntSlot | FloatSlot, idx: int) -> int | float:
    """Slot value at the ``idx``-th step (matches the sandbox encoder)."""
    if isinstance(slot, IntSlot):
        return slot.lo + idx
    span = slot.hi - slot.lo
    return slot.lo + (idx / (slot.levels - 1)) * span


def _slot_levels(slot: IntSlot | FloatSlot) -> int:
    if isinstance(slot, IntSlot):
        return slot.hi - slot.lo
    return slot.levels


def _enumerate_slot_combos(
    t: TokenType,
) -> Iterable[dict[str, int | float]]:
    """Yield one ``{slot_name: slot_value}`` dict per row of type ``t``."""
    if not t.slots:
        yield {}
        return
    slot_items = list(t.slots.items())
    iterators = [
        [_slot_value_for_index(slot, idx) for idx in range(_slot_levels(slot))]
        for _, slot in slot_items
    ]
    for combo in itertools.product(*iterators):
        yield {slot_items[i][0]: combo[i] for i in range(len(slot_items))}


# ---------------------------------------------------------------------------
# W_EMBED construction
# ---------------------------------------------------------------------------


def _build_w_embed(vocab: TokenVocab) -> torch.Tensor:
    """Construct W_EMBED in numpy column-by-column.

    The per-row, per-derived-column Python loop the docstring describes
    is too slow on the full ~120k-row × ~330-derived vocab (35s+ at
    import time, dominated by torch element-assign overhead). The
    implementation below does the same writes column-batched in numpy
    and only converts to torch once at the end.
    """
    layout = vocab.layout
    w = np.zeros((vocab.n_rows, layout.d_embed), dtype=np.float32)

    e8_codes: dict[str, np.ndarray] = {
        t.name: index_to_vector(layout.e8_indices[t.name]).numpy()
        for t in vocab.types
    }

    for t in vocab.types:
        start, end = vocab.type_to_row_range[t]
        if start == end:
            continue

        w[start:end, 0:D_CATEGORY] = e8_codes[t.name]

        if not t.slots:
            continue

        # `slot_indices[:, s]` is the integer step index along slot s,
        # row-aligned with rows [start, end).
        slot_indices = _slot_index_table(t)
        slot_value_arrays: dict[str, np.ndarray] = {}
        for s_idx, (slot_name, slot) in enumerate(t.slots.items()):
            idxs = slot_indices[:, s_idx]
            slot_value_arrays[slot_name] = _slot_values_from_indices(slot, idxs)

            raw_col = layout.slot_columns[(t.name, slot_name)]
            w[start:end, raw_col] = _normalized_slot_column(slot, idxs)

            dq_start, dq_n = layout.digit_quad_columns[(t.name, slot_name)]
            w[start:end, dq_start : dq_start + dq_n] = _digit_quad_block(
                idxs, dq_n
            )

            for derived_name, fn in slot.derived.items():
                col = layout.derived_columns[derived_name]
                w[start:end, col] = _evaluate_derived(
                    fn, slot_value_arrays[slot_name]
                )

    return torch.from_numpy(w)


def _slot_index_table(t: TokenType) -> np.ndarray:
    """`(n_rows_for_type, n_slots)` integer step indices in declaration order.

    Mirrors `itertools.product` over slot-step ranges, but as a numpy
    array. The product is built C-order (last slot fastest), matching
    `itertools.product(*[range(n0), range(n1), ...])`.
    """
    slots = list(t.slots.values())
    if not slots:
        return np.zeros((1, 0), dtype=np.int64)
    sizes = [_slot_levels(s) for s in slots]
    total = 1
    for n in sizes:
        total *= n
    out = np.empty((total, len(sizes)), dtype=np.int64)
    repeat = 1
    tile = total
    for s_idx, n in enumerate(sizes):
        tile //= n
        # rows alternate over slot s_idx every `tile` rows, and the
        # pattern of values 0..n-1 repeats `repeat` times.
        base = np.repeat(np.arange(n), tile)
        out[:, s_idx] = np.tile(base, repeat)
        repeat *= n
    return out


def _slot_values_from_indices(
    slot: IntSlot | FloatSlot, indices: np.ndarray
) -> np.ndarray:
    if isinstance(slot, IntSlot):
        return (slot.lo + indices).astype(np.int64)
    span = slot.hi - slot.lo
    levels = slot.levels
    return (slot.lo + (indices.astype(np.float64) / (levels - 1)) * span).astype(
        np.float64
    )


def _normalized_slot_column(
    slot: IntSlot | FloatSlot, indices: np.ndarray
) -> np.ndarray:
    """Vectorized analog of the per-row normalization in the docstring."""
    if isinstance(slot, IntSlot):
        span = slot.hi - slot.lo
        return (indices.astype(np.float32) / float(span)).astype(np.float32)
    levels = slot.levels
    return ((2 * indices + 1) / (2.0 * levels)).astype(np.float32)


def _evaluate_derived(
    fn: Callable[[int | float], float], slot_values: np.ndarray
) -> np.ndarray:
    """Evaluate ``fn`` elementwise over ``slot_values`` as fast as
    possible.

    First tries to call ``fn(slot_values)`` directly — many of our
    derived helpers compose math ops that broadcast over numpy arrays
    (the ``ANGLE_VALUE`` sin/cos chain is the load-bearing case at
    ~2.6M scalar evals). Falls back to a tight ``fromiter`` loop when
    the function isn't array-compatible (the ``_SCREEN_X_ONE_HOT_DERIVED``
    lambdas use ``int(value) == column`` which raises on arrays).
    """
    try:
        result = fn(slot_values)
        arr = np.asarray(result, dtype=np.float32)
        if arr.shape != slot_values.shape:
            # ``fn`` produced a scalar from an array input — fall through.
            raise ValueError("scalar-from-array")
        return arr
    except Exception:
        return np.fromiter(
            (float(fn(v)) for v in slot_values.tolist()),
            dtype=np.float32,
            count=slot_values.shape[0],
        )


def _digit_quad_block(indices: np.ndarray, n_cols: int) -> np.ndarray:
    """Vectorized digit-quadratic block matching ``digit_quad_row``.

    ``n_cols`` is 2 (one digit) or 4 (two digits). Rows are indexed by
    integer step ``k = indices[row]`` along the slot's grid.
    """
    if n_cols == 2:
        lo_c = indices.astype(np.float32) - np.float32(CENTER)
        return np.stack([lo_c, -lo_c * lo_c], axis=1).astype(np.float32)
    if n_cols == 4:
        hi = (indices // BASE).astype(np.float32)
        lo = (indices % BASE).astype(np.float32)
        hi_c = hi - np.float32(CENTER)
        lo_c = lo - np.float32(CENTER)
        return np.stack(
            [hi_c, -hi_c * hi_c, lo_c, -lo_c * lo_c], axis=1
        ).astype(np.float32)
    raise ValueError(f"Unsupported digit-quad block width: {n_cols}")


# ---------------------------------------------------------------------------
# Module-level vocab and embedding table
# ---------------------------------------------------------------------------


TOKEN_VOCAB: TokenVocab = TokenVocab(VOCAB_TYPES)
W_EMBED: torch.Tensor = _build_w_embed(TOKEN_VOCAB)
assert W_EMBED.shape == (TOKEN_VOCAB.n_rows, TOKEN_VOCAB.layout.d_embed)


def build_doom_embedding(input_name: str = "token_ids") -> Embedding:
    """Factory for the DOOM ``Embedding`` graph node.

    ``input_name`` is the slot ``Embedding.compute`` reads from
    ``input_values`` — wired to the 1-wide integer ``token_ids`` input
    declared on the DOOM graph.
    """
    # ``Embedding`` carries a vocabulary list for introspection; build
    # one-name-per-row entries from the (type, slot_values) table.
    vocab_names = [
        _row_label(t, values) for t, values in TOKEN_VOCAB.row_to_token
    ]
    return Embedding(
        vocab=vocab_names,
        d_embed=TOKEN_VOCAB.layout.d_embed,
        table=W_EMBED,
        input_name=input_name,
        special_tokens=[],
    )


def _row_label(t: TokenType, values: Mapping[str, int | float]) -> str:
    if not values:
        return t.name
    parts = [
        f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in values.items()
    ]
    return f"{t.name}({','.join(parts)})"
