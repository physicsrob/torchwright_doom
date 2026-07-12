"""Read-side std/helper shim.

The std surface the ported renderer files import from, lowering each to an
existing torchwright op. The helpers fall into these categories:

- Column plumbing (residual-column reshaping): ``concat``, ``split``,
  ``linear``, ``reduce_sum``.
- Booleans over ±1 predicates: ``bool_and``, ``bool_or``, ``bool_to_01``.
- Mutually-exclusive selection: ``select`` (two-way), ``type_switch`` (one
  of N gated branches), ``pick_by_one_hot`` (slot value by one-hot mask),
  ``pick_by_index`` (slot value by scalar index).
- Token emission: ``make_token`` / ``make_token_head`` (next-token rows /
  heads), ``clamp`` / ``clamp_to_slot`` (pin a value to a slot's range).
- Piecewise-linear and table builders: ``one_hot``, ``gate``, ``constant``,
  ``pwl_def`` (reusable 1D piecewise-linear function), ``table_lookup_2d``.
- Emit-deferral re-exports from :mod:`.emit`: ``ScalarEmit`` /
  ``AngleInputEmit`` (the deferred-carrier wrappers) and ``value_scalar`` /
  ``angle_scalar`` / ``angle_inputs`` (their builders).

``type_switch`` and ``pick_by_one_hot`` are the two helpers the output-head
dispatch is built from: ``render_main.dispatch_next_token`` uses
``type_switch`` to select the winning branch's head, and ``pick_by_one_hot``
to pick a numeric carrier / world-angle scalar float-exact from its
candidates.

Two helpers that already exist on the real side — ``extract_derived`` and
``indicator_to_bool`` (in :mod:`.extract`) — are re-exported here so a ported
file has a single import source for the whole std surface.

These are graph-construction helpers: each returns a torchwright
``Node``. There is no runtime state.
"""

from __future__ import annotations

from typing import Callable

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.ops.linear import bool_to_01, sum_nodes
from torchwright.ops.swiglu.logic_ops import (
    bool_all_true,
    bool_any_true,
    bool_not as _bool_not,
    cond_gate,
)
from torchwright.ops.swiglu.map_select import in_range
from torchwright.ops.swiglu.arithmetic_ops import compare as _compare
from torchwright.ops.swiglu.arithmetic_ops import clamp as _clamp
from torchwright.ops.swiglu.arithmetic_ops import piecewise_linear as _piecewise_linear
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.swiglu.onehot_table import onehot_lookup
from torchwright.ops.swiglu.map_select import (
    broadcast_select as _broadcast_select,
    dynamic_extract as _dynamic_extract,
    select as _select,
    switch as _switch,
    table_lookup_2d as _table_lookup_2d,
)

from .emit import (
    AngleInputEmit,
    ScalarEmit,
    angle_inputs,
    angle_scalar,
    value_scalar,
)
from .extract import extract_derived, indicator_to_bool
from .tokens import FloatSlot, IntSlot, TokenType

__all__ = [
    "concat",
    "split",
    "one_hot",
    "gate",
    "linear",
    "constant",
    "bool_and",
    "bool_or",
    "bool_not",
    "compare",
    "sum",
    "select",
    "type_switch",
    "reduce_sum",
    "make_token",
    "bool_to_01",
    "extract_derived",
    "indicator_to_bool",
    "pick_by_one_hot",
    "pick_by_index",
    "pick_const_by_index",
    "pwl_def",
    "table_lookup_2d",
    "clamp",
    "clamp_to_slot",
    "ScalarEmit",
    "value_scalar",
    "angle_scalar",
    "AngleInputEmit",
    "angle_inputs",
]


def concat(*nodes: Node) -> Node:
    """Join node column ranges end to end.

    Lowers to :class:`Concatenate`. A single argument passes through
    unchanged (``Concatenate`` of one node would be a no-op anyway).
    """
    if not nodes:
        raise ValueError("concat requires at least one node")
    if len(nodes) == 1:
        return nodes[0]
    return Concatenate(list(nodes))


def _column_selector(d_in: int, start: int, width: int) -> torch.Tensor:
    """Identity selector reading ``[start:start + width]`` of a ``d_in`` row."""
    weights = torch.zeros((d_in, width), dtype=torch.float32)
    for off in range(width):
        weights[start + off, off] = 1.0
    return weights


def split(node: Node, sizes: list[int]) -> list[Node]:
    """Slice ``node`` into consecutive sub-ranges of the given ``sizes``.

    The original ``split`` is free metadata; on the real graph each piece
    is a fused identity ``Linear`` reading exactly its column range (the
    same column-select pattern :mod:`.extract` uses), which folds into
    its consumer.
    """
    total = 0
    for size in sizes:
        total += size
    if total != len(node):
        raise ValueError(
            f"split sizes {sizes} sum to {total}, not node width {len(node)}"
        )
    pieces: list[Node] = []
    start = 0
    for i, width in enumerate(sizes):
        pieces.append(
            Linear(node, _column_selector(len(node), start, width), name=f"split_{i}")
        )
        start += width
    return pieces


def linear(node: Node, output_matrix: list[list[float]]) -> Node:
    """Fixed ``(d_input, d_output)`` projection.

    ``output_matrix`` is plain numpy/list weight data, not a node — the
    real-graph weight analog — matching the original contract exactly.
    """
    weights = torch.tensor(output_matrix, dtype=torch.float32)
    return Linear(node, weights)


def constant(value: float | int | list[float]) -> Node:
    """Module-level fixed value.

    Scalars become 1-wide literals; lists become width-N literals.
    """
    if isinstance(value, (int, float)):
        tensor = torch.tensor([float(value)], dtype=torch.float32)
    else:
        tensor = torch.tensor([float(v) for v in value], dtype=torch.float32)
    return create_literal_value(tensor)


def _plus_one(scalar: Node) -> Node:
    """``scalar + 1`` as one fused ``Linear`` (the ``one_hot`` upper bound)."""
    return Linear(
        scalar,
        torch.ones((1, 1), dtype=torch.float32),
        torch.ones((1,), dtype=torch.float32),
        name="plus_one",
    )


def one_hot(scalar: Node, n: int) -> Node:
    """Width-``n`` numeric 0/1 one-hot of an integer scalar in ``[0, n)``.

    Lowers to ``bool_to_01(in_range(scalar, scalar + 1, n))``: ``in_range``
    marks position ``i`` true when ``scalar <= i + 0.5 < scalar + 1`` —
    i.e. ``i - 0.5 < scalar <= i + 0.5`` — so an integer ``scalar = k``
    selects index ``k`` cleanly, with the same ``1/step_sharpness``
    half-integer blend zones as the original's trapezoidal kernel.
    """
    if len(scalar) != 1:
        raise ValueError(f"one_hot expects a scalar node, got width {len(scalar)}")
    return bool_to_01(in_range(scalar, _plus_one(scalar), n))


def gate(cond: Node, value: Node) -> Node:
    """Zero-false value masking (lowers to ``cond_gate``).

    ``cond`` is a width-1 ±1 boolean; ``value`` passes through when
    ``cond`` is +1 and is zeroed when ``cond`` is -1.
    """
    return cond_gate(cond, value)


def bool_and(*conds: Node) -> Node:
    """Boolean conjunction over ±1 predicates."""
    return bool_all_true(list(conds))


def bool_or(*conds: Node) -> Node:
    """Boolean disjunction over ±1 predicates."""
    return bool_any_true(list(conds))


def bool_not(cond: Node) -> Node:
    """Boolean negation over a ±1 predicate."""
    return _bool_not(cond)


def compare(node: Node, thresh: float) -> Node:
    """±1 sharpened step: +1 where ``node`` is above ``thresh``, else -1."""
    return _compare(node, thresh=thresh, true_level=1.0, false_level=-1.0)


def sum(*nodes: Node) -> Node:  # noqa: A001 - intentional std-surface shadow
    """Pointwise N-ary sum of same-width nodes.

    Shadows ``builtins.sum`` for callers that ``from .std import sum``.
    """
    return sum_nodes(list(nodes))


def select(cond: Node, true_value: Node, false_value: Node) -> Node:
    """Two-way ±1-boolean branch."""
    return _select(cond, true_value, false_value)


def type_switch(*pairs: tuple[Node, Node], max_fanout: int | None = None) -> Node:
    """Mutually-exclusive branch selection.

    Each argument is a ``(condition, value)`` pair; exactly one condition is
    +1. Lowers to a sum of type-gated values (``cond_gate``). ``max_fanout``
    caps how many gated operands are summed at once: ``None`` is a single flat
    ``Linear`` (shallow, but all gated copies live together); ``k >= 2`` chains
    the reduction so at most ``k`` gated copies sit on the residual stream at a
    time, trading depth for a lower peak width. The dispatch uses the dial
    because each gated copy is a full emit head.
    """
    conditions = [c for c, _v in pairs]
    values = [v for _c, v in pairs]
    if max_fanout is None:
        return _switch(conditions, values)
    return sum_nodes([cond_gate(c, v) for c, v in pairs], max_fanout=max_fanout)


def reduce_sum(node: Node) -> Node:
    """Sum across a node's components to a 1-wide node.

    A single ``Linear`` with an all-ones column vector.
    """
    weights = torch.ones((len(node), 1), dtype=torch.float32)
    return Linear(node, weights, name="reduce_sum")


def pick_by_one_hot(mask: Node, table: Node, d_fill: int = 1) -> Node:
    """Select slot value(s) from a slot-major runtime table by a one-hot mask.

    ``mask`` is a width-``n_slots`` 0/1 one-hot; ``table`` is width
    ``n_slots * d_fill`` (slot-major). Lowers to ``broadcast_select`` (each
    slot picks its ``table`` value where the ±1 mask is +1, else 0) followed by
    a fixed ``Linear`` that sums across slots. Result width is ``d_fill``.
    """
    n_slots = len(mask)
    if len(table) != n_slots * d_fill:
        raise ValueError(
            f"pick_by_one_hot: table width {len(table)} != n_slots {n_slots} "
            f"* d_fill {d_fill}"
        )
    mask_pm1 = indicator_to_bool(mask)
    zero = create_literal_value(torch.zeros(d_fill, dtype=torch.float32))
    # This helper is built eagerly at every position (the renderer builds all
    # branch candidates and masks by token type), so a recovered one-hot mask
    # can be fractional at the *discarded* rows. That is the swiglu
    # broadcast_select's documented contract — no ±1 assert; a fractional mask
    # blends within the hull of zero and the branch ranges — and the
    # literal-zero false branch satisfies its one condition (zero lies in the
    # branch-range union) by construction. The winning row's clean one-hot
    # picks its value within ~1 ulp relative (gate×value directly; no additive
    # offset, so nothing scales with the branches' range).
    selected = _broadcast_select(mask_pm1, table, zero, n_slots, d_fill)
    # Sum slot-major selected[s*d_fill + c] over s, into channel c.
    weights = torch.zeros((n_slots * d_fill, d_fill), dtype=torch.float32)
    for s in range(n_slots):
        for c in range(d_fill):
            weights[s * d_fill + c, c] = 1.0
    return Linear(selected, weights, name="pick_by_one_hot_sum")


def pick_by_index(index: Node, table: Node, n_slots: int, d_fill: int = 1) -> Node:
    """Select slot value(s) from a slot-major runtime table by a scalar index.

    ``index`` is a width-1 scalar carrying an integer in ``[0, n_slots)``;
    ``table`` is width ``n_slots * d_fill`` (slot-major). Lowers directly to
    torchwright ``dynamic_extract``, which is itself
    ``in_range`` -> ``broadcast_select`` -> fixed ``Linear`` sum. Result width
    is ``d_fill``.

    For a compile-time-constant table prefer :func:`pick_const_by_index`,
    which bakes the table into a selection Linear instead of gating a live
    table copy on the residual stream.
    """
    return _dynamic_extract(table, index, n_slots, d_fill)


def pick_const_by_index(
    index: Node, values: list[float], n_slots: int, d_fill: int = 1
) -> Node:
    """Slot value(s) from a compile-time-CONSTANT slot-major table by index.

    Same selection semantics as ``pick_by_index(index, constant(values), ...)``
    — integer ``index`` in ``[0, n_slots)`` picks slot ``index``, out-of-range
    picks zeros — but the table lives in weights: lowers to ``one_hot`` ->
    ``onehot_lookup``, whose single-block case is one selection ``Linear``
    (zero hidden lanes, and one FFN sublayer shallower than the gated pick).
    The ``one_hot`` blend-zone behavior at half-integer inputs is the same
    trapezoidal blend of adjacent slots as the gated form.
    """
    table = torch.tensor([float(v) for v in values], dtype=torch.float32)
    if len(table) != n_slots * d_fill:
        raise ValueError(
            f"pick_const_by_index: table length {len(table)} != n_slots "
            f"{n_slots} * d_fill {d_fill}"
        )
    rows = table.reshape(n_slots, d_fill)
    key_to_value = {}
    for slot in range(n_slots):
        key = torch.zeros(n_slots, dtype=torch.float32)
        key[slot] = 1.0
        key_to_value[key] = rows[slot]
    return onehot_lookup(
        one_hot(index, n_slots),
        key_to_value,
        torch.zeros(d_fill, dtype=torch.float32),
    )


def pwl_def(
    fn, breakpoints: int, input_range: tuple[float, float], *, name: str = "pwl"
) -> Callable[[Node], Node]:
    """Build a reusable 1D piecewise-linear function.

    Returns a callable that applies the PWL to a scalar node, mirroring the
    original ``PWLDef`` pattern: construct once at module level, apply many times
    inside the graph builders. ``breakpoints`` is the grid resolution (an int);
    the grid spans ``input_range`` uniformly. ``fn`` is sampled at each
    breakpoint and the result linearly interpolates between them, lowering to
    torchwright ``piecewise_linear``.

    The returned callable — not this factory — is what builds a graph node, so
    the tuple-of-PWLs module-level pattern (``_U_MOD_BY_BANK``,
    ``_COLORMAP_ROW_PWLS``) stays node-free at import (no ``constant``/op nodes;
    ``global_node_id`` unchanged), satisfying the no-import-time-nodes rule.
    """
    lo, hi = float(input_range[0]), float(input_range[1])
    if breakpoints < 2:
        raise ValueError(f"pwl_def breakpoints must be >= 2, got {breakpoints}")
    grid = [lo + (hi - lo) * i / (breakpoints - 1) for i in range(breakpoints)]

    def apply(node: Node) -> Node:
        return _piecewise_linear(node, grid, fn, name=name)

    return apply


def table_lookup_2d(
    i: Node, j: Node, table, *, index_scale: float = 1.0, sharpness: float = 100.0
) -> Node:
    """Compile-time constant 2D table lookup by scaled integer indices.

    Thin re-export of torchwright core ``table_lookup_2d`` so a ported renderer
    file reaches it through the same ``std`` surface as
    its other ops. ``table`` is plain numpy/array weight data (not a node), the
    real-graph weight analog — torchwright's builder accepts the raw array.
    Inputs near integer ``k`` select index ``k``; out-of-range indices clamp to
    the table edge (cancellation-free after the I0 fix). The ``eps = 1 /
    sharpness`` transition bands are centered at half-integer boundaries.
    """
    return _table_lookup_2d(i, j, table, index_scale=index_scale, sharpness=sharpness)


def make_token(token_type: TokenType, **slot_value_nodes: Node) -> Node:
    """Build a next-token residual row (lowers to ``emit_token``).

    The renderer builds every branch's next-token eagerly at every position and
    masks by token type in ``dispatch``, so a branch's slot inputs are only
    valid at the rows that branch actually fires on; elsewhere a computed slot
    (e.g. ``child_u - N_NODES_MAX`` for a node child, or ``last_node + 1``) goes
    out of the slot's range. The original ``make_token`` tolerates that via the
    clamping one-hot encoder; the real ``emit_token``'s digit-quad payload does
    not, and an out-of-range value blows up the row (and the downstream
    ``select`` / ``type_switch`` value-range guards). Clamping each slot value
    to its declared range here restores that tolerance — it is a no-op at the
    rows the branch is selected on, and bounds the discarded garbage rows.
    """
    from .emit import emit_token

    return emit_token(token_type, **_clamp_slot_values(token_type, slot_value_nodes))


def make_token_head(token_type: TokenType, **slot_value_nodes: Node) -> Node:
    """Like :func:`make_token`, but returns only the emit *head* (no derived
    tail). The renderer's dispatch folds over heads and stamps one shared
    ``emit_derived_zero`` after selecting the winning branch — building each
    candidate at head width instead of full ``d_embed`` is what keeps the
    dispatch's live residual width small enough to compile. Slot-value clamping
    is identical to :func:`make_token`.
    """
    from .emit import emit_token_head

    return emit_token_head(
        token_type, **_clamp_slot_values(token_type, slot_value_nodes)
    )


def clamp(node: Node, lo: float, hi: float) -> Node:
    """Clamp a 1-wide scalar to ``[lo, hi]`` in one MLP sublayer.

    A thin re-export of :func:`torchwright.ops.swiglu.arithmetic_ops.clamp`, so
    a ported renderer file reaches its clamp through the same
    ``std`` surface as its other ops instead of importing ``torchwright.ops``
    directly. Used by the dispatch's world-angle collapse to pin each candidate
    ``(dx, dy)`` to the atan square before the pick."""
    return _clamp(node, lo, hi)


def clamp_to_slot(token_type: TokenType, slot_name: str, value: Node) -> Node:
    """Clamp ``value`` to ``token_type.slots[slot_name]``'s declared range — the
    same clamp :func:`make_token_head` applies internally.

    The dispatch's numeric-carrier collapse uses this to bound each candidate
    scalar *before* the pick (``pick_by_one_hot``). Historically this kept the
    relu-era pick's additive offset (derived from the union of candidate value
    ranges) inside its sanity bound; the swiglu pick multiplies gate×value
    directly and has no such offset, so the clamp's stated reason is gone.
    Retained at the cutover as a cheap idempotent bound on the value-range
    bookkeeping — removal is the cutover plan's D4 audit, not flip work. It is
    a no-op at the winning row (the per-branch head clamped there too, and the
    clamp is idempotent under the shared head's re-clamp)."""
    return _clamp_slot_values(token_type, {slot_name: value})[slot_name]


def _clamp_slot_values(
    token_type: TokenType, slot_value_nodes: dict[str, Node]
) -> dict[str, Node]:
    """Clamp each slot value to its declared range (see :func:`make_token`)."""
    clamped: dict[str, Node] = {}
    for name, value_node in slot_value_nodes.items():
        slot = token_type.slots[name]
        if isinstance(slot, IntSlot):
            lo, hi = float(slot.lo), float(slot.hi) - 1.0  # IntSlot is [lo, hi)
        else:
            assert isinstance(slot, FloatSlot)
            lo, hi = float(slot.lo), float(slot.hi)
        clamped[name] = _clamp(value_node, lo, hi)
    return clamped
