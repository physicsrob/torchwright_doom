"""Past-published solid wall fragments used as horizontal occlusion state
(Plan F / F3).

Ported from ``doom_sandbox/implementation/forward/solid_intervals.py``. The
seg scan fills this channel at ``R_STORE_WALL_RANGE`` and queries it at
``FIND_RUN`` to decide which screen columns a later seg may still draw —
DOOM's ``solidsegs`` horizontal occlusion, expressed as a queryable union of
prior one-sided drawseg fragments.

Changes from the sandbox source: ``Vec`` -> ``Node``; the sandbox-``api``
imports map to the real ``std`` / ``past`` / ``render_ops`` shims; and the
module-level ``constant(...)`` sentinels (``_SENTINEL_KEY`` etc.) are built
*inside* the methods — a module-level ``constant`` is a graph node whose low
id aliases test-built nodes after the conftest id-counter reset
(``reference_eval`` / ``probe_compiled`` memoisation). The integer index
tables ``_SCALED_START`` / ``_NEXT_START_INDICATORS`` are plain weight data,
so they stay at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .attention_handles import RecentMarkerHandle
from .constants import SCREEN_WIDTH
from .past import PastHandle, PastHandleScope
from .protocol_tokens import ProtocolTokenView
from .render_ops import MUL_SCREEN, add_const, and_, one_minus, snap_bool
from .scene_index import SceneIndex
from .std import concat, constant, linear, one_hot, select, split

# Per-position start-score scale and the threshold indicator basis for
# ``pick_argmin_above``. Plain weight data (no graph nodes), so module scope is
# fine; both are sized by SCREEN_WIDTH, which is fixed at import.
_SCALED_START = [[0.75]]
_NEXT_START_INDICATORS = [
    [1.0 if start > threshold else 0.0 for threshold in range(SCREEN_WIDTH + 1)]
    for start in range(SCREEN_WIDTH + 1)
]


@dataclass(frozen=True)
class SolidIntervals:
    """Queryable union of prior one-sided drawseg fragments."""

    past: PastHandleScope
    key: PastHandle
    interval_state: PastHandle
    start_score: PastHandle
    start_indicators: PastHandle
    start_value: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        scene: SceneIndex,
    ) -> "SolidIntervals":
        """Publish the current input fragment if it is a solid drawseg."""
        # R_STORE_WALL_RANGE rows publish the completed fragment. The store row
        # carries the seg id directly; x1 is the most recent FIND_RUN column
        # and x2 is recovered from the most recent semantic EMIT_X2 row.
        find_run_row = RecentMarkerHandle.publish(
            past,
            "solid_find_run",
            inp.is_find_run,
        )
        emit_x2_row = RecentMarkerHandle.publish(
            past,
            "solid_emit_x2",
            inp.is_emit_x2,
        )
        input_x_or_zero = past.publish("solid_input_x_or_zero", inp.seg_x_or_zero)
        emit_x2_x = past.publish("emit_x2_x_or_zero", inp.emit_x2_x)
        x1 = find_run_row.pick(past, input_x_or_zero)
        x2 = emit_x2_row.pick(past, emit_x2_x)
        is_open_portal = scene.segs.is_portal(inp.store_i)
        solid_emit = and_(inp.is_store_wall_range, one_minus(is_open_portal))

        sentinel_key = constant([0.0, 0.0, 1.0])
        sentinel_start = constant(float(SCREEN_WIDTH))

        interval_key = _interval_key(x1, x2)
        key = past.publish(
            "solid_interval_key",
            select(solid_emit, interval_key, sentinel_key),
        )
        interval_state = past.publish(
            "solid_interval_state",
            concat(solid_emit, select(solid_emit, x2, sentinel_start)),
        )
        start_value_vec = select(solid_emit, x1, sentinel_start)
        start_value = past.publish("solid_interval_start", start_value_vec)
        start_score = past.publish(
            "solid_interval_start_score",
            linear(start_value_vec, _SCALED_START),
        )
        start_indicators = past.publish(
            "solid_interval_start_above",
            linear(one_hot(start_value_vec, SCREEN_WIDTH + 1), _NEXT_START_INDICATORS),
        )
        return cls(
            past=past,
            key=key,
            interval_state=interval_state,
            start_score=start_score,
            start_indicators=start_indicators,
            start_value=start_value,
        )

    # DOOM: solidsegs scan predicate (r_bsp.c:R_ClipSolidWallSegment/R_ClipPassWallSegment)
    def covered_and_end(
        self,
        column: Node,
        column_square: Node,
    ) -> tuple[Node, Node]:
        """Return coverage and covering interval end using one shared query.

        The query ``[col², col, 1]`` dotted with a published key
        ``[-2, 2(a+b), -2ab]`` scores ``-2(col-a)(col-b)`` — positive when
        ``col`` lies inside the padded interval ``[a, b] = [x1-1, x2+1]``. The
        sentinel key ``[0, 0, 1]`` scores a flat 1, so a column outside every
        interval (or only at a padded edge) picks the sentinel and reads "not
        covered".
        """
        query = concat(column_square, column, constant(1.0))
        covered, end = split(
            self.past.pick_argmax(
                query,
                self.key,
                self.interval_state,
            ),
            [1, 1],
        )
        # ``covered`` is the matched key's ±1 ``solid_emit``; the fp32 softmax
        # recovers it as ≈±1 with ~1e-5 noise. Both consumers use it only as a
        # ``select`` cond, so snap it to a clean ±1 — the noise would otherwise
        # leak the discarded branch's head and flip a thin integer-slot argmax
        # (see ``render_ops.snap_bool``). ``end`` stays raw (a screen column).
        return snap_bool(covered), end

    def next_start_after(self, column: Node) -> Node:
        """Return the nearest solid interval start strictly after `column`."""
        return self.past.pick_argmin_above(
            self.start_score,
            self.start_indicators,
            one_hot(column, SCREEN_WIDTH + 1),
            self.start_value,
        )


# Encode solid interval (x1, x2) as query key for argmax matching (DOOM: cliprange_t, r_bsp.c:88-92)
def _interval_key(x1: Node, x2: Node) -> Node:
    a = add_const(x1, -1.0)
    b = add_const(x2, 1.0)
    ab = MUL_SCREEN(a, b)
    minus_two_ab = linear(ab, [[-2.0]])
    sum_ab = linear(concat(a, b), [[2.0], [2.0]])
    return concat(constant(-2.0), sum_ab, minus_two_ab)
