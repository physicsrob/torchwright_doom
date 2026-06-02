"""SEG-keyed wall-range facts (Plan F / F4 + F5).

Ported from ``doom_sandbox/implementation/forward/wall_range_state.py``. The
reduced Phase-F build needs only :class:`RecentDrawsegState` — the cached
drawseg values (seg id, range start/stop, scale1) that the seg-scan loop and
the drawseg-scalar chain reuse within one ``R_STORE_WALL_RANGE`` cycle.

``SegLevelFacts`` (the SEG-keyed lifted lookups DOOM's ``R_StoreWallRange``
derives for later wall-column access — texturemid, rw_distance, texture ids) is
**Phase H** (``seg_projection`` Phase 11) and is deferred; it is added when the
wall-column rasterizer lands.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .attention_handles import RecentMarkerHandle
from .past import PastHandle, PastHandleScope
from .render_ops import SCREEN_X_CLAMP
from .std import split


@dataclass(frozen=True)
class RecentDrawsegState:
    """Recent drawseg values reused by eager branch builders."""

    store_i: Node
    store_key: Node
    store_x1: Node
    stop_x: Node
    scale1: Node

    @classmethod
    def read_from_recent_rows(
        cls,
        past: PastHandleScope,
        *,
        store_range_row: RecentMarkerHandle,
        emit_x2_row: RecentMarkerHandle,
        drawseg_scale1_row: RecentMarkerHandle,
        input_i_state_or_zero: PastHandle,
        store_x1: Node,
        input_x_or_zero: PastHandle,
        input_drawseg_scale_or_zero: PastHandle,
    ) -> "RecentDrawsegState":
        store_i, store_key = split(
            store_range_row.pick(past, input_i_state_or_zero),
            [1, 3],
        )
        return cls(
            store_i=store_i,
            store_key=store_key,
            store_x1=store_x1,
            stop_x=SCREEN_X_CLAMP(emit_x2_row.pick(past, input_x_or_zero)),
            scale1=drawseg_scale1_row.pick(
                past,
                input_drawseg_scale_or_zero,
            ),
        )
