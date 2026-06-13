"""SEG-keyed wall-range facts.

Ported from ``doom_sandbox/implementation/forward/wall_range_state.py``. Defines
two records. :class:`RecentDrawsegState` holds the cached drawseg values (seg
id, range start/stop, scale1) that the seg-scan loop and the drawseg-scalar
chain reuse within one ``R_STORE_WALL_RANGE`` cycle.

:class:`SegLevelFacts` holds the SEG-keyed lifted lookups DOOM's
``R_StoreWallRange`` derives for later wall-column access — texturemid,
rw_distance, texture ids, texture height-index; the wall-column rasterizer
recovers these per-seg by lifted-id lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchwright.graph import Node

from .attention_handles import (
    LiftedKeyValueHandle,
    LiftedKeyValueLookup,
    RecentMarkerHandle,
)
from .past import PastHandle, PastHandleScope
from .render_ops import RW_DISTANCE_CLAMP, SCREEN_X_CLAMP, mul_normal_coord, sub
from .std import split
from .std import sum as vec_sum

if TYPE_CHECKING:
    from .protocol_tokens import ProtocolTokenView
    from .scene_index import SceneIndex


def rw_distance_for(scene: SceneIndex, seg_i: Node) -> Node:
    """DOOM: rw_distance (r_segs.c:R_StoreWallRange) — perpendicular distance
    from the viewpoint to the seg's line: the view-relative endpoint dotted
    with the seg normal, clamped to the R5 range.

    The single definition for both consumers — ``WallRangeBuilder`` (the
    drawseg store path) and :meth:`SegLevelFacts.publish` (the seg-level
    fact row) previously built this identical chain independently.
    """
    ax, ay = scene.segs.endpoint_a(seg_i)
    rel_x = sub(ax, scene.view.x)
    rel_y = sub(ay, scene.view.y)
    distance = vec_sum(
        mul_normal_coord(scene.segs.normal_cos(seg_i), rel_x),
        mul_normal_coord(scene.segs.normal_sin(seg_i), rel_y),
    )
    return RW_DISTANCE_CLAMP(distance)


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


@dataclass(frozen=True)
class SegLevelFacts:
    """SEG-keyed computed facts published at R_STORE_WALL_RANGE positions.

    DOOM: R_StoreWallRange (r_segs.c) — derives per-seg drawseg_t fields
    (texturemid, rw_distance, texture ids, texture height-index) for wall-column
    rendering. Each field is a callable lifted-id lookup the wall-column pass
    recovers by the lifted seg id of the seg being rasterized.
    """

    dc_tmid_mid: LiftedKeyValueLookup
    dc_tmid_upper: LiftedKeyValueLookup
    dc_tmid_lower: LiftedKeyValueLookup
    has_mid: LiftedKeyValueLookup
    has_upper: LiftedKeyValueLookup
    has_lower: LiftedKeyValueLookup
    K_part_0: LiftedKeyValueLookup
    K_part_1: LiftedKeyValueLookup
    K_part_2: LiftedKeyValueLookup
    rw_distance: LiftedKeyValueLookup
    h_idx_oh_mid: LiftedKeyValueLookup
    h_idx_oh_upper: LiftedKeyValueLookup
    h_idx_oh_lower: LiftedKeyValueLookup

    @classmethod
    def publish(
        cls,
        past: PastHandleScope,
        inp: "ProtocolTokenView",
        scene: "SceneIndex",
        seg_key_at_kpart_row: Node,
    ) -> "SegLevelFacts":
        seg_i = inp.store_i
        active = inp.is_store_wall_range
        seg_key = inp.id_lifted_key

        mid_tex_id = scene.segs.mid_tex_id(seg_i)
        upper_tex_id = scene.segs.upper_tex_id(seg_i)
        lower_tex_id = scene.segs.lower_tex_id(seg_i)
        h_idx_oh_mid_val = scene.assets.walls.h_idx_oh(mid_tex_id)
        h_idx_oh_upper_val = scene.assets.walls.h_idx_oh(upper_tex_id)
        h_idx_oh_lower_val = scene.assets.walls.h_idx_oh(lower_tex_id)

        rw_distance_value = rw_distance_for(scene, seg_i)

        active_kpart = inp.is_seg_kpart
        seg_key_kpart = seg_key_at_kpart_row
        active_dc_mid = inp.is_value_after_seg_dc_tmid_mid
        active_dc_upper = inp.is_value_after_seg_dc_tmid_upper
        active_dc_lower = inp.is_value_after_seg_dc_tmid_lower

        return cls(
            dc_tmid_mid=_seg_keyed_lookup(
                past,
                "seg_dc_tmid_mid",
                active_dc_mid,
                seg_key_kpart,
                inp.value_v3,
            ),
            dc_tmid_upper=_seg_keyed_lookup(
                past,
                "seg_dc_tmid_upper",
                active_dc_upper,
                seg_key_kpart,
                inp.value_v4,
            ),
            dc_tmid_lower=_seg_keyed_lookup(
                past,
                "seg_dc_tmid_lower",
                active_dc_lower,
                seg_key_kpart,
                inp.value_v4,
            ),
            has_mid=_seg_keyed_lookup(
                past,
                "seg_has_mid",
                active_kpart,
                seg_key_kpart,
                inp.seg_kpart_has_mid,
            ),
            has_upper=_seg_keyed_lookup(
                past,
                "seg_has_upper",
                active_kpart,
                seg_key_kpart,
                inp.seg_kpart_has_upper,
            ),
            has_lower=_seg_keyed_lookup(
                past,
                "seg_has_lower",
                active_kpart,
                seg_key_kpart,
                inp.seg_kpart_has_lower,
            ),
            K_part_0=_seg_keyed_lookup(
                past,
                "seg_K_part_0",
                active_kpart,
                seg_key_kpart,
                inp.seg_kpart_K_part_0,
            ),
            K_part_1=_seg_keyed_lookup(
                past,
                "seg_K_part_1",
                active_kpart,
                seg_key_kpart,
                inp.seg_kpart_K_part_1,
            ),
            K_part_2=_seg_keyed_lookup(
                past,
                "seg_K_part_2",
                active_kpart,
                seg_key_kpart,
                inp.seg_kpart_K_part_2,
            ),
            rw_distance=_seg_keyed_lookup(
                past,
                "seg_rw_distance",
                active,
                seg_key,
                rw_distance_value,
            ),
            h_idx_oh_mid=_seg_keyed_lookup(
                past,
                "seg_h_idx_oh_mid",
                active,
                seg_key,
                h_idx_oh_mid_val,
            ),
            h_idx_oh_upper=_seg_keyed_lookup(
                past,
                "seg_h_idx_oh_upper",
                active,
                seg_key,
                h_idx_oh_upper_val,
            ),
            h_idx_oh_lower=_seg_keyed_lookup(
                past,
                "seg_h_idx_oh_lower",
                active,
                seg_key,
                h_idx_oh_lower_val,
            ),
        )


def _seg_keyed_lookup(
    past: PastHandleScope,
    name: str,
    active: Node,
    seg_key: Node,
    value: Node,
) -> LiftedKeyValueLookup:
    """Publish a SEG-keyed value channel as a callable lifted-id lookup."""
    handle = LiftedKeyValueHandle.publish(past, name, active, seg_key, value)
    return LiftedKeyValueLookup(past, handle)
