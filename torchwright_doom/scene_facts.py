"""Scene facts exposed to render code.

This module turns token/header interpretation into queryable scene data. It
does not parse raw token embeddings itself; `SceneTokenView` names the current
token pieces, and `HeaderContext` supplies the current NODE/SS/SEG context.

The five public groups mirror the data render code needs:

- `PlayerView`: unkeyed player pose from marker/value payloads.
- `NodeIndex`: node geometry and children keyed by node id.
- `SubsectorIndex`: first owned seg keyed by subsector id.
- `SegIndex`: seg ownership, presence, and endpoints keyed by seg id.
- `PlaneIndex`: visplane height, flat, and light startmap keyed by plane id.

Channel contract used below:

- `*_key` is an exact-match key only on producer rows and zero elsewhere.
  Both the mandatory value tables and the presence tables key on lifted
  scalar-id equality (the id is encoded as ``[id, -id^2, 1]`` so one
  attention dot-product peaks at exact id equality; see ``GLOSSARY.md``).
  A lifted key has no "no-match" state -- a query always returns the nearest
  matched id -- so absence is detected by recovering that nearest id and
  comparing it to the query, NOT by a one-hot probe scoring zero (see
  ``LiftedKeyPresenceHandle``). The one exception is ``PlaneIndex``, which
  keys on a width-N one-hot over plane id because plane ids are dense and
  small.
- `*_value` is published on every row; it is meaningful only where the matching
  key/validity marker says the row is a producer.
- `*_marker` is the same-row presence bit used by optional lookups to
  distinguish a real match from a no-match fallback.

Changes from the original: the import block (``Vec`` -> ``Node``; ``Past`` ->
``GraphPast``; ``one_hot`` from the real-side shim, token declarations from
``vocab``, constants/ops from the real-side render shim). The five index
dataclasses, their ``publish`` classmethods, the 13 module-level lookup
helpers, and ``SegIndex.is_portal`` are a line-for-line port -- except that
the node and seg VALUE-backed lookups, which the original kept as separate
helpers, are merged here into the single ``_keyed_value_lookup`` (it differs
only in which header context supplies the key).
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .attention_handles import (
    KeyValueHandle,
    KeyValueLookup,
    LiftedKeyPresenceHandle,
    LiftedKeyPresenceLookup,
    LiftedKeyValueHandle,
    LiftedKeyValueLookup,
    ValidValueHandle,
)
from .past import GraphPast
from .render_constants import MATCH_GAIN_LONG
from .render_ops import and_, one_minus
from .scene_headers import HeaderContext
from .scene_tokens import SceneTokenView
from .std import constant, one_hot
from .tokens import TokenType
from .value_ranges import ValueRange
from .vocab import (
    BBOX_BOT_BACK,
    BBOX_BOT_FRONT,
    BBOX_LEFT_BACK,
    BBOX_LEFT_FRONT,
    BBOX_RIGHT_BACK,
    BBOX_RIGHT_FRONT,
    BBOX_TOP_BACK,
    BBOX_TOP_FRONT,
    N_PLANES_MAX,
    NODE_DX,
    NODE_DY,
    NODE_PX,
    NODE_PY,
    PLANE_HEIGHT,
    PLAYER_ANGLE_MARK,
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    PLAYER_Z_MARK,
    SEG_AX,
    SEG_AY,
    SEG_BACK_CEILING,
    SEG_BACK_FLOOR,
    SEG_BX,
    SEG_BY,
    SEG_FRONT_CEILING,
    SEG_FRONT_FLOOR,
    SEG_NORMAL_ANGLE,
    SEG_ROWOFFSET,
)


@dataclass(frozen=True)
class PlayerView:
    """Player pose vectors consumed by render code as `scene.view`."""

    x: Node
    y: Node
    z: Node
    angle: Node
    angle_sin: Node
    angle_cos: Node
    ray_x_by_screen: Node
    ray_y_by_screen: Node

    @classmethod
    def publish(cls, past: GraphPast, token: SceneTokenView) -> "PlayerView":
        """Publish player marker/value channels and return the recovered pose."""
        return cls(
            x=_mean_player_value(
                past,
                token,
                "player_x",
                PLAYER_X_MARK,
                ValueRange.R1,
            ),
            y=_mean_player_value(
                past,
                token,
                "player_y",
                PLAYER_Y_MARK,
                ValueRange.R1,
            ),
            z=_mean_player_value(
                past,
                token,
                "player_z",
                PLAYER_Z_MARK,
                ValueRange.R3,
            ),
            angle=_mean_player_angle(
                past,
                token,
                "player_angle",
                PLAYER_ANGLE_MARK,
            ),
            angle_sin=_mean_player_angle_derived(
                past,
                token,
                "player_angle_sin",
                PLAYER_ANGLE_MARK,
                token.angle_sin,
            ),
            angle_cos=_mean_player_angle_derived(
                past,
                token,
                "player_angle_cos",
                PLAYER_ANGLE_MARK,
                token.angle_cos,
            ),
            ray_x_by_screen=_mean_player_angle_derived(
                past,
                token,
                "player_angle_ray_x",
                PLAYER_ANGLE_MARK,
                token.angle_ray_x_by_screen,
            ),
            ray_y_by_screen=_mean_player_angle_derived(
                past,
                token,
                "player_angle_ray_y",
                PLAYER_ANGLE_MARK,
                token.angle_ray_y_by_screen,
            ),
        )


def _mean_player_value(
    past: GraphPast,
    token: SceneTokenView,
    name: str,
    marker_type: TokenType,
    range_id: ValueRange,
) -> Node:
    """Publish and recover one VALUE-backed player field."""
    return ValidValueHandle.publish(
        past,
        name,
        token.value_after(marker_type),
        token.payload_value(range_id),
    ).mean(past)


def _mean_player_angle(
    past: GraphPast,
    token: SceneTokenView,
    name: str,
    marker_type: TokenType,
) -> Node:
    """Publish and recover one ANGLE_VALUE-backed player field."""
    return ValidValueHandle.publish(
        past,
        name,
        token.angle_after(marker_type),
        token.angle,
    ).mean(past)


def _mean_player_angle_derived(
    past: GraphPast,
    token: SceneTokenView,
    name: str,
    marker_type: TokenType,
    derived_value: Node,
) -> Node:
    return ValidValueHandle.publish(
        past,
        name,
        token.angle_after(marker_type),
        derived_value,
    ).mean(past)


@dataclass(frozen=True)
class NodeIndex:
    """BSP node facts consumed as `scene.nodes`.

    `root` is the root node id. The remaining fields are callable lookups keyed
    by a node id vector, for example `scene.nodes.px(node)`.

    DOOM: node_t (r_defs.h) — BSP partition line (px, py, dx, dy) plus front/back
    child references and bounding boxes, walked by R_RenderBSPNode (r_bsp.c).
    """

    has_any: Node
    exists: LiftedKeyPresenceLookup
    root: Node
    px: LiftedKeyValueLookup
    py: LiftedKeyValueLookup
    dx: LiftedKeyValueLookup
    dy: LiftedKeyValueLookup
    front_child: LiftedKeyValueLookup
    back_child: LiftedKeyValueLookup
    front_child_lifted: LiftedKeyValueLookup
    back_child_lifted: LiftedKeyValueLookup
    bbox_top_front: LiftedKeyValueLookup
    bbox_bottom_front: LiftedKeyValueLookup
    bbox_left_front: LiftedKeyValueLookup
    bbox_right_front: LiftedKeyValueLookup
    bbox_top_back: LiftedKeyValueLookup
    bbox_bottom_back: LiftedKeyValueLookup
    bbox_left_back: LiftedKeyValueLookup
    bbox_right_back: LiftedKeyValueLookup

    @classmethod
    def publish(
        cls,
        past: GraphPast,
        token: SceneTokenView,
        node_context: HeaderContext,
    ) -> "NodeIndex":
        """Publish NODE-backed channels and return callable node lookups."""
        return cls(
            has_any=_node_has_any(past, token, node_context),
            exists=_node_presence_lookup(past, token),
            root=_node_root(past, node_context),
            px=_keyed_value_lookup(
                past,
                token,
                node_context,
                "node_px",
                NODE_PX,
                ValueRange.R1,
            ),
            py=_keyed_value_lookup(
                past,
                token,
                node_context,
                "node_py",
                NODE_PY,
                ValueRange.R1,
            ),
            dx=_keyed_value_lookup(
                past,
                token,
                node_context,
                "node_dx",
                NODE_DX,
                ValueRange.R2,
            ),
            dy=_keyed_value_lookup(
                past,
                token,
                node_context,
                "node_dy",
                NODE_DY,
                ValueRange.R2,
            ),
            front_child=_node_child_lookup(
                past,
                node_context,
                "node_front_child",
                token.is_front_child,
                token.front_child_u,
            ),
            back_child=_node_child_lookup(
                past,
                node_context,
                "node_back_child",
                token.is_back_child,
                token.back_child_u,
            ),
            front_child_lifted=_node_child_lookup(
                past,
                node_context,
                "node_front_child_lifted",
                token.is_front_child,
                token.child_lifted_key,
            ),
            back_child_lifted=_node_child_lookup(
                past,
                node_context,
                "node_back_child_lifted",
                token.is_back_child,
                token.child_lifted_key,
            ),
            bbox_top_front=_keyed_value_lookup(
                past,
                token,
                node_context,
                "bbox_top_front",
                BBOX_TOP_FRONT,
                ValueRange.R0,
            ),
            bbox_bottom_front=_keyed_value_lookup(
                past,
                token,
                node_context,
                "bbox_bottom_front",
                BBOX_BOT_FRONT,
                ValueRange.R0,
            ),
            bbox_left_front=_keyed_value_lookup(
                past,
                token,
                node_context,
                "bbox_left_front",
                BBOX_LEFT_FRONT,
                ValueRange.R0,
            ),
            bbox_right_front=_keyed_value_lookup(
                past,
                token,
                node_context,
                "bbox_right_front",
                BBOX_RIGHT_FRONT,
                ValueRange.R0,
            ),
            bbox_top_back=_keyed_value_lookup(
                past,
                token,
                node_context,
                "bbox_top_back",
                BBOX_TOP_BACK,
                ValueRange.R0,
            ),
            bbox_bottom_back=_keyed_value_lookup(
                past,
                token,
                node_context,
                "bbox_bottom_back",
                BBOX_BOT_BACK,
                ValueRange.R0,
            ),
            bbox_left_back=_keyed_value_lookup(
                past,
                token,
                node_context,
                "bbox_left_back",
                BBOX_LEFT_BACK,
                ValueRange.R0,
            ),
            bbox_right_back=_keyed_value_lookup(
                past,
                token,
                node_context,
                "bbox_right_back",
                BBOX_RIGHT_BACK,
                ValueRange.R0,
            ),
        )


def _node_root(past: GraphPast, node_context: HeaderContext) -> Node:
    """The BSP root is the last NODE header in the prefill."""
    return past.pick_most_recent(
        constant(1.0),
        node_context.header_active,
        node_context.header_id,
        match_gain=MATCH_GAIN_LONG,
    )


def _node_has_any(
    past: GraphPast,
    token: SceneTokenView,
    node_context: HeaderContext,
) -> Node:
    """Return 1 iff the prefill contains at least one NODE header."""
    node_value = past.publish("node_any_value", token.is_node)
    return past.pick_argmax(constant(1.0), node_context.header_active, node_value)


def _node_presence_lookup(
    past: GraphPast, token: SceneTokenView
) -> LiftedKeyPresenceLookup:
    """Publish NODE presence keyed by node id (lifted-equality, width-3 key).

    Lifted form: recover the matched node id and test equality to the query
    (see ``LiftedKeyPresenceHandle``). A sentinel/early-exit query id past the
    last real node has no exact producer, so its nearest-neighbour recovery
    reads absent.
    """
    handle = LiftedKeyPresenceHandle.publish(
        past,
        "node_exists",
        token.is_node,
        token.id_lifted_key,
        token.node_j,
    )
    return LiftedKeyPresenceLookup(past, handle)


def _keyed_value_lookup(
    past: GraphPast,
    token: SceneTokenView,
    context: HeaderContext,
    name: str,
    marker_type: TokenType,
    range_id: ValueRange,
) -> LiftedKeyValueLookup:
    """Publish one VALUE-backed scene lookup keyed by the current entity
    (node or seg) header context — the NODE and SEG sides are the same
    publish, differing only in which header context supplies the key."""
    handle = LiftedKeyValueHandle.publish(
        past,
        name,
        token.value_after(marker_type),
        context.current_key,
        token.payload_value(range_id),
    )
    return LiftedKeyValueLookup(past, handle)


def _node_child_lookup(
    past: GraphPast,
    node_context: HeaderContext,
    name: str,
    is_child: Node,
    child_u: Node,
) -> LiftedKeyValueLookup:
    """Publish one direct child lookup keyed by current node."""
    handle = LiftedKeyValueHandle.publish(
        past,
        name,
        is_child,
        node_context.current_key,
        child_u,
    )
    return LiftedKeyValueLookup(past, handle)


@dataclass(frozen=True)
class SubsectorIndex:
    """Subsector facts consumed as `scene.subsectors`.

    The implementation only needs the first seg for each subsector.
    """

    first_seg: LiftedKeyValueLookup
    present: LiftedKeyPresenceLookup

    @classmethod
    def publish(
        cls,
        past: GraphPast,
        token: SceneTokenView,
        subsector_context: HeaderContext,
    ) -> "SubsectorIndex":
        """Publish SS-backed channels and return subsector lookups.

        Keyed by the lifted subsector id (``current_key`` = ``[ss,-ss^2,1]``,
        already carried by the header context).
        The first-seg VALUE rides a lifted value lookup (queried only for present
        subsectors, so the bare lifted query is fine); presence rides a lifted
        presence lookup that recovers the matched subsector id and compares it to
        the query (see ``LiftedKeyPresenceHandle``).
        """
        first_seg_mask = and_(
            token.is_seg,
            token.is_first_seg_of_subsector,
        )
        value_handle = LiftedKeyValueHandle.publish(
            past,
            "first_seg",
            first_seg_mask,
            subsector_context.current_key,
            token.seg_i,
        )
        present_handle = LiftedKeyPresenceHandle.publish(
            past,
            "first_seg_present",
            first_seg_mask,
            subsector_context.current_key,
            subsector_context.current_id,
        )
        return cls(
            first_seg=LiftedKeyValueLookup(past, value_handle),
            present=LiftedKeyPresenceLookup(past, present_handle),
        )

    def has_first_seg(self, ss: Node) -> Node:
        """Return whether subsector `ss` has at least one SEG."""
        return self.present(ss)


@dataclass(frozen=True)
class SegIndex:
    """SEG facts consumed as `scene.segs`.

    Fields are callable lookups keyed by seg id. `endpoint_a()` and
    `endpoint_b()` group the coordinate lookups used by render code.

    DOOM: seg_t (r_defs.h) — pegging flags dontpegtop/dontpegbottom come from
    line_t.flags ML_DONTPEGTOP/ML_DONTPEGBOTTOM and control vertical texture
    alignment on portal walls (r_segs.c).
    """

    subsector: LiftedKeyValueLookup
    exists: LiftedKeyPresenceLookup
    ax: LiftedKeyValueLookup
    ay: LiftedKeyValueLookup
    bx: LiftedKeyValueLookup
    by: LiftedKeyValueLookup
    two_sided: LiftedKeyValueLookup
    normal_angle: LiftedKeyValueLookup
    normal_sin: LiftedKeyValueLookup
    normal_cos: LiftedKeyValueLookup
    front_floor: LiftedKeyValueLookup
    front_ceiling: LiftedKeyValueLookup
    back_floor: LiftedKeyValueLookup
    back_ceiling: LiftedKeyValueLookup
    upper_texture: LiftedKeyValueLookup
    lower_texture: LiftedKeyValueLookup
    mid_texture: LiftedKeyValueLookup
    mid_tex_id: LiftedKeyValueLookup
    upper_tex_id: LiftedKeyValueLookup
    lower_tex_id: LiftedKeyValueLookup
    dontpegtop: LiftedKeyValueLookup
    dontpegbottom: LiftedKeyValueLookup
    rowoffset: LiftedKeyValueLookup
    light_static: LiftedKeyValueLookup
    empty_line: LiftedKeyValueLookup
    closed_door: LiftedKeyValueLookup

    def is_portal(self, seg_i: Node) -> Node:
        """A see-through portal: a two-sided line that is not a closed door."""
        return and_(self.two_sided(seg_i), one_minus(self.closed_door(seg_i)))

    @classmethod
    def publish(
        cls,
        past: GraphPast,
        token: SceneTokenView,
        subsector_context: HeaderContext,
        seg_context: HeaderContext,
    ) -> "SegIndex":
        """Publish SEG-backed channels and return callable seg lookups."""
        return cls(
            subsector=_seg_subsector_lookup(past, token, subsector_context),
            exists=_seg_presence_lookup(past, token),
            ax=_keyed_value_lookup(
                past,
                token,
                seg_context,
                "seg_ax",
                SEG_AX,
                ValueRange.R1,
            ),
            ay=_keyed_value_lookup(
                past,
                token,
                seg_context,
                "seg_ay",
                SEG_AY,
                ValueRange.R1,
            ),
            bx=_keyed_value_lookup(
                past,
                token,
                seg_context,
                "seg_bx",
                SEG_BX,
                ValueRange.R1,
            ),
            by=_keyed_value_lookup(
                past,
                token,
                seg_context,
                "seg_by",
                SEG_BY,
                ValueRange.R1,
            ),
            two_sided=_seg_two_sided_lookup(past, token, seg_context),
            # DOOM: rw_normalangle (r_segs.c) — seg angle from its linedef partition (seg_t.linedef, r_defs.h).
            normal_angle=_seg_angle_lookup(
                past,
                token,
                seg_context,
                "seg_normal_angle",
                token.angle,
            ),
            normal_sin=_seg_angle_lookup(
                past,
                token,
                seg_context,
                "seg_normal_sin",
                token.angle_sin,
            ),
            normal_cos=_seg_angle_lookup(
                past,
                token,
                seg_context,
                "seg_normal_cos",
                token.angle_cos,
            ),
            front_floor=_keyed_value_lookup(
                past,
                token,
                seg_context,
                "seg_front_floor",
                SEG_FRONT_FLOOR,
                ValueRange.R3,
            ),
            front_ceiling=_keyed_value_lookup(
                past,
                token,
                seg_context,
                "seg_front_ceiling",
                SEG_FRONT_CEILING,
                ValueRange.R3,
            ),
            back_floor=_keyed_value_lookup(
                past,
                token,
                seg_context,
                "seg_back_floor",
                SEG_BACK_FLOOR,
                ValueRange.R4,
            ),
            back_ceiling=_keyed_value_lookup(
                past,
                token,
                seg_context,
                "seg_back_ceiling",
                SEG_BACK_CEILING,
                ValueRange.R4,
            ),
            upper_texture=_seg_flag_lookup(
                past,
                "seg_upper_texture",
                token.is_seg_upper_texture,
                seg_context,
                token.seg_upper_texture_present,
            ),
            lower_texture=_seg_flag_lookup(
                past,
                "seg_lower_texture",
                token.is_seg_lower_texture,
                seg_context,
                token.seg_lower_texture_present,
            ),
            mid_texture=_seg_flag_lookup(
                past,
                "seg_mid_texture",
                token.is_seg_mid_texture,
                seg_context,
                token.seg_mid_texture_present,
            ),
            mid_tex_id=_seg_flag_lookup(
                past,
                "seg_mid_tex_id",
                token.is_seg_mid_texture,
                seg_context,
                token.seg_mid_tex_id,
            ),
            upper_tex_id=_seg_flag_lookup(
                past,
                "seg_upper_tex_id",
                token.is_seg_upper_texture,
                seg_context,
                token.seg_upper_tex_id,
            ),
            lower_tex_id=_seg_flag_lookup(
                past,
                "seg_lower_tex_id",
                token.is_seg_lower_texture,
                seg_context,
                token.seg_lower_tex_id,
            ),
            # DOOM: ML_DONTPEGTOP from line_t.flags (r_segs.c) — aligns upper wall texture to ceiling vs backsector ceiling.
            dontpegtop=_seg_flag_lookup(
                past,
                "seg_dontpegtop",
                token.is_seg_pegging,
                seg_context,
                token.seg_dontpegtop_flag,
            ),
            # DOOM: ML_DONTPEGBOTTOM from line_t.flags (r_segs.c) — aligns lower/mid wall texture to floor vs backsector floor.
            dontpegbottom=_seg_flag_lookup(
                past,
                "seg_dontpegbottom",
                token.is_seg_pegging,
                seg_context,
                token.seg_dontpegbottom_flag,
            ),
            # DOOM: side_t.rowoffset (r_defs.h) — vertical offset added to all wall textures on this side.
            rowoffset=_keyed_value_lookup(
                past,
                token,
                seg_context,
                "seg_rowoffset",
                SEG_ROWOFFSET,
                ValueRange.R3,
            ),
            light_static=_seg_flag_lookup(
                past,
                "seg_light_static",
                token.is_seg_light_static,
                seg_context,
                token.seg_light_static,
            ),
            empty_line=_seg_flag_lookup(
                past,
                "seg_empty_line",
                token.is_seg_empty_line,
                seg_context,
                token.seg_empty_line_flag,
            ),
            closed_door=_seg_flag_lookup(
                past,
                "seg_closed_door",
                token.is_seg_closed_door,
                seg_context,
                token.seg_closed_door_flag,
            ),
        )

    def endpoint_a(self, seg_i: Node) -> tuple[Node, Node]:
        """Return the A endpoint coordinates for `seg_i`."""
        return self.ax(seg_i), self.ay(seg_i)

    def endpoint_b(self, seg_i: Node) -> tuple[Node, Node]:
        """Return the B endpoint coordinates for `seg_i`."""
        return self.bx(seg_i), self.by(seg_i)


def _seg_subsector_lookup(
    past: GraphPast,
    token: SceneTokenView,
    subsector_context: HeaderContext,
) -> LiftedKeyValueLookup:
    """Publish subsector id keyed by seg id."""
    handle = LiftedKeyValueHandle.publish(
        past,
        "seg_ss",
        token.is_seg,
        token.id_lifted_key,
        subsector_context.current_id,
    )
    return LiftedKeyValueLookup(past, handle)


def _seg_presence_lookup(
    past: GraphPast, token: SceneTokenView
) -> LiftedKeyPresenceLookup:
    """Publish SEG presence keyed by seg id (lifted-equality, width-3 key).

    Lifted form: recover the matched seg id and test equality to the query (see
    ``LiftedKeyPresenceHandle``). A query for an absent seg recovers its nearest
    neighbour and reads absent.
    """
    handle = LiftedKeyPresenceHandle.publish(
        past,
        "seg_exists",
        token.is_seg,
        token.id_lifted_key,
        token.seg_i,
    )
    return LiftedKeyPresenceLookup(past, handle)


def _seg_angle_lookup(
    past: GraphPast,
    token: SceneTokenView,
    seg_context: HeaderContext,
    name: str,
    value: Node,
) -> LiftedKeyValueLookup:
    """Publish one ANGLE_VALUE-backed SEG lookup keyed by current seg.

    DOOM: rw_normalangle (r_segs.c) — the seg's angle derives from its linedef
    partition dx/dy (seg_t.linedef in r_defs.h).
    """
    handle = LiftedKeyValueHandle.publish(
        past,
        name,
        token.angle_after(SEG_NORMAL_ANGLE),
        seg_context.current_key,
        value,
    )
    return LiftedKeyValueLookup(past, handle)


def _seg_two_sided_lookup(
    past: GraphPast,
    token: SceneTokenView,
    seg_context: HeaderContext,
) -> LiftedKeyValueLookup:
    """Publish the SEG_TWO_SIDED flag keyed by seg id."""
    return _seg_flag_lookup(
        past,
        "seg_two_sided",
        token.is_seg_two_sided,
        seg_context,
        token.seg_two_sided_flag,
    )


def _seg_flag_lookup(
    past: GraphPast,
    name: str,
    active: Node,
    seg_context: HeaderContext,
    flag: Node,
) -> LiftedKeyValueLookup:
    """Publish one inline integer SEG flag keyed by seg id."""
    handle = LiftedKeyValueHandle.publish(
        past,
        name,
        active,
        seg_context.current_key,
        flag,
    )
    return LiftedKeyValueLookup(past, handle)


@dataclass(frozen=True)
class PlaneIndex:
    """Physical visplane facts keyed by plane id."""

    height: KeyValueLookup
    flat_id: KeyValueLookup
    light_static: KeyValueLookup

    @classmethod
    def publish(
        cls,
        past: GraphPast,
        token: SceneTokenView,
        plane_context: HeaderContext,
    ) -> "PlaneIndex":
        height_handle = KeyValueHandle.publish(
            past,
            "plane_height",
            token.value_after(PLANE_HEIGHT),
            plane_context.current_key,
            token.payload_value(ValueRange.R3),
        )
        flat_id_handle = KeyValueHandle.publish(
            past,
            "plane_flat_id",
            token.is_plane_def,
            one_hot(token.plane_def_p, N_PLANES_MAX),
            token.plane_def_flat_id,
        )
        light_static_handle = KeyValueHandle.publish(
            past,
            "plane_light_static",
            token.is_plane_light,
            plane_context.current_key,
            token.plane_light_static,
        )
        return cls(
            height=KeyValueLookup(past, height_handle, N_PLANES_MAX),
            flat_id=KeyValueLookup(past, flat_id_handle, N_PLANES_MAX),
            light_static=KeyValueLookup(past, light_static_handle, N_PLANES_MAX),
        )
