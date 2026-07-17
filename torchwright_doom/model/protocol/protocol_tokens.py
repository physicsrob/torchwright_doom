"""Current-token interpretation for the autoregressive protocol.

This module runs once at compile time: it builds part of the computation graph
that torchwright lowers into the transformer's weights. Nothing here executes
during inference — at render time, only the compiled transformer runs. Coined
terms: see GLOSSARY.md.

`SceneTokenView` interprets prefill tokens as serialized scene facts. This file
interprets the same `input_vec` as the runtime control/render protocol:
traversal states, numeric payload carriers, angle phases, emit tokens, and
inert rows that should produce `NO_OP`.

Important convention: protocol token types name renderer states; carrier token
types only name numeric encodings. `PROCESS_SEG` means "run the R_AddLine
decision for this seg", even though slots and scene state still affect the next
token. `VALUE` and `ANGLE_VALUE` only mean "this row carries a number"; their
immediately preceding marker says whether that number is scene data, projection
state, or some future AR thinking payload. This is why carrier rows are kept out
of the broad "inert prefill" branch below and routed separately.

The one-token dispatch predicates are installed from the registry by
``_install_registry_token_checks()`` — an import-time side effect that mutates
the class. The context-sensitive cached properties below are defined by hand.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from torchwright.graph import Node
from torchwright.graph import annotate, annotated


def _input_cached_property(fn):
    """``cached_property`` whose body runs under ``annotate("input")``.

    ``ProtocolTokenView`` is the current-token typed decode; every node a
    property builds is the ``input`` subsystem (token decode + memory fetch).
    The properties are lazily evaluated at first access from many subsystems, so
    wrapping the body in the ContextVar (rather than relying on the caller's
    label) is what pins the decode nodes to ``input`` deterministically. The
    cache, lazy timing, and node creation are otherwise identical to a bare
    ``cached_property`` — this only sets ``.annotation`` on the nodes built."""

    def wrapper(self):
        with annotate("input"):
            return fn(self)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return cached_property(wrapper)


from .protocol_registry import (
    BBOX_ANGLE_MARKERS as _BBOX_ANGLE_MARKERS,
    INERT_NON_PAYLOAD_TYPES as _INERT_NON_PAYLOAD_TYPES,
    PROJECTION_ANGLE_MARKERS as _PROJECTION_ANGLE_MARKERS,
    SCENE_ANGLE_MARKERS as _SCENE_ANGLE_MARKERS,
    SCENE_VALUE_MARKERS as _SCENE_VALUE_MARKERS,
    TOKEN_CHECK_PREDICATES as _TOKEN_CHECK_PREDICATES,
)
from ..render_ops import and_
from ..std import bool_or, concat, extract_derived, indicator_to_bool, sum as vec_sum
from ..token_match import input_type_matches as _input_type_matches
from ..tokens import TokenType
from ..value_ranges import ValueRange, value_derived
from ..vocab import (
    ADVANCE_SEG,
    BBOX_BOXPOS,
    BBOX_CORNER_X_MARK_A,
    BBOX_CORNER_X_MARK_B,
    BBOX_CORNER_Y_MARK_A,
    BBOX_CORNER_Y_MARK_B,
    BBOX_SCAN,
    BBOX_THETA_MARK_A,
    BBOX_THETA_MARK_B,
    BBOX_WORLD_ANGLE_MARK_A,
    BBOX_WORLD_ANGLE_MARK_B,
    CLIP_UPDATE,
    DRAWSEG_BSILHEIGHT,
    DRAWSEG_META,
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE1_DEN,
    DRAWSEG_SCALE2,
    DRAWSEG_SCALE2_DEN,
    DRAWSEG_SCALESTEP,
    DRAWSEG_SCALESTEP_DEN,
    DRAWSEG_TSILHEIGHT,
    DRAWSEG_U_PHASE,
    EMIT_X2,
    FIND_RUN,
    FLAT_NEXT_PLANE,
    FLAT_NEXT_VP,
    FLAT_VISPLANE_BEGIN,
    HUD_ITEM,
    ID_LIFTED_KEY_DERIVED_NAME,
    MAKE_SPANS_COL,
    PIXEL,
    PLANE_DEF,
    PLANE_MARK,
    PROCESS_SEG,
    R_CHECK_PLANE,
    R_CHECK_PLANE_RESULT,
    R_STORE_WALL_RANGE,
    SCREEN_RANGE,
    SCREEN_X_ONE_HOT_DERIVED_NAMES,
    SCREEN_Y_VALUE,
    SEG_DC_TMID_LOWER,
    SEG_DC_TMID_MID,
    SEG_DC_TMID_UPPER,
    SET_CURSOR_X,
    SET_CURSOR_Y,
    SIDE_RECORD,
    SPAN_CLOSE_SLOT,
    SPAN_ROW,
    SS_CEILING_PLANE,
    SS_FLOOR_PLANE,
    THETA_MARK_A,
    THETA_MARK_B,
    THINK_SIDE,
    TRAVERSE_BETWEEN,
    TRAVERSE_ENTER,
    TRAVERSE_RETURN,
    VALUE,
    VISIT_SUBSECTOR,
    WALL_COL_U,
    WALL_SPAN_META,
    WORLD_ANGLE_MARK_A,
    WORLD_ANGLE_MARK_B,
)


class ProtocolTokenView:
    """Lazy typed view over the current AR protocol token.

    The body holds 100+ cached properties. They fall into a handful of groups;
    the body is banner-commented (``# ---``) in the same order:

    - One-token dispatch predicates ``is_*`` — installed from the registry at
      import time by :func:`_install_registry_token_checks` (declared in the
      ``TYPE_CHECKING`` block above, not written out in the body). Each is
      exactly ``token.check(input_vec)``: a ±1 test of the current row's type.
    - Marker-relative phase predicates ``is_value_after_*`` — true when the
      current row is a ``VALUE`` carrier AND the previous row's type was a
      specific marker, so a shared carrier is disambiguated by what preceded it.
    - Range-bank numeric ``VALUE`` decoders (``value_v*`` / ``value_inv*``) —
      decode the carrier's number through a given value range; call sites give
      them role-specific local names.
    - ``seg_kpart_*`` accessors — the wall texture-part bits and offsets read
      off a single ``SEG_KPART`` row.
    - Payload-group membership tests (``is_inert_non_payload``,
      ``is_scene_value_payload``, ``is_scene_angle_payload``,
      ``is_projection_angle_payload``, ``is_bbox_angle_payload``) — classify a
      carrier row by the marker group its previous token belongs to.
    - Angle after-marker phase predicates ``angle_after_*`` — like the
      ``is_value_after_*`` group but for ``ANGLE_VALUE`` carriers.
    - ``screen_*_after_*`` phase predicates — disambiguate ``SCREEN_Y_VALUE`` /
      ``SCREEN_RANGE`` rows by their one or two preceding markers.
    - Per-token slot accessors — typed reads of a single named slot off the
      current row (``plane_mark_*``, ``r_check_plane_*``, ``flat_*``,
      ``span_*``, ``think_node``, ``cursor_*``, ``pixel_*``, ``screen_*``, …);
      each returns 0 when the current row is not that token's type.
    - bbox sub-protocol tests and accessors (``bbox_scan_*``, ``bbox_boxpos``,
      ``bbox_corner_*_boxpos``, ``boxpos_check_*``, ``boxpos_fails_open``) —
      the bounding-box visibility sub-protocol's reads.
    - The ``*_or_zero`` value accessors — sum a set of mutually-exclusive
      per-token extractors so the result is the one populated value, or zero.

    See ``GLOSSARY.md`` for the coined vocabulary (carrier, marker, …).
    """

    if TYPE_CHECKING:
        # These cached properties are installed from TOKEN_CHECK_PREDICATES at
        # module import time; declare them here so mypy sees the generated API.
        is_value: Node
        is_angle_value: Node
        is_begin: Node
        is_think_side: Node
        is_side_record: Node
        is_enter: Node
        is_between: Node
        is_return: Node
        is_visit_subsector: Node
        is_process_seg: Node
        is_find_run: Node
        is_world_a_mark: Node
        is_theta_a_mark: Node
        is_world_b_mark: Node
        is_theta_b_mark: Node
        is_bbox_boxpos: Node
        is_bbox_corner_x_a_mark: Node
        is_bbox_corner_y_a_mark: Node
        is_bbox_corner_x_b_mark: Node
        is_bbox_corner_y_b_mark: Node
        is_bbox_world_a_mark: Node
        is_bbox_theta_a_mark: Node
        is_bbox_world_b_mark: Node
        is_bbox_theta_b_mark: Node
        is_bbox_scan: Node
        is_advance_seg: Node
        is_emit_x2: Node
        is_store_wall_range: Node
        is_seg_kpart: Node
        is_seg_dc_tmid_mid: Node
        is_seg_dc_tmid_upper: Node
        is_seg_dc_tmid_lower: Node
        is_drawseg_meta: Node
        is_drawseg_scale1_den: Node
        is_drawseg_scale1: Node
        is_drawseg_scale2_den: Node
        is_drawseg_scale2: Node
        is_drawseg_scalestep_den: Node
        is_drawseg_scalestep: Node
        is_drawseg_bsilheight: Node
        is_drawseg_tsilheight: Node
        is_drawseg_u_phase: Node
        is_r_check_plane: Node
        is_r_check_plane_result: Node
        is_wall_column: Node
        is_wall_col_u: Node
        is_wall_span_meta: Node
        is_set_cursor_y: Node
        is_pixel_color: Node
        is_clip_update: Node
        is_plane_mark: Node
        is_draw_planes_begin: Node
        is_flat_next_plane: Node
        is_flat_next_vp: Node
        is_flat_visplane_begin: Node
        is_make_spans_col: Node
        is_span_close_slot: Node
        is_span_row: Node
        is_set_cursor_direction_y: Node
        is_set_cursor_direction_x: Node
        is_screen_y_value: Node
        is_screen_range: Node
        is_done: Node
        is_draw_psprites_begin: Node
        is_hud_begin: Node
        is_hud_item: Node

    def __init__(
        self,
        input_vec: Node,
        prev_input_type: Node,
        prev_prev_input_type: Node,
    ) -> None:
        self.input_vec: Node = input_vec
        self.prev_input_type: Node = prev_input_type
        self.prev_prev_input_type: Node = prev_prev_input_type

    # One-token dispatch predicates are installed from protocol_registry after
    # class creation. Context-sensitive predicates below remain hand-written.

    # --- Marker-relative phase predicates: is this the VALUE row after marker M? ---
    @_input_cached_property
    def is_value_after_seg_dc_tmid_mid(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, SEG_DC_TMID_MID),
        )

    @_input_cached_property
    def is_value_after_seg_dc_tmid_upper(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, SEG_DC_TMID_UPPER),
        )

    @_input_cached_property
    def is_value_after_seg_dc_tmid_lower(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, SEG_DC_TMID_LOWER),
        )

    # --- Range-bank numeric VALUE decoders (value_v* / value_inv*) ---
    # VALUE reads stay indexed by the range-bank id. Call sites give these
    # numeric accessors role-specific local names when the surrounding marker
    # protocol determines the meaning.
    @_input_cached_property
    def value_v0(self) -> Node:
        return value_derived(self.input_vec, ValueRange.R0)

    @_input_cached_property
    def value_v3(self) -> Node:
        return value_derived(self.input_vec, ValueRange.R3)

    @_input_cached_property
    def value_v4(self) -> Node:
        return value_derived(self.input_vec, ValueRange.R4)

    @_input_cached_property
    def value_v5(self) -> Node:
        return value_derived(self.input_vec, ValueRange.R5)

    @_input_cached_property
    def value_inv5(self) -> Node:
        return value_derived(self.input_vec, ValueRange.R5, "inv")

    @_input_cached_property
    def value_wall_scale_diminish5(self) -> Node:
        return value_derived(self.input_vec, ValueRange.R5, "wall_scale_diminish")

    @_input_cached_property
    def value_inv6(self) -> Node:
        return value_derived(self.input_vec, ValueRange.R6, "inv")

    @_input_cached_property
    def value_inv7(self) -> Node:
        return value_derived(self.input_vec, ValueRange.R7, "inv")

    @_input_cached_property
    def value_v8(self) -> Node:
        return value_derived(self.input_vec, ValueRange.R8)

    # --- SEG_KPART accessors: wall texture-part bits and offsets ---
    @_input_cached_property
    def seg_kpart_K_part_0(self) -> Node:
        return extract_derived(self.input_vec, "K_part_0")

    @_input_cached_property
    def seg_kpart_K_part_1(self) -> Node:
        return extract_derived(self.input_vec, "K_part_1")

    @_input_cached_property
    def seg_kpart_K_part_2(self) -> Node:
        return extract_derived(self.input_vec, "K_part_2")

    @_input_cached_property
    def seg_kpart_has_mid(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "has_mid"))

    @_input_cached_property
    def seg_kpart_has_upper(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "has_upper"))

    @_input_cached_property
    def seg_kpart_has_lower(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "has_lower"))

    # --- Current-type tests and same-row slot accessors (visplane setup) ---
    @_input_cached_property
    def is_set_cursor_x(self) -> Node:
        return SET_CURSOR_X.check(self.input_vec)

    @_input_cached_property
    def hud_item_value(self) -> Node:
        """The draw-list item index carried by a HUD_ITEM row (else 0)."""
        return HUD_ITEM.extract(self.input_vec, "item")

    @_input_cached_property
    def wall_col_u_idx(self) -> Node:
        return WALL_COL_U.extract(self.input_vec, "u_idx")

    @_input_cached_property
    def is_plane_def(self) -> Node:
        return PLANE_DEF.check(self.input_vec)

    @_input_cached_property
    def is_ss_floor_plane(self) -> Node:
        return SS_FLOOR_PLANE.check(self.input_vec)

    @_input_cached_property
    def is_ss_ceiling_plane(self) -> Node:
        return SS_CEILING_PLANE.check(self.input_vec)

    @_input_cached_property
    def ss_floor_plane_p(self) -> Node:
        return SS_FLOOR_PLANE.extract(self.input_vec, "p")

    @_input_cached_property
    def ss_ceiling_plane_p(self) -> Node:
        return SS_CEILING_PLANE.extract(self.input_vec, "p")

    # --- Payload-group membership tests: classify a carrier by its marker ---
    @_input_cached_property
    def is_inert_non_payload(self) -> Node:
        """Rows whose type is intrinsically inert and should emit `NO_OP`.

        Numeric payload carriers are excluded on purpose. Their token type does
        not name a renderer state, so the payload router must inspect marker
        context instead of having this broad branch swallow every carrier row.
        """
        return _current_type_matches_any(self.input_vec, _INERT_NON_PAYLOAD_TYPES)

    @_input_cached_property
    def is_scene_value_payload(self) -> Node:
        """Whether this `VALUE` row carries scene/prefill data."""
        return and_(
            self.is_value,
            _prev_type_matches_any(self.prev_input_type, _SCENE_VALUE_MARKERS),
        )

    @_input_cached_property
    def is_scene_angle_payload(self) -> Node:
        """Whether this `ANGLE_VALUE` row carries scene/prefill data."""
        return and_(
            self.is_angle_value,
            _prev_type_matches_any(self.prev_input_type, _SCENE_ANGLE_MARKERS),
        )

    @_input_cached_property
    def is_projection_angle_payload(self) -> Node:
        """Whether this `ANGLE_VALUE` row is part of the AR projection cycle."""
        return and_(
            self.is_angle_value,
            _prev_type_matches_any(self.prev_input_type, _PROJECTION_ANGLE_MARKERS),
        )

    @_input_cached_property
    def is_bbox_angle_payload(self) -> Node:
        """Whether this `ANGLE_VALUE` row is part of bbox projection."""
        return and_(
            self.is_angle_value,
            _prev_type_matches_any(self.prev_input_type, _BBOX_ANGLE_MARKERS),
        )

    # --- Angle after-marker phase predicates: ANGLE_VALUE row after marker M ---
    @_input_cached_property
    def angle_after_world_a(self) -> Node:
        return and_(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, WORLD_ANGLE_MARK_A),
        )

    @_input_cached_property
    def angle_after_theta_a(self) -> Node:
        return and_(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, THETA_MARK_A),
        )

    @_input_cached_property
    def angle_after_world_b(self) -> Node:
        return and_(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, WORLD_ANGLE_MARK_B),
        )

    @_input_cached_property
    def angle_after_theta_b(self) -> Node:
        return and_(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, THETA_MARK_B),
        )

    @_input_cached_property
    def angle_after_drawseg_u_phase(self) -> Node:
        return and_(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, DRAWSEG_U_PHASE),
        )

    @_input_cached_property
    def angle_after_bbox_world_a(self) -> Node:
        return and_(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, BBOX_WORLD_ANGLE_MARK_A),
        )

    @_input_cached_property
    def angle_after_bbox_theta_a(self) -> Node:
        return and_(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, BBOX_THETA_MARK_A),
        )

    @_input_cached_property
    def angle_after_bbox_world_b(self) -> Node:
        return and_(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, BBOX_WORLD_ANGLE_MARK_B),
        )

    @_input_cached_property
    def angle_after_bbox_theta_b(self) -> Node:
        return and_(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, BBOX_THETA_MARK_B),
        )

    # --- More VALUE after-marker phase predicates (bbox corners, drawseg) ---
    @_input_cached_property
    def is_value_after_bbox_corner_x_a(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, BBOX_CORNER_X_MARK_A),
        )

    @_input_cached_property
    def is_value_after_bbox_corner_y_a(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, BBOX_CORNER_Y_MARK_A),
        )

    @_input_cached_property
    def is_value_after_bbox_corner_x_b(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, BBOX_CORNER_X_MARK_B),
        )

    @_input_cached_property
    def is_value_after_bbox_corner_y_b(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, BBOX_CORNER_Y_MARK_B),
        )

    @_input_cached_property
    def is_value_after_drawseg_scale1_den(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, DRAWSEG_SCALE1_DEN),
        )

    @_input_cached_property
    def is_value_after_drawseg_scale1(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, DRAWSEG_SCALE1),
        )

    @_input_cached_property
    def is_value_after_drawseg_scale2_den(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, DRAWSEG_SCALE2_DEN),
        )

    @_input_cached_property
    def is_value_after_drawseg_scale2(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, DRAWSEG_SCALE2),
        )

    @_input_cached_property
    def is_value_after_drawseg_scalestep_den(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, DRAWSEG_SCALESTEP_DEN),
        )

    @_input_cached_property
    def is_value_after_drawseg_scalestep(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, DRAWSEG_SCALESTEP),
        )

    @_input_cached_property
    def is_value_after_drawseg_bsilheight(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, DRAWSEG_BSILHEIGHT),
        )

    @_input_cached_property
    def is_value_after_drawseg_tsilheight(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, DRAWSEG_TSILHEIGHT),
        )

    @_input_cached_property
    def is_value_after_wall_column(self) -> Node:
        # The rw_scale VALUE follows WALL_COL_U (not SET_CURSOR_X
        # directly). The semantic name is preserved — this is still the
        # VALUE row that carries the wall-column scale.
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, WALL_COL_U),
        )

    @_input_cached_property
    def is_value_after_set_cursor_y(self) -> Node:
        return and_(
            self.is_value,
            _input_type_matches(self.prev_input_type, SET_CURSOR_Y),
        )

    # --- SCREEN_Y / SCREEN_RANGE after-marker phase predicates ---
    @_input_cached_property
    def screen_y_after_wall_column_scale(self) -> Node:
        # The wall-column rw_scale carrier follows WALL_COL_U; its
        # SCREEN_Y_VALUE emits the staged ceiling.
        return and_(
            self.is_screen_y_value,
            and_(
                _input_type_matches(self.prev_input_type, VALUE),
                _input_type_matches(self.prev_prev_input_type, WALL_COL_U),
            ),
        )

    @_input_cached_property
    def screen_range_after_clip_update(self) -> Node:
        return and_(
            self.is_screen_range,
            _input_type_matches(self.prev_input_type, CLIP_UPDATE),
        )

    @_input_cached_property
    def screen_range_after_plane_mark(self) -> Node:
        return and_(
            self.is_screen_range,
            _input_type_matches(self.prev_input_type, PLANE_MARK),
        )

    # --- Per-token slot accessors: typed reads of one named slot off the
    # current row (each returns 0 when the active row is not that type) ---
    @_input_cached_property
    def plane_mark_kind(self) -> Node:
        return PLANE_MARK.extract(self.input_vec, "kind")

    @_input_cached_property
    def plane_mark_p(self) -> Node:
        return PLANE_MARK.extract(self.input_vec, "p")

    @_input_cached_property
    def plane_mark_vp(self) -> Node:
        return PLANE_MARK.extract(self.input_vec, "vp")

    @_input_cached_property
    def r_check_plane_kind(self) -> Node:
        return R_CHECK_PLANE.extract(self.input_vec, "kind")

    @_input_cached_property
    def r_check_plane_vp(self) -> Node:
        return R_CHECK_PLANE.extract(self.input_vec, "vp")

    @_input_cached_property
    def r_check_plane_result_kind(self) -> Node:
        return R_CHECK_PLANE_RESULT.extract(self.input_vec, "kind")

    @_input_cached_property
    def r_check_plane_result_vp(self) -> Node:
        return R_CHECK_PLANE_RESULT.extract(self.input_vec, "vp")

    @_input_cached_property
    def plane_def_p(self) -> Node:
        return PLANE_DEF.extract(self.input_vec, "p")

    @_input_cached_property
    def flat_next_plane_p(self) -> Node:
        return FLAT_NEXT_PLANE.extract(self.input_vec, "p")

    @_input_cached_property
    def flat_next_vp_p(self) -> Node:
        return FLAT_NEXT_VP.extract(self.input_vec, "p")

    @_input_cached_property
    def flat_next_vp_vp(self) -> Node:
        return FLAT_NEXT_VP.extract(self.input_vec, "vp")

    @_input_cached_property
    def flat_visplane_p(self) -> Node:
        return FLAT_VISPLANE_BEGIN.extract(self.input_vec, "p")

    @_input_cached_property
    def flat_visplane_vp(self) -> Node:
        return FLAT_VISPLANE_BEGIN.extract(self.input_vec, "vp")

    @_input_cached_property
    def make_spans_x(self) -> Node:
        return MAKE_SPANS_COL.extract(self.input_vec, "x")

    @_input_cached_property
    def span_close_slot(self) -> Node:
        return SPAN_CLOSE_SLOT.extract(self.input_vec, "slot")

    @_input_cached_property
    def span_row_y(self) -> Node:
        return SPAN_ROW.extract(self.input_vec, "y")

    @_input_cached_property
    def span_row_yslope(self) -> Node:
        return extract_derived(self.input_vec, "yslope")

    @_input_cached_property
    def think_node(self) -> Node:
        return THINK_SIDE.extract(self.input_vec, "node")

    @_input_cached_property
    def side_record_node(self) -> Node:
        return SIDE_RECORD.extract(self.input_vec, "node")

    @_input_cached_property
    def side_record_side(self) -> Node:
        return SIDE_RECORD.extract(self.input_vec, "side")

    @_input_cached_property
    def enter_node(self) -> Node:
        return TRAVERSE_ENTER.extract(self.input_vec, "node")

    @_input_cached_property
    def enter_depth(self) -> Node:
        return TRAVERSE_ENTER.extract(self.input_vec, "depth")

    @_input_cached_property
    def between_node(self) -> Node:
        return TRAVERSE_BETWEEN.extract(self.input_vec, "node")

    @_input_cached_property
    def between_depth(self) -> Node:
        return TRAVERSE_BETWEEN.extract(self.input_vec, "depth")

    @_input_cached_property
    def return_entity(self) -> Node:
        return TRAVERSE_RETURN.extract(self.input_vec, "entity_u")

    @_input_cached_property
    def return_depth(self) -> Node:
        return TRAVERSE_RETURN.extract(self.input_vec, "depth")

    @_input_cached_property
    def visit_ss(self) -> Node:
        return VISIT_SUBSECTOR.extract(self.input_vec, "s")

    @_input_cached_property
    def visit_depth(self) -> Node:
        return VISIT_SUBSECTOR.extract(self.input_vec, "depth")

    @_input_cached_property
    def process_i(self) -> Node:
        return PROCESS_SEG.extract(self.input_vec, "i")

    @_input_cached_property
    def find_run_x(self) -> Node:
        return FIND_RUN.extract(self.input_vec, "x")

    @_input_cached_property
    def find_run_x_square(self) -> Node:
        return extract_derived(self.input_vec, "x_square")

    @_input_cached_property
    def advance_seg_i(self) -> Node:
        return ADVANCE_SEG.extract(self.input_vec, "i")

    @_input_cached_property
    def emit_x2_x(self) -> Node:
        return EMIT_X2.extract(self.input_vec, "x")

    @_input_cached_property
    def emit_x2_xtova_cos(self) -> Node:
        return extract_derived(self.input_vec, "xtova_cos")

    @_input_cached_property
    def store_i(self) -> Node:
        return R_STORE_WALL_RANGE.extract(self.input_vec, "i")

    @_input_cached_property
    def id_lifted_key(self) -> Node:
        return extract_derived(self.input_vec, ID_LIFTED_KEY_DERIVED_NAME)

    @_input_cached_property
    def drawseg_meta_i(self) -> Node:
        return DRAWSEG_META.extract(self.input_vec, "i")

    @_input_cached_property
    def cursor_x(self) -> Node:
        return SET_CURSOR_X.extract(self.input_vec, "x")

    @_input_cached_property
    def cursor_x_distscale(self) -> Node:
        return extract_derived(self.input_vec, "distscale")

    @_input_cached_property
    def wall_column_x(self) -> Node:
        return self.cursor_x

    @_input_cached_property
    def cursor_y(self) -> Node:
        return SET_CURSOR_Y.extract(self.input_vec, "y")

    @_input_cached_property
    def wall_span_meta_y(self) -> Node:
        return WALL_SPAN_META.extract(self.input_vec, "y")

    @_input_cached_property
    def wall_span_meta_ordinal(self) -> Node:
        return WALL_SPAN_META.extract(self.input_vec, "ordinal")

    @_input_cached_property
    def pixel_color(self) -> Node:
        return PIXEL.extract(self.input_vec, "color")

    @_input_cached_property
    def pixel_r(self) -> Node:
        return extract_derived(self.input_vec, "pixel_r")

    @_input_cached_property
    def pixel_g(self) -> Node:
        return extract_derived(self.input_vec, "pixel_g")

    @_input_cached_property
    def pixel_b(self) -> Node:
        return extract_derived(self.input_vec, "pixel_b")

    @_input_cached_property
    def screen_y(self) -> Node:
        return SCREEN_Y_VALUE.extract(self.input_vec, "y")

    @_input_cached_property
    def screen_range_y1(self) -> Node:
        return SCREEN_RANGE.extract(self.input_vec, "y1")

    @_input_cached_property
    def screen_range_y2(self) -> Node:
        return SCREEN_RANGE.extract(self.input_vec, "y2")

    # --- bbox sub-protocol tests and accessors (bounding-box visibility) ---
    @_input_cached_property
    def bbox_scan_x(self) -> Node:
        return BBOX_SCAN.extract(self.input_vec, "x")

    @_input_cached_property
    def bbox_scan_x_square(self) -> Node:
        return extract_derived(self.input_vec, "x_square")

    @_input_cached_property
    def bbox_boxpos(self) -> Node:
        return BBOX_BOXPOS.extract(self.input_vec, "boxpos")

    @_input_cached_property
    def bbox_corner_x_a_boxpos(self) -> Node:
        return BBOX_CORNER_X_MARK_A.extract(self.input_vec, "boxpos")

    @_input_cached_property
    def bbox_corner_y_a_boxpos(self) -> Node:
        return BBOX_CORNER_Y_MARK_A.extract(self.input_vec, "boxpos")

    @_input_cached_property
    def bbox_corner_x_b_boxpos(self) -> Node:
        return BBOX_CORNER_X_MARK_B.extract(self.input_vec, "boxpos")

    @_input_cached_property
    def bbox_corner_y_b_boxpos(self) -> Node:
        return BBOX_CORNER_Y_MARK_B.extract(self.input_vec, "boxpos")

    @_input_cached_property
    def boxpos_check_a_x_right(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "check_a_x_right"))

    @_input_cached_property
    def boxpos_check_a_y_bottom(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "check_a_y_bottom"))

    @_input_cached_property
    def boxpos_check_b_x_right(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "check_b_x_right"))

    @_input_cached_property
    def boxpos_check_b_y_bottom(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "check_b_y_bottom"))

    @_input_cached_property
    def boxpos_fails_open(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "fails_open"))

    # --- The *_or_zero value accessors ---
    # Each candidate is a per-token .extract(), which returns its slot value
    # only when the current row is that token's type and 0 otherwise. Exactly
    # one source slot is populated per row (a row has one type), so summing the
    # candidates yields that one value, or zero when none of them is the active
    # type — a marker-free way to read "whichever of these tokens this row is".
    @_input_cached_property
    def boxpos_or_zero(self) -> Node:
        return vec_sum(
            self.bbox_boxpos,
            self.bbox_corner_x_a_boxpos,
            self.bbox_corner_y_a_boxpos,
            self.bbox_corner_x_b_boxpos,
            self.bbox_corner_y_b_boxpos,
        )

    @_input_cached_property
    def seg_i_or_zero(self) -> Node:
        return vec_sum(
            self.process_i,
            self.advance_seg_i,
            self.store_i,
            self.drawseg_meta_i,
        )

    @_input_cached_property
    def seg_x_or_zero(self) -> Node:
        return vec_sum(
            self.find_run_x,
            self.emit_x2_x,
            self.cursor_x,
        )


def _token_check_property(token_type: TokenType):
    @_input_cached_property  # type: ignore[misc]
    def predicate(self: ProtocolTokenView) -> Node:
        return token_type.check(self.input_vec)

    return predicate


def _install_registry_token_checks() -> None:
    for spec in _TOKEN_CHECK_PREDICATES:
        if hasattr(ProtocolTokenView, spec.predicate):
            raise ValueError(
                f"{spec.predicate} is registry-derived but is already defined "
                "on ProtocolTokenView"
            )
        predicate = _token_check_property(spec.token)
        predicate.__set_name__(ProtocolTokenView, spec.predicate)
        setattr(ProtocolTokenView, spec.predicate, predicate)


_install_registry_token_checks()


def _prev_type_matches_any(
    prev_input_type: Node,
    token_types: tuple[TokenType, ...],
) -> Node:
    return bool_or(
        *(
            _input_type_matches(prev_input_type, token_type)
            for token_type in token_types
        )
    )


def _current_type_matches_any(
    input_vec: Node,
    token_types: tuple[TokenType, ...],
) -> Node:
    return bool_or(*(token_type.check(input_vec) for token_type in token_types))


@annotated("input")
def screen_column_one_hot(input_vec: Node) -> Node:
    return concat(
        *(extract_derived(input_vec, name) for name in SCREEN_X_ONE_HOT_DERIVED_NAMES)
    )
