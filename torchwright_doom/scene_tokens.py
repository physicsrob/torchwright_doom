"""Prefill token interpretation for scene indexing.

The scene prefill stream is compact, so most facts are not self-contained
tokens. Structural headers (`NODE`, `SS`, `SEG`) establish the current record,
then marker tokens identify which field the following `VALUE` or `ANGLE_VALUE`
payload belongs to. This module gives those raw token checks/extractions names
that match the scene-index story.

Ported from ``doom_sandbox/implementation/forward/scene_tokens.py``. The only
changes from the sandbox source are the import block (``Vec`` -> ``Node``; the
std helpers + token declarations now come from the real-side shim / vocab) and
the shared ``input_type_matches`` body (``type_matches``); the
``@cached_property`` token accessors are a line-for-line port. Those names are
the prefill protocol's documentation — keep them explicit even where
repetitive.
"""

from __future__ import annotations

from functools import cached_property

from torchwright.graph import Node

from .std import bool_and, concat, extract_derived, indicator_to_bool
from .token_match import input_type_matches as _input_type_matches
from .tokens import TokenType
from .value_ranges import ValueRange, value_derived
from .vocab import (
    ANGLE_RAY_X_DERIVED_NAMES,
    ANGLE_RAY_Y_DERIVED_NAMES,
    ANGLE_VALUE,
    ID_LIFTED_KEY_DERIVED_NAME,
    NODE,
    NODE_BACK_CHILD,
    NODE_FRONT_CHILD,
    PLANE_DEF,
    PLANE_LIGHT,
    SEG,
    SEG_CLOSED_DOOR,
    SEG_EMPTY_LINE,
    SEG_LIGHT_STATIC,
    SEG_LOWER_TEXTURE,
    SEG_MID_TEXTURE,
    SEG_PEGGING,
    SEG_TWO_SIDED,
    SEG_UPPER_TEXTURE,
    SS,
    VALUE,
)


class SceneTokenView:
    """Lazy typed view over the current prefill/input token.

    `input_vec` is the token currently being processed. `prev_input_type` is
    the previous token's type slice, recovered in `SceneIndex.build()`, and is
    used only for marker/value pairs such as `NODE_PX VALUE`.
    """

    def __init__(self, input_vec: Node, prev_input_type: Node) -> None:
        self.input_vec: Node = input_vec
        self.prev_input_type: Node = prev_input_type

    @cached_property
    def is_value(self) -> Node:
        """Whether this token carries a normal scalar payload."""
        return VALUE.check(self.input_vec)

    @cached_property
    def is_angle_value(self) -> Node:
        """Whether this token carries an angle payload."""
        return ANGLE_VALUE.check(self.input_vec)

    def payload_value(self, range_id: ValueRange) -> Node:
        """The scalar payload decoded with the marker-selected range."""
        return value_derived(self.input_vec, range_id)

    @cached_property
    def angle(self) -> Node:
        """The angle payload from an `ANGLE_VALUE` token."""
        return ANGLE_VALUE.extract(self.input_vec, "angle")

    @cached_property
    def angle_sin(self) -> Node:
        """Derived sine for the current `ANGLE_VALUE` token."""
        return extract_derived(self.input_vec, "sin")

    @cached_property
    def angle_cos(self) -> Node:
        """Derived cosine for the current `ANGLE_VALUE` token."""
        return extract_derived(self.input_vec, "cos")

    @cached_property
    def angle_ray_x_by_screen(self) -> Node:
        return concat(
            *(
                extract_derived(self.input_vec, name)
                for name in ANGLE_RAY_X_DERIVED_NAMES
            )
        )

    @cached_property
    def angle_ray_y_by_screen(self) -> Node:
        return concat(
            *(
                extract_derived(self.input_vec, name)
                for name in ANGLE_RAY_Y_DERIVED_NAMES
            )
        )

    @cached_property
    def is_node(self) -> Node:
        """Whether this token opens a NODE record."""
        return NODE.check(self.input_vec)

    @cached_property
    def node_j(self) -> Node:
        """The NODE record id from a NODE header token."""
        return NODE.extract(self.input_vec, "j")

    @cached_property
    def is_subsector(self) -> Node:
        """Whether this token opens an SS/subsector record."""
        return SS.check(self.input_vec)

    @cached_property
    def subsector_s(self) -> Node:
        """The subsector id from an SS header token."""
        return SS.extract(self.input_vec, "s")

    @cached_property
    def is_seg(self) -> Node:
        """Whether this token opens a SEG record."""
        return SEG.check(self.input_vec)

    @cached_property
    def seg_i(self) -> Node:
        """The seg id from a SEG header token."""
        return SEG.extract(self.input_vec, "i")

    @cached_property
    def is_first_seg_of_subsector(self) -> Node:
        """Whether this SEG header is the first SEG owned by its subsector."""
        return indicator_to_bool(SEG.extract(self.input_vec, "is_first_of_ss"))

    @cached_property
    def is_seg_two_sided(self) -> Node:
        """Whether this token names the current SEG's solid/portal flag."""
        return SEG_TWO_SIDED.check(self.input_vec)

    @cached_property
    def seg_two_sided_flag(self) -> Node:
        """The current SEG_TWO_SIDED flag."""
        return indicator_to_bool(SEG_TWO_SIDED.extract(self.input_vec, "flag"))

    @cached_property
    def is_seg_upper_texture(self) -> Node:
        return SEG_UPPER_TEXTURE.check(self.input_vec)

    @cached_property
    def seg_upper_texture_present(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "present"))

    @cached_property
    def is_seg_lower_texture(self) -> Node:
        return SEG_LOWER_TEXTURE.check(self.input_vec)

    @cached_property
    def seg_lower_texture_present(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "present"))

    @cached_property
    def is_seg_mid_texture(self) -> Node:
        return SEG_MID_TEXTURE.check(self.input_vec)

    @cached_property
    def seg_mid_texture_present(self) -> Node:
        return indicator_to_bool(extract_derived(self.input_vec, "present"))

    @cached_property
    def seg_mid_tex_id(self) -> Node:
        return SEG_MID_TEXTURE.extract(self.input_vec, "tex_id")

    @cached_property
    def seg_upper_tex_id(self) -> Node:
        return SEG_UPPER_TEXTURE.extract(self.input_vec, "tex_id")

    @cached_property
    def seg_lower_tex_id(self) -> Node:
        return SEG_LOWER_TEXTURE.extract(self.input_vec, "tex_id")

    @cached_property
    def is_seg_light_static(self) -> Node:
        return SEG_LIGHT_STATIC.check(self.input_vec)

    @cached_property
    def seg_light_static(self) -> Node:
        return SEG_LIGHT_STATIC.extract(self.input_vec, "light")

    @cached_property
    def id_lifted_key(self) -> Node:
        return extract_derived(self.input_vec, ID_LIFTED_KEY_DERIVED_NAME)

    @cached_property
    def is_plane_def(self) -> Node:
        return PLANE_DEF.check(self.input_vec)

    @cached_property
    def plane_def_p(self) -> Node:
        return PLANE_DEF.extract(self.input_vec, "p")

    @cached_property
    def plane_def_flat_id(self) -> Node:
        return PLANE_DEF.extract(self.input_vec, "flat_id")

    @cached_property
    def is_plane_light(self) -> Node:
        return PLANE_LIGHT.check(self.input_vec)

    @cached_property
    def plane_light_static(self) -> Node:
        """Precomputed Doom ``zlight`` startmap from PLANE_LIGHT.light."""
        return extract_derived(self.input_vec, "light_static")

    @cached_property
    def is_seg_pegging(self) -> Node:
        # DOOM: line_t.flags contains ML_DONTPEGTOP and ML_DONTPEGBOTTOM (r_defs.h, used in r_segs.c).
        return SEG_PEGGING.check(self.input_vec)

    @cached_property
    def seg_dontpegtop_flag(self) -> Node:
        return indicator_to_bool(SEG_PEGGING.extract(self.input_vec, "dontpegtop"))

    @cached_property
    def seg_dontpegbottom_flag(self) -> Node:
        return indicator_to_bool(SEG_PEGGING.extract(self.input_vec, "dontpegbottom"))

    @cached_property
    def is_seg_empty_line(self) -> Node:
        return SEG_EMPTY_LINE.check(self.input_vec)

    @cached_property
    def seg_empty_line_flag(self) -> Node:
        return indicator_to_bool(SEG_EMPTY_LINE.extract(self.input_vec, "flag"))

    @cached_property
    def is_seg_closed_door(self) -> Node:
        return SEG_CLOSED_DOOR.check(self.input_vec)

    @cached_property
    def seg_closed_door_flag(self) -> Node:
        return indicator_to_bool(SEG_CLOSED_DOOR.extract(self.input_vec, "flag"))

    @cached_property
    def is_front_child(self) -> Node:
        """Whether this token directly names a NODE front child."""
        return NODE_FRONT_CHILD.check(self.input_vec)

    @cached_property
    def front_child_u(self) -> Node:
        """The encoded front-child entity from a NODE_FRONT_CHILD token."""
        return NODE_FRONT_CHILD.extract(self.input_vec, "child_u")

    @cached_property
    def is_back_child(self) -> Node:
        """Whether this token directly names a NODE back child."""
        return NODE_BACK_CHILD.check(self.input_vec)

    @cached_property
    def back_child_u(self) -> Node:
        """The encoded back-child entity from a NODE_BACK_CHILD token."""
        return NODE_BACK_CHILD.extract(self.input_vec, "child_u")

    @cached_property
    def child_lifted_key(self) -> Node:
        """This row's child lifted equality key ``[child, -child^2, 1]``.

        The ``id_lifted_key`` derived span is declared on both NODE child slots
        (and the structural id slots). Each ``(type, slot, name)`` declaration
        owns its own span, so on a NODE_FRONT_CHILD row this recovers the front
        child's key and on a NODE_BACK_CHILD row the back child's; on any other
        row it is the off-type zero. The front/back child lookups gate this read
        by ``is_front_child`` / ``is_back_child``, so one accessor serves both.
        Mirrors ``front_child_u`` / ``back_child_u`` in the width-3 lifted form
        used as a producer key.
        """
        return extract_derived(self.input_vec, ID_LIFTED_KEY_DERIVED_NAME)

    def value_after(self, marker_type: TokenType) -> Node:
        """Mask scalar VALUE payloads that immediately follow `marker_type`.

        This is the central marker/value convention: the marker token says what
        the next payload means, while the payload token contains only the value.
        """
        return bool_and(
            self.is_value,
            _input_type_matches(self.prev_input_type, marker_type),
        )

    def angle_after(self, marker_type: TokenType) -> Node:
        """Mask angle payloads that immediately follow `marker_type`."""
        return bool_and(
            self.is_angle_value,
            _input_type_matches(self.prev_input_type, marker_type),
        )
