"""Doom wall-lighting row selection — constant anchors against DOOM's C source.

Pinned against the
in-tree ``torchwright_doom.doom_lighting``. After the renderer/drafter were
vendored, a compiled-vs-Python-renderer gate can no longer catch a *lighting*
transcription bug (both descend from the same hand-port). This pins the
scalelight table, the lightnum sector-shift, and the scale-diminish distance
term directly against DOOM's ``r_segs.c`` / ``r_plane.c`` integer formulas — the
only independent guard on the lighting constants.
"""

from __future__ import annotations

import pytest

from torchwright_doom.doom_lighting import (
    DEFAULT_SCREEN_WIDTH,
    FRACUNIT,
    MAXLIGHTSCALE,
    NUMCOLORMAPS,
    apply_doom_colormap,
    doom_wall_base_colormap_row,
    doom_wall_colormap_row,
    doom_wall_colormap_row_from_index,
    doom_wall_light_static,
    doom_wall_lightnum,
    doom_wall_orientation_light_bias,
    doom_wall_scale_diminish,
    doom_wall_scale_diminish_from_index,
    doom_wall_scale_index,
    doom_wall_scale_index_from_fixed,
    doom_wall_startmap,
)


def test_wall_orientation_bias_matches_r_segs_axis_checks():
    assert doom_wall_orientation_light_bias(0, 10, 64, 10) == -1
    assert doom_wall_orientation_light_bias(10, 0, 10, 64) == 1
    assert doom_wall_orientation_light_bias(0, 0, 64, 64) == 0

    # r_segs.c checks y equality first.
    assert doom_wall_orientation_light_bias(4, 4, 4, 4) == -1


def test_wall_lightnum_uses_sector_shift_bias_extralight_and_clamp():
    assert doom_wall_lightnum(160) == 10
    assert doom_wall_lightnum(160, orientation_bias=-1) == 9
    assert doom_wall_lightnum(160, orientation_bias=1) == 11
    assert doom_wall_lightnum(144, orientation_bias=-1, extralight=2) == 10

    assert doom_wall_lightnum(0, orientation_bias=-1) == 0
    assert doom_wall_lightnum(255, orientation_bias=1, extralight=4) == 15


def test_scale_index_matches_doom_fixed_shift_and_clamp():
    assert doom_wall_scale_index_from_fixed(0) == 0
    assert doom_wall_scale_index_from_fixed(4095) == 0
    assert doom_wall_scale_index_from_fixed(4096) == 1
    assert doom_wall_scale_index_from_fixed(47 << 12) == 47
    assert doom_wall_scale_index_from_fixed(100 << 12) == MAXLIGHTSCALE - 1

    assert doom_wall_scale_index(0.0) == 0
    assert doom_wall_scale_index((1 << 12) / FRACUNIT) == 1
    assert doom_wall_scale_index(2.5) == 40
    assert doom_wall_scale_index(10.0) == MAXLIGHTSCALE - 1

    with pytest.raises(ValueError):
        doom_wall_scale_index(-0.01)
    with pytest.raises(ValueError):
        doom_wall_scale_index_from_fixed(-1)


def test_scalelight_table_rows_match_doom_integer_formula_full_width():
    # Full-width Doom view: distance term is scale_index // DISTMAP.
    assert [
        doom_wall_colormap_row_from_index(8, scale_index)
        for scale_index in (0, 1, 2, 3, 10, 47)
    ] == [28, 28, 27, 27, 23, 5]

    assert doom_wall_base_colormap_row(8) == 28
    assert doom_wall_startmap(0) == 60
    assert doom_wall_base_colormap_row(0) == NUMCOLORMAPS - 1
    assert doom_wall_colormap_row_from_index(0, 0) == NUMCOLORMAPS - 1
    assert doom_wall_colormap_row_from_index(0, 47) == NUMCOLORMAPS - 1
    assert doom_wall_colormap_row_from_index(15, 0) == 0
    assert doom_wall_colormap_row_from_index(15, 47) == 0


def test_scalelight_table_uses_view_width_and_detailshift_integer_order():
    # The C expression is:
    #   j * SCREENWIDTH / (viewwidth << detailshift) / DISTMAP
    # with integer truncation after each division.
    half_view = DEFAULT_SCREEN_WIDTH // 2

    assert (
        doom_wall_colormap_row_from_index(
            8,
            10,
            view_width=half_view,
        )
        == 18
    )
    assert (
        doom_wall_colormap_row_from_index(
            8,
            10,
            view_width=half_view,
            detailshift=1,
        )
        == 23
    )


def test_wall_scale_diminish_is_the_additive_distance_term():
    assert [
        doom_wall_scale_diminish_from_index(scale_index)
        for scale_index in (0, 1, 2, 3, 10, 47)
    ] == [0, 0, -1, -1, -5, -23]

    scale_for_index_32 = (32 << 12) / FRACUNIT
    assert doom_wall_scale_diminish(scale_for_index_32) == -16


def test_wall_colormap_row_combines_sector_light_axis_bias_and_scale():
    scale_for_index_32 = (32 << 12) / FRACUNIT

    assert doom_wall_light_static(144) == 24
    assert doom_wall_light_static(144, orientation_bias=-1) == 28
    assert doom_wall_light_static(144, orientation_bias=1) == 20

    assert doom_wall_colormap_row(144, scale_for_index_32) == 8
    assert (
        doom_wall_colormap_row(
            144,
            scale_for_index_32,
            orientation_bias=-1,
        )
        == 12
    )
    assert (
        doom_wall_colormap_row(
            144,
            scale_for_index_32,
            orientation_bias=1,
        )
        == 4
    )


def test_apply_doom_colormap_uses_wall_rows_only():
    colormap = [[row * 256 + palette for palette in range(256)] for row in range(33)]

    assert apply_doom_colormap(colormap, 0, 42) == 42
    assert apply_doom_colormap(colormap, 31, 7) == 31 * 256 + 7

    with pytest.raises(ValueError):
        apply_doom_colormap(colormap, 32, 7)
    with pytest.raises(ValueError):
        apply_doom_colormap(colormap, 0, 256)
