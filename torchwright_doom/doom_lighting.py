"""Doom wall-lighting helpers (ported verbatim from the sandbox).

These functions mirror the integer row-selection logic used by Doom's
``scalelight`` / ``zlight`` tables for textured walls and flats. They do
not apply PLAYPAL; the output is the COLORMAP row to use (or the palette
index after COLORMAP indirection).

Pure integer/float math with no project dependencies — ported as the
source of truth for the ``wall_scale_diminish`` VALUE-derived column
(see :func:`value_ranges.value_derived_columns`) and for the forward
renderer's lighting. Kept faithful to the sandbox so the
translation table maps 1:1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

LIGHTLEVELS = 16
LIGHTSEGSHIFT = 4
MAXLIGHTSCALE = 48
LIGHTSCALESHIFT = 12
NUMCOLORMAPS = 32
DISTMAP = 2
FRACBITS = 16
FRACUNIT = 1 << FRACBITS
DEFAULT_SCREEN_WIDTH = 320

# Flats use Doom's ``zlight`` distance table indexed by
# ``distance >> LIGHTZSHIFT`` (real-valued: ``int(distance) >> 4``,
# since FRACBITS=16 and LIGHTZSHIFT=20 differ by 4).
MAXLIGHTZ = 128
LIGHTZSHIFT = 20


def doom_wall_orientation_light_bias(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> int:
    """Return Doom's axis-based wall light-number adjustment.

    Horizontal linedefs are one light level darker; vertical linedefs are one
    light level brighter. The check order matches ``r_segs.c``: a degenerate
    line with both coordinates equal is treated as horizontal.
    """

    if y1 == y2:
        return -1
    if x1 == x2:
        return 1
    return 0


def doom_wall_lightnum(
    sector_light: int,
    *,
    orientation_bias: int = 0,
    extralight: int = 0,
) -> int:
    """Return the clamped ``scalelight`` light-number index for a wall."""

    lightnum = (
        (int(sector_light) >> LIGHTSEGSHIFT) + int(extralight) + int(orientation_bias)
    )
    return _clamp_int(lightnum, 0, LIGHTLEVELS - 1)


def doom_wall_scale_index_from_fixed(rw_scale_fixed: int) -> int:
    """Return Doom's clamped ``rw_scale >> LIGHTSCALESHIFT`` table index."""

    rw_scale_fixed = int(rw_scale_fixed)
    if rw_scale_fixed < 0:
        raise ValueError("rw_scale_fixed must be non-negative")
    index = rw_scale_fixed >> LIGHTSCALESHIFT
    return _clamp_int(index, 0, MAXLIGHTSCALE - 1)


def doom_wall_scale_index(scale: float) -> int:
    """Return the light-scale index for a real-valued sandbox wall scale.

    The sandbox reference stores Doom fixed-point scales as real values where
    ``1.0`` corresponds to ``FRACUNIT``.
    """

    scale = float(scale)
    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    fixed = math.floor(scale * FRACUNIT)
    return doom_wall_scale_index_from_fixed(fixed)


def doom_wall_base_colormap_row(lightnum: int) -> int:
    """Return the sector/orientation COLORMAP row before distance scaling."""

    return _clamp_colormap_row(doom_wall_startmap(lightnum))


def doom_wall_startmap(lightnum: int) -> int:
    """Return Doom's unbounded ``startmap`` for a light-number index.

    This is the value to combine with the negative scale-diminish term before
    the final COLORMAP clamp. It intentionally ranges above 31 for dark
    sectors.
    """

    lightnum = _clamp_int(int(lightnum), 0, LIGHTLEVELS - 1)
    return _doom_scalelight_startmap(lightnum)


def doom_wall_light_static(
    sector_light: int,
    *,
    orientation_bias: int = 0,
    extralight: int = 0,
) -> int:
    """Return the unbounded sector/orientation term for wall lighting."""

    return doom_wall_startmap(
        doom_wall_lightnum(
            sector_light,
            orientation_bias=orientation_bias,
            extralight=extralight,
        )
    )


def doom_wall_scale_diminish(
    scale: float,
    *,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    view_width: int | None = None,
    detailshift: int = 0,
) -> int:
    """Return the non-positive row adjustment contributed by wall scale."""

    return doom_wall_scale_diminish_from_index(
        doom_wall_scale_index(scale),
        screen_width=screen_width,
        view_width=view_width,
        detailshift=detailshift,
    )


def doom_wall_scale_diminish_from_index(
    scale_index: int,
    *,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    view_width: int | None = None,
    detailshift: int = 0,
) -> int:
    """Return the non-positive row adjustment for a clamped scale index."""

    scale_index = _clamp_int(int(scale_index), 0, MAXLIGHTSCALE - 1)
    return -_doom_scalelight_distance_term(
        scale_index,
        screen_width=screen_width,
        view_width=view_width,
        detailshift=detailshift,
    )


def doom_wall_colormap_row_from_index(
    lightnum: int,
    scale_index: int,
    *,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    view_width: int | None = None,
    detailshift: int = 0,
) -> int:
    """Return Doom's COLORMAP row for ``scalelight[lightnum][scale_index]``."""

    base = _doom_scalelight_startmap(_clamp_int(int(lightnum), 0, LIGHTLEVELS - 1))
    distance = _doom_scalelight_distance_term(
        _clamp_int(int(scale_index), 0, MAXLIGHTSCALE - 1),
        screen_width=screen_width,
        view_width=view_width,
        detailshift=detailshift,
    )
    return _clamp_colormap_row(base - distance)


def doom_wall_colormap_row(
    sector_light: int,
    scale: float,
    *,
    orientation_bias: int = 0,
    extralight: int = 0,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    view_width: int | None = None,
    detailshift: int = 0,
) -> int:
    """Return the COLORMAP row selected for a textured wall column."""

    lightnum = doom_wall_lightnum(
        sector_light,
        orientation_bias=orientation_bias,
        extralight=extralight,
    )
    return doom_wall_colormap_row_from_index(
        lightnum,
        doom_wall_scale_index(scale),
        screen_width=screen_width,
        view_width=view_width,
        detailshift=detailshift,
    )


def doom_flat_lightnum(
    sector_light: int,
    *,
    extralight: int = 0,
) -> int:
    """Return Doom's clamped ``zlight`` light-number index for a flat.

    Flats have no orientation bias — only the sector light level (plus
    ``extralight``) drives the light-num index, unlike walls which also
    pick up an axis bias from the linedef direction.
    """

    lightnum = (int(sector_light) >> LIGHTSEGSHIFT) + int(extralight)
    return _clamp_int(lightnum, 0, LIGHTLEVELS - 1)


def doom_flat_startmap(
    sector_light: int,
    *,
    extralight: int = 0,
) -> int:
    """Return the unbounded sector-light term for flat ``zlight``."""

    return _doom_scalelight_startmap(
        doom_flat_lightnum(sector_light, extralight=extralight)
    )


def doom_flat_distance_index(distance: float) -> int:
    """Return the clamped ``zlight`` distance bucket for a real distance.

    DOOM does ``distance_fixed >> LIGHTZSHIFT`` on a fixed-point distance;
    the real-valued sandbox equivalent is ``int(distance) >> 4``.
    """

    if distance < 0.0:
        distance = 0.0
    raw_index = int(distance) >> 4
    return _clamp_int(raw_index, 0, MAXLIGHTZ - 1)


def doom_flat_scale_diminish(
    distance: float,
    *,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
) -> int:
    """Return the non-positive ``zlight`` row adjustment for a distance.

    The integer shifts unwind cleanly in real values:
    ``scale = (screen_width / 2) * 4096 / (index + 1)`` (fixed), and
    ``scale >> LIGHTSCALESHIFT`` collapses to ``(screen_width // 2) //
    (index + 1)`` with the final ``// DISTMAP`` giving the per-step row
    increment.
    """

    index = doom_flat_distance_index(distance)
    scale_shifted = (int(screen_width) // 2) // (index + 1)
    return -(scale_shifted // DISTMAP)


def doom_flat_colormap_row(
    sector_light: int,
    distance: float,
    *,
    extralight: int = 0,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
) -> int:
    """Return the clamped COLORMAP row for a flat span at a distance."""

    base = doom_flat_startmap(sector_light, extralight=extralight)
    diminish = doom_flat_scale_diminish(distance, screen_width=screen_width)
    return _clamp_colormap_row(base + diminish)


def apply_doom_colormap(
    colormap: Sequence[Sequence[int]],
    row: int,
    palette_index: int,
) -> int:
    """Return ``COLORMAP[row][palette_index]`` with wall-lighting bounds."""

    row = int(row)
    palette_index = int(palette_index)
    if not 0 <= row < NUMCOLORMAPS:
        raise ValueError(f"wall COLORMAP row must be in [0, {NUMCOLORMAPS})")
    if not 0 <= palette_index < 256:
        raise ValueError("palette_index must be in [0, 256)")
    return int(colormap[row][palette_index])


def _doom_scalelight_startmap(lightnum: int) -> int:
    return ((LIGHTLEVELS - 1 - int(lightnum)) * 2 * NUMCOLORMAPS) // LIGHTLEVELS


def _doom_scalelight_distance_term(
    scale_index: int,
    *,
    screen_width: int,
    view_width: int | None,
    detailshift: int,
) -> int:
    screen_width = int(screen_width)
    view_width = screen_width if view_width is None else int(view_width)
    detailshift = int(detailshift)
    if screen_width <= 0:
        raise ValueError("screen_width must be positive")
    if view_width <= 0:
        raise ValueError("view_width must be positive")
    if detailshift < 0:
        raise ValueError("detailshift must be non-negative")
    view_denom = view_width << detailshift
    if view_denom <= 0:
        raise ValueError("view_width << detailshift must be positive")
    return ((int(scale_index) * screen_width) // view_denom) // DISTMAP


def _clamp_colormap_row(value: int) -> int:
    return _clamp_int(int(value), 0, NUMCOLORMAPS - 1)


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))
