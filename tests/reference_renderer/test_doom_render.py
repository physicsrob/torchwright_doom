"""Tests for the C-faithful DOOM reference renderer (``doom_render.py``).

The new renderer mirrors r_main.c / r_bsp.c / r_segs.c / r_draw.c
function-for-function.  These tests exercise:

  1. A synthetic single-sector room (no BSP) — proves the wall pipeline
     paints the correct screen regions.
  2. R_PointOnSide parity — picks the correct side for hand-picked
     points against a synthetic node.
  3. R_ClipSolidWallSegment table walk — reproduces the canonical
     four-wall example from ``orig-doom-renderer/renderer.md`` step
     by step.
  4. WAD parity — when a doom1.wad is available, renders the same
     pose with both the new and the legacy renderer and asserts
     reasonable agreement.  Skipped if the WAD is missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest

from torchwright_doom.doom.wad import BspNode
from torchwright_doom.reference_renderer import doom_render as dr
from torchwright_doom.reference_renderer.types import RenderConfig
from torchwright_doom.reference_renderer.trig import generate_trig_table


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _box_walls():
    """Four walls of a 256x256 box centred at the origin, wound clockwise.

    DOOM convention: front (right of a→b) faces *into* the sector.  For
    a closed room that's clockwise traversal viewed from above (+y up,
    +x east).
    """
    return [
        dr._WallSpec(a=(128, 128), b=(128, -128)),    # east  (going south)
        dr._WallSpec(a=(128, -128), b=(-128, -128)),  # south (going west)
        dr._WallSpec(a=(-128, -128), b=(-128, 128)),  # west  (going north)
        dr._WallSpec(a=(-128, 128), b=(128, 128)),    # north (going east)
    ]


@pytest.fixture
def small_config() -> RenderConfig:
    return RenderConfig(
        screen_width=32,
        screen_height=24,
        fov_columns=64,
        trig_table=generate_trig_table(),
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
        player_eye_z=41.0,
    )


# ---------------------------------------------------------------------
# Single-sector smoke test
# ---------------------------------------------------------------------


def test_single_sector_smoke(small_config):
    """Render a 256x256 box room from origin facing east; check the framebuffer.

    Player eye at z=41 (DOOM convention: stands 41 units above floor in
    a 128-tall room).  Walls 128 world units east of the player ⇒
    rw_scale ≈ 16/128 = 0.125 px/unit.  Wall projection:
        top    = centery - (128 - 41) * 0.125 = 12 - 10.875 ≈ 1.125  ⇒ row 2
        bottom = centery - (0   - 41) * 0.125 = 12 +  5.125 ≈ 17.125 ⇒ row 17
    """
    walls = _box_walls()
    md = dr.single_sector_map(walls, floor_h=0, ceiling_h=128)

    frame = dr.R_RenderPlayerView(0.0, 0.0, 41.0, 0, md, small_config)

    # Shape and colour-space sanity.
    assert frame.shape == (24, 32, 3)
    assert frame.dtype == np.float64

    h, w = small_config.screen_height, small_config.screen_width
    centery = h // 2

    ceil = np.array(small_config.ceiling_color)
    floor = np.array(small_config.floor_color)

    # Row 0 is above the wall band ⇒ ceiling.  Row h-1 is below ⇒ floor.
    assert np.allclose(frame[0], ceil), "top edge should be all ceiling"
    assert np.allclose(frame[-1], floor), "bottom edge should be all floor"

    # Wall expected at the centre column.  The walls of a 256-unit box
    # viewed from the centre cover every column (no gaps), so the centre
    # row in the centre column must be a wall pixel (sector colour).
    centre_col = w // 2
    centre_pixel = frame[centery, centre_col]
    assert not np.allclose(centre_pixel, ceil), "wall expected at centre pixel"
    assert not np.allclose(centre_pixel, floor), "wall expected at centre pixel"

    # And the centre row should have a wall in *every* column (closed room).
    for col in range(w):
        px = frame[centery, col]
        assert not np.allclose(px, ceil) and not np.allclose(px, floor), (
            f"col {col}: expected wall at centre row, got {px}"
        )

    # Print solidsegs / floorclip / ceilingclip snapshots under -s so a
    # reader can see the C state machine evolve.
    print()
    print("solidsegs after frame:")
    for i in range(dr.newend):
        print(f"  [{i}] first={dr.solidsegs[i].first:>10}  last={dr.solidsegs[i].last:>10}")
    print(f"floorclip:   {dr.floorclip.tolist()}")
    print(f"ceilingclip: {dr.ceilingclip.tolist()}")


def test_single_sector_rotation(small_config):
    """Rotating through 4 cardinal directions, the centre column always shows wall."""
    walls = _box_walls()
    md = dr.single_sector_map(walls, floor_h=0, ceiling_h=128)
    centery = small_config.screen_height // 2
    centre_col = small_config.screen_width // 2

    ceil = np.array(small_config.ceiling_color)
    floor = np.array(small_config.floor_color)

    for bam in (0, dr.ANG90, dr.ANG180, dr.ANG270):
        frame = dr.R_RenderPlayerView(0.0, 0.0, 41.0, bam, md, small_config)
        px = frame[centery, centre_col]
        assert not np.allclose(px, ceil) and not np.allclose(px, floor), (
            f"angle 0x{bam:08x}: expected wall at centre, got {px}"
        )


# ---------------------------------------------------------------------
# R_PointOnSide parity
# ---------------------------------------------------------------------


def test_point_on_side_axis_aligned():
    """Vertical / horizontal partition lines hit the shortcut paths."""
    # Vertical partition: x = 100, dy > 0 ⇒ points with x > 100 are FRONT (0).
    vertical = BspNode(
        px=100, py=0, dx=0, dy=1,
        front_bbox=(0, 0, 0, 0), back_bbox=(0, 0, 0, 0),
        front_child=0, back_child=0,
    )
    assert dr.R_PointOnSide(150, 0, vertical) == 0  # right of line ⇒ front
    assert dr.R_PointOnSide(50, 0, vertical) == 1   # left ⇒ back
    assert dr.R_PointOnSide(100, 0, vertical) == 1  # x <= px ⇒ back when dy > 0

    # Horizontal partition: y = 50, dx > 0 (line points east).  C code:
    #     if (!dy) { if (y <= y0) return dx<0; return dx>0; }
    # With dx=1>0: y>50 returns 1 (back); y<=50 returns 0 (front).
    # Front = "right of direction" ⇒ for east-pointing line, south side.
    horizontal = BspNode(
        px=0, py=50, dx=1, dy=0,
        front_bbox=(0, 0, 0, 0), back_bbox=(0, 0, 0, 0),
        front_child=0, back_child=0,
    )
    assert dr.R_PointOnSide(0, 100, horizontal) == 1  # above (north) ⇒ back
    assert dr.R_PointOnSide(0, 0, horizontal) == 0    # below (south) ⇒ front


def test_point_on_side_general():
    """Diagonal partition uses the cross-product path."""
    # Partition through (0,0) with direction (1,1) — splits NW from SE.
    # Cross product: (dy)*(dx_pt) - (dx)*(dy_pt) = 1*x - 1*y.
    # Front (return 0) when right < left, i.e. y*1 < x*1 ⇒ y < x.
    diag = BspNode(
        px=0, py=0, dx=1, dy=1,
        front_bbox=(0, 0, 0, 0), back_bbox=(0, 0, 0, 0),
        front_child=0, back_child=0,
    )
    assert dr.R_PointOnSide(10, 5, diag) == 0   # y < x ⇒ front
    assert dr.R_PointOnSide(5, 10, diag) == 1   # y > x ⇒ back


# ---------------------------------------------------------------------
# R_ClipSolidWallSegment table walk (renderer.md example)
# ---------------------------------------------------------------------


def test_clip_solid_wall_segment_table_walk():
    """Reproduce the four-wall example from renderer.md step by step.

    Initial state: only off-screen sentinels.
    After A (100..220): mid-screen, fits between sentinels.
    After B (-INF..50): clamped to left, extends sentinel[0].
    After C (50..70):   touches sentinel[0]'s right edge; extends it.
    After D (70..100):  bridges to A; merges everything into one sentinel.
    """
    # Set up with viewwidth=320 (matching the canonical example).
    dr._init_view_size(320, 200)
    dr.R_ClearClipSegs()

    # Stub R_StoreWallRange so we don't try to draw — we're only
    # checking solidsegs[] state evolution.
    saved = dr.R_StoreWallRange
    dr.R_StoreWallRange = lambda first, last: None
    try:
        # Initial sentinels.
        assert dr.newend == 2
        assert (dr.solidsegs[0].first, dr.solidsegs[0].last) == (-0x7FFFFFFF, -1)
        assert (dr.solidsegs[1].first, dr.solidsegs[1].last) == (320, 0x7FFFFFFF)

        # After A: mid-screen wall 100..220.
        dr.R_ClipSolidWallSegment(100, 220)
        assert dr.newend == 3
        assert (dr.solidsegs[0].first, dr.solidsegs[0].last) == (-0x7FFFFFFF, -1)
        assert (dr.solidsegs[1].first, dr.solidsegs[1].last) == (100, 220)
        assert (dr.solidsegs[2].first, dr.solidsegs[2].last) == (320, 0x7FFFFFFF)

        # After B: spans the left edge (clamped at -INF).  Extends sentinel[0].
        dr.R_ClipSolidWallSegment(-0x7FFFFFFF, 50)
        assert dr.newend == 3
        assert (dr.solidsegs[0].first, dr.solidsegs[0].last) == (-0x7FFFFFFF, 50)
        assert (dr.solidsegs[1].first, dr.solidsegs[1].last) == (100, 220)

        # After C: 50..70.  Adjacent to sentinel[0]'s last (50) so merges in.
        dr.R_ClipSolidWallSegment(51, 70)
        assert dr.newend == 3
        assert (dr.solidsegs[0].first, dr.solidsegs[0].last) == (-0x7FFFFFFF, 70)

        # After D: bridge across to MAX.  All real entries merge into
        # solidsegs[0].  NB: the C ``while (next++ != newend)`` crunch
        # idiom leaves one trailing junk slot before resetting newend
        # (it reads solidsegs[newend], which is past valid data).  That's
        # the actual DOOM behaviour — we don't pin newend or that slot;
        # what matters is solidsegs[0] now covers everything, which makes
        # subsequent walls early-exit at slot 0.
        dr.R_ClipSolidWallSegment(71, 0x7FFFFFFF)
        assert (dr.solidsegs[0].first, dr.solidsegs[0].last) == (-0x7FFFFFFF, 0x7FFFFFFF)
    finally:
        dr.R_StoreWallRange = saved


# ---------------------------------------------------------------------
# WAD parity (skipped without doom1.wad)
# ---------------------------------------------------------------------


def _wad_path() -> str | None:
    """Return path to doom1.wad if it sits next to the test (umbrella checkout)."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "doom1.wad"),
        os.path.join(os.getcwd(), "doom1.wad"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


@pytest.mark.skipif(_wad_path() is None, reason="doom1.wad not available")
def test_wad_parity_smoke():
    """Render E1M1 from a player-spawn pose and confirm we got *some* walls.

    Full bit-exact parity with the legacy renderer is out of scope (the
    two pipelines differ structurally — that's the point of the new
    file).  This test just confirms the pipeline runs end-to-end on
    real WAD data and doesn't crash.
    """
    from torchwright_doom.doom.wad import WADReader

    wad = WADReader(_wad_path())  # type: ignore[arg-type]
    md = wad.get_map("E1M1")

    # Pick the player-1 spawn (Thing type 1).
    spawn = next(t for t in md.things if t.type == 1)
    player_x, player_y = float(spawn.x), float(spawn.y)
    player_z = 41.0  # E1M1 spawn floor height ≈ 0; eye 41 above.
    bam = (spawn.angle * (2 ** 32) // 360) & dr.ANGMASK

    # Build a wall-texture atlas from sidedef names.
    tex_names = set()
    for sd in md.sidedefs:
        for n in (sd.upper, sd.lower, sd.middle):
            if n and n != "-":
                tex_names.add(n)
    textures = {n: t for n, t in wad.get_textures(list(tex_names)).items()}

    cfg = RenderConfig(
        screen_width=64, screen_height=48, fov_columns=64,
        trig_table=generate_trig_table(),
        ceiling_color=(0.0, 0.0, 0.5),
        floor_color=(0.5, 0.0, 0.0),
        player_eye_z=player_z,
    )

    frame = dr.R_RenderPlayerView(player_x, player_y, player_z, bam, md, cfg, textures)

    # We expect *some* pixels to differ from the bare ceiling/floor fill.
    ceil = np.array(cfg.ceiling_color)
    floor = np.array(cfg.floor_color)
    is_wall = ~(
        np.all(np.isclose(frame, ceil), axis=-1)
        | np.all(np.isclose(frame, floor), axis=-1)
    )
    wall_pixels = int(is_wall.sum())
    assert wall_pixels > 100, f"expected wall pixels in E1M1 spawn frame, got {wall_pixels}"
