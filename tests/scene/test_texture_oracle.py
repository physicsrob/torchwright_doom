"""The texel database is readable through the real graph.

A ``reference_eval``-only direct value-match probe: build the
asset accessors with literal query inputs, ``reference_eval`` the channel, and
compare to the WAD-loaded ``ASSET_BOOK`` pixels / ``COLORMAP_ROWS`` within
``1e-3``. No compile, no GPU, no rollout, no geometry.

Covers a wall lookup that exercises the ``u``-wrap, one that
hits the missing-texture bank 0, a flat, and a non-zero ``colormap_row``; plus
the ``WallAssets.height`` / ``h_idx_oh`` metadata and the
``AssetIndex`` wiring ``SceneIndex`` exposes.
"""

from __future__ import annotations

import subprocess
import sys

import torch

from torchwright.debug.probe import reference_eval

from torchwright_doom import asset_banks as ab
from torchwright_doom.assets import AssetIndex, FlatAssets, WallAssets
from torchwright_doom.doom_lighting import NUMCOLORMAPS, apply_doom_colormap
from torchwright_doom.lighting import apply_colormap_row
from torchwright_doom.scene_index import SceneIndex
from torchwright_doom.std import constant

BOOK = ab.ASSET_BOOK
ATOL = 1e-3


def _vec(node) -> torch.Tensor:
    return reference_eval(node, {}, 1)[node].reshape(-1)


def _scalar(node) -> float:
    return float(_vec(node)[0])


def _v_mods(v: float) -> tuple:
    """Teacher-forced per-height v_mod tuple: the same already-wrapped value
    in every WALL_HEIGHT_BANK slot (each bank reads its own height's entry,
    so a uniform tuple reproduces the old single-scalar semantics)."""
    return tuple(constant(v) for _ in ab.WALL_HEIGHT_BANK)


def test_wall_palette_index_matches_wad():
    walls = WallAssets()
    # tex 1 = BROWN1 (native 128 wide); u=128 wraps to native column 0 via the
    # per-bank floor-mod sawtooth.
    brown1 = BOOK.wall_textures[0]
    assert _scalar(
        walls.palette_index(constant(1.0), constant(128.0), _v_mods(0.0))
    ) == float(brown1.pixels[0][0])
    assert _scalar(
        walls.palette_index(constant(1.0), constant(128.0), _v_mods(10.0))
    ) == float(brown1.pixels[0][10])
    # u in range (no wrap), and a wrap past one period (u = 133 -> column 5).
    assert _scalar(
        walls.palette_index(constant(1.0), constant(5.0), _v_mods(3.0))
    ) == float(brown1.pixels[5][3])
    assert _scalar(
        walls.palette_index(constant(1.0), constant(133.0), _v_mods(3.0))
    ) == float(brown1.pixels[5][3])
    # tex 3 = COMPUTE2, a different (256x56) bank, to exercise the bank_mask.
    comp = BOOK.wall_textures[2]
    assert _scalar(
        walls.palette_index(constant(3.0), constant(10.0), _v_mods(4.0))
    ) == float(comp.pixels[10][4])


def test_wall_missing_texture_returns_zero():
    # tex 0 = MISSING_TEXTURE_ID routes to bank 0 (the 1x1x1 zero bank).
    walls = WallAssets()
    assert (
        abs(_scalar(walls.palette_index(constant(0.0), constant(7.0), _v_mods(2.0))))
        < ATOL
    )


def test_wall_metadata_accessors():
    walls = WallAssets()
    # BROWN1 (global id 1) lives at local 0 of the 128x128 bank (bank 6).
    assert _scalar(walls.bank_id(constant(1.0))) == 6.0
    assert _scalar(walls.local_id(constant(1.0))) == 0.0
    assert _scalar(walls.width(constant(1.0))) == 128.0
    # Texture height and height-bank identity.
    assert _scalar(walls.height(constant(1.0))) == 128.0
    # h_idx_oh: WALL_HEIGHT_BANK = (16, 56, 72, 128); height 128 -> index 3.
    assert ab.WALL_HEIGHT_BANK == (16, 56, 72, 128)
    h_oh = _vec(walls.h_idx_oh(constant(1.0)))
    assert torch.allclose(h_oh, torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=ATOL)


def test_flat_palette_index_and_is_sky():
    flats = FlatAssets()
    for flat_id in range(ab.N_FLATS):
        flat = BOOK.flat_textures[flat_id]
        got = _scalar(
            flats.palette_index(constant(float(flat_id)), constant(9.0), constant(13.0))
        )
        assert abs(got - float(flat.pixels[9][13])) < ATOL, flat.name
        # F_SKY1 (flat 5) is the only sky flat; is_sky returns the ±1 boolean.
        is_sky = _scalar(flats.is_sky(constant(float(flat_id))))
        expected = 1.0 if flat.name.upper().startswith("F_SKY") else -1.0
        assert abs(is_sky - expected) < ATOL, flat.name


def test_apply_colormap_row_matches_reference():
    # row 0 is the identity (full bright); higher rows darken. Cross-check the
    # in-graph application against the pure-Python apply_doom_colormap.
    cases = [(0, 0), (139, 0), (139, 8), (100, 16), (255, 31), (42, 1)]
    for raw, row in cases:
        got = _scalar(apply_colormap_row(constant(float(raw)), constant(float(row))))
        expected = float(ab.COLORMAP_ROWS[row][raw])
        assert abs(got - expected) < ATOL, (raw, row)
        # same value the ported row-selection helper would apply.
        assert expected == float(apply_doom_colormap(BOOK.colormap, row, raw))
    assert NUMCOLORMAPS == 32


def test_assets_exposed_through_scene_index():
    # SceneIndex.assets is the working constant-backed AssetIndex surface.
    assert SceneIndex.__dataclass_fields__["assets"].type == "AssetIndex"
    index = AssetIndex()
    assert isinstance(index.walls, WallAssets)
    assert isinstance(index.flats, FlatAssets)
    # the surface is callable and matches the WAD through SceneIndex's field type.
    brown1 = BOOK.wall_textures[0]
    assert _scalar(
        index.walls.palette_index(constant(1.0), constant(0.0), _v_mods(0.0))
    ) == float(brown1.pixels[0][0])


def test_no_import_time_graph_nodes():
    # The no-import-time-nodes rule: importing the asset modules must not build
    # any graph node (module-level constant(...) would alias under the conftest
    # node-id reset). Checked in a clean interpreter so the in-test node counter
    # does not mask leakage.
    code = (
        "import torchwright_doom.wad_assets, torchwright_doom.asset_banks, "
        "torchwright_doom.assets, torchwright_doom.lighting, torchwright_doom.std, "
        "torchwright_doom.scene_index;"
        "from torchwright.graph import node as n;"
        "print(n.global_node_id)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "0", out.stdout + out.stderr
