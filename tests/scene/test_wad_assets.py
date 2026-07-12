"""WAD-backed asset loading — real DOOM1.WAD byte anchors.

Pinned against the in-tree
``asset_banks.ASSET_BOOK`` (the module-level book the real graph reads, loaded
from the committed ``doom1.wad`` with the compiled allowlist). After the
renderer/drafter were vendored, this is the only test that pins absolute WAD
byte values — palette entries, colormap rows, native texture dimensions —
rather than internal consistency, so it anchors the loader against real game
bytes.
"""

from __future__ import annotations

from pathlib import Path

from torchwright_doom import asset_banks as ab
from torchwright_doom.asset_config import FLAT_NAMES, WALL_TEXTURE_NAMES
from torchwright_doom.config import load_render_config
from torchwright_doom.prompt.scene import load_render_scene

_SUBMODULE_ROOT = Path(__file__).resolve().parents[2]


def test_wad_asset_book_loads_start_room_allowlist():
    book = ab.ASSET_BOOK

    assert [texture.name for texture in book.wall_textures] == list(WALL_TEXTURE_NAMES)
    assert [texture.name for texture in book.flat_textures] == list(FLAT_NAMES)
    assert len(book.palette) == 256
    assert len(book.colormap) == 33
    assert all(len(row) == 256 for row in book.colormap)


def test_wad_wall_textures_keep_native_dimensions():
    expected_shapes = {
        "BROWN1": (128, 128),
        "BROWN144": (128, 128),
        "COMPUTE2": (256, 56),
        "DOOR3": (64, 72),
        "DOORSTOP": (8, 128),
        "LITE3": (32, 128),
        "STARTAN3": (128, 128),
        "STEP6": (32, 16),
        "SUPPORT2": (64, 128),
    }
    book = ab.ASSET_BOOK

    assert {
        texture.name: (texture.width, texture.height) for texture in book.wall_textures
    } == expected_shapes
    for texture in book.wall_textures:
        _assert_palette_texture_shape(texture)


def test_wad_flats_palette_and_colormap_load_from_doom1_wad():
    book = ab.ASSET_BOOK

    assert {
        texture.name: (texture.width, texture.height) for texture in book.flat_textures
    } == {name: (64, 64) for name in FLAT_NAMES}
    for texture in book.flat_textures:
        _assert_palette_texture_shape(texture)

    assert tuple(tuple(c) for c in book.palette[:5]) == (
        (0, 0, 0),
        (31, 23, 11),
        (23, 15, 7),
        (75, 75, 75),
        (255, 255, 255),
    )
    assert book.colormap[0][247] == 0
    assert book.colormap[31][255] == 8
    assert book.colormap[32][0] == 4
    assert any(
        book.colormap[row][palette_index] != palette_index
        for row in range(33)
        for palette_index in range(256)
    )


def test_start_room_map_asset_names_are_in_compiled_allowlist():
    config = load_render_config(_SUBMODULE_ROOT / "configs" / "e1m1.yaml")
    scene = load_render_scene(config, base_dir=_SUBMODULE_ROOT)
    map_data = scene.map_data

    wall_names = {
        name
        for sidedef in map_data.sidedefs
        for name in (sidedef.upper, sidedef.lower, sidedef.middle)
        if name and name != "-"
    }
    flat_names = {
        name
        for sector in map_data.sectors
        for name in (sector.floor_tex, sector.ceiling_tex)
        if name and name != "-"
    }

    assert wall_names <= set(WALL_TEXTURE_NAMES)
    assert flat_names <= set(FLAT_NAMES)


def _assert_palette_texture_shape(texture) -> None:
    assert len(texture.pixels) == texture.width
    assert texture.width > 0
    assert texture.height > 0
    for column in texture.pixels:
        assert len(column) == texture.height
        assert all(0 <= palette_index < 256 for palette_index in column)
