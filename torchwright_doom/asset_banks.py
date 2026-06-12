"""Compile-time WAD asset banks for the texture/flat read surface.

Real-side port of ``doom_sandbox/implementation/asset_banks.py``. Builds the
per-``(width, height)`` wall banks, the per-texture metadata tables, the flat
table, and the COLORMAP rows from the WAD-loaded ``ASSET_BOOK`` (data-source
**B**).

Everything here is **plain numpy / Python data, never a graph node** — the
no-import-time-nodes rule forbids module-level ``constant(...)`` / op nodes
(they alias under the test-suite node-id reset). ``table_lookup_2d`` consumes
the raw numpy ``table`` arrays directly; the ``WALL_TEX_*`` metadata tuples are
wrapped in ``constant(...)`` only *inside* the accessor methods in
:mod:`.assets`. Importing this module leaves ``global_node_id == 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asset_config import FLAT_NAMES, WALL_TEXTURE_NAMES
from .doom_lighting import NUMCOLORMAPS
from .wad_assets import (
    DOOM1_WAD_PATH,
    FLAT_SIZE,
    AssetBook,
    TextureImage,
    load_asset_book,
)


def _is_sky_flat(flat_name: str) -> bool:
    name = flat_name.upper()
    return name.startswith("F_SKY")


@dataclass(frozen=True)
class WallBank:
    """Native-size wall textures sharing one table shape."""

    bank_id: int
    width: int
    height: int
    global_ids: tuple[int, ...]
    names: tuple[str, ...]
    # table[local_id, v, u] -> palette index
    table: np.ndarray


def _wall_table(textures) -> np.ndarray:
    height = textures[0].height
    width = textures[0].width
    table = np.zeros((len(textures), height, width), dtype=np.float32)
    for local_id, texture in enumerate(textures):
        for u in range(width):
            for v in range(height):
                table[local_id, v, u] = float(texture.pixels[u][v])
    return table


@dataclass(frozen=True)
class AssetBanks:
    """All raw texture/flat tables for one ordered asset set."""

    asset_book: AssetBook
    wall_names: tuple[str, ...]
    flat_names: tuple[str, ...]
    playpal: tuple[tuple[int, int, int], ...]
    colormap_rows: tuple[tuple[int, ...], ...]
    wall_banks: tuple[WallBank, ...]
    wall_height_bank: tuple[int, ...]
    wall_width_bank: tuple[int, ...]
    wall_max_width: int
    wall_max_height: int
    n_wall_textures: int
    wall_tex_bank_id: tuple[float, ...]
    wall_tex_local_id: tuple[float, ...]
    wall_tex_width: tuple[float, ...]
    wall_tex_height: tuple[float, ...]
    wall_tex_h_idx_oh: tuple[float, ...]
    flat_table: np.ndarray
    flat_is_sky: tuple[float, ...]
    n_flats: int
    wall_local_id_values_by_bank: tuple[list[float], ...]
    wall_bank_table_2d: tuple[np.ndarray, ...]
    wall_bank_row_addr: tuple[list[list[float]], ...]
    flat_id_values: list[float]
    flat_table_2d: np.ndarray
    flat_row_addr: list[list[float]]


def _build_wall_banks(asset_book: AssetBook) -> tuple[WallBank, ...]:
    by_size: dict[tuple[int, int], list[tuple[int, "TextureImage"]]] = {}
    for global_id, texture in enumerate(asset_book.wall_textures, start=1):
        by_size.setdefault((texture.width, texture.height), []).append(
            (global_id, texture)
        )

    banks: list[WallBank] = [
        WallBank(
            bank_id=0,
            width=1,
            height=1,
            global_ids=(0,),
            names=("<missing>",),
            table=np.zeros((1, 1, 1), dtype=np.float32),
        )
    ]
    for bank_id, ((width, height), entries) in enumerate(
        sorted(by_size.items()),
        start=1,
    ):
        textures = [texture for _global_id, texture in entries]
        banks.append(
            WallBank(
                bank_id=bank_id,
                width=width,
                height=height,
                global_ids=tuple(global_id for global_id, _texture in entries),
                names=tuple(texture.name for texture in textures),
                table=_wall_table(textures),
            )
        )
    return tuple(banks)


def _build_wall_metadata(
    wall_banks: tuple[WallBank, ...],
    wall_height_bank: tuple[int, ...],
    n_wall_textures: int,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    bank_by_global: dict[int, tuple[int, int, int, int]] = {
        0: (0, 0, 0, 0),
    }
    for bank in wall_banks:
        for local_id, global_id in enumerate(bank.global_ids):
            bank_by_global[global_id] = (
                bank.bank_id,
                local_id,
                bank.width,
                bank.height,
            )

    bank_ids: list[float] = []
    local_ids: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    h_idx_oh: list[float] = []
    for global_id in range(n_wall_textures + 1):
        bank_id, local_id, width, height = bank_by_global[global_id]
        bank_ids.append(float(bank_id))
        local_ids.append(float(local_id))
        widths.append(float(width))
        heights.append(float(height))
        h_idx = wall_height_bank.index(height) if height in wall_height_bank else -1
        h_idx_oh.extend(
            1.0 if h_idx == idx else 0.0 for idx in range(len(wall_height_bank))
        )
    return (
        tuple(bank_ids),
        tuple(local_ids),
        tuple(widths),
        tuple(heights),
        tuple(h_idx_oh),
    )


def _build_flat_table(asset_book: AssetBook) -> np.ndarray:
    flat_table = np.zeros(
        (len(asset_book.flat_textures), FLAT_SIZE, FLAT_SIZE), dtype=np.float32
    )
    for flat_id, texture in enumerate(asset_book.flat_textures):
        if texture.width != FLAT_SIZE or texture.height != FLAT_SIZE:
            raise ValueError(
                f"flat {texture.name!r} is {texture.width}x{texture.height}, "
                "expected 64x64"
            )
        for u in range(FLAT_SIZE):
            for v in range(FLAT_SIZE):
                flat_table[flat_id, v, u] = float(texture.pixels[u][v])
    return flat_table


def build_asset_banks(
    asset_book: AssetBook | None = None,
    *,
    wad_path=DOOM1_WAD_PATH,
    wall_names: tuple[str, ...] | list[str] = WALL_TEXTURE_NAMES,
    flat_names: tuple[str, ...] | list[str] = FLAT_NAMES,
) -> AssetBanks:
    """Build all compile-time lookup banks for an ordered asset set."""
    wall_names = tuple(name.upper() for name in wall_names)
    flat_names = tuple(name.upper() for name in flat_names)
    if asset_book is None:
        wad_path = DOOM1_WAD_PATH if wad_path is None else wad_path
        asset_book = load_asset_book(
            wad_path,
            wall_texture_names=wall_names,
            flat_names=flat_names,
        )

    wall_banks = _build_wall_banks(asset_book)
    wall_height_bank = tuple(
        sorted({texture.height for texture in asset_book.wall_textures})
    )
    wall_width_bank = tuple(
        sorted({texture.width for texture in asset_book.wall_textures})
    )
    wall_max_width = max(wall_width_bank) if wall_width_bank else 0
    wall_max_height = max(wall_height_bank) if wall_height_bank else 0
    n_wall_textures = len(wall_names)
    (
        wall_tex_bank_id,
        wall_tex_local_id,
        wall_tex_width,
        wall_tex_height,
        wall_tex_h_idx_oh,
    ) = _build_wall_metadata(wall_banks, wall_height_bank, n_wall_textures)

    flat_table = _build_flat_table(asset_book)
    flat_is_sky = tuple(1.0 if _is_sky_flat(name) else 0.0 for name in flat_names)
    n_flats = len(flat_names)

    return AssetBanks(
        asset_book=asset_book,
        wall_names=wall_names,
        flat_names=flat_names,
        playpal=tuple((int(r), int(g), int(b)) for r, g, b in asset_book.palette),
        colormap_rows=tuple(
            tuple(int(v) for v in row) for row in asset_book.colormap[:NUMCOLORMAPS]
        ),
        wall_banks=wall_banks,
        wall_height_bank=wall_height_bank,
        wall_width_bank=wall_width_bank,
        wall_max_width=wall_max_width,
        wall_max_height=wall_max_height,
        n_wall_textures=n_wall_textures,
        wall_tex_bank_id=wall_tex_bank_id,
        wall_tex_local_id=wall_tex_local_id,
        wall_tex_width=wall_tex_width,
        wall_tex_height=wall_tex_height,
        wall_tex_h_idx_oh=wall_tex_h_idx_oh,
        flat_table=flat_table,
        flat_is_sky=flat_is_sky,
        n_flats=n_flats,
        wall_local_id_values_by_bank=tuple(
            [float(i) for i in range(len(bank.global_ids))] for bank in wall_banks
        ),
        wall_bank_table_2d=tuple(
            bank.table.reshape(len(bank.global_ids) * bank.height, bank.width)
            for bank in wall_banks
        ),
        wall_bank_row_addr=tuple([[float(bank.height)], [1.0]] for bank in wall_banks),
        flat_id_values=[float(i) for i in range(n_flats)],
        flat_table_2d=flat_table.reshape(n_flats * FLAT_SIZE, FLAT_SIZE),
        flat_row_addr=[[float(FLAT_SIZE)], [1.0]],
    )


DEFAULT_ASSET_BANKS = build_asset_banks()

ASSET_BOOK = DEFAULT_ASSET_BANKS.asset_book
PLAYPAL = DEFAULT_ASSET_BANKS.playpal
# COLORMAP has NUMCOLORMAPS light maps (rows 0..NUMCOLORMAPS-1) followed by the
# invulnerability map; the renderer only uses the light maps.
COLORMAP_ROWS = DEFAULT_ASSET_BANKS.colormap_rows
WALL_BANKS = DEFAULT_ASSET_BANKS.wall_banks
WALL_HEIGHT_BANK = DEFAULT_ASSET_BANKS.wall_height_bank
WALL_WIDTH_BANK = DEFAULT_ASSET_BANKS.wall_width_bank
WALL_MAX_WIDTH = DEFAULT_ASSET_BANKS.wall_max_width
WALL_MAX_HEIGHT = DEFAULT_ASSET_BANKS.wall_max_height
N_WALL_TEXTURES = DEFAULT_ASSET_BANKS.n_wall_textures
WALL_TEX_BANK_ID = DEFAULT_ASSET_BANKS.wall_tex_bank_id
WALL_TEX_LOCAL_ID = DEFAULT_ASSET_BANKS.wall_tex_local_id
WALL_TEX_WIDTH = DEFAULT_ASSET_BANKS.wall_tex_width
WALL_TEX_HEIGHT = DEFAULT_ASSET_BANKS.wall_tex_height
WALL_TEX_H_IDX_OH = DEFAULT_ASSET_BANKS.wall_tex_h_idx_oh
FLAT_TABLE = DEFAULT_ASSET_BANKS.flat_table
FLAT_IS_SKY = DEFAULT_ASSET_BANKS.flat_is_sky
N_FLATS = DEFAULT_ASSET_BANKS.n_flats
