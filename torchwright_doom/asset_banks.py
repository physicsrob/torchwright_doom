"""Compile-time WAD asset banks for the texture/flat read surface (Plan I1).

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
from .wad_assets import load_asset_book

ASSET_BOOK = load_asset_book()
PLAYPAL = tuple(tuple(int(c) for c in rgb) for rgb in ASSET_BOOK.palette)
COLORMAP_ROWS = tuple(tuple(int(v) for v in row) for row in ASSET_BOOK.colormap[:32])


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


def _build_wall_banks() -> tuple[WallBank, ...]:
    by_size: dict[tuple[int, int], list[tuple[int, object]]] = {}
    for global_id, texture in enumerate(ASSET_BOOK.wall_textures, start=1):
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


WALL_BANKS = _build_wall_banks()
WALL_HEIGHT_BANK = tuple(
    sorted({texture.height for texture in ASSET_BOOK.wall_textures})
)
WALL_WIDTH_BANK = tuple(sorted({texture.width for texture in ASSET_BOOK.wall_textures}))
WALL_MAX_WIDTH = max(WALL_WIDTH_BANK)
WALL_MAX_HEIGHT = max(WALL_HEIGHT_BANK)
N_WALL_TEXTURES = len(WALL_TEXTURE_NAMES)


def _build_wall_metadata() -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    bank_by_global: dict[int, tuple[int, int, int, int]] = {
        0: (0, 0, 0, 0),
    }
    for bank in WALL_BANKS:
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
    for global_id in range(N_WALL_TEXTURES + 1):
        bank_id, local_id, width, height = bank_by_global[global_id]
        bank_ids.append(float(bank_id))
        local_ids.append(float(local_id))
        widths.append(float(width))
        heights.append(float(height))
        h_idx = WALL_HEIGHT_BANK.index(height) if height in WALL_HEIGHT_BANK else -1
        h_idx_oh.extend(
            1.0 if h_idx == idx else 0.0 for idx in range(len(WALL_HEIGHT_BANK))
        )
    return (
        tuple(bank_ids),
        tuple(local_ids),
        tuple(widths),
        tuple(heights),
        tuple(h_idx_oh),
    )


(
    WALL_TEX_BANK_ID,
    WALL_TEX_LOCAL_ID,
    WALL_TEX_WIDTH,
    WALL_TEX_HEIGHT,
    WALL_TEX_H_IDX_OH,
) = _build_wall_metadata()


FLAT_TABLE = np.zeros((len(FLAT_NAMES), 64, 64), dtype=np.float32)
for flat_id, texture in enumerate(ASSET_BOOK.flat_textures):
    if texture.width != 64 or texture.height != 64:
        raise ValueError(
            f"flat {texture.name!r} is {texture.width}x{texture.height}, "
            "expected 64x64"
        )
    for u in range(64):
        for v in range(64):
            FLAT_TABLE[flat_id, v, u] = float(texture.pixels[u][v])

FLAT_IS_SKY = tuple(1.0 if _is_sky_flat(name) else 0.0 for name in FLAT_NAMES)
N_FLATS = len(FLAT_NAMES)
