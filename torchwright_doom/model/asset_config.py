"""Minimal compiled-asset surface for the prefill vocab + prompt builder.

This module runs once at compile time: it builds part of the computation
graph that torchwright lowers into the transformer's weights. Nothing here
executes during inference — at render time, only the compiled transformer
runs. Coined terms: see GLOSSARY.md.

The prefill contract is asset-coupled in three places:
``PLANE_DEF.flat_id`` ranges over ``N_FLATS``, ``seg.texture.*`` over
``N_WALL_TEXTURES + 1``, and ``PIXEL.color`` carries ``pixel_r/g/b``
derived columns read off the 256-entry ``PLAYPAL`` palette. The prompt
builder also needs the texture/flat **name -> id** maps to emit the same
ids the pydoom reference renderer uses.

This module holds only the WAD-independent name/id/palette data the vocab
and prompt builder need: the texture/flat name lists, the id maps, the
counts, and a snapshot of ``PLAYPAL`` (the static DOOM1 palette pydoom
loads from the WAD — copied as a literal so the prefill side does not pull
in the WAD-loading machinery). The forward-path pixel and dimension banks
(wall/flat pixel tables, ``WALL_HEIGHT_BANK`` etc., the ``table_lookup``
data) live in ``model/assets/asset_banks.py`` and its ``table_lookup``
consumers, NOT here.
"""

from __future__ import annotations

from dataclasses import dataclass

WALL_TEXTURE_NAMES: tuple[str, ...] = (
    "BROWN1",
    "BROWN144",
    "COMPUTE2",
    "DOOR3",
    "DOORSTOP",
    "LITE3",
    "STARTAN3",
    "STEP6",
    "SUPPORT2",
)

FLAT_NAMES: tuple[str, ...] = (
    "CEIL3_5",
    "FLAT14",
    "FLAT5_5",
    "FLOOR4_8",
    "FLOOR7_1",
    "F_SKY1",
)

MISSING_TEXTURE_ID = 0
WALL_TEXTURE_ID_BY_NAME = {name: idx + 1 for idx, name in enumerate(WALL_TEXTURE_NAMES)}
FLAT_ID_BY_NAME = {name: idx for idx, name in enumerate(FLAT_NAMES)}

N_WALL_TEXTURES = len(WALL_TEXTURE_NAMES)
N_FLATS = len(FLAT_NAMES)


@dataclass(frozen=True)
class AssetConfig:
    """Ordered wall/flat names for one compiled renderer artifact."""

    wall_names: tuple[str, ...] = WALL_TEXTURE_NAMES
    flat_names: tuple[str, ...] = FLAT_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "wall_names", tuple(name.upper() for name in self.wall_names)
        )
        object.__setattr__(
            self, "flat_names", tuple(name.upper() for name in self.flat_names)
        )
        if len(set(self.wall_names)) != len(self.wall_names):
            raise ValueError(f"duplicate wall texture names: {self.wall_names!r}")
        if len(set(self.flat_names)) != len(self.flat_names):
            raise ValueError(f"duplicate flat names: {self.flat_names!r}")

    @property
    def wall_id_by_name(self) -> dict[str, int]:
        return {name: idx + 1 for idx, name in enumerate(self.wall_names)}

    @property
    def flat_id_by_name(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.flat_names)}

    @property
    def n_wall_textures(self) -> int:
        return len(self.wall_names)

    @property
    def n_flats(self) -> int:
        return len(self.flat_names)

    def wall_name_by_id(self) -> dict[int, str]:
        return {idx + 1: name for idx, name in enumerate(self.wall_names)}

    def flat_name_by_id(self) -> dict[int, str]:
        return {idx: name for idx, name in enumerate(self.flat_names)}


DEFAULT_ASSET_CONFIG = AssetConfig()

# Static DOOM1 PLAYPAL (256 RGB triples) — snapshot of pydoom's
# WAD-loaded palette (asset_banks.PLAYPAL; checksum sum-of-channels =
# 83712). Used by PIXEL.color's pixel_r/g/b derived columns.
PLAYPAL: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (31, 23, 11),
    (23, 15, 7),
    (75, 75, 75),
    (255, 255, 255),
    (27, 27, 27),
    (19, 19, 19),
    (11, 11, 11),
    (7, 7, 7),
    (47, 55, 31),
    (35, 43, 15),
    (23, 31, 7),
    (15, 23, 0),
    (79, 59, 43),
    (71, 51, 35),
    (63, 43, 27),
    (255, 183, 183),
    (247, 171, 171),
    (243, 163, 163),
    (235, 151, 151),
    (231, 143, 143),
    (223, 135, 135),
    (219, 123, 123),
    (211, 115, 115),
    (203, 107, 107),
    (199, 99, 99),
    (191, 91, 91),
    (187, 87, 87),
    (179, 79, 79),
    (175, 71, 71),
    (167, 63, 63),
    (163, 59, 59),
    (155, 51, 51),
    (151, 47, 47),
    (143, 43, 43),
    (139, 35, 35),
    (131, 31, 31),
    (127, 27, 27),
    (119, 23, 23),
    (115, 19, 19),
    (107, 15, 15),
    (103, 11, 11),
    (95, 7, 7),
    (91, 7, 7),
    (83, 7, 7),
    (79, 0, 0),
    (71, 0, 0),
    (67, 0, 0),
    (255, 235, 223),
    (255, 227, 211),
    (255, 219, 199),
    (255, 211, 187),
    (255, 207, 179),
    (255, 199, 167),
    (255, 191, 155),
    (255, 187, 147),
    (255, 179, 131),
    (247, 171, 123),
    (239, 163, 115),
    (231, 155, 107),
    (223, 147, 99),
    (215, 139, 91),
    (207, 131, 83),
    (203, 127, 79),
    (191, 123, 75),
    (179, 115, 71),
    (171, 111, 67),
    (163, 107, 63),
    (155, 99, 59),
    (143, 95, 55),
    (135, 87, 51),
    (127, 83, 47),
    (119, 79, 43),
    (107, 71, 39),
    (95, 67, 35),
    (83, 63, 31),
    (75, 55, 27),
    (63, 47, 23),
    (51, 43, 19),
    (43, 35, 15),
    (239, 239, 239),
    (231, 231, 231),
    (223, 223, 223),
    (219, 219, 219),
    (211, 211, 211),
    (203, 203, 203),
    (199, 199, 199),
    (191, 191, 191),
    (183, 183, 183),
    (179, 179, 179),
    (171, 171, 171),
    (167, 167, 167),
    (159, 159, 159),
    (151, 151, 151),
    (147, 147, 147),
    (139, 139, 139),
    (131, 131, 131),
    (127, 127, 127),
    (119, 119, 119),
    (111, 111, 111),
    (107, 107, 107),
    (99, 99, 99),
    (91, 91, 91),
    (87, 87, 87),
    (79, 79, 79),
    (71, 71, 71),
    (67, 67, 67),
    (59, 59, 59),
    (55, 55, 55),
    (47, 47, 47),
    (39, 39, 39),
    (35, 35, 35),
    (119, 255, 111),
    (111, 239, 103),
    (103, 223, 95),
    (95, 207, 87),
    (91, 191, 79),
    (83, 175, 71),
    (75, 159, 63),
    (67, 147, 55),
    (63, 131, 47),
    (55, 115, 43),
    (47, 99, 35),
    (39, 83, 27),
    (31, 67, 23),
    (23, 51, 15),
    (19, 35, 11),
    (11, 23, 7),
    (191, 167, 143),
    (183, 159, 135),
    (175, 151, 127),
    (167, 143, 119),
    (159, 135, 111),
    (155, 127, 107),
    (147, 123, 99),
    (139, 115, 91),
    (131, 107, 87),
    (123, 99, 79),
    (119, 95, 75),
    (111, 87, 67),
    (103, 83, 63),
    (95, 75, 55),
    (87, 67, 51),
    (83, 63, 47),
    (159, 131, 99),
    (143, 119, 83),
    (131, 107, 75),
    (119, 95, 63),
    (103, 83, 51),
    (91, 71, 43),
    (79, 59, 35),
    (67, 51, 27),
    (123, 127, 99),
    (111, 115, 87),
    (103, 107, 79),
    (91, 99, 71),
    (83, 87, 59),
    (71, 79, 51),
    (63, 71, 43),
    (55, 63, 39),
    (255, 255, 115),
    (235, 219, 87),
    (215, 187, 67),
    (195, 155, 47),
    (175, 123, 31),
    (155, 91, 19),
    (135, 67, 7),
    (115, 43, 0),
    (255, 255, 255),
    (255, 219, 219),
    (255, 187, 187),
    (255, 155, 155),
    (255, 123, 123),
    (255, 95, 95),
    (255, 63, 63),
    (255, 31, 31),
    (255, 0, 0),
    (239, 0, 0),
    (227, 0, 0),
    (215, 0, 0),
    (203, 0, 0),
    (191, 0, 0),
    (179, 0, 0),
    (167, 0, 0),
    (155, 0, 0),
    (139, 0, 0),
    (127, 0, 0),
    (115, 0, 0),
    (103, 0, 0),
    (91, 0, 0),
    (79, 0, 0),
    (67, 0, 0),
    (231, 231, 255),
    (199, 199, 255),
    (171, 171, 255),
    (143, 143, 255),
    (115, 115, 255),
    (83, 83, 255),
    (55, 55, 255),
    (27, 27, 255),
    (0, 0, 255),
    (0, 0, 227),
    (0, 0, 203),
    (0, 0, 179),
    (0, 0, 155),
    (0, 0, 131),
    (0, 0, 107),
    (0, 0, 83),
    (255, 255, 255),
    (255, 235, 219),
    (255, 215, 187),
    (255, 199, 155),
    (255, 179, 123),
    (255, 163, 91),
    (255, 143, 59),
    (255, 127, 27),
    (243, 115, 23),
    (235, 111, 15),
    (223, 103, 15),
    (215, 95, 11),
    (203, 87, 7),
    (195, 79, 0),
    (183, 71, 0),
    (175, 67, 0),
    (255, 255, 255),
    (255, 255, 215),
    (255, 255, 179),
    (255, 255, 143),
    (255, 255, 107),
    (255, 255, 71),
    (255, 255, 35),
    (255, 255, 0),
    (167, 63, 0),
    (159, 55, 0),
    (147, 47, 0),
    (135, 35, 0),
    (79, 59, 39),
    (67, 47, 27),
    (55, 35, 19),
    (47, 27, 11),
    (0, 0, 83),
    (0, 0, 71),
    (0, 0, 59),
    (0, 0, 47),
    (0, 0, 35),
    (0, 0, 23),
    (0, 0, 11),
    (0, 0, 0),
    (255, 159, 67),
    (255, 231, 75),
    (255, 123, 255),
    (255, 0, 255),
    (207, 0, 207),
    (159, 0, 155),
    (111, 0, 107),
    (167, 107, 107),
)
