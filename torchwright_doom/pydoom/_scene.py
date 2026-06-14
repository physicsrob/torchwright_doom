"""Vendored value objects the Python renderer + drafter read.

These are the pieces of the original ``types`` / ``api`` that have no
``torchwright_doom`` equivalent: the ``Scene`` wrapper (it carries ``.map_data``
and ``.test_poses``, which the native scene does not), its ``TextureImage`` and
``GameState`` members (the native ``GameState`` drops ``move_speed`` /
``turn_speed``), and the decoded ``Pixel``.

The heavy map types (``MapData`` / ``Sector`` / ``Seg`` / ``SUBSECTOR_FLAG``)
are deliberately NOT vendored — they retarget to ``..prompt.types``, a proven
field-identical mirror (``inference/wad_scene`` already round-trips
native<->pydoom through ``model_dump``). So one ``MapData`` threads through the
renderer, ``bake_segments``, and the BSP scalars.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from ..prompt.types import MapData


class GameState(BaseModel):
    """Per-frame player state. Vendored rather than retargeted because the
    native ``prompt.types.GameState`` drops ``move_speed`` / ``turn_speed``."""

    model_config = ConfigDict(extra="ignore")

    x: float
    y: float
    angle: int = Field(ge=0, lt=256)
    viewz: float = 41.0
    move_speed: float = 0.3
    turn_speed: int = 4


class TextureImage(BaseModel):
    """A committed WAD image asset; ``pixels[x][y] == palette_index`` (column
    major). Wall textures keep native WAD dims; flats are 64x64."""

    model_config = ConfigDict(extra="ignore")

    name: str
    width: int
    height: int
    pixels: list[list[int]]


class Scene(BaseModel):
    """A loaded fixture: map data, test poses, and optional WAD assets.

    Textured fixtures carry a 256-entry RGB palette and a 33x256 colormap;
    geometry-only fixtures leave these empty.
    """

    model_config = ConfigDict(extra="ignore")

    map_data: MapData
    test_poses: list[GameState] = []
    wall_textures: list[TextureImage] = []
    flat_textures: list[TextureImage] = []
    palette: list[tuple[int, int, int]] = []
    colormap: list[list[int]] = []


@dataclass(frozen=True)
class Pixel:
    """A pixel emission decoded from a render-emitting position."""

    x: int
    y: int
    color: tuple[int, int, int]  # RGB
