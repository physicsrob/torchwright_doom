"""Renumbered, mean-centred DOOM map geometry + per-frame player state.

The :class:`MapData` schema mirrors what DOOM's reference renderer
consumes after WAD ingestion: dense vertex / linedef / sidedef /
sector / subsector / seg / node indices with valid cross-references.
A WAD-loaded :class:`MapData` (raw) carries integer coords and
``scene_origin == (0.0, 0.0)``; the subset step (see
:mod:`.subset`) renumbers, mean-centres, and stores the centroid in
``scene_origin``.

Cross-references (``Linedef.front_sidedef``, ``Sidedef.sector``,
``Seg.linedef``, ``Subsector.first_seg``, ``BspNode.front_child`` /
``back_child``) are dense indices into the corresponding lists,
valid in the renumbered numbering.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SUBSECTOR_FLAG = 0x8000


class Vertex(BaseModel):
    model_config = ConfigDict(extra="ignore")

    x: float
    y: float


class Linedef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    v1: int
    v2: int
    flags: int
    special: int
    tag: int
    front_sidedef: int
    back_sidedef: int


class Sidedef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    x_offset: int
    y_offset: int
    upper: str
    lower: str
    middle: str
    sector: int


class Sector(BaseModel):
    model_config = ConfigDict(extra="ignore")

    floor_h: float
    ceiling_h: float
    floor_tex: str
    ceiling_tex: str
    light: int
    special: int
    tag: int


class Seg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    v1: int
    v2: int
    angle: int
    linedef: int
    side: int
    offset: int


class Subsector(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seg_count: int
    first_seg: int


class BspNode(BaseModel):
    """High bit of ``front_child`` / ``back_child`` is :data:`SUBSECTOR_FLAG`."""

    model_config = ConfigDict(extra="ignore")

    px: float
    py: float
    dx: float
    dy: float
    front_bbox: tuple[float, float, float, float]
    back_bbox: tuple[float, float, float, float]
    front_child: int
    back_child: int


class Thing(BaseModel):
    model_config = ConfigDict(extra="ignore")

    x: float
    y: float
    angle: int
    type: int
    flags: int


class MapData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    vertices: list[Vertex]
    linedefs: list[Linedef]
    sidedefs: list[Sidedef]
    sectors: list[Sector]
    segs: list[Seg]
    subsectors: list[Subsector]
    nodes: list[BspNode]
    things: list[Thing] = []
    scene_origin: tuple[float, float] = (0.0, 0.0)


class GameState(BaseModel):
    """Player state in subset (mean-centred) coordinates."""

    model_config = ConfigDict(extra="ignore")

    x: float
    y: float
    angle: int = Field(ge=0, lt=256)
    viewz: float = 41.0
