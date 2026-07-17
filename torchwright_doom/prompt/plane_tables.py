"""Canonical visplane tables for a :class:`MapData`.

Host-side prompt preparation (view-independent — the plane list depends
only on map data; see the input-side boundary in README). Deduplicates
floors and ceilings into a stable plane list and tags each subsector
with its floor/ceiling plane ID. Sole consumer: the prompt builder
(``build.py``), which emits these as the prefill's ``planeDef`` /
``ssFloorPlane`` / ``ssCeilingPlane`` tokens — the in-model plane-marking
stage then reads those *tokens*. The pydoom oracle keeps its own private
``_build_plane_tables`` (``pydoom/renderer.py``) with the same dedup
rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.asset_config import FLAT_ID_BY_NAME
from .types import MapData, Sector


@dataclass(frozen=True)
class PlaneRecord:
    plane_id: int
    height: float
    flat_id: int
    light: int
    is_sky: int


@dataclass(frozen=True)
class SubsectorPlaneInfo:
    floor_plane_id: int
    ceiling_plane_id: int


@dataclass(frozen=True)
class PlaneTables:
    planes: list[PlaneRecord]
    subsectors: list[SubsectorPlaneInfo | None]


@dataclass(frozen=True)
class _PlaneKey:
    height: float
    flat_id: int
    light: int
    is_sky: int


def is_sky_flat(flat_name: str) -> bool:
    name = flat_name.upper()
    return name == "F_SKY1" or name.startswith("F_SKY")


def _plane_key(
    height: float, flat_name: str, light: int, flat_ids: dict[str, int]
) -> _PlaneKey:
    sky = is_sky_flat(flat_name)
    return _PlaneKey(
        height=0.0 if sky else float(height),
        flat_id=flat_ids[flat_name],
        light=0 if sky else int(light),
        is_sky=1 if sky else 0,
    )


def _subsector_front_sector(md: MapData, s: int) -> Sector | None:
    sub = md.subsectors[s]
    if sub.seg_count <= 0:
        return None
    raw_seg = md.segs[sub.first_seg]
    linedef = md.linedefs[raw_seg.linedef]
    sd_idx = linedef.front_sidedef if raw_seg.side == 0 else linedef.back_sidedef
    if sd_idx < 0:
        return None
    return md.sectors[md.sidedefs[sd_idx].sector]


def build_plane_tables(
    md: MapData, flat_ids: dict[str, int] | None = None
) -> PlaneTables:
    # Global asset flat numbering, so PLANE_DEF.flat_id *and* plane ordering
    # (sorted by flat_id) match the PROTOCOL.md prefill spec and the pydoom
    # oracle's numbering. Assumes the map's flats are compiled into
    # asset_config.FLAT_ID_BY_NAME.
    flat_ids = FLAT_ID_BY_NAME if flat_ids is None else flat_ids

    keys: set[_PlaneKey] = set()
    for sector in md.sectors:
        keys.add(_plane_key(sector.floor_h, sector.floor_tex, sector.light, flat_ids))
        keys.add(
            _plane_key(sector.ceiling_h, sector.ceiling_tex, sector.light, flat_ids)
        )
    sorted_keys = sorted(keys, key=lambda k: (k.is_sky, k.height, k.flat_id, k.light))
    plane_id_by_key = {key: idx for idx, key in enumerate(sorted_keys)}
    planes = [
        PlaneRecord(
            plane_id=idx,
            height=key.height,
            flat_id=key.flat_id,
            light=key.light,
            is_sky=key.is_sky,
        )
        for idx, key in enumerate(sorted_keys)
    ]

    subsectors: list[SubsectorPlaneInfo | None] = []
    for s in range(len(md.subsectors)):
        front_sector = _subsector_front_sector(md, s)
        if front_sector is None:
            subsectors.append(None)
            continue
        floor_key = _plane_key(
            front_sector.floor_h,
            front_sector.floor_tex,
            front_sector.light,
            flat_ids,
        )
        ceiling_key = _plane_key(
            front_sector.ceiling_h,
            front_sector.ceiling_tex,
            front_sector.light,
            flat_ids,
        )
        subsectors.append(
            SubsectorPlaneInfo(
                floor_plane_id=plane_id_by_key[floor_key],
                ceiling_plane_id=plane_id_by_key[ceiling_key],
            )
        )
    return PlaneTables(planes=planes, subsectors=subsectors)
