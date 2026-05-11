"""Canonical visplane tables for a :class:`MapData`.

Deduplicates floors and ceilings into a stable plane list and tags
each subsector with its floor/ceiling plane ID. Both the prompt's
visplane section and the renderer's plane-marking stage read from
this.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    flat_names: list[str]
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


def build_plane_tables(md: MapData) -> PlaneTables:
    flat_names = sorted(
        {flat for sector in md.sectors for flat in (sector.floor_tex, sector.ceiling_tex)}
    )
    flat_ids = {name: idx for idx, name in enumerate(flat_names)}

    keys: set[_PlaneKey] = set()
    for sector in md.sectors:
        keys.add(_plane_key(sector.floor_h, sector.floor_tex, sector.light, flat_ids))
        keys.add(_plane_key(sector.ceiling_h, sector.ceiling_tex, sector.light, flat_ids))
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
        sector = _subsector_front_sector(md, s)
        if sector is None:
            subsectors.append(None)
            continue
        floor_key = _plane_key(sector.floor_h, sector.floor_tex, sector.light, flat_ids)
        ceiling_key = _plane_key(
            sector.ceiling_h, sector.ceiling_tex, sector.light, flat_ids
        )
        subsectors.append(
            SubsectorPlaneInfo(
                floor_plane_id=plane_id_by_key[floor_key],
                ceiling_plane_id=plane_id_by_key[ceiling_key],
            )
        )
    return PlaneTables(flat_names=flat_names, planes=planes, subsectors=subsectors)
