"""Parse DOOM WAD geometry into a raw :class:`MapData`.

This loader handles the seven geometry lumps used by the renderer
(``VERTEXES``, ``LINEDEFS``, ``SIDEDEFS``, ``SECTORS``, ``SEGS``,
``SSECTORS``, ``NODES``) plus ``THINGS``. Texture and patch lumps are
not parsed here — the prefill pipeline only references texture names,
not pixels.

The returned :class:`MapData` is WAD-shaped: integer coords cast to
floats, ``scene_origin == (0.0, 0.0)``. :func:`.subset.subset_by_bbox`
turns it into a renumbered, mean-centred subset.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

from .types import (
    BspNode,
    Linedef,
    MapData,
    Sector,
    Seg,
    Sidedef,
    Subsector,
    Thing,
    Vertex,
)


_MAP_LUMP_NAMES = frozenset(
    {
        "THINGS",
        "LINEDEFS",
        "SIDEDEFS",
        "VERTEXES",
        "SEGS",
        "SSECTORS",
        "NODES",
        "SECTORS",
        "REJECT",
        "BLOCKMAP",
    }
)


def _decode_name(raw: bytes) -> str:
    return raw.rstrip(b"\x00").decode("ascii", errors="replace")


class WADReader:
    def __init__(self, path: str | Path):
        with open(path, "rb") as f:
            self._data = f.read()
        numlumps = struct.unpack_from("<I", self._data, 4)[0]
        dir_offset = struct.unpack_from("<I", self._data, 8)[0]
        self._lump_order: list[tuple[str, int, int]] = []
        for i in range(numlumps):
            base = dir_offset + i * 16
            offset = struct.unpack_from("<I", self._data, base)[0]
            size = struct.unpack_from("<I", self._data, base + 4)[0]
            name = _decode_name(self._data[base + 8 : base + 16])
            self._lump_order.append((name, offset, size))

    def _find_map_lumps(self, map_name: str) -> dict[str, tuple[int, int]]:
        found_marker = False
        result: dict[str, tuple[int, int]] = {}
        for name, off, size in self._lump_order:
            if not found_marker:
                if name == map_name and size == 0:
                    found_marker = True
                continue
            if name in _MAP_LUMP_NAMES:
                if name not in result:
                    result[name] = (off, size)
            else:
                break
        if not found_marker:
            raise KeyError(f"Map marker {map_name!r} not found in WAD")
        return result

    def get_map(self, map_name: str) -> MapData:
        lumps = self._find_map_lumps(map_name)
        required = ("VERTEXES", "LINEDEFS", "SIDEDEFS", "SECTORS", "SEGS", "SSECTORS", "NODES")
        missing = [n for n in required if n not in lumps]
        if missing:
            raise KeyError(f"Map {map_name!r} missing required lumps: {missing}")
        things = _parse_things(self._slice(lumps["THINGS"])) if "THINGS" in lumps else []
        return MapData(
            name=map_name,
            vertices=_parse_vertexes(self._slice(lumps["VERTEXES"])),
            linedefs=_parse_linedefs(self._slice(lumps["LINEDEFS"])),
            sidedefs=_parse_sidedefs(self._slice(lumps["SIDEDEFS"])),
            sectors=_parse_sectors(self._slice(lumps["SECTORS"])),
            segs=_parse_segs(self._slice(lumps["SEGS"])),
            subsectors=_parse_subsectors(self._slice(lumps["SSECTORS"])),
            nodes=_parse_nodes(self._slice(lumps["NODES"])),
            things=things,
        )

    def _slice(self, off_size: tuple[int, int]) -> bytes:
        off, size = off_size
        return self._data[off : off + size]


def _parse_vertexes(buf: bytes) -> list[Vertex]:
    return [Vertex(x=float(x), y=float(y)) for x, y in struct.iter_unpack("<hh", buf)]


def _parse_linedefs(buf: bytes) -> list[Linedef]:
    out: list[Linedef] = []
    for v1, v2, flags, special, tag, fs, bs in struct.iter_unpack("<HHHHHHH", buf):
        out.append(
            Linedef(
                v1=v1,
                v2=v2,
                flags=flags,
                special=special,
                tag=tag,
                front_sidedef=-1 if fs == 0xFFFF else fs,
                back_sidedef=-1 if bs == 0xFFFF else bs,
            )
        )
    return out


def _parse_sidedefs(buf: bytes) -> list[Sidedef]:
    out: list[Sidedef] = []
    for xo, yo, u, lo, mi, sec in struct.iter_unpack("<hh8s8s8sH", buf):
        out.append(
            Sidedef(
                x_offset=xo,
                y_offset=yo,
                upper=_decode_name(u),
                lower=_decode_name(lo),
                middle=_decode_name(mi),
                sector=sec,
            )
        )
    return out


def _parse_sectors(buf: bytes) -> list[Sector]:
    out: list[Sector] = []
    for fh, ch, ft, ct, light, special, tag in struct.iter_unpack("<hh8s8shhh", buf):
        out.append(
            Sector(
                floor_h=float(fh),
                ceiling_h=float(ch),
                floor_tex=_decode_name(ft),
                ceiling_tex=_decode_name(ct),
                light=light,
                special=special,
                tag=tag,
            )
        )
    return out


def _parse_segs(buf: bytes) -> list[Seg]:
    return [
        Seg(v1=v1, v2=v2, angle=angle, linedef=ld, side=side, offset=offset)
        for v1, v2, angle, ld, side, offset in struct.iter_unpack("<HHhHhh", buf)
    ]


def _parse_subsectors(buf: bytes) -> list[Subsector]:
    return [
        Subsector(seg_count=count, first_seg=first)
        for count, first in struct.iter_unpack("<HH", buf)
    ]


def _parse_nodes(buf: bytes) -> list[BspNode]:
    out: list[BspNode] = []
    for fields in struct.iter_unpack("<hhhh" "hhhh" "hhhh" "HH", buf):
        px, py, dx, dy, ft, fb, fl, fr, bt, bb, bl, br, fc, bc = fields
        out.append(
            BspNode(
                px=float(px),
                py=float(py),
                dx=float(dx),
                dy=float(dy),
                front_bbox=(float(ft), float(fb), float(fl), float(fr)),
                back_bbox=(float(bt), float(bb), float(bl), float(br)),
                front_child=fc,
                back_child=bc,
            )
        )
    return out


def _parse_things(buf: bytes) -> list[Thing]:
    return [
        Thing(x=float(x), y=float(y), angle=angle, type=type_, flags=flags)
        for x, y, angle, type_, flags in struct.iter_unpack("<hhHHH", buf)
    ]
