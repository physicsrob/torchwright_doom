"""Resolve per-seg renderer-friendly fields from a :class:`MapData`.

:func:`bake_segments` walks the seg → linedef → sidedef → sector
chain once and returns one :class:`Segment` per ``md.segs`` entry, in
seg order. Each :class:`Segment` carries the seg's two endpoints, its
front-sector floor/ceiling heights, the back-sector heights for
two-sided segs (``None`` otherwise), and the three texture names
attached to the seg's "viewing-side" sidedef.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import MapData


@dataclass(frozen=True)
class Segment:
    ax: float
    ay: float
    bx: float
    by: float
    front_floor: float
    front_ceiling: float
    back_floor: float | None = None
    back_ceiling: float | None = None
    middle_texture_name: str = "-"
    upper_texture_name: str = "-"
    lower_texture_name: str = "-"

    @property
    def is_two_sided(self) -> bool:
        return self.back_floor is not None and self.back_ceiling is not None


def bake_segments(md: MapData) -> list[Segment]:
    n_sd = len(md.sidedefs)
    out: list[Segment] = []
    for seg in md.segs:
        ld = md.linedefs[seg.linedef]
        front_sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
        back_sd_idx = ld.back_sidedef if seg.side == 0 else ld.front_sidedef

        sd_front = md.sidedefs[front_sd_idx]
        front_sec = md.sectors[sd_front.sector]

        is_two_sided = 0 <= back_sd_idx < n_sd
        if is_two_sided:
            sd_back = md.sidedefs[back_sd_idx]
            back_sec = md.sectors[sd_back.sector]
            back_floor: float | None = float(back_sec.floor_h)
            back_ceiling: float | None = float(back_sec.ceiling_h)
            upper_name = sd_front.upper
            lower_name = sd_front.lower
            mid_name = sd_front.middle
        else:
            back_floor = None
            back_ceiling = None
            upper_name = "-"
            lower_name = "-"
            mid_name = (
                sd_front.middle
                if sd_front.middle != "-"
                else (sd_front.lower if sd_front.lower != "-" else sd_front.upper)
            )

        v1 = md.vertices[seg.v1]
        v2 = md.vertices[seg.v2]
        out.append(
            Segment(
                ax=float(v1.x),
                ay=float(v1.y),
                bx=float(v2.x),
                by=float(v2.y),
                front_floor=float(front_sec.floor_h),
                front_ceiling=float(front_sec.ceiling_h),
                back_floor=back_floor,
                back_ceiling=back_ceiling,
                middle_texture_name=mid_name,
                upper_texture_name=upper_name,
                lower_texture_name=lower_name,
            )
        )
    return out
