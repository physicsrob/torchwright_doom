"""Bbox-subset a WAD :class:`MapData` to a renumbered, mean-centred slice.

:func:`subset_by_bbox` keeps every full WAD SEG whose endpoint-bbox
intersects the given world-space rectangle, the subsectors containing
those segs, and the minimal BSP subtree that addresses those
subsectors. The returned :class:`MapData` is structurally identical
to a WAD-loaded one (dense indices, valid cross-references) but
renumbered, mean-centred (every coordinate shifted by the centroid of
kept vertices), and with ``scene_origin`` set to that centroid.

Any BSP child reference into a pruned subtree is redirected to a
synthetic empty subsector (``seg_count == 0``) appended at the end of
the subsector list — the renderer's BSP traversal terminates cleanly
at it.
"""

from __future__ import annotations

from .types import (
    BspNode,
    Linedef,
    MapData,
    Seg,
    Sidedef,
    Subsector,
    Vertex,
    SUBSECTOR_FLAG,
)


def _decode_child(child_ref: int) -> tuple[bool, int]:
    if child_ref & SUBSECTOR_FLAG:
        return True, child_ref & ~SUBSECTOR_FLAG
    return False, child_ref


def _walk_paths(md: MapData, root_node_idx: int) -> dict[int, list[int]]:
    """For every subsector, the list of BSP node indices from root to its leaf."""
    paths: dict[int, list[int]] = {}

    def visit(is_ss: bool, idx: int, path: list[int]) -> None:
        if is_ss:
            paths[idx] = list(path)
            return
        node = md.nodes[idx]
        path.append(idx)
        visit(*_decode_child(node.front_child), path)
        visit(*_decode_child(node.back_child), path)
        path.pop()

    visit(False, root_node_idx, [])
    return paths


def _build_seg_to_subsector(md: MapData) -> dict[int, int]:
    out: dict[int, int] = {}
    for ss_idx, ss in enumerate(md.subsectors):
        for seg_idx in range(ss.first_seg, ss.first_seg + ss.seg_count):
            out[seg_idx] = ss_idx
    return out


def _seg_intersects_bbox(
    md: MapData,
    seg_idx: int,
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> bool:
    seg = md.segs[seg_idx]
    if seg.v1 >= len(md.vertices) or seg.v2 >= len(md.vertices):
        return False
    v1 = md.vertices[seg.v1]
    v2 = md.vertices[seg.v2]
    seg_left = min(v1.x, v2.x)
    seg_right = max(v1.x, v2.x)
    seg_bottom = min(v1.y, v2.y)
    seg_top = max(v1.y, v2.y)
    return not (
        seg_right < left or seg_left > right or seg_top < bottom or seg_bottom > top
    )


def _select_segs_in_bbox(
    md: MapData,
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> list[int]:
    selected: list[int] = []
    for seg_idx in range(len(md.segs)):
        seg = md.segs[seg_idx]
        if seg.linedef >= len(md.linedefs):
            continue
        ld = md.linedefs[seg.linedef]
        sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
        if sd_idx < 0 or sd_idx >= len(md.sidedefs):
            continue
        if _seg_intersects_bbox(md, seg_idx, left, bottom, right, top):
            selected.append(seg_idx)
    return selected


def subset_by_bbox(
    md: MapData,
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> MapData:
    """Renumber ``md`` to the subset of segs intersecting the given world-space box.

    Coordinates are world-space (raw WAD frame, before mean-centring).
    The returned :class:`MapData` carries the centroid in
    ``scene_origin`` so callers can shift world-frame inputs (player
    pose, etc.) into the subset frame consistently.
    """
    if not md.nodes:
        raise ValueError("subset_by_bbox requires a MapData with a non-trivial BSP")

    selected_orig = _select_segs_in_bbox(md, left, bottom, right, top)
    if not selected_orig:
        raise ValueError(
            f"no segs intersect bbox (l={left}, b={bottom}, r={right}, t={top}) "
            f"in {md.name!r}"
        )
    selected_set = set(selected_orig)

    seg_to_ss = _build_seg_to_subsector(md)
    selected_subsectors: set[int] = {seg_to_ss[s] for s in selected_orig}

    root_idx = len(md.nodes) - 1
    paths = _walk_paths(md, root_idx)
    subset_node_ids: set[int] = set()
    for ss_idx in selected_subsectors:
        for node_idx in paths[ss_idx]:
            subset_node_ids.add(node_idx)

    sorted_old_ss = sorted(selected_subsectors)
    old_to_new_ss = {old: new for new, old in enumerate(sorted_old_ss)}
    empty_ss_new_idx = len(sorted_old_ss)

    new_segs_old_indices: list[int] = []
    new_subsector_first_seg: list[int] = []
    new_subsector_seg_count: list[int] = []
    for old_ss_idx in sorted_old_ss:
        subsector = md.subsectors[old_ss_idx]
        first_new_seg = len(new_segs_old_indices)
        for old_seg_idx in range(
            subsector.first_seg, subsector.first_seg + subsector.seg_count
        ):
            if old_seg_idx in selected_set:
                new_segs_old_indices.append(old_seg_idx)
        new_subsector_first_seg.append(first_new_seg)
        new_subsector_seg_count.append(len(new_segs_old_indices) - first_new_seg)

    kept_ld_old = {md.segs[old].linedef for old in new_segs_old_indices}
    sorted_old_ld = sorted(kept_ld_old)
    old_to_new_ld = {old: new for new, old in enumerate(sorted_old_ld)}

    kept_sd_old: set[int] = set()
    for old in sorted_old_ld:
        ld = md.linedefs[old]
        if 0 <= ld.front_sidedef < len(md.sidedefs):
            kept_sd_old.add(ld.front_sidedef)
        if 0 <= ld.back_sidedef < len(md.sidedefs):
            kept_sd_old.add(ld.back_sidedef)
    sorted_old_sd = sorted(kept_sd_old)
    old_to_new_sd = {old: new for new, old in enumerate(sorted_old_sd)}

    kept_sec_old: set[int] = set()
    for old in sorted_old_sd:
        sec = md.sidedefs[old].sector
        if 0 <= sec < len(md.sectors):
            kept_sec_old.add(sec)
    sorted_old_sec = sorted(kept_sec_old)
    old_to_new_sec = {old: new for new, old in enumerate(sorted_old_sec)}

    kept_v_old: set[int] = set()
    for old_seg_idx in new_segs_old_indices:
        seg = md.segs[old_seg_idx]
        kept_v_old.add(seg.v1)
        kept_v_old.add(seg.v2)
    for old_ld_idx in sorted_old_ld:
        ld = md.linedefs[old_ld_idx]
        if 0 <= ld.v1 < len(md.vertices):
            kept_v_old.add(ld.v1)
        if 0 <= ld.v2 < len(md.vertices):
            kept_v_old.add(ld.v2)
    sorted_old_v = sorted(kept_v_old)
    old_to_new_v = {old: new for new, old in enumerate(sorted_old_v)}

    sorted_old_node = sorted(subset_node_ids)
    old_to_new_node = {old: new for new, old in enumerate(sorted_old_node)}

    centroid_x = sum(md.vertices[old].x for old in sorted_old_v) / len(sorted_old_v)
    centroid_y = sum(md.vertices[old].y for old in sorted_old_v) / len(sorted_old_v)

    new_vertices = [
        Vertex(
            x=md.vertices[old].x - centroid_x,
            y=md.vertices[old].y - centroid_y,
        )
        for old in sorted_old_v
    ]

    new_sectors = [md.sectors[old] for old in sorted_old_sec]

    def _remap_sd(old: int) -> int:
        if old < 0:
            return -1
        return old_to_new_sd.get(old, -1)

    new_sidedefs = [
        Sidedef(
            x_offset=md.sidedefs[old].x_offset,
            y_offset=md.sidedefs[old].y_offset,
            upper=md.sidedefs[old].upper,
            lower=md.sidedefs[old].lower,
            middle=md.sidedefs[old].middle,
            sector=old_to_new_sec.get(md.sidedefs[old].sector, 0),
        )
        for old in sorted_old_sd
    ]

    new_linedefs = [
        Linedef(
            v1=old_to_new_v[md.linedefs[old].v1],
            v2=old_to_new_v[md.linedefs[old].v2],
            flags=md.linedefs[old].flags,
            special=md.linedefs[old].special,
            tag=md.linedefs[old].tag,
            front_sidedef=_remap_sd(md.linedefs[old].front_sidedef),
            back_sidedef=_remap_sd(md.linedefs[old].back_sidedef),
        )
        for old in sorted_old_ld
    ]

    new_segs = [
        Seg(
            v1=old_to_new_v[md.segs[old].v1],
            v2=old_to_new_v[md.segs[old].v2],
            angle=md.segs[old].angle,
            linedef=old_to_new_ld[md.segs[old].linedef],
            side=md.segs[old].side,
            offset=md.segs[old].offset,
        )
        for old in new_segs_old_indices
    ]

    new_subsectors = [
        Subsector(
            seg_count=new_subsector_seg_count[i],
            first_seg=new_subsector_first_seg[i],
        )
        for i in range(len(sorted_old_ss))
    ]
    new_subsectors.append(Subsector(seg_count=0, first_seg=0))

    def _remap_child(child_ref: int) -> int:
        is_ss, ref = _decode_child(child_ref)
        if is_ss:
            new_ss = old_to_new_ss.get(ref)
            if new_ss is None:
                return SUBSECTOR_FLAG | empty_ss_new_idx
            return SUBSECTOR_FLAG | new_ss
        new_node = old_to_new_node.get(ref)
        if new_node is None:
            return SUBSECTOR_FLAG | empty_ss_new_idx
        return new_node

    def _shift_bbox(
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        t, b, l, r = bbox
        return (t - centroid_y, b - centroid_y, l - centroid_x, r - centroid_x)

    new_nodes = [
        BspNode(
            px=md.nodes[old].px - centroid_x,
            py=md.nodes[old].py - centroid_y,
            dx=md.nodes[old].dx,
            dy=md.nodes[old].dy,
            front_bbox=_shift_bbox(md.nodes[old].front_bbox),
            back_bbox=_shift_bbox(md.nodes[old].back_bbox),
            front_child=_remap_child(md.nodes[old].front_child),
            back_child=_remap_child(md.nodes[old].back_child),
        )
        for old in sorted_old_node
    ]

    return MapData(
        name=md.name,
        vertices=new_vertices,
        linedefs=new_linedefs,
        sidedefs=new_sidedefs,
        sectors=new_sectors,
        segs=new_segs,
        subsectors=new_subsectors,
        nodes=new_nodes,
        things=list(md.things),
        scene_origin=(centroid_x, centroid_y),
    )
