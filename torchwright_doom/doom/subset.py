"""Subset a :class:`MapData` to the closest N walls around a player.

The transformer's input slots are fixed at compile time — at most
``max_walls`` SEGS and ``max_bsp_nodes`` BSP nodes per frame.  The
subsetter takes a full :class:`MapData` (loaded from a WAD or built
from hand-authored segments) and returns a *renumbered* :class:`MapData`
that's structurally identical — dense vertex / linedef / sidedef /
sector / subsector / seg / node indices, valid cross-references — but
restricted to the closest N segs and the minimal BSP subtree covering
their subsectors.

The renumbered MapData is what the C-faithful reference renderer
consumes (so the reference render shows exactly what the transformer
sees), and what :func:`torchwright_doom.doom.graph_inputs.build_graph_inputs`
turns into the transformer's ``GraphInputs``.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from torchwright_doom.doom.wad import (
    SUBSECTOR_FLAG,
    BspNode,
    Linedef,
    MapData,
    Seg,
    Sector,
    Sidedef,
    Subsector,
    Vertex,
    WADReader,
)
from torchwright_doom.reference_renderer.types import Segment

# DOOM's player eye height (world units above the sector floor the
# player is standing on).  The full player capsule is 56 units tall;
# the eye sits 41 units above the feet.
DOOM_PLAYER_EYE_HEIGHT = 41.0


# ---------------------------------------------------------------------------
# BSP helpers (shared with graph_inputs.py)
# ---------------------------------------------------------------------------


def _decode_child(child_ref: int) -> Tuple[bool, int]:
    """Decode a BSP child reference. Returns (is_subsector, index)."""
    if child_ref & SUBSECTOR_FLAG:
        return True, child_ref & ~SUBSECTOR_FLAG
    return False, child_ref


def find_sector_at(md: MapData, x: float, y: float) -> int:
    """Return the sector index containing ``(x, y)`` by BSP descent.

    Uses DOOM's classification rule (point on FRONT side iff
    ``dx*(y-py) < dy*(x-px)``) at each internal node, recurses into
    front or back child accordingly, and reads the sector off the
    first seg of the resulting subsector via its sidedef.
    """
    idx = len(md.nodes) - 1  # root
    while True:
        node = md.nodes[idx]
        on_front = node.dx * (y - node.py) < node.dy * (x - node.px)
        child = node.front_child if on_front else node.back_child
        is_ss, ref = _decode_child(child)
        if is_ss:
            ss = md.subsectors[ref]
            seg = md.segs[ss.first_seg]
            ld = md.linedefs[seg.linedef]
            sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
            return md.sidedefs[sd_idx].sector
        idx = ref


def _walk_paths(md: MapData, root_node_idx: int) -> Dict[int, List[Tuple[int, int]]]:
    """Return, for every subsector, its path from the BSP root.

    A path is a list of ``(node_idx, side)`` pairs where ``side`` is
    ``0`` when the traversal descended into the front child, ``1`` into
    the back child.  Paths are ordered root-first.
    """
    paths: Dict[int, List[Tuple[int, int]]] = {}

    def visit(is_ss: bool, idx: int, path: List[Tuple[int, int]]) -> None:
        if is_ss:
            paths[idx] = list(path)
            return
        node = md.nodes[idx]
        path.append((idx, 0))
        visit(*_decode_child(node.front_child), path)
        path.pop()
        path.append((idx, 1))
        visit(*_decode_child(node.back_child), path)
        path.pop()

    visit(False, root_node_idx, [])
    return paths


def _build_seg_to_subsector(md: MapData) -> Dict[int, int]:
    """Map each seg index to its owning subsector index."""
    mapping: Dict[int, int] = {}
    for ss_idx, ss in enumerate(md.subsectors):
        for seg_idx in range(ss.first_seg, ss.first_seg + ss.seg_count):
            mapping[seg_idx] = ss_idx
    return mapping


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _seg_midpoint(md: MapData, seg_idx: int) -> Tuple[float, float]:
    seg = md.segs[seg_idx]
    v1 = md.vertices[seg.v1]
    v2 = md.vertices[seg.v2]
    return (v1.x + v2.x) * 0.5, (v1.y + v2.y) * 0.5


def _select_closest_segs(
    md: MapData,
    px: float,
    py: float,
    max_walls: int,
) -> List[int]:
    """Return original seg indices of up to ``max_walls`` closest segs."""
    ranked: List[Tuple[float, int]] = []
    for seg_idx in range(len(md.segs)):
        seg = md.segs[seg_idx]
        if seg.v1 >= len(md.vertices) or seg.v2 >= len(md.vertices):
            continue
        if seg.linedef >= len(md.linedefs):
            continue
        ld = md.linedefs[seg.linedef]
        sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
        if sd_idx < 0 or sd_idx >= len(md.sidedefs):
            continue
        mx, my = _seg_midpoint(md, seg_idx)
        dist2 = (mx - px) ** 2 + (my - py) ** 2
        ranked.append((dist2, seg_idx))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [idx for _, idx in ranked[:max_walls]]


# ---------------------------------------------------------------------------
# Renumber
# ---------------------------------------------------------------------------


# Marker used in BSP child redirection: any pruned subtree (NODE or
# SUBSECTOR child not in the kept set) becomes a SUBSECTOR-flagged
# reference to the synthetic empty subsector that we append to the
# renumbered subsector list.  ``R_Subsector`` no-ops on a 0-count
# subsector, and ``R_CheckBBox`` stays conservative-loose, so the
# renderer's BSP traversal terminates cleanly at the stub.


def subset_map_data(
    md: MapData,
    px: float,
    py: float,
    max_walls: int = 32,
    max_bsp_nodes: int = 48,
) -> Tuple[MapData, List[int]]:
    """Renumber ``md`` to the closest ``max_walls`` segs around ``(px, py)``.

    The returned :class:`MapData` is mean-centred (every vertex / BSP
    node origin / bbox shifted by the centroid of the kept vertex
    coords) and carries ``scene_origin`` set to that centroid.  All
    cross-references are dense and valid in the new numbering.

    BSP children pointing into pruned subtrees are redirected to a
    synthetic empty subsector (``seg_count == 0``) appended at the end
    of ``subsectors`` — ``R_Subsector`` no-ops on it and
    ``R_CheckBBox`` stays conservative-loose, so the reference
    renderer's traversal terminates correctly at the stub.

    Returns ``(renumbered_md, original_seg_indices)`` — the second
    element maps new seg index → original seg index in ``md.segs``,
    for tests / fixtures that need the provenance.
    """
    if len(md.nodes) == 0:
        raise ValueError(
            "subset_map_data requires a MapData with a non-trivial BSP; "
            "got nodes=[] (use build_scene_map_data for hand-authored scenes)"
        )

    # --- 1. Select N closest segs ---
    selected_orig: List[int] = _select_closest_segs(md, px, py, max_walls)
    if not selected_orig:
        raise ValueError(f"no valid segs found near ({px}, {py}) in {md.name!r}")
    selected_set: Set[int] = set(selected_orig)

    # --- 2. Map segs to subsectors ---
    seg_to_ss = _build_seg_to_subsector(md)
    selected_subsectors: Set[int] = {seg_to_ss[s] for s in selected_orig}

    # --- 3. Walk BSP tree to get paths ---
    root_idx = len(md.nodes) - 1
    paths = _walk_paths(md, root_idx)

    # --- 4. Find minimal BSP subtree covering selected subsectors ---
    subset_node_ids: Set[int] = set()
    for ss in selected_subsectors:
        for node_idx, _side in paths[ss]:
            subset_node_ids.add(node_idx)
    if len(subset_node_ids) > max_bsp_nodes:
        raise ValueError(
            f"BSP subtree has {len(subset_node_ids)} nodes but "
            f"max_bsp_nodes={max_bsp_nodes}; reduce max_walls or "
            "increase max_bsp_nodes"
        )

    # --- 5. Build old→new index maps ---

    # Subsectors: the kept ones, ordered by their old index (deterministic),
    # plus a synthetic empty subsector appended at the end.
    sorted_old_ss = sorted(selected_subsectors)
    old_to_new_ss: Dict[int, int] = {old: new for new, old in enumerate(sorted_old_ss)}
    EMPTY_SS_NEW_IDX = len(sorted_old_ss)

    # Segs: walk new subsectors in order, take their kept segs in old
    # order within each subsector (this gives each subsector a
    # contiguous run of segs in the new numbering).
    new_segs_old_indices: List[int] = []
    new_subsector_first_seg: List[int] = []
    new_subsector_seg_count: List[int] = []
    for old_ss_idx in sorted_old_ss:
        ss = md.subsectors[old_ss_idx]
        first_new_seg = len(new_segs_old_indices)
        for old_seg_idx in range(ss.first_seg, ss.first_seg + ss.seg_count):
            if old_seg_idx in selected_set:
                new_segs_old_indices.append(old_seg_idx)
        new_subsector_first_seg.append(first_new_seg)
        new_subsector_seg_count.append(len(new_segs_old_indices) - first_new_seg)

    # Linedefs: those referenced by kept segs.
    kept_ld_old: Set[int] = set(md.segs[old].linedef for old in new_segs_old_indices)
    sorted_old_ld = sorted(kept_ld_old)
    old_to_new_ld: Dict[int, int] = {old: new for new, old in enumerate(sorted_old_ld)}

    # Sidedefs: front/back of kept linedefs.
    kept_sd_old: Set[int] = set()
    for old in sorted_old_ld:
        ld = md.linedefs[old]
        if 0 <= ld.front_sidedef < len(md.sidedefs):
            kept_sd_old.add(ld.front_sidedef)
        if 0 <= ld.back_sidedef < len(md.sidedefs):
            kept_sd_old.add(ld.back_sidedef)
    sorted_old_sd = sorted(kept_sd_old)
    old_to_new_sd: Dict[int, int] = {old: new for new, old in enumerate(sorted_old_sd)}

    # Sectors: those referenced by kept sidedefs.
    kept_sec_old: Set[int] = set()
    for old in sorted_old_sd:
        sec = md.sidedefs[old].sector
        if 0 <= sec < len(md.sectors):
            kept_sec_old.add(sec)
    sorted_old_sec = sorted(kept_sec_old)
    old_to_new_sec: Dict[int, int] = {old: new for new, old in enumerate(sorted_old_sec)}

    # Vertices: those referenced by kept segs and kept linedefs.
    kept_v_old: Set[int] = set()
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
    old_to_new_v: Dict[int, int] = {old: new for new, old in enumerate(sorted_old_v)}

    # Nodes.
    sorted_old_node = sorted(subset_node_ids)
    old_to_new_node: Dict[int, int] = {
        old: new for new, old in enumerate(sorted_old_node)
    }

    # --- 6. Compute centroid of kept vertices ---
    sum_x = sum(md.vertices[old].x for old in sorted_old_v)
    sum_y = sum(md.vertices[old].y for old in sorted_old_v)
    n_v = len(sorted_old_v)
    centroid_x = sum_x / n_v
    centroid_y = sum_y / n_v

    # --- 7. Build renumbered lists ---

    new_vertices = [
        Vertex(
            x=md.vertices[old].x - centroid_x,
            y=md.vertices[old].y - centroid_y,
        )
        for old in sorted_old_v
    ]

    new_sectors = [md.sectors[old] for old in sorted_old_sec]

    def _remap_sector(old: int) -> int:
        return old_to_new_sec.get(old, 0)

    new_sidedefs = [
        Sidedef(
            x_offset=md.sidedefs[old].x_offset,
            y_offset=md.sidedefs[old].y_offset,
            upper=md.sidedefs[old].upper,
            lower=md.sidedefs[old].lower,
            middle=md.sidedefs[old].middle,
            sector=_remap_sector(md.sidedefs[old].sector),
        )
        for old in sorted_old_sd
    ]

    def _remap_sd(old: int) -> int:
        if old < 0:
            return -1
        return old_to_new_sd.get(old, -1)

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
        Subsector(seg_count=new_subsector_seg_count[i], first_seg=new_subsector_first_seg[i])
        for i in range(len(sorted_old_ss))
    ]
    # Synthetic empty subsector at the end — destination of any pruned
    # BSP child reference.
    new_subsectors.append(Subsector(seg_count=0, first_seg=0))

    def _remap_child(child_ref: int) -> int:
        is_ss, ref = _decode_child(child_ref)
        if is_ss:
            new_ss = old_to_new_ss.get(ref)
            if new_ss is None:
                return SUBSECTOR_FLAG | EMPTY_SS_NEW_IDX
            return SUBSECTOR_FLAG | new_ss
        # NODE reference
        new_node = old_to_new_node.get(ref)
        if new_node is None:
            return SUBSECTOR_FLAG | EMPTY_SS_NEW_IDX
        return new_node

    def _shift_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        top, bot, left, right = bbox
        return (
            top - centroid_y,
            bot - centroid_y,
            left - centroid_x,
            right - centroid_x,
        )

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

    new_md = MapData(
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

    return new_md, list(new_segs_old_indices)


# ---------------------------------------------------------------------------
# Hand-authored scenes — synthetic balanced BSP
# ---------------------------------------------------------------------------


@dataclass
class _BspTreeNode:
    """In-memory BSP tree node used by :func:`build_scene_map_data`.

    Either a leaf (``subsector_idx`` set, ``front``/``back`` None) or
    an internal node (``front``/``back`` set, ``subsector_idx`` None).
    """

    subsector_idx: Optional[int] = None
    # Plane stored as the BspNode-style (px, py, dx, dy) so the
    # flatten step writes :class:`BspNode` directly.
    px: float = 0.0
    py: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    front: Optional["_BspTreeNode"] = None
    back: Optional["_BspTreeNode"] = None
    # Loose bbox covering the subtree's geometry.  Computed during
    # flatten; used only by R_CheckBBox for occlusion culling.
    front_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    back_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def is_leaf(self) -> bool:
        return self.subsector_idx is not None


def _build_balanced_bsp_tree(
    subsector_indices: List[int],
    segments: List[Segment],
    depth: int = 0,
) -> _BspTreeNode:
    """Build an axis-aligned balanced BSP over single-seg subsectors.

    ``subsector_indices`` is a list of synthetic subsector indices —
    each subsector i contains exactly seg i.  At each internal node
    the split axis alternates (x at even depth, y at odd).  Subsectors
    are partitioned by their seg's midpoint.

    Encoded so that lower midpoint goes BACK and higher goes FRONT,
    matching the original ``_build_balanced_bsp``.  In BspNode form:

    - x split at v: ``BspNode(px=v, py=0, dx=0, dy=1)`` ⇒ front iff x > v
    - y split at v: ``BspNode(px=0, py=v, dx=-1, dy=0)`` ⇒ front iff y > v
    """
    if len(subsector_indices) == 1:
        return _BspTreeNode(subsector_idx=subsector_indices[0])

    axis = depth % 2  # 0 = x, 1 = y

    def midpoint_axis(ss_idx: int) -> float:
        s = segments[ss_idx]
        if axis == 0:
            return (s.ax + s.bx) * 0.5
        return (s.ay + s.by) * 0.5

    sorted_indices = sorted(subsector_indices, key=midpoint_axis)
    split_pos = len(sorted_indices) // 2
    back = sorted_indices[:split_pos]
    front = sorted_indices[split_pos:]

    split_value = (midpoint_axis(back[-1]) + midpoint_axis(front[0])) * 0.5

    if axis == 0:
        px, py, dx, dy = float(split_value), 0.0, 0.0, 1.0
    else:
        px, py, dx, dy = 0.0, float(split_value), -1.0, 0.0

    return _BspTreeNode(
        px=px, py=py, dx=dx, dy=dy,
        front=_build_balanced_bsp_tree(front, segments, depth + 1),
        back=_build_balanced_bsp_tree(back, segments, depth + 1),
    )


def _segs_bbox(
    seg_indices: List[int],
    segments: List[Segment],
) -> Tuple[float, float, float, float]:
    """Compute (top, bottom, left, right) bbox over the named segs."""
    xs: List[float] = []
    ys: List[float] = []
    for i in seg_indices:
        s = segments[i]
        xs.extend((s.ax, s.bx))
        ys.extend((s.ay, s.by))
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (max(ys), min(ys), min(xs), max(xs))


def _flatten_bsp_to_nodes(
    root: _BspTreeNode,
    segments: List[Segment],
) -> List[BspNode]:
    """Flatten the in-memory BSP tree into a list of :class:`BspNode`.

    Nodes are emitted in post-order so the root lands last (matching
    DOOM's convention: ``root_idx == len(nodes) - 1``).  Leaves
    encode as SUBSECTOR_FLAG | subsector_index.

    Bboxes are computed by collecting every leaf subsector's seg and
    taking the axis-aligned hull — loose enough that R_CheckBBox stays
    conservative.
    """
    nodes: List[BspNode] = []

    def visit(t: _BspTreeNode) -> Tuple[int, List[int]]:
        """Return (child_ref, contained_seg_indices)."""
        if t.is_leaf:
            assert t.subsector_idx is not None
            return SUBSECTOR_FLAG | t.subsector_idx, [t.subsector_idx]
        assert t.front is not None and t.back is not None
        front_ref, front_segs = visit(t.front)
        back_ref, back_segs = visit(t.back)
        front_bbox = _segs_bbox(front_segs, segments)
        back_bbox = _segs_bbox(back_segs, segments)
        node_idx = len(nodes)
        nodes.append(
            BspNode(
                px=t.px, py=t.py, dx=t.dx, dy=t.dy,
                front_bbox=front_bbox,
                back_bbox=back_bbox,
                front_child=front_ref,
                back_child=back_ref,
            )
        )
        return node_idx, front_segs + back_segs

    visit(root)
    return nodes


def build_scene_map_data(segments: List[Segment]) -> MapData:
    """Build a :class:`MapData` from hand-authored segments.

    Each segment becomes its own subsector (so BSP-rank sorting can
    discriminate them).  An axis-aligned balanced BSP partitions the
    subsectors by their seg midpoints, alternating x / y at each
    depth level.  ``scene_origin`` stays at ``(0.0, 0.0)`` — hand-
    authored scenes are typically centered already, so no shift is
    applied here.

    Reuses :func:`mapdata_from_segments` (in
    ``reference_renderer/doom_render.py``) for vertex / linedef /
    sidedef / sector wiring; replaces its single subsector + empty
    BSP with the per-seg subsectors and balanced tree.
    """
    from torchwright_doom.reference_renderer.doom_render import (
        mapdata_from_segments,
    )

    if not segments:
        raise ValueError("build_scene_map_data requires at least 1 segment")

    md, _ = mapdata_from_segments(segments, None)
    # md.subsectors == [Subsector(seg_count=N, first_seg=0)] and
    # md.nodes == [].  Replace with N single-seg subsectors and a
    # balanced BSP.
    n = len(md.segs)
    new_subsectors = [Subsector(seg_count=1, first_seg=i) for i in range(n)]
    if n == 1:
        # Edge case: one seg, no BSP needed.  R_RenderBSPNode(-1)
        # falls through to R_Subsector(0).
        return MapData(
            name=md.name,
            vertices=md.vertices,
            linedefs=md.linedefs,
            sidedefs=md.sidedefs,
            sectors=md.sectors,
            segs=md.segs,
            subsectors=new_subsectors,
            nodes=[],
            things=md.things,
            scene_origin=(0.0, 0.0),
        )

    tree = _build_balanced_bsp_tree(list(range(n)), segments, depth=0)
    nodes = _flatten_bsp_to_nodes(tree, segments)

    return MapData(
        name=md.name,
        vertices=md.vertices,
        linedefs=md.linedefs,
        sidedefs=md.sidedefs,
        sectors=md.sectors,
        segs=md.segs,
        subsectors=new_subsectors,
        nodes=nodes,
        things=md.things,
        scene_origin=(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# WAD convenience
# ---------------------------------------------------------------------------


def subset_from_wad(
    wad_path: str,
    map_name: str,
    px: float,
    py: float,
    max_walls: int = 32,
    max_bsp_nodes: int = 48,
) -> Tuple[MapData, List[int]]:
    """Load a WAD map and return ``(subset_map_data, original_seg_indices)``.

    Convenience wrapper for the common case — see :func:`subset_map_data`
    for parameter semantics.  The texture pixel data is not loaded
    here; pass the returned :class:`MapData` plus a name→pixels dict
    (built by walking ``subset_md.sidedefs`` and calling
    :meth:`WADReader.get_texture` on each name) into
    :func:`torchwright_doom.doom.graph_inputs.build_graph_inputs`.
    """
    wad = WADReader(wad_path)
    md = wad.get_map(map_name)
    return subset_map_data(md, px, py, max_walls, max_bsp_nodes)


def load_wad_textures_for_subset(
    wad_path: str,
    subset_md: MapData,
    tex_size: int = 8,
) -> Dict[str, "object"]:
    """Build a name→pixels dict covering every texture referenced by a subset.

    Walks ``subset_md.sidedefs`` for upper / lower / middle texture
    names, looks each one up in the WAD, and downscales to ``tex_size``.
    Names not found in the WAD are silently dropped (the corresponding
    Segment slot will get ``texture_id = -1`` in
    :func:`build_graph_inputs`).
    """
    from torchwright_doom.reference_renderer.textures import downscale_texture

    wad = WADReader(wad_path)
    names: Set[str] = set()
    for sd in subset_md.sidedefs:
        for n in (sd.upper, sd.lower, sd.middle):
            if n and n != "-":
                names.add(n)
    out: Dict[str, object] = {}
    for n in sorted(names):
        tex = wad.get_texture(n)
        if tex is None:
            continue
        out[n] = downscale_texture(tex, tex_size, tex_size)
    return out
