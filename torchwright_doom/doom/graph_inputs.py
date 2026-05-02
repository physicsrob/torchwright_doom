"""Build the transformer's per-frame :class:`GraphInputs` from a :class:`MapData`.

A :class:`GraphInputs` is the wall-budget-capped, mean-centred,
pre-rank-precomputed slice of a map that the compiled DOOM transformer
consumes.  It is built from a renumbered :class:`MapData` (typically
the output of :func:`torchwright_doom.doom.subset.subset_map_data` or
:func:`build_scene_map_data`); the transformer never sees raw WAD
geometry.

The key identity for the BSP-rank precompute (derived in ``DOOM.md``)
is:

    rank(W) = dot(coeffs_W, side_P_vec) + const_W

where ``side_P_vec[i] ∈ {0, 1}`` is the runtime "which side of node
i's splitting plane is the player on?" and ``coeffs_W`` / ``const_W``
are baked from the BSP structure.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from torchwright_doom.doom.subset import (
    _build_seg_to_subsector,
    _decode_child,
    _walk_paths,
)
from torchwright_doom.doom.wad import (
    BspNode,
    MapData,
    sector_color,
    _assign_tex_id,
    _pick_seg_texture,
)
from torchwright_doom.reference_renderer.types import Segment


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BspPlane:
    """A BSP node's splitting plane expressed as an implicit line.

    The plane equation is ``nx*x + ny*y + d == 0``, with sign
    convention ``side_P = 1`` iff ``nx*x + ny*y + d > 0`` (FRONT).

    Planes inside a :class:`GraphInputs` are **unit-normalized**
    (``nx² + ny² == 1``), so ``d`` is the signed distance from the
    origin (in subset frame) to the plane.
    """

    nx: float
    ny: float
    d: float


@dataclass
class GraphInputs:
    """Transformer-ready slice of a DOOM map.

    Built from a renumbered :class:`MapData` (the kind returned by
    :func:`subset_map_data` or :func:`build_scene_map_data`) plus a
    name→pixels texture dictionary.

    ``segments`` order matches the rows of ``seg_bsp_coeffs`` /
    ``seg_bsp_consts`` and the order of ``map_data.segs`` from which
    this was built.

    ``seg_bsp_coeffs`` has shape ``(N, max_bsp_nodes)`` — rows match
    ``segments``, columns match ``bsp_planes``.  Padding columns
    (when fewer than ``max_bsp_nodes`` real planes exist) are zeros
    and contribute nothing to rank.  Among real columns, every
    column has at least one non-zero entry — zero-contribution
    planes (high-level partitions where every kept subsector lies on
    the same side) are pruned by :func:`build_graph_inputs`.

    ``scene_origin`` is forwarded from ``map_data.scene_origin`` —
    the offset that converts world coords to subset coords.
    Segments and BSP planes here are stored in the shifted (subset)
    frame; the host shifts the player position into subset frame
    before feeding the graph and adds it back to ``RESOLVED_X/Y`` on
    the way out.
    """

    segments: List[Segment]
    textures: List[np.ndarray]
    tex_name_to_id: Dict[str, int]
    bsp_planes: List[BspPlane]
    seg_bsp_coeffs: np.ndarray
    seg_bsp_consts: np.ndarray
    scene_origin: Tuple[float, float] = (0.0, 0.0)


# ---------------------------------------------------------------------------
# BSP plane math
# ---------------------------------------------------------------------------


def _make_plane(node: BspNode) -> BspPlane:
    """Convert DOOM's (px, py, dx, dy) representation to (nx, ny, d).

    Derivation: DOOM classifies a point on the FRONT side when
    ``dx*(y - py) < dy*(x - px)``.  Rearranging:

        dy*x - dx*y + (dx*py - dy*px) > 0   (front)

    So (nx, ny, d) = (dy, -dx, dx*py - dy*px) with the rule that
    ``nx*x + ny*y + d > 0 ⇒ FRONT``.

    The returned plane is **not** unit-normalized; the caller (e.g.
    :func:`build_graph_inputs`) normalizes during the final pass.
    """
    return BspPlane(
        nx=float(node.dy),
        ny=float(-node.dx),
        d=float(node.dx) * float(node.py) - float(node.dy) * float(node.px),
    )


def side_P(plane: BspPlane, px: float, py: float) -> int:
    """Classify a point against a BSP plane.

    Returns 1 if on the FRONT side (``nx*x + ny*y + d > 0``), else 0.
    """
    raw = plane.nx * px + plane.ny * py + plane.d
    return 1 if raw > 0 else 0


# ---------------------------------------------------------------------------
# Coefficient precomputation (operates on a renumbered MapData)
# ---------------------------------------------------------------------------


def _count_segs_in_subtree(
    md: MapData,
    root_node_idx: int,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Walk the BSP and count segs per subtree.

    Returns three dicts:
    - ``ss_count[ss_idx]``: ``Subsector.seg_count`` (cached for symmetry)
    - ``front_count[node_idx]``: total segs in node's front subtree
    - ``back_count[node_idx]``: total segs in node's back subtree
    """
    ss_count: Dict[int, int] = {}
    front_count: Dict[int, int] = {}
    back_count: Dict[int, int] = {}

    def count(is_ss: bool, idx: int) -> int:
        if is_ss:
            if idx in ss_count:
                return ss_count[idx]
            c = md.subsectors[idx].seg_count
            ss_count[idx] = c
            return c
        node = md.nodes[idx]
        f = count(*_decode_child(node.front_child))
        b = count(*_decode_child(node.back_child))
        front_count[idx] = f
        back_count[idx] = b
        return f + b

    count(False, root_node_idx)
    return ss_count, front_count, back_count


def _compute_coefficients(
    md: MapData,
    max_bsp_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (N, max_bsp_nodes) coefficient matrix and (N,) consts.

    Operates on a renumbered :class:`MapData` whose nodes / subsectors
    / segs are already densely indexed.  For each seg W at path
    ``[(node_i, side_W_i)]``:

    - ``side_W_i = 0`` (W in front subtree of node_i):
        coeffs[W, i] = -back_count[node_i]
        consts[W]   += +back_count[node_i]
    - ``side_W_i = 1`` (W in back subtree of node_i):
        coeffs[W, i] = +front_count[node_i]
        consts[W]   += 0
    """
    N = len(md.segs)
    coeffs = np.zeros((N, max_bsp_nodes), dtype=np.float64)
    consts = np.zeros(N, dtype=np.float64)
    if N == 0 or len(md.nodes) == 0:
        return coeffs, consts

    seg_to_ss = _build_seg_to_subsector(md)
    paths = _walk_paths(md, len(md.nodes) - 1)
    _ss_count, front_count, back_count = _count_segs_in_subtree(
        md,
        len(md.nodes) - 1,
    )

    for seg_idx in range(N):
        ss_idx = seg_to_ss.get(seg_idx)
        if ss_idx is None or ss_idx not in paths:
            continue
        path = paths[ss_idx]
        for node_idx, side_W in path:
            if node_idx >= max_bsp_nodes:
                # Caller declared a smaller cap than the renumbered
                # MapData has nodes — surface this rather than
                # silently truncating.
                raise ValueError(
                    f"map_data has {len(md.nodes)} BSP nodes but "
                    f"max_bsp_nodes={max_bsp_nodes}; increase the cap"
                )
            if side_W == 0:
                bc = back_count.get(node_idx, 0)
                coeffs[seg_idx, node_idx] = -float(bc)
                consts[seg_idx] += float(bc)
            else:
                fc = front_count.get(node_idx, 0)
                coeffs[seg_idx, node_idx] = float(fc)
    return coeffs, consts


def _prune_zero_coeff_columns(
    coeffs: np.ndarray,
    bsp_planes: List[BspPlane],
) -> Tuple[np.ndarray, List[BspPlane]]:
    """Drop BSP planes whose coefficient column is uniformly zero.

    A column of all zeros in ``seg_bsp_coeffs`` means the
    corresponding plane contributes zero to every segment's rank,
    regardless of player position.  This happens naturally for
    high-level BSP partitions where every kept subsector lies on the
    same side.

    Pruning is safe — it changes neither side bits nor rank values —
    and tightens the contract on ``GraphInputs.bsp_planes``: every
    plane participates in at least one segment's rank.

    Preserves the matrix shape ``(N, max_bsp_nodes)``: surviving
    columns shift left, the rest become padding zeros.
    """
    n_real = len(bsp_planes)
    if n_real == 0:
        return coeffs, bsp_planes
    nonzero_mask = np.any(coeffs[:, :n_real] != 0.0, axis=0)
    nonzero_idx = np.where(nonzero_mask)[0]
    if len(nonzero_idx) == n_real:
        return coeffs, bsp_planes
    n_kept = len(nonzero_idx)
    new_coeffs = np.zeros_like(coeffs)
    new_coeffs[:, :n_kept] = coeffs[:, nonzero_idx]
    new_planes = [bsp_planes[i] for i in nonzero_idx]
    return new_coeffs, new_planes


# ---------------------------------------------------------------------------
# Segment + texture conversion
# ---------------------------------------------------------------------------


def _segments_from_map_data(
    md: MapData,
) -> Tuple[List[Segment], Dict[str, int], List[str]]:
    """Build sector-aware ``Segment`` objects from a renumbered MapData.

    Returns ``(segments, name_to_id, name_order)``:

    - ``segments``: one ``Segment`` per ``md.segs`` entry, with
      front (and back, when two-sided) sector heights baked in and
      texture ids assigned via :func:`_assign_tex_id`.
    - ``name_to_id``: name → texture id used by the segments above.
    - ``name_order``: texture names in the order they were first
      seen, used by :func:`build_graph_inputs` to prioritise
      keep-order when capping at ``max_textures``.
    """
    segments: List[Segment] = []
    name_to_id: Dict[str, int] = {}
    name_order: List[str] = []
    n_sd = len(md.sidedefs)

    for seg in md.segs:
        ld = md.linedefs[seg.linedef]
        front_sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
        back_sd_idx = ld.back_sidedef if seg.side == 0 else ld.front_sidedef
        sd_front = md.sidedefs[front_sd_idx]
        front_sec = md.sectors[sd_front.sector]
        color = sector_color(sd_front.sector)

        is_two_sided = 0 <= back_sd_idx < n_sd
        if is_two_sided:
            sd_back = md.sidedefs[back_sd_idx]
            back_sec = md.sectors[sd_back.sector]
            mid_name = sd_front.middle  # usually '-' for two-sided
            upper_name = sd_front.upper
            lower_name = sd_front.lower
        else:
            sd_back = None
            back_sec = None
            mid_name = _pick_seg_texture(sd_front)
            upper_name = "-"
            lower_name = "-"

        mid_id = _assign_tex_id(mid_name, name_to_id)
        upper_id = _assign_tex_id(upper_name, name_to_id)
        lower_id = _assign_tex_id(lower_name, name_to_id)
        for n in (mid_name, upper_name, lower_name):
            if n not in ("-", "") and n not in name_order:
                name_order.append(n)

        v1 = md.vertices[seg.v1]
        v2 = md.vertices[seg.v2]
        seg_kw: Dict[str, object] = dict(
            ax=float(v1.x),
            ay=float(v1.y),
            bx=float(v2.x),
            by=float(v2.y),
            color=color,
            texture_id=mid_id,
            front_floor=float(front_sec.floor_h),
            front_ceiling=float(front_sec.ceiling_h),
        )
        if is_two_sided and back_sec is not None:
            seg_kw["back_floor"] = float(back_sec.floor_h)
            seg_kw["back_ceiling"] = float(back_sec.ceiling_h)
            seg_kw["upper_texture_id"] = upper_id
            seg_kw["lower_texture_id"] = lower_id
        segments.append(Segment(**seg_kw))

    return segments, name_to_id, name_order


def _reverse_lookup(d: Dict[str, int], value: int) -> str:
    for k, v in d.items():
        if v == value:
            return k
    return "-"


def _cap_textures(
    segments: List[Segment],
    name_to_id: Dict[str, int],
    name_order: List[str],
    textures_dict: Dict[str, np.ndarray],
    max_textures: int,
) -> Tuple[List[Segment], List[np.ndarray], Dict[str, int]]:
    """Cap the texture set at ``max_textures`` and remap segment IDs.

    Names not present in ``textures_dict`` are dropped; remaining
    names are kept in ``name_order`` until the cap is reached.
    Segment texture IDs are remapped to the compacted index;
    references to dropped names become ``-1``.
    """
    kept_names: List[str] = []
    textures: List[np.ndarray] = []
    for name in name_order:
        if len(kept_names) >= max_textures:
            break
        tex = textures_dict.get(name)
        if tex is None:
            continue
        kept_names.append(name)
        textures.append(tex)

    new_name_to_id: Dict[str, int] = {name: i for i, name in enumerate(kept_names)}

    def _remap(old_id: int) -> int:
        if old_id < 0:
            return -1
        old_name = _reverse_lookup(name_to_id, old_id)
        return new_name_to_id.get(old_name, -1)

    remapped: List[Segment] = []
    for s in segments:
        remapped.append(
            Segment(
                ax=s.ax,
                ay=s.ay,
                bx=s.bx,
                by=s.by,
                color=s.color,
                texture_id=_remap(s.texture_id),
                front_floor=s.front_floor,
                front_ceiling=s.front_ceiling,
                back_floor=s.back_floor,
                back_ceiling=s.back_ceiling,
                upper_texture_id=_remap(s.upper_texture_id),
                lower_texture_id=_remap(s.lower_texture_id),
            )
        )
    return remapped, textures, new_name_to_id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_graph_inputs(
    map_data: MapData,
    textures_dict: Optional[Dict[str, np.ndarray]] = None,
    max_textures: int = 32,
    max_bsp_nodes: int = 48,
) -> GraphInputs:
    """Build a :class:`GraphInputs` from a renumbered :class:`MapData`.

    ``map_data`` should be the output of
    :func:`torchwright_doom.doom.subset.subset_map_data` or
    :func:`torchwright_doom.doom.subset.build_scene_map_data` — its
    indices must be dense and its BSP children must resolve.

    ``textures_dict`` maps texture name → ``(W, H, 3)`` float array
    (already at the desired resolution).  Names not in the dict are
    dropped from the texture atlas; the corresponding segment slots
    get ``texture_id = -1``.  Pass ``None`` for an empty texture set.

    ``max_textures`` caps the texture atlas (priority by
    first-appearance order in ``map_data.segs``).

    ``max_bsp_nodes`` is the width of the precomputed coefficient
    matrix.  Must be at least ``len(map_data.nodes)``; raises
    ``ValueError`` otherwise.

    BSP planes are unit-normalized; high-level planes whose
    coefficient column is uniformly zero (every kept subsector lies
    on the same side) are pruned.
    """
    if textures_dict is None:
        textures_dict = {}

    # --- 1. Build sector-aware Segments and collect texture names ---
    segments, name_to_id, name_order = _segments_from_map_data(map_data)

    # --- 2. Cap textures at max_textures and remap segment ids ---
    segments, textures, tex_name_to_id = _cap_textures(
        segments, name_to_id, name_order, textures_dict, max_textures,
    )

    # --- 3. Build BSP planes (un-normalized) ---
    bsp_planes: List[BspPlane] = [_make_plane(n) for n in map_data.nodes]
    if len(bsp_planes) > max_bsp_nodes:
        raise ValueError(
            f"map_data has {len(bsp_planes)} BSP nodes but "
            f"max_bsp_nodes={max_bsp_nodes}; increase the cap"
        )

    # --- 4. Compute rank coefficients ---
    coeffs, consts = _compute_coefficients(map_data, max_bsp_nodes)

    # --- 5. Prune zero-contribution columns ---
    coeffs, bsp_planes = _prune_zero_coeff_columns(coeffs, bsp_planes)

    # --- 6. Unit-normalize plane normals ---
    # Plane equation and side classification are invariant under
    # positive scaling of (nx, ny, d).  After normalization (nx, ny)
    # is on the unit circle and d is the signed distance from the
    # origin to the plane — same envelope as wall coords, which the
    # graph's bsp_plane_nx/ny inputs declare as [-1, 1].
    normalized: List[BspPlane] = []
    for i, p in enumerate(bsp_planes):
        mag = math.sqrt(p.nx * p.nx + p.ny * p.ny)
        if mag == 0.0:
            raise ValueError(
                f"degenerate BSP plane at index {i}: (nx, ny) = (0, 0)"
            )
        normalized.append(BspPlane(nx=p.nx / mag, ny=p.ny / mag, d=p.d / mag))

    return GraphInputs(
        segments=segments,
        textures=textures,
        tex_name_to_id=tex_name_to_id,
        bsp_planes=normalized,
        seg_bsp_coeffs=coeffs,
        seg_bsp_consts=consts,
        scene_origin=tuple(map_data.scene_origin),
    )


# ---------------------------------------------------------------------------
# Helpers for tests (reference BSP traversal)
# ---------------------------------------------------------------------------


def bsp_traversal_order(
    md: MapData,
    px: float,
    py: float,
    selected_seg_indices: Optional[Set[int]] = None,
) -> List[int]:
    """Reference BSP front-to-back traversal for testing.

    Walks the full BSP and returns the seg indices in the order DOOM
    would render them.  If ``selected_seg_indices`` is provided, the
    output is filtered to just those segs (preserving order).
    """
    out: List[int] = []

    def visit(is_ss: bool, idx: int) -> None:
        if is_ss:
            ss = md.subsectors[idx]
            for seg_idx in range(ss.first_seg, ss.first_seg + ss.seg_count):
                if selected_seg_indices is None or seg_idx in selected_seg_indices:
                    out.append(seg_idx)
            return
        node = md.nodes[idx]
        plane = _make_plane(node)
        if side_P(plane, px, py) == 1:
            visit(*_decode_child(node.front_child))
            visit(*_decode_child(node.back_child))
        else:
            visit(*_decode_child(node.back_child))
            visit(*_decode_child(node.front_child))

    if len(md.nodes) == 0:
        # Single-subsector edge case — R_RenderBSPNode(-1) → R_Subsector(0).
        visit(True, 0)
    else:
        visit(False, len(md.nodes) - 1)
    return out
