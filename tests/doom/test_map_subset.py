"""Tests for the subsetter (``torchwright_doom.doom.subset``) and the
transformer-input builder (``torchwright_doom.doom.graph_inputs``).

The subsetter takes a full WAD :class:`MapData` and returns a
renumbered :class:`MapData` covering the closest N walls (mean-centred).
The graph-inputs builder turns that renumbered MapData into a
transformer-ready :class:`GraphInputs` with rank-precomputed BSP
coefficients.

The critical correctness test is
:func:`test_rank_formula_matches_bsp_traversal`: the precomputed
coefficients must reproduce DOOM's actual BSP front-to-back ordering
when dotted against the runtime player-side decisions.
"""

from typing import List, Set

import numpy as np
import pytest

from torchwright_doom.doom.graph_inputs import (
    BspPlane,
    GraphInputs,
    _make_plane,
    bsp_traversal_order,
    build_graph_inputs,
    side_P,
)
from torchwright_doom.doom.subset import (
    SUBSECTOR_FLAG,
    _decode_child,
    build_scene_map_data,
    subset_from_wad,
    subset_map_data,
)
from torchwright_doom.doom.wad import MapData, WADReader
from torchwright_doom.reference_renderer.scenes import box_room_textured
from torchwright_doom.reference_renderer.types import Segment

# E1M1's canonical player spawn (Doomguy, THING type 1).
E1M1_START = (1056.0, -3616.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_e1m1() -> MapData:
    """Raw E1M1 MapData, used for cross-checking against BSP traversal."""
    return WADReader("doom1.wad").get_map("E1M1")


@pytest.fixture(scope="module")
def subset_md_default(raw_e1m1) -> MapData:
    """Renumbered E1M1 subset at the spawn position."""
    md, _orig = subset_map_data(
        raw_e1m1,
        px=E1M1_START[0],
        py=E1M1_START[1],
        max_walls=32,
        max_bsp_nodes=64,
    )
    return md


@pytest.fixture(scope="module")
def gi_default(subset_md_default) -> GraphInputs:
    """GraphInputs over the default E1M1 subset (no textures)."""
    return build_graph_inputs(
        subset_md_default,
        textures_dict={},
        max_bsp_nodes=64,
    )


# ---------------------------------------------------------------------------
# subset_map_data — invariants on the renumbered MapData
# ---------------------------------------------------------------------------


def _check_dense_indices(md: MapData) -> None:
    """All cross-references in ``md`` resolve to valid in-range indices."""
    nv, nl, nsd, nsec, nss, nseg, nnode = (
        len(md.vertices),
        len(md.linedefs),
        len(md.sidedefs),
        len(md.sectors),
        len(md.subsectors),
        len(md.segs),
        len(md.nodes),
    )
    for sd in md.sidedefs:
        assert 0 <= sd.sector < nsec, f"sidedef sector {sd.sector} out of range"
    for ld in md.linedefs:
        assert 0 <= ld.v1 < nv and 0 <= ld.v2 < nv
        assert ld.front_sidedef == -1 or 0 <= ld.front_sidedef < nsd
        assert ld.back_sidedef == -1 or 0 <= ld.back_sidedef < nsd
    for s in md.segs:
        assert 0 <= s.v1 < nv and 0 <= s.v2 < nv
        assert 0 <= s.linedef < nl
    for ss in md.subsectors:
        assert ss.first_seg + ss.seg_count <= nseg
    for n in md.nodes:
        for child in (n.front_child, n.back_child):
            is_ss, ref = _decode_child(child)
            if is_ss:
                assert 0 <= ref < nss
            else:
                assert 0 <= ref < nnode


def test_subset_md_is_dense(subset_md_default: MapData) -> None:
    """The renumbered MapData has dense, valid cross-refs throughout."""
    _check_dense_indices(subset_md_default)


def test_subset_md_respects_max_walls(raw_e1m1) -> None:
    for n in (4, 8, 16, 32):
        md, orig = subset_map_data(
            raw_e1m1,
            px=E1M1_START[0],
            py=E1M1_START[1],
            max_walls=n,
            max_bsp_nodes=64,
        )
        assert len(md.segs) <= n
        assert len(orig) == len(md.segs)
        _check_dense_indices(md)


def test_subset_md_too_small_max_bsp_nodes_raises(raw_e1m1) -> None:
    with pytest.raises(ValueError, match="BSP subtree"):
        subset_map_data(
            raw_e1m1,
            px=E1M1_START[0],
            py=E1M1_START[1],
            max_walls=32,
            max_bsp_nodes=2,  # impossibly small
        )


def test_subset_md_mean_centred(subset_md_default: MapData) -> None:
    """Vertex coords average to (approximately) the origin."""
    xs = [v.x for v in subset_md_default.vertices]
    ys = [v.y for v in subset_md_default.vertices]
    assert abs(sum(xs) / len(xs)) < 1e-6
    assert abs(sum(ys) / len(ys)) < 1e-6


def test_subset_md_scene_origin_is_centroid(raw_e1m1) -> None:
    """``scene_origin`` records the centroid of kept vertex coords."""
    md, _ = subset_map_data(
        raw_e1m1,
        px=E1M1_START[0],
        py=E1M1_START[1],
        max_walls=8,
        max_bsp_nodes=64,
    )
    # Every kept vertex coord plus origin recovers a world-frame coord.
    ox, oy = md.scene_origin
    # All kept world-frame coords must land within the WAD's int16
    # envelope (±32768).
    for v in md.vertices:
        assert -32768 <= v.x + ox <= 32768
        assert -32768 <= v.y + oy <= 32768


def test_subset_md_empty_subsector_appended(subset_md_default: MapData) -> None:
    """Pruned BSP children resolve to a synthetic empty subsector."""
    # The subsetter always appends an empty subsector at the end; if
    # any BSP child references it, that's the redirect target.
    last = subset_md_default.subsectors[-1]
    assert last.seg_count == 0


def test_subset_md_selected_segs_are_closest(raw_e1m1) -> None:
    """Returned segs are the closest segs by midpoint distance."""
    _md, orig = subset_map_data(
        raw_e1m1,
        px=E1M1_START[0],
        py=E1M1_START[1],
        max_walls=8,
        max_bsp_nodes=64,
    )
    selected_dists = []
    for seg_idx in orig:
        seg = raw_e1m1.segs[seg_idx]
        v1 = raw_e1m1.vertices[seg.v1]
        v2 = raw_e1m1.vertices[seg.v2]
        mx = (v1.x + v2.x) / 2.0
        my = (v1.y + v2.y) / 2.0
        d2 = (mx - E1M1_START[0]) ** 2 + (my - E1M1_START[1]) ** 2
        selected_dists.append(d2)
    selected_set = set(orig)
    max_selected = max(selected_dists)
    for seg_idx in range(len(raw_e1m1.segs)):
        if seg_idx in selected_set:
            continue
        seg = raw_e1m1.segs[seg_idx]
        if seg.v1 >= len(raw_e1m1.vertices) or seg.v2 >= len(raw_e1m1.vertices):
            continue
        if seg.linedef >= len(raw_e1m1.linedefs):
            continue
        ld = raw_e1m1.linedefs[seg.linedef]
        sd_idx = ld.front_sidedef if seg.side == 0 else ld.back_sidedef
        if sd_idx < 0 or sd_idx >= len(raw_e1m1.sidedefs):
            continue
        v1 = raw_e1m1.vertices[seg.v1]
        v2 = raw_e1m1.vertices[seg.v2]
        mx = (v1.x + v2.x) / 2.0
        my = (v1.y + v2.y) / 2.0
        d2 = (mx - E1M1_START[0]) ** 2 + (my - E1M1_START[1]) ** 2
        assert d2 >= max_selected - 1e-6


# ---------------------------------------------------------------------------
# build_graph_inputs — shape + invariants
# ---------------------------------------------------------------------------


def test_gi_nonempty(gi_default: GraphInputs) -> None:
    assert len(gi_default.segments) > 0
    assert len(gi_default.bsp_planes) > 0
    assert gi_default.seg_bsp_coeffs.shape[0] == len(gi_default.segments)
    assert gi_default.seg_bsp_consts.shape[0] == len(gi_default.segments)


def test_gi_coeffs_shape_matches_max_bsp_nodes(subset_md_default) -> None:
    gi = build_graph_inputs(subset_md_default, {}, max_bsp_nodes=64)
    assert gi.seg_bsp_coeffs.shape[1] == 64
    assert len(gi.bsp_planes) <= 64


def test_gi_padding_is_zero(gi_default: GraphInputs) -> None:
    """Columns past the real BSP-plane count are exact zeros."""
    n_real = len(gi_default.bsp_planes)
    padding = gi_default.seg_bsp_coeffs[:, n_real:]
    assert np.all(padding == 0.0)


def test_gi_planes_are_unit_normalized(gi_default: GraphInputs) -> None:
    """Every plane has ``nx² + ny² == 1`` to within FP noise."""
    for p in gi_default.bsp_planes:
        mag2 = p.nx * p.nx + p.ny * p.ny
        assert abs(mag2 - 1.0) < 1e-9


def test_gi_no_zero_columns_among_real_planes(gi_default: GraphInputs) -> None:
    """Every real plane participates in at least one segment's rank."""
    n_real = len(gi_default.bsp_planes)
    real_cols = gi_default.seg_bsp_coeffs[:, :n_real]
    nonzero = np.any(real_cols != 0.0, axis=0)
    assert nonzero.all(), (
        f"plane(s) {np.where(~nonzero)[0].tolist()} have all-zero "
        "coefficient columns and should have been pruned"
    )


def test_gi_scene_origin_forwarded(subset_md_default) -> None:
    gi = build_graph_inputs(subset_md_default, {}, max_bsp_nodes=64)
    assert tuple(gi.scene_origin) == tuple(subset_md_default.scene_origin)


# ---------------------------------------------------------------------------
# Plane math
# ---------------------------------------------------------------------------


def test_plane_passes_through_node_point(raw_e1m1) -> None:
    """For each BSP node, ``(nx, ny, d)`` plane passes through DOOM's
    original ``(px, py)``.
    """
    for node in raw_e1m1.nodes:
        plane = _make_plane(node)
        raw = plane.nx * node.px + plane.ny * node.py + plane.d
        assert abs(raw) < 1e-6


def test_plane_sign_matches_doom_side_classification(raw_e1m1) -> None:
    """Our sign convention (``side_P = 1`` for front) agrees with
    DOOM's original ``R_PointOnSide`` formulation.
    """
    rng = np.random.default_rng(0)
    for node in raw_e1m1.nodes[:20]:
        plane = _make_plane(node)
        for _ in range(10):
            x = rng.uniform(-5000, 5000)
            y = rng.uniform(-5000, 5000)
            ours = side_P(plane, x, y)
            doom_front = node.dx * (y - node.py) < node.dy * (x - node.px)
            assert bool(ours) == bool(doom_front)


# ---------------------------------------------------------------------------
# Rank formula verification (the critical correctness test)
# ---------------------------------------------------------------------------


def _rank_order(gi: GraphInputs, px: float, py: float) -> List[int]:
    """Return seg indices in rank-sorted order at ``(px, py)`` (world frame).

    Player coords are shifted into subset frame before evaluating
    side_P (subset BSP planes are stored in shifted frame).  Tie-break
    on seg index, matching DOOM's in-subsector rendering order.
    """
    ox, oy = gi.scene_origin
    px_s = px - ox
    py_s = py - oy
    side_P_vec = np.zeros(gi.seg_bsp_coeffs.shape[1], dtype=np.float64)
    for i, plane in enumerate(gi.bsp_planes):
        side_P_vec[i] = float(side_P(plane, px_s, py_s))
    ranks = gi.seg_bsp_coeffs @ side_P_vec + gi.seg_bsp_consts
    seg_idx_arr = np.arange(len(gi.segments))
    order = np.lexsort((seg_idx_arr, ranks))
    return [int(i) for i in order]


def test_rank_formula_matches_subset_bsp_traversal(
    subset_md_default: MapData,
    gi_default: GraphInputs,
) -> None:
    """Sort segs by rank; must equal a BSP walk on the renumbered MapData.

    The renumbered MapData carries its own BSP, and ``bsp_traversal_order``
    walks that tree.  The graph_inputs' rank formula (computed from the
    same renumbered MapData) must reproduce that exact ordering.
    """
    ox, oy = subset_md_default.scene_origin
    px_s = E1M1_START[0] - ox
    py_s = E1M1_START[1] - oy
    reference = bsp_traversal_order(subset_md_default, px_s, py_s)
    computed = _rank_order(gi_default, E1M1_START[0], E1M1_START[1])
    assert computed == reference, (
        f"\nrank-sorted segs: {computed}\nBSP traversal:    {reference}"
    )


def test_rank_formula_at_multiple_positions(raw_e1m1) -> None:
    """The rank formula must reproduce BSP order across player positions."""
    test_positions = [
        E1M1_START,
        (1000.0, -3500.0),
        (1500.0, -3300.0),
        (2000.0, -3000.0),
    ]
    for px, py in test_positions:
        md, _orig = subset_map_data(
            raw_e1m1,
            px=px,
            py=py,
            max_walls=16,
            max_bsp_nodes=64,
        )
        gi = build_graph_inputs(md, {}, max_bsp_nodes=64)
        ox, oy = md.scene_origin
        reference = bsp_traversal_order(md, px - ox, py - oy)
        computed = _rank_order(gi, px, py)
        assert computed == reference, (
            f"at ({px}, {py}):\n"
            f"  rank-sorted: {computed}\n"
            f"  BSP:         {reference}"
        )


# ---------------------------------------------------------------------------
# subset_from_wad — convenience wrapper
# ---------------------------------------------------------------------------


def test_subset_from_wad_matches_direct() -> None:
    """The convenience wrapper produces the same MapData as direct subsetting."""
    md_direct, orig_direct = subset_map_data(
        WADReader("doom1.wad").get_map("E1M1"),
        px=E1M1_START[0],
        py=E1M1_START[1],
        max_walls=8,
        max_bsp_nodes=64,
    )
    md_wad, orig_wad = subset_from_wad(
        wad_path="doom1.wad",
        map_name="E1M1",
        px=E1M1_START[0],
        py=E1M1_START[1],
        max_walls=8,
        max_bsp_nodes=64,
    )
    assert orig_wad == orig_direct
    assert len(md_wad.segs) == len(md_direct.segs)
    assert tuple(md_wad.scene_origin) == tuple(md_direct.scene_origin)


# ---------------------------------------------------------------------------
# build_scene_map_data — hand-authored scenes
# ---------------------------------------------------------------------------


def test_build_scene_map_data_box_room() -> None:
    segments, _textures = box_room_textured(wad_path="doom1.wad", tex_size=8)
    md = build_scene_map_data(segments)
    _check_dense_indices(md)
    # Each seg gets its own subsector; balanced BSP has N-1 internal nodes.
    assert len(md.segs) == len(segments)
    # subsectors = N segs + (no synthetic empty for build_scene_map_data;
    # every BSP child is reachable in the balanced tree)
    assert len(md.subsectors) == len(segments)
    assert len(md.nodes) == len(segments) - 1


def test_build_scene_map_data_single_seg() -> None:
    segments = [
        Segment(
            ax=0,
            ay=0,
            bx=1,
            by=0,
            color=(0.5, 0.5, 0.5),
            front_floor=-1.0,
            front_ceiling=1.0,
            texture_id=0,
        )
    ]
    md = build_scene_map_data(segments)
    assert len(md.segs) == 1
    assert len(md.subsectors) == 1
    assert len(md.nodes) == 0  # single subsector → R_RenderBSPNode(-1)


def test_build_scene_map_data_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_scene_map_data([])


def test_build_scene_map_data_rank_matches_traversal() -> None:
    """Rank-sorted segs match the synthetic BSP's traversal at multiple poses.

    Critical correctness check for the synthetic balanced BSP — if the
    coefficients are right, the runtime rank formula must reproduce
    the same ordering that a front-to-back BSP walk would produce.
    """
    segments, textures = box_room_textured(wad_path="doom1.wad", tex_size=8)
    md = build_scene_map_data(segments)
    gi = build_graph_inputs(
        md,
        {f"TEX{i}": t for i, t in enumerate(textures)},
        max_bsp_nodes=16,
    )

    # A few interior poses.
    for px, py in [(0.0, 0.0), (3.0, 2.0), (-2.5, 4.0), (4.9, -4.9)]:
        # build_scene_map_data leaves scene_origin at (0, 0).
        reference = bsp_traversal_order(md, px, py)
        computed = _rank_order(gi, px, py)
        assert computed == reference, (
            f"at ({px}, {py}): rank order {computed} != BSP traversal {reference}"
        )


# ---------------------------------------------------------------------------
# Texture handling
# ---------------------------------------------------------------------------


def test_textures_capped_in_graph_inputs(subset_md_default) -> None:
    """``max_textures`` caps the atlas; segments referencing dropped
    textures get ``texture_id == -1``."""
    # Build a synthetic textures_dict containing every name from the
    # subset's sidedefs.
    names: Set[str] = set()
    for sd in subset_md_default.sidedefs:
        for n in (sd.upper, sd.lower, sd.middle):
            if n and n != "-":
                names.add(n)
    textures_dict = {n: np.zeros((4, 4, 3), dtype=np.float64) for n in names}

    gi = build_graph_inputs(
        subset_md_default,
        textures_dict,
        max_textures=2,
        max_bsp_nodes=64,
    )
    assert len(gi.textures) <= 2
    assert len(gi.tex_name_to_id) == len(gi.textures)
    for seg in gi.segments:
        assert -1 <= seg.texture_id < len(gi.textures)
