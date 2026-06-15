"""Pure-Python reference for textured wall pixel emission.

This reference extends the wall-column pass with texture sampling and
the floor and ceiling openings Doom writes into ``visplane_t.top[]`` /
``visplane_t.bottom[]`` during ``R_RenderSegLoop``. It intentionally
does not expose a literal mutable ``visplane_t``; the output is an
append-only stream of pixel and plane-column ownership facts.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from ._bsp import decode_child, make_plane, side_P
from ._scene import GameState, Pixel, Scene
from ..prompt.geometry import Segment, bake_segments
from ..prompt.types import MapData, Sector, Seg
from ..doom_lighting import (
    NUMCOLORMAPS,
    apply_doom_colormap,
    doom_flat_colormap_row,
    doom_wall_colormap_row,
    doom_wall_orientation_light_bias,
)
from ..asset_banks import ASSET_BOOK, COLORMAP_ROWS, PLAYPAL
from ..asset_config import FLAT_ID_BY_NAME, WALL_TEXTURE_ID_BY_NAME

_DEFAULT_SCREEN_WIDTH = 60
_DEFAULT_SCREEN_HEIGHT = 50


def _screen_dim_from_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


# Test/profiling harnesses may reduce the compile-time screen dimensions by
# launching a fresh Python process with these env vars set before the implementation is
# imported. Normal tests use the defaults above.
SCREEN_WIDTH = _screen_dim_from_env(
    "TORCHWRIGHT_DOOM_SCREEN_WIDTH", _DEFAULT_SCREEN_WIDTH, minimum=2
)
SCREEN_HEIGHT = _screen_dim_from_env(
    "TORCHWRIGHT_DOOM_SCREEN_HEIGHT", _DEFAULT_SCREEN_HEIGHT, minimum=2
)
# Status-bar / viewport split — mirror of constants.py. When the bar is on, the
# 3D view occupies the top VIEW_HEIGHT rows and the horizon sits at the view
# centre; off (default) => BAR_HEIGHT 0 => VIEW_HEIGHT == SCREEN_HEIGHT, i.e.
# today's full-screen reference, bit-identical. Bar height scales with the
# screen (DOOM ST_HEIGHT 32 of 200): 32 at scale 1, 16 at scale 2.
_FULL_SCREEN_HEIGHT = 200
_FULL_BAR_HEIGHT = 32
_HUD = os.environ.get("TORCHWRIGHT_DOOM_HUD", "0")
if _HUD not in ("0", "1"):
    raise ValueError(f"TORCHWRIGHT_DOOM_HUD must be '0' or '1', got {_HUD!r}")
HUD_ENABLED = _HUD == "1"
BAR_HEIGHT = (
    (_FULL_BAR_HEIGHT * SCREEN_HEIGHT) // _FULL_SCREEN_HEIGHT if HUD_ENABLED else 0
)
VIEW_HEIGHT = SCREEN_HEIGHT - BAR_HEIGHT
HUD_TOP = VIEW_HEIGHT
CENTER_Y = VIEW_HEIGHT / 2.0
_DETAIL = os.environ.get("TORCHWRIGHT_DOOM_DETAIL", "high")
if _DETAIL not in ("low", "high"):
    raise ValueError(
        f"TORCHWRIGHT_DOOM_DETAIL must be 'low' or 'high', got {_DETAIL!r}"
    )
PIXEL_WIDTH = 2 if _DETAIL == "low" else 1
COLUMN_COUNT = SCREEN_WIDTH // PIXEL_WIDTH
DEFAULT_VIEW_Z = 41.0

ANGLE_BAM = 8192
FOV_HALF_BAM = 1024
_FIXTURE_TO_BAM = ANGLE_BAM // 256
_TAN_FOV_HALF = math.tan(FOV_HALF_BAM * 2 * math.pi / ANGLE_BAM)
_PROJECTION = (SCREEN_WIDTH - 1) / (2.0 * _TAN_FOV_HALF)
_NEAR_DEPTH = 1e-3
_MIN_SCALE = 1.0 / 256.0
_MAX_SCALE = 64.0
N_VP_PER_PLANE_MAX = 8

SIL_NONE = 0
SIL_BOTTOM = 1
SIL_TOP = 2
SIL_BOTH = SIL_BOTTOM | SIL_TOP
SIL_HEIGHT_MAX = 1.0e9
SIL_HEIGHT_MIN = -1.0e9


@dataclass(frozen=True)
class StoreWallRangeRecord:
    seg_idx: int
    x1: int
    x2: int


@dataclass(frozen=True)
class DrawsegMetaRecord:
    seg_idx: int
    x1: int
    x2: int
    wall_kind: str
    scale1: float
    scale2: float
    scalestep: float
    silhouette: int
    bsilheight: float
    tsilheight: float


@dataclass(frozen=True)
class WallSpanRecord:
    seg_idx: int
    x: int
    part: str
    y1: int
    y2: int


@dataclass(frozen=True)
class PixelColorOptions:
    x: int
    y: int
    colors: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class PixelStructureTolerance:
    optional_missing_xy: frozenset[tuple[int, int]]
    optional_extra_xy: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class ClipUpdateRecord:
    x: int
    ceilingclip: int
    floorclip: int


@dataclass(frozen=True)
class WallColumnPass:
    ranges: list[StoreWallRangeRecord]
    drawsegs: list[DrawsegMetaRecord]
    spans: list[WallSpanRecord]
    clip_updates: list[ClipUpdateRecord]


@dataclass(frozen=True)
class PlaneDefRecord:
    plane_id: int
    height: float
    flat_id: int
    light: int
    is_sky: int


@dataclass(frozen=True)
class PlaneColumnMarkRecord:
    plane_id: int
    vp: int
    plane_kind: str
    x: int
    y1: int
    y2: int


@dataclass(frozen=True)
class WallPlaneMarkPass:
    wall_columns: WallColumnPass
    planes: list[PlaneDefRecord]
    plane_marks: list[PlaneColumnMarkRecord]


@dataclass(frozen=True)
class FlatSpanRecord:
    plane_id: int
    vp: int
    y: int
    x1: int
    x2: int


@dataclass(frozen=True)
class _StoredRangeContext:
    record: StoreWallRangeRecord
    subsector_idx: int


@dataclass
class _ClipRange:
    """One entry in Doom's horizontal ``solidsegs`` list."""

    first: int
    last: int


@dataclass
class _RuntimeVisplanes:
    """Reference-side runtime visplane allocator.

    ``R_CheckPlane`` assigns a candidate runtime instance, but only actual
    plane marks publish occupancy. This mirrors Doom's split between checking
    a visplane and later writing ``top[]`` / ``bottom[]`` for a column.
    """

    occupied_columns: dict[tuple[int, int], set[int]]

    def __init__(self) -> None:
        self.occupied_columns = {}

    def check_plane(self, plane_id: int, x1: int, x2: int) -> int:
        current = set(range(int(x1), int(x2) + 1))
        for vp in range(N_VP_PER_PLANE_MAX):
            occupied = self.occupied_columns.get((int(plane_id), vp), set())
            if occupied.isdisjoint(current):
                return vp
        raise AssertionError(
            "N_VP_PER_PLANE_MAX exceeded while assigning "
            f"plane={plane_id} range=[{x1},{x2}]"
        )

    def publish_occupancy(self, plane_id: int, vp: int, x: int) -> None:
        key = (int(plane_id), int(vp))
        occupied = self.occupied_columns.setdefault(key, set())
        if int(x) in occupied:
            raise AssertionError(
                "duplicate runtime visplane column " f"(p={plane_id}, vp={vp}, x={x})"
            )
        occupied.add(int(x))


def _check_plane(
    runtime: _RuntimeVisplanes,
    *,
    plane_id: int,
    x1: int,
    x2: int,
) -> int:
    """Reference helper named after Doom's ``R_CheckPlane``."""
    return runtime.check_plane(plane_id, x1, x2)


_CHECKCOORD: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (0, 0): (3, 0, 2, 1),
    (0, 1): (3, 0, 2, 0),
    (0, 2): (3, 1, 2, 0),
    (1, 0): (2, 0, 2, 1),
    (1, 2): (3, 1, 3, 0),
    (2, 0): (2, 0, 3, 1),
    (2, 1): (2, 1, 3, 1),
    (2, 2): (2, 1, 3, 0),
}


# DOOM: viewangletox[] (r_main.c) — the viewing angle -> screen column lookup
def viewangletox(theta_bam: int) -> int:
    """Map signed 8192-scale BAM view angle to an integer screen column."""

    if theta_bam >= FOV_HALF_BAM:
        return 0
    if theta_bam <= -FOV_HALF_BAM:
        return COLUMN_COUNT - 1
    t = math.tan(theta_bam * 2 * math.pi / ANGLE_BAM)
    return round((COLUMN_COUNT - 1) * (1 - t / _TAN_FOV_HALF) / 2)


def expected_wall_column_pass(scene: Scene, state: GameState) -> WallColumnPass:
    """Compute the structured wall-column output."""

    md = scene.map_data
    segments = bake_segments(md)
    viewz = _state_viewz(state)
    ranges = _horizontal_wall_ranges(scene, state)
    ceilingclip = [-1] * COLUMN_COUNT
    floorclip = [VIEW_HEIGHT] * COLUMN_COUNT
    drawsegs: list[DrawsegMetaRecord] = []
    spans: list[WallSpanRecord] = []
    clip_updates: list[ClipUpdateRecord] = []

    for record in ranges:
        seg = segments[record.seg_idx]
        meta = _drawseg_meta(seg, md.segs[record.seg_idx], record, state, viewz)
        drawsegs.append(meta)
        _render_wall_columns(
            seg=seg,
            record=record,
            meta=meta,
            viewz=viewz,
            ceilingclip=ceilingclip,
            floorclip=floorclip,
            spans=spans,
            clip_updates=clip_updates,
        )

    return WallColumnPass(
        ranges=ranges,
        drawsegs=drawsegs,
        spans=spans,
        clip_updates=clip_updates,
    )


ML_DONTPEGTOP = 0x0008
ML_DONTPEGBOTTOM = 0x0010


def expected_pixel_pass(scene: Scene, state: GameState) -> list[Pixel]:
    """Per-pixel RGB samples.

    Wall pixels use texture sampling and Doom COLORMAP lighting. Flat
    pixels are appended afterward with the same atlas-sampled, COLORMAP-
    lit pipeline (milestones D and E).
    """
    md = scene.map_data
    segments = bake_segments(md)
    viewz = _state_viewz(state)
    contexts = _horizontal_wall_range_contexts(scene, state)
    name_to_id = _build_uv_texture_id_map()
    h_scaled_by_id = _texture_height_by_id()
    ceilingclip = [-1] * COLUMN_COUNT
    floorclip = [VIEW_HEIGHT] * COLUMN_COUNT
    out: list[Pixel] = []

    for context in contexts:
        record = context.record
        seg = segments[record.seg_idx]
        raw_seg = md.segs[record.seg_idx]
        meta = _drawseg_meta(seg, raw_seg, record, state, viewz)
        scale_ctx = _doom_scale_context(seg, raw_seg, state)
        flags, rowoffset = _seg_flags_and_rowoffset(md, record.seg_idx)
        mid_id, upper_id, lower_id = _seg_part_tex_ids(md, record.seg_idx, name_to_id)

        _render_wall_columns_pixels(
            scene=scene,
            seg=seg,
            raw_seg=raw_seg,
            record=record,
            meta=meta,
            scale_ctx=scale_ctx,
            flags=flags,
            rowoffset=rowoffset,
            mid_id=mid_id,
            upper_id=upper_id,
            lower_id=lower_id,
            h_scaled_by_id=h_scaled_by_id,
            viewz=viewz,
            ceilingclip=ceilingclip,
            floorclip=floorclip,
            out=out,
            accumulate_column=_accumulate_pixel_column,
        )

    out.extend(expected_flat_pixel_pass(scene, state))
    return out


def expected_pixel_color_options(
    scene: Scene, state: GameState
) -> list[PixelColorOptions]:
    """Accepted one-step texture-neighborhood colors after wall lighting."""
    md = scene.map_data
    segments = bake_segments(md)
    viewz = _state_viewz(state)
    contexts = _horizontal_wall_range_contexts(scene, state)
    name_to_id = _build_uv_texture_id_map()
    h_scaled_by_id = _texture_height_by_id()
    ceilingclip = [-1] * COLUMN_COUNT
    floorclip = [VIEW_HEIGHT] * COLUMN_COUNT
    out: list[PixelColorOptions] = []

    for context in contexts:
        record = context.record
        seg = segments[record.seg_idx]
        raw_seg = md.segs[record.seg_idx]
        meta = _drawseg_meta(seg, raw_seg, record, state, viewz)
        scale_ctx = _doom_scale_context(seg, raw_seg, state)
        flags, rowoffset = _seg_flags_and_rowoffset(md, record.seg_idx)
        mid_id, upper_id, lower_id = _seg_part_tex_ids(md, record.seg_idx, name_to_id)

        _render_wall_columns_pixels(
            scene=scene,
            seg=seg,
            raw_seg=raw_seg,
            record=record,
            meta=meta,
            scale_ctx=scale_ctx,
            flags=flags,
            rowoffset=rowoffset,
            mid_id=mid_id,
            upper_id=upper_id,
            lower_id=lower_id,
            h_scaled_by_id=h_scaled_by_id,
            viewz=viewz,
            ceilingclip=ceilingclip,
            floorclip=floorclip,
            out=out,
            accumulate_column=_accumulate_pixel_options_column,
        )

    out.extend(_flat_pixel_color_options(scene, state))
    return out


def expected_pixel_structure_tolerance(
    scene: Scene, state: GameState
) -> PixelStructureTolerance:
    """Rows that may be inserted/deleted by near-integer geometry edges.

    Continuous projection values feed Doom's integer wall coverage through
    ceil/floor and through the per-column ceiling/floor clip arrays. If a
    reference boundary is within this small band around an integer, a faithful
    approximate implementation can flip exactly one edge row. Only those edge
    rows are optional; stable span interiors remain strict.
    """
    md = scene.map_data
    segments = bake_segments(md)
    viewz = _state_viewz(state)
    contexts = _horizontal_wall_range_contexts(scene, state)
    name_to_id = _build_uv_texture_id_map()
    h_scaled_by_id = _texture_height_by_id()
    ceilingclip = [-1] * COLUMN_COUNT
    floorclip = [VIEW_HEIGHT] * COLUMN_COUNT
    edge_trace = _PixelEdgeTrace(COLUMN_COUNT)
    out: list[None] = []

    for context in contexts:
        record = context.record
        seg = segments[record.seg_idx]
        raw_seg = md.segs[record.seg_idx]
        meta = _drawseg_meta(seg, raw_seg, record, state, viewz)
        flags, rowoffset = _seg_flags_and_rowoffset(md, record.seg_idx)
        mid_id, upper_id, lower_id = _seg_part_tex_ids(md, record.seg_idx, name_to_id)

        _render_wall_columns_pixels(
            scene=scene,
            seg=seg,
            raw_seg=raw_seg,
            record=record,
            meta=meta,
            scale_ctx=_doom_scale_context(seg, raw_seg, state),
            flags=flags,
            rowoffset=rowoffset,
            mid_id=mid_id,
            upper_id=upper_id,
            lower_id=lower_id,
            h_scaled_by_id=h_scaled_by_id,
            viewz=viewz,
            ceilingclip=ceilingclip,
            floorclip=floorclip,
            out=out,
            accumulate_column=_ignore_pixel_column,
            edge_trace=edge_trace,
        )

    return edge_trace.tolerance()


def expected_flat_pixel_pass(scene: Scene, state: GameState) -> list[Pixel]:
    """Milestone-E flat pixels in R_DrawPlanes order.

    Each span's per-pixel `(u, v)` is computed via Doom's R_MapPlane /
    R_DrawSpan math at native 64x64 flat scale, then a per-row Doom
    ``zlight`` COLORMAP row is applied to the raw palette index before
    PLAYPAL conversion. Sky planes are skipped.
    """
    flat_atlas = _build_flat_atlas(scene)
    wall_pass = expected_wall_plane_mark_pass(scene, state)
    planes_by_id = {plane.plane_id: plane for plane in wall_pass.planes}
    view_x, view_y, view_z, view_angle_rad = _state_view(state)
    out: list[Pixel] = []
    for span in expected_flat_spans(scene, state):
        plane = planes_by_id.get(span.plane_id)
        if plane is None or plane.is_sky:
            continue
        xfrac0, yfrac0, xstep, ystep = _map_plane_setup(
            plane_height=plane.height,
            view_x=view_x,
            view_y=view_y,
            view_z=view_z,
            view_angle_rad=view_angle_rad,
            y=span.y,
            x1=span.x1,
        )
        atlas = flat_atlas[plane.flat_id]
        colormap_row = _flat_colormap_row_for(plane, view_z, span.y)
        for k, raw_idx in enumerate(
            _draw_span(atlas, xfrac0, yfrac0, xstep, ystep, span.x2 - span.x1)
        ):
            lit_idx = apply_doom_colormap(COLORMAP_ROWS, colormap_row, raw_idx)
            out.append(
                Pixel(
                    x=(span.x1 + k) * PIXEL_WIDTH,
                    y=span.y,
                    color=tuple(int(c) for c in PLAYPAL[lit_idx]),
                )
            )
    return out


def _flat_colormap_row_for(plane: PlaneDefRecord, view_z: float, y: int) -> int:
    """Per-row COLORMAP row for a flat span at screen row `y`."""
    planeheight = abs(plane.height - view_z)
    yslope = _PROJECTION / max(0.5, abs(float(y) - CENTER_Y))
    distance = planeheight * yslope
    return doom_flat_colormap_row(
        plane.light,
        distance,
        screen_width=SCREEN_WIDTH,
    )


def expected_flat_spans(scene: Scene, state: GameState) -> list[FlatSpanRecord]:
    """Run the literal R_DrawPlanes/R_MakeSpans coverage pass."""
    wall_pass = expected_wall_plane_mark_pass(scene, state)
    planes_by_id = {plane.plane_id: plane for plane in wall_pass.planes}
    columns = _runtime_visplane_columns(wall_pass.plane_marks)
    spans: list[FlatSpanRecord] = []
    for plane_id, vp in sorted(columns):
        plane = planes_by_id.get(plane_id)
        if plane is None or plane.is_sky:
            continue
        table = columns[(plane_id, vp)]
        xs = [x for x, (top, bottom) in enumerate(table) if top <= bottom]
        if not xs:
            continue
        for y, x1, x2 in _make_spans(table, min(xs), max(xs)):
            spans.append(
                FlatSpanRecord(
                    plane_id=plane_id,
                    vp=vp,
                    y=y,
                    x1=x1,
                    x2=x2,
                )
            )
    return spans


def _flat_pixel_color_options(
    scene: Scene, state: GameState
) -> list[PixelColorOptions]:
    """3×3 UV-neighborhood palette options for each flat span pixel."""
    flat_atlas = _build_flat_atlas(scene)
    wall_pass = expected_wall_plane_mark_pass(scene, state)
    planes_by_id = {plane.plane_id: plane for plane in wall_pass.planes}
    view_x, view_y, view_z, view_angle_rad = _state_view(state)
    out: list[PixelColorOptions] = []
    for span in expected_flat_spans(scene, state):
        plane = planes_by_id.get(span.plane_id)
        if plane is None or plane.is_sky:
            continue
        xfrac0, yfrac0, xstep, ystep = _map_plane_setup(
            plane_height=plane.height,
            view_x=view_x,
            view_y=view_y,
            view_z=view_z,
            view_angle_rad=view_angle_rad,
            y=span.y,
            x1=span.x1,
        )
        atlas = flat_atlas[plane.flat_id]
        colormap_row = _flat_colormap_row_for(plane, view_z, span.y)
        # Flat lighting's per-row colormap is derived from a runtime
        # multiply (distance), and tiny boundary drift around the
        # `int(distance) >> 4` step can shift the picked colormap row by
        # one. Accept the ±1 colormap_row neighborhood alongside the
        # 3×3 UV neighborhood — the same "one-step" tolerance shape used
        # for wall U/V drift.
        candidate_rows = [
            max(0, min(NUMCOLORMAPS - 1, colormap_row + dr)) for dr in (-1, 0, 1)
        ]
        for k in range(span.x2 - span.x1 + 1):
            xfrac = xfrac0 + k * xstep
            yfrac = yfrac0 + k * ystep
            u = int(math.floor(xfrac)) % 64
            v = int(math.floor(yfrac)) % 64
            colors = {
                tuple(
                    int(c)
                    for c in PLAYPAL[
                        apply_doom_colormap(
                            COLORMAP_ROWS,
                            row,
                            atlas[(u + du) % 64][(v + dv) % 64],
                        )
                    ]
                )
                for du in range(-4, 5)
                for dv in range(-4, 5)
                for row in candidate_rows
            }
            out.append(
                PixelColorOptions(
                    x=(span.x1 + k) * PIXEL_WIDTH,
                    y=span.y,
                    colors=tuple(sorted(colors)),
                )
            )
    return out


def _build_flat_atlas(scene: Scene) -> dict[int, list[list[int]]]:
    """Map flat_id to native 64x64 palette indices."""

    _assert_required_flats_compiled(scene.map_data)
    out: dict[int, list[list[int]]] = {}
    for flat_id, ft in enumerate(ASSET_BOOK.flat_textures):
        out[flat_id] = [[int(ft.pixels[u][v]) for v in range(64)] for u in range(64)]
    return out


def _state_view(state: GameState) -> tuple[float, float, float, float]:
    """Unpack (view_x, view_y, view_z, view_angle_rad) from GameState."""
    return (
        float(state.x),
        float(state.y),
        _state_viewz(state),
        state.angle * 2.0 * math.pi / 256.0,
    )


# DOOM: R_MapPlane setup (r_plane.c:121)
def _map_plane_setup(
    *,
    plane_height: float,
    view_x: float,
    view_y: float,
    view_z: float,
    view_angle_rad: float,
    y: int,
    x1: int,
) -> tuple[float, float, float, float]:
    """Return native-scale (xfrac0, yfrac0, xstep, ystep) for a span row."""
    planeheight = abs(plane_height - view_z)
    yslope = _PROJECTION / max(0.5, abs(float(y) - CENTER_Y))
    distance = planeheight * yslope
    distscale_x1 = 1.0 / max(0.1, math.cos(_xtoviewangle_rad(x1)))
    length = distance * distscale_x1
    angle = view_angle_rad + _xtoviewangle_rad(x1)
    half_screen = (COLUMN_COUNT - 1) / 2.0
    xfrac0_native = view_x + length * math.cos(angle)
    yfrac0_native = -view_y - length * math.sin(angle)
    # Doom steps horizontal flat spans along viewangle - 90deg. Since this
    # renderer stores yfrac as -world_y, the y step becomes +cos(viewangle).
    xstep_native = distance * math.sin(view_angle_rad) / half_screen
    ystep_native = distance * math.cos(view_angle_rad) / half_screen
    return (xfrac0_native, yfrac0_native, xstep_native, ystep_native)


# DOOM: R_DrawSpan (r_draw.c) — the flat span texel-fetch loop (R_MapPlane sets it up, calls this)
def _draw_span(
    atlas: list[list[int]],
    xfrac0: float,
    yfrac0: float,
    xstep: float,
    ystep: float,
    last_k: int,
) -> list[int]:
    """Sample the 64x64 flat at k = 0..last_k along xstep/ystep."""
    out: list[int] = []
    for k in range(last_k + 1):
        xfrac = xfrac0 + k * xstep
        yfrac = yfrac0 + k * ystep
        u = int(math.floor(xfrac)) % 64
        v = int(math.floor(yfrac)) % 64
        out.append(atlas[u][v])
    return out


def _runtime_visplane_columns(
    marks: list[PlaneColumnMarkRecord],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    seen: set[tuple[int, int, int]] = set()
    for mark in marks:
        key = (mark.plane_id, mark.vp, mark.x)
        if key in seen:
            raise AssertionError(
                "duplicate runtime visplane column "
                f"(p={mark.plane_id}, vp={mark.vp}, x={mark.x})"
            )
        seen.add(key)
        table = out.setdefault(
            (mark.plane_id, mark.vp),
            [(SCREEN_HEIGHT, -1) for _ in range(COLUMN_COUNT)],
        )
        table[mark.x] = (mark.y1, mark.y2)
    return out


# DOOM: R_MakeSpans (r_plane.c) — convert visplane coverage to horizontal spans
def _make_spans(
    table: list[tuple[int, int]],
    minx: int,
    maxx: int,
) -> list[tuple[int, int, int]]:
    open_x_by_y: dict[int, int] = {}
    spans: list[tuple[int, int, int]] = []

    def close_rows(lo: int, hi: int, x_close: int) -> None:
        if lo > hi:
            return
        for y in range(lo, hi + 1):
            x1 = open_x_by_y.pop(y, minx)
            spans.append((y, x1, x_close))

    def open_rows(lo: int, hi: int, x_open: int) -> None:
        if lo > hi:
            return
        for y in range(lo, hi + 1):
            open_x_by_y[y] = x_open

    for x in range(minx, maxx + 2):
        if x == minx:
            t1, b1 = SCREEN_HEIGHT, -1
        else:
            t1, b1 = table[x - 1]
        if x == maxx + 1:
            t2, b2 = SCREEN_HEIGHT, -1
        else:
            t2, b2 = table[x]

        close_top_lo = t1
        close_top_hi = min(t2 - 1, b1)
        if close_top_lo <= close_top_hi:
            close_rows(close_top_lo, close_top_hi, x - 1)
            t1_after = min(t2, b1 + 1)
        else:
            t1_after = t1

        close_bottom_lo = max(b2 + 1, t1_after)
        close_bottom_hi = b1
        if close_bottom_lo <= close_bottom_hi:
            close_rows(close_bottom_lo, close_bottom_hi, x - 1)
            b1_after = max(b2, t1_after - 1)
        else:
            b1_after = b1

        open_top_lo = t2
        open_top_hi = min(t1_after - 1, b2)
        if open_top_lo <= open_top_hi:
            open_rows(open_top_lo, open_top_hi, x)
            t2_after = min(t1_after, b2 + 1)
        else:
            t2_after = t2

        open_bottom_lo = max(b1_after + 1, t2_after)
        open_bottom_hi = b2
        if open_bottom_lo <= open_bottom_hi:
            open_rows(open_bottom_lo, open_bottom_hi, x)

    return spans


def _build_uv_texture_id_map(_scene_textures=None) -> dict[str, int]:
    return {"-": 0, "": 0, **WALL_TEXTURE_ID_BY_NAME}


def _texture_h_scaled(tex) -> int:
    """Compatibility name for native wall texture height."""

    if not tex.pixels:
        return 0
    return len(tex.pixels[0])


def _texture_height_by_id() -> dict[int, int]:
    return {
        idx + 1: len(tex.pixels[0]) if tex.pixels else 0
        for idx, tex in enumerate(ASSET_BOOK.wall_textures)
    }


def _assert_required_flats_compiled(md: MapData) -> None:
    for sector in md.sectors:
        for flat_name in (sector.floor_tex, sector.ceiling_tex):
            if flat_name not in FLAT_ID_BY_NAME:
                raise ValueError(
                    f"flat {flat_name!r} is used by the map but is not in " "FLAT_NAMES"
                )


def _seg_flags_and_rowoffset(md: MapData, seg_idx: int) -> tuple[int, int]:
    raw_seg = md.segs[seg_idx]
    linedef = md.linedefs[raw_seg.linedef]
    sd_idx = linedef.front_sidedef if raw_seg.side == 0 else linedef.back_sidedef
    y_offset = md.sidedefs[sd_idx].y_offset if sd_idx >= 0 else 0
    return (linedef.flags, y_offset)


def _seg_part_tex_ids(
    md: MapData, seg_idx: int, name_to_id: dict[str, int]
) -> tuple[int, int, int]:
    raw_seg = md.segs[seg_idx]
    linedef = md.linedefs[raw_seg.linedef]
    sd_idx = linedef.front_sidedef if raw_seg.side == 0 else linedef.back_sidedef
    if sd_idx < 0:
        return (0, 0, 0)
    sd = md.sidedefs[sd_idx]
    return (
        _texture_id(sd.middle, name_to_id),
        _texture_id(sd.upper, name_to_id),
        _texture_id(sd.lower, name_to_id),
    )


def _texture_id(name: str, name_to_id: dict[str, int]) -> int:
    if not name or name == "-":
        return 0
    key = name.upper()
    try:
        return name_to_id[key]
    except KeyError as exc:
        raise ValueError(
            f"texture {key!r} is used by the map but is not in " "WALL_TEXTURE_NAMES"
        ) from exc


# DOOM: dc_texturemid pegging (r_segs.c, R_StoreWallRange lines 462-604)
def _dc_texturemid_for(
    *,
    part: str,
    seg: Segment,
    flags: int,
    rowoffset: int,
    h_scaled: int,
    viewz: float,
) -> float:
    texture_height_native = h_scaled
    worldtop = seg.front_ceiling - viewz
    if part == "mid":
        if flags & ML_DONTPEGBOTTOM:
            base = seg.front_floor + texture_height_native - viewz
        else:
            base = worldtop
    elif part == "upper":
        if flags & ML_DONTPEGTOP:
            base = worldtop
        else:
            assert seg.back_ceiling is not None
            base = seg.back_ceiling + texture_height_native - viewz
    elif part == "lower":
        if flags & ML_DONTPEGBOTTOM:
            base = worldtop
        else:
            assert seg.back_floor is not None
            base = seg.back_floor - viewz
    else:
        raise ValueError(part)
    return base + rowoffset


def _texturecolumn_native(x: int, scale_ctx: "_DoomScaleContext") -> int:
    """Doom's ``texturecolumn`` for a screen column.

    Computed as ``floor(rw_offset - tan(angle - rw_normalangle) * rw_distance)``
    in native texture units, where ``rw_offset`` reduces to ``0`` for
    Phase 1 (sidedef.x_offset is not currently surfaced).
    """
    visangle = _normalize_angle(scale_ctx.view_angle + _xtoviewangle_rad(x))
    angle_rel = _angle_diff(visangle, scale_ctx.rw_normalangle)
    return math.floor(-math.tan(angle_rel) * scale_ctx.rw_distance)


_PIXEL_EDGE_EPS = 0.01


class _PixelEdgeTrace:
    def __init__(self, width: int) -> None:
        self.ceiling_fragile = [False] * width
        self.floor_fragile = [False] * width
        self.optional_missing: set[tuple[int, int]] = set()
        self.optional_extra: set[tuple[int, int]] = set()

    def tolerance(self) -> PixelStructureTolerance:
        return PixelStructureTolerance(
            optional_missing_xy=frozenset(self.optional_missing),
            optional_extra_xy=frozenset(self.optional_extra),
        )

    def ceiling_start_fragile(
        self,
        x: int,
        raw_value: float,
        raw_y: int,
        clip_y: int,
        y: int,
    ) -> bool:
        return (raw_y == y and _near_pixel_edge(raw_value)) or (
            clip_y == y and self.ceiling_fragile[x]
        )

    def floor_end_fragile(
        self,
        x: int,
        raw_value: float,
        raw_y: int,
        clip_y: int,
        y: int,
    ) -> bool:
        return (raw_y == y and _near_pixel_edge(raw_value)) or (
            clip_y == y and self.floor_fragile[x]
        )

    def mark_span(
        self,
        x: int,
        y_start: int,
        y_end: int,
        *,
        start_fragile: bool,
        end_fragile: bool,
    ) -> None:
        _mark_pixel_span_edges(
            x,
            y_start,
            y_end,
            start_fragile=start_fragile,
            end_fragile=end_fragile,
            optional_missing=self.optional_missing,
            optional_extra=self.optional_extra,
        )

    def set_ceiling(self, x: int, value: int, fragile: bool) -> None:
        clamped = _clamp_ceilingclip(value)
        self.ceiling_fragile[x] = fragile and clamped == value

    def set_floor(self, x: int, value: int, fragile: bool) -> None:
        clamped = _clamp_floorclip(value)
        self.floor_fragile[x] = fragile and clamped == value

    def set_solid(self, x: int) -> None:
        self.ceiling_fragile[x] = False
        self.floor_fragile[x] = False


def _render_wall_columns_pixels(
    *,
    scene: Scene,
    seg: Segment,
    raw_seg: Seg,
    record: StoreWallRangeRecord,
    meta: DrawsegMetaRecord,
    scale_ctx: "_DoomScaleContext",
    flags: int,
    rowoffset: int,
    mid_id: int,
    upper_id: int,
    lower_id: int,
    h_scaled_by_id: dict[int, int],
    viewz: float,
    ceilingclip: list[int],
    floorclip: list[int],
    out: list,
    accumulate_column,
    edge_trace: _PixelEdgeTrace | None = None,
) -> None:
    worldtop = seg.front_ceiling - viewz
    worldbottom = seg.front_floor - viewz
    worldhigh = (seg.back_ceiling - viewz) if seg.back_ceiling is not None else worldtop
    worldlow = (seg.back_floor - viewz) if seg.back_floor is not None else worldbottom

    upper_texture_present = upper_id != 0
    lower_texture_present = lower_id != 0
    mid_texture_present = mid_id != 0

    if not seg.is_two_sided:
        markceiling = True
        markfloor = True
    else:
        markceiling = seg.back_ceiling != seg.front_ceiling
        markfloor = seg.back_floor != seg.front_floor
    if meta.wall_kind == "closed":
        markceiling = True
        markfloor = True

    for x in range(record.x1, record.x2 + 1):
        scale_x = meta.scale1 + (x - record.x1) * meta.scalestep
        colormap_row = _wall_colormap_row(
            md=scene.map_data,
            seg=seg,
            raw_seg=raw_seg,
            scale=scale_x,
        )
        dc_iscale = 1.0 / scale_x if scale_x != 0 else 0.0
        top_y_raw = CENTER_Y - worldtop * scale_x
        bot_y_raw = CENTER_Y - worldbottom * scale_x
        yl_raw = math.ceil(top_y_raw)
        yl_clip = ceilingclip[x] + 1
        yl = max(yl_raw, yl_clip)
        yh_raw = math.floor(bot_y_raw)
        yh_clip = floorclip[x] - 1
        yh = min(yh_raw, yh_clip)
        yl_fragile = (
            edge_trace.ceiling_start_fragile(x, top_y_raw, yl_raw, yl_clip, yl)
            if edge_trace is not None
            else False
        )
        yh_fragile = (
            edge_trace.floor_end_fragile(x, bot_y_raw, yh_raw, yh_clip, yh)
            if edge_trace is not None
            else False
        )

        u_scaled = _texturecolumn_native(x, scale_ctx)

        if meta.wall_kind in {"solid", "closed"}:
            if mid_texture_present:
                h = h_scaled_by_id.get(mid_id, 0)
                if edge_trace is not None and h > 0:
                    edge_trace.mark_span(
                        x,
                        max(0, yl),
                        min(VIEW_HEIGHT - 1, yh),
                        start_fragile=yl_fragile,
                        end_fragile=yh_fragile,
                    )
                dc_tmid = _dc_texturemid_for(
                    part="mid",
                    seg=seg,
                    flags=flags,
                    rowoffset=rowoffset,
                    h_scaled=h,
                    viewz=viewz,
                )
                accumulate_column(
                    out,
                    scene,
                    mid_id,
                    x,
                    max(0, yl),
                    min(VIEW_HEIGHT - 1, yh),
                    dc_tmid,
                    dc_iscale,
                    h,
                    u_scaled,
                    colormap_row,
                )
            ceilingclip[x] = VIEW_HEIGHT
            floorclip[x] = -1
            if edge_trace is not None:
                edge_trace.set_solid(x)
            continue

        if worldhigh < worldtop:
            high_y_raw = CENTER_Y - worldhigh * scale_x
            mid_raw = math.floor(high_y_raw)
            mid_clip = floorclip[x] - 1
            mid_y = min(mid_raw, mid_clip)
            mid_end_fragile = (
                edge_trace.floor_end_fragile(x, high_y_raw, mid_raw, mid_clip, mid_y)
                if edge_trace is not None
                else False
            )
            if upper_texture_present:
                h = h_scaled_by_id.get(upper_id, 0)
                if edge_trace is not None and h > 0:
                    edge_trace.mark_span(
                        x,
                        max(0, yl),
                        min(VIEW_HEIGHT - 1, mid_y),
                        start_fragile=yl_fragile,
                        end_fragile=mid_end_fragile,
                    )
                dc_tmid = _dc_texturemid_for(
                    part="upper",
                    seg=seg,
                    flags=flags,
                    rowoffset=rowoffset,
                    h_scaled=h,
                    viewz=viewz,
                )
                accumulate_column(
                    out,
                    scene,
                    upper_id,
                    x,
                    max(0, yl),
                    min(VIEW_HEIGHT - 1, mid_y),
                    dc_tmid,
                    dc_iscale,
                    h,
                    u_scaled,
                    colormap_row,
                )
                if mid_y >= yl:
                    ceilingclip[x] = _clamp_ceilingclip(mid_y)
                    if edge_trace is not None:
                        edge_trace.set_ceiling(x, mid_y, mid_end_fragile)
                else:
                    ceilingclip[x] = _clamp_ceilingclip(yl - 1)
                    if edge_trace is not None:
                        edge_trace.set_ceiling(x, yl - 1, yl_fragile)
            elif markceiling:
                ceilingclip[x] = _clamp_ceilingclip(yl - 1)
                if edge_trace is not None:
                    edge_trace.set_ceiling(x, yl - 1, yl_fragile)
        elif markceiling:
            ceilingclip[x] = _clamp_ceilingclip(yl - 1)
            if edge_trace is not None:
                edge_trace.set_ceiling(x, yl - 1, yl_fragile)

        if worldlow > worldbottom:
            low_y_raw = CENTER_Y - worldlow * scale_x
            mid_y = math.ceil(low_y_raw)
            mid_start_fragile = (
                _near_pixel_edge(low_y_raw) if edge_trace is not None else False
            )
            if mid_y <= ceilingclip[x]:
                mid_y = ceilingclip[x] + 1
                mid_start_fragile = (
                    edge_trace.ceiling_fragile[x] if edge_trace is not None else False
                )
            if lower_texture_present:
                h = h_scaled_by_id.get(lower_id, 0)
                if edge_trace is not None and h > 0:
                    edge_trace.mark_span(
                        x,
                        max(0, mid_y),
                        min(VIEW_HEIGHT - 1, yh),
                        start_fragile=mid_start_fragile,
                        end_fragile=yh_fragile,
                    )
                dc_tmid = _dc_texturemid_for(
                    part="lower",
                    seg=seg,
                    flags=flags,
                    rowoffset=rowoffset,
                    h_scaled=h,
                    viewz=viewz,
                )
                accumulate_column(
                    out,
                    scene,
                    lower_id,
                    x,
                    max(0, mid_y),
                    min(VIEW_HEIGHT - 1, yh),
                    dc_tmid,
                    dc_iscale,
                    h,
                    u_scaled,
                    colormap_row,
                )
                if mid_y <= yh:
                    floorclip[x] = _clamp_floorclip(mid_y)
                    if edge_trace is not None:
                        edge_trace.set_floor(x, mid_y, mid_start_fragile)
                else:
                    floorclip[x] = _clamp_floorclip(yh + 1)
                    if edge_trace is not None:
                        edge_trace.set_floor(x, yh + 1, yh_fragile)
            elif markfloor:
                floorclip[x] = _clamp_floorclip(yh + 1)
                if edge_trace is not None:
                    edge_trace.set_floor(x, yh + 1, yh_fragile)
        elif markfloor:
            floorclip[x] = _clamp_floorclip(yh + 1)
            if edge_trace is not None:
                edge_trace.set_floor(x, yh + 1, yh_fragile)


def _accumulate_pixel_column(
    out: list[Pixel],
    scene: Scene,
    tex_id: int,
    x: int,
    y_start: int,
    y_end: int,
    dc_texturemid: float,
    dc_iscale: float,
    h_scaled: int,
    u_scaled: int,
    colormap_row: int,
) -> None:
    if h_scaled <= 0 or tex_id <= 0 or y_end < y_start:
        return
    tex = ASSET_BOOK.wall_textures[tex_id - 1]
    w_native = tex.width
    src_x = u_scaled % w_native
    for y in range(y_start, y_end + 1):
        v_native = dc_texturemid + (y - CENTER_Y) * dc_iscale
        v_scaled = math.floor(v_native)
        v_scaled_mod_h = v_scaled % h_scaled
        palette_idx = tex.pixels[src_x][v_scaled_mod_h]
        lit_idx = apply_doom_colormap(COLORMAP_ROWS, colormap_row, palette_idx)
        out.append(Pixel(x=x * PIXEL_WIDTH, y=y, color=PLAYPAL[lit_idx]))


def _ignore_pixel_column(*args, **kwargs) -> None:
    return None


def _accumulate_pixel_options_column(
    out: list[PixelColorOptions],
    scene: Scene,
    tex_id: int,
    x: int,
    y_start: int,
    y_end: int,
    dc_texturemid: float,
    dc_iscale: float,
    h_scaled: int,
    u_scaled: int,
    colormap_row: int,
) -> None:
    if h_scaled <= 0 or tex_id <= 0 or y_end < y_start:
        return
    tex = ASSET_BOOK.wall_textures[tex_id - 1]
    w_native = tex.width
    for y in range(y_start, y_end + 1):
        v_native = dc_texturemid + (y - CENTER_Y) * dc_iscale
        v_scaled = math.floor(v_native)
        v_scaled_mod_h = v_scaled % h_scaled
        colors = {
            PLAYPAL[
                apply_doom_colormap(
                    COLORMAP_ROWS,
                    colormap_row,
                    tex.pixels[(u_scaled + du) % w_native][
                        (v_scaled_mod_h + dv) % h_scaled
                    ],
                )
            ]
            for du in range(-4, 5)
            for dv in range(-4, 5)
        }
        out.append(
            PixelColorOptions(x=x * PIXEL_WIDTH, y=y, colors=tuple(sorted(colors)))
        )


def _near_pixel_edge(value: float) -> bool:
    return abs(value - round(value)) <= _PIXEL_EDGE_EPS


def _mark_pixel_span_edges(
    x: int,
    y_start: int,
    y_end: int,
    *,
    start_fragile: bool,
    end_fragile: bool,
    optional_missing: set[tuple[int, int]],
    optional_extra: set[tuple[int, int]],
) -> None:
    if y_end < y_start:
        return
    if start_fragile:
        optional_missing.add((x, y_start))
        if y_start > 0:
            optional_extra.add((x, y_start - 1))
    if end_fragile:
        optional_missing.add((x, y_end))
        if y_end < VIEW_HEIGHT - 1:
            optional_extra.add((x, y_end + 1))


# DOOM: wall lighting colormap row (r_segs.c:R_RenderMaskedSegRange lines 113-135)
def _wall_colormap_row(
    *,
    md: MapData,
    seg: Segment,
    raw_seg: Seg,
    scale: float,
) -> int:
    sector_light = _front_sector(md, raw_seg).light
    orientation_bias = doom_wall_orientation_light_bias(
        seg.ax,
        seg.ay,
        seg.bx,
        seg.by,
    )
    return doom_wall_colormap_row(
        sector_light,
        scale,
        orientation_bias=orientation_bias,
        screen_width=SCREEN_WIDTH,
    )


def _front_sector(md: MapData, raw_seg: Seg) -> Sector:
    linedef = md.linedefs[raw_seg.linedef]
    sidedef_idx = linedef.front_sidedef if raw_seg.side == 0 else linedef.back_sidedef
    return md.sectors[md.sidedefs[sidedef_idx].sector]


def expected_wall_plane_mark_pass(scene: Scene, state: GameState) -> WallPlaneMarkPass:
    """Compute the wall-column + plane-column-mark output."""

    md = scene.map_data
    segments = bake_segments(md)
    viewz = _state_viewz(state)
    plane_tables = _build_plane_tables(md)
    contexts = _horizontal_wall_range_contexts(scene, state)
    ceilingclip = [-1] * COLUMN_COUNT
    floorclip = [VIEW_HEIGHT] * COLUMN_COUNT
    runtime_visplanes = _RuntimeVisplanes()
    ranges: list[StoreWallRangeRecord] = []
    drawsegs: list[DrawsegMetaRecord] = []
    spans: list[WallSpanRecord] = []
    clip_updates: list[ClipUpdateRecord] = []
    plane_marks: list[PlaneColumnMarkRecord] = []

    for context in contexts:
        record = context.record
        seg = segments[record.seg_idx]
        meta = _drawseg_meta(seg, md.segs[record.seg_idx], record, state, viewz)
        plane_context = plane_tables.context_for_subsector(context.subsector_idx, viewz)
        ceiling_vp = (
            _check_plane(
                runtime_visplanes,
                plane_id=plane_context.ceiling_plane_id,
                x1=record.x1,
                x2=record.x2,
            )
            if plane_context.ceiling_plane_id is not None
            else None
        )
        floor_vp = (
            _check_plane(
                runtime_visplanes,
                plane_id=plane_context.floor_plane_id,
                x1=record.x1,
                x2=record.x2,
            )
            if plane_context.floor_plane_id is not None
            else None
        )
        ranges.append(record)
        drawsegs.append(meta)
        _render_wall_columns(
            seg=seg,
            record=record,
            meta=meta,
            viewz=viewz,
            ceilingclip=ceilingclip,
            floorclip=floorclip,
            spans=spans,
            clip_updates=clip_updates,
            plane_marks=plane_marks,
            runtime_visplanes=runtime_visplanes,
            ceiling_plane_id=plane_context.ceiling_plane_id,
            ceiling_vp=ceiling_vp,
            floor_plane_id=plane_context.floor_plane_id,
            floor_vp=floor_vp,
            ceiling_is_sky=plane_context.ceiling_is_sky,
        )

    return WallPlaneMarkPass(
        wall_columns=WallColumnPass(
            ranges=ranges,
            drawsegs=drawsegs,
            spans=spans,
            clip_updates=clip_updates,
        ),
        planes=plane_tables.planes,
        plane_marks=plane_marks,
    )


def _horizontal_wall_ranges(
    scene: Scene, state: GameState
) -> list[StoreWallRangeRecord]:
    return [context.record for context in _horizontal_wall_range_contexts(scene, state)]


def _horizontal_wall_range_contexts(
    scene: Scene, state: GameState
) -> list[_StoredRangeContext]:
    """``R_RenderBSPNode`` through horizontal clipping."""

    md = scene.map_data
    segments = bake_segments(md)
    px, py = state.x, state.y
    view_angle_bam = state.angle * _FIXTURE_TO_BAM
    solidsegs: list[_ClipRange] = [
        _ClipRange(-(10**9), -1),
        _ClipRange(COLUMN_COUNT, 10**9),
    ]
    out: list[_StoredRangeContext] = []

    def visit(is_ss: bool, idx: int) -> None:
        if is_ss:
            ss = md.subsectors[idx]
            for seg_idx in range(ss.first_seg, ss.first_seg + ss.seg_count):
                seg = segments[seg_idx]
                cross_z = (seg.bx - seg.ax) * (py - seg.ay) - (seg.by - seg.ay) * (
                    px - seg.ax
                )
                if cross_z >= 0:
                    continue
                if _is_empty_line(seg):
                    continue
                # Doom-style FOV / behind-viewer cull. An earlier
                # version of this code projected each vertex
                # independently, clamping
                # out-of-FOV theta to the screen edges (sx=0 or
                # sx=SCREEN_WIDTH-1). For a wall whose endpoints
                # are both outside the FOV but on opposite sides —
                # exactly the geometry of a wall behind the player —
                # that produced sx1=159, sx2=0, the wall got
                # treated as "fills the entire screen" and added to
                # `solidsegs` first, which then occluded every
                # subsequently-projected wall. `_project_seg_endpoints`
                # mirrors Doom's R_AddLine logic: it computes the
                # signed angular span and bails out when both
                # endpoints sit beyond the same FOV edge.
                projected = _project_seg_endpoints(
                    seg.ax,
                    seg.ay,
                    seg.bx,
                    seg.by,
                    px,
                    py,
                    view_angle_bam,
                )
                if projected is None:
                    continue
                first, last = projected
                if seg.is_two_sided and not _is_closed(seg):
                    fragments = _clip_pass_wall_segment(first, last, solidsegs)
                else:
                    fragments = _clip_solid_wall_segment(first, last, solidsegs)
                for f1, f2 in fragments:
                    out.append(
                        _StoredRangeContext(
                            record=StoreWallRangeRecord(seg_idx, f1, f2),
                            subsector_idx=idx,
                        )
                    )
            return

        node = md.nodes[idx]
        plane = make_plane(node)
        side = side_P(plane, px, py)
        if side == 1:
            front_child = decode_child(node.front_child)
            back_child = decode_child(node.back_child)
            back_bbox = node.back_bbox
        else:
            front_child = decode_child(node.back_child)
            back_child = decode_child(node.front_child)
            back_bbox = node.front_bbox

        visit(*front_child)
        if _check_bbox(back_bbox, px, py, view_angle_bam, solidsegs):
            visit(*back_child)

    if len(md.nodes) == 0:
        if len(md.subsectors) > 0:
            visit(True, 0)
    else:
        visit(False, len(md.nodes) - 1)
    return out


# DOOM: helper supporting R_PointToAngle (r_main.c) / R_AddLine (r_bsp.c) angle projection
def _theta_bam(vx: float, vy: float, px: float, py: float, view_angle_bam: int) -> int:
    dx = vx - px
    dy = vy - py
    angle_world = math.atan2(dy, dx)
    angle_world_bam = round(angle_world * ANGLE_BAM / (2 * math.pi)) % ANGLE_BAM
    theta = (angle_world_bam - view_angle_bam) % ANGLE_BAM
    if theta >= ANGLE_BAM // 2:
        theta -= ANGLE_BAM
    return theta


def _world_angle_bam_unsigned(vx: float, vy: float, px: float, py: float) -> int:
    """Unsigned BAM-8192 bearing from `(px, py)` to `(vx, vy)`."""
    angle_world = math.atan2(vy - py, vx - px)
    return round(angle_world * ANGLE_BAM / (2 * math.pi)) % ANGLE_BAM


def _fov_clip(
    angle1: int,
    angle2: int,
    *,
    on_span_wrap: str,
) -> tuple[int, int] | None:
    """Mirror of Doom's R_AddLine / R_CheckBBox FOV clipper.

    `angle1`, `angle2` are unsigned BAM-8192 angles already brought
    into the player frame (i.e. world angle minus view angle, mod
    `ANGLE_BAM`). The two endpoints describe a wall (or the two
    far-corner extremes of a bbox) and `angle1 - angle2` (mod
    ANGLE_BAM) is the wall's angular span as seen from the player
    going CCW from V2 to V1.

    When the seg/bbox is fully behind the player or off the same
    edge of the FOV the corresponding Doom branch returns early. When
    one endpoint is outside the FOV but the other is inside, the
    outside endpoint is clipped to the FOV edge so the screen
    projection stays inside the FOV cone (otherwise the unsigned
    `theta > FOV_HALF_BAM` branch in `viewangletox` flips the wall to
    the wrong screen edge).

    Returns `(clipped_a1, clipped_a2)` (still unsigned) or `None` to
    indicate "fully off-screen, cull". `on_span_wrap` mirrors Doom's
    convention split between R_AddLine ("backface, return") and
    R_CheckBBox ("sitting on a line, return visible"):
      * `"cull"` → return None (used by seg projection).
      * `"keep"` → return the unclipped angles (used by bbox check).
    """
    span = (angle1 - angle2) % ANGLE_BAM
    if span >= ANGLE_BAM // 2:
        if on_span_wrap == "keep":
            return (angle1, angle2)
        return None

    clipangle = FOV_HALF_BAM
    two_clip = 2 * clipangle

    # Off the *left* edge of the FOV?
    tspan = (angle1 + clipangle) % ANGLE_BAM
    if tspan > two_clip:
        tspan -= two_clip
        if tspan >= span:
            return None
        angle1 = clipangle  # clip V1 to the left FOV edge

    # Off the *right* edge?
    tspan = (clipangle - angle2) % ANGLE_BAM
    if tspan > two_clip:
        tspan -= two_clip
        if tspan >= span:
            return None
        angle2 = ANGLE_BAM - clipangle  # = -clipangle in signed form

    return (angle1, angle2)


def _project_seg_endpoints(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    px: float,
    py: float,
    view_angle_bam: int,
    *,
    on_span_wrap: str = "cull",
) -> tuple[int, int] | None:
    """Doom R_AddLine / R_CheckBBox screen-column projection.

    Returns `(sx_left, sx_right)` (always sorted) screen-column
    endpoints of the seg or bbox-extent edge, or `None` if it is
    entirely culled (off-screen / behind / backface). Use
    `on_span_wrap="keep"` for bbox checks; the default `"cull"` is
    correct for wall segs.
    """
    a1 = (_world_angle_bam_unsigned(ax, ay, px, py) - view_angle_bam) % ANGLE_BAM
    a2 = (_world_angle_bam_unsigned(bx, by, px, py) - view_angle_bam) % ANGLE_BAM

    clipped = _fov_clip(a1, a2, on_span_wrap=on_span_wrap)
    if clipped is None:
        return None
    a1, a2 = clipped

    # Convert unsigned [0, ANGLE_BAM) → signed [-ANGLE_BAM/2, ANGLE_BAM/2)
    # so `viewangletox` sees the angle in its expected domain.
    a1_signed = a1 - ANGLE_BAM if a1 >= ANGLE_BAM // 2 else a1
    a2_signed = a2 - ANGLE_BAM if a2 >= ANGLE_BAM // 2 else a2

    sx1 = viewangletox(a1_signed)
    sx2 = viewangletox(a2_signed)
    if sx1 == sx2:
        return None
    return (sx1, sx2) if sx1 <= sx2 else (sx2, sx1)


def _viewport_region(
    px: float, py: float, top: float, bottom: float, left: float, right: float
) -> tuple[int, int]:
    if px <= left:
        boxx = 0
    elif px < right:
        boxx = 1
    else:
        boxx = 2
    if py >= top:
        boxy = 0
    elif py > bottom:
        boxy = 1
    else:
        boxy = 2
    return boxy, boxx


def _check_bbox(
    bbox: tuple[float, float, float, float],
    px: float,
    py: float,
    view_angle_bam: int,
    solidsegs: list[_ClipRange],
) -> bool:
    top, bottom, left, right = bbox
    region = _viewport_region(px, py, top, bottom, left, right)
    if region == (1, 1):
        return True

    bspcoord = (top, bottom, left, right)
    x1_idx, y1_idx, x2_idx, y2_idx = _CHECKCOORD[region]
    cx1, cy1 = bspcoord[x1_idx], bspcoord[y1_idx]
    cx2, cy2 = bspcoord[x2_idx], bspcoord[y2_idx]

    # Same FOV-cull pattern as the seg projector, but with Doom's
    # R_CheckBBox's "sitting on a line" rule: span >= ANG180 means
    # one corner has wrapped past the other from the player's POV;
    # Doom returns *true* (treat as visible) in that case.
    projected = _project_seg_endpoints(
        cx1,
        cy1,
        cx2,
        cy2,
        px,
        py,
        view_angle_bam,
        on_span_wrap="keep",
    )
    if projected is None:
        return False
    first, last = projected

    return not any(entry.first <= first and last <= entry.last for entry in solidsegs)


# DOOM: R_ClipSolidWallSegment (r_bsp.c) — solid wall clipping against openings
def _clip_solid_wall_segment(
    first: int, last: int, solidsegs: list[_ClipRange]
) -> list[tuple[int, int]]:
    i = 0
    while solidsegs[i].last < first - 1:
        i += 1

    fragments: list[tuple[int, int]] = []
    start = solidsegs[i]

    if first < start.first:
        if last < start.first - 1:
            fragments.append((first, last))
            solidsegs.insert(i, _ClipRange(first, last))
            return fragments
        fragments.append((first, start.first - 1))
        start.first = first

    if last <= start.last:
        return fragments

    j = i
    while j + 1 < len(solidsegs) and last >= solidsegs[j + 1].first - 1:
        fragments.append((solidsegs[j].last + 1, solidsegs[j + 1].first - 1))
        j += 1
        if last <= solidsegs[j].last:
            start.last = solidsegs[j].last
            del solidsegs[i + 1 : j + 1]
            return fragments

    fragments.append((solidsegs[j].last + 1, last))
    start.last = last
    del solidsegs[i + 1 : j + 1]
    return fragments


# DOOM: R_ClipPassWallSegment (r_bsp.c) — portal (two-sided) wall clipping
def _clip_pass_wall_segment(
    first: int, last: int, solidsegs: list[_ClipRange]
) -> list[tuple[int, int]]:
    i = 0
    while solidsegs[i].last < first - 1:
        i += 1

    fragments: list[tuple[int, int]] = []
    start = solidsegs[i]

    if first < start.first:
        if last < start.first - 1:
            fragments.append((first, last))
            return fragments
        fragments.append((first, start.first - 1))

    if last <= start.last:
        return fragments

    j = i
    while j + 1 < len(solidsegs) and last >= solidsegs[j + 1].first - 1:
        fragments.append((solidsegs[j].last + 1, solidsegs[j + 1].first - 1))
        j += 1
        if last <= solidsegs[j].last:
            return fragments

    fragments.append((solidsegs[j].last + 1, last))
    return fragments


def _state_viewz(state: GameState) -> float:
    return float(getattr(state, "viewz", DEFAULT_VIEW_Z))


def _has_texture(name: str) -> bool:
    return bool(name and name != "-")


def _is_closed(seg: Segment) -> bool:
    return (
        seg.is_two_sided
        and seg.back_floor is not None
        and seg.back_ceiling is not None
        and (seg.back_ceiling <= seg.front_floor or seg.back_floor >= seg.front_ceiling)
    )


def _is_empty_line(seg: Segment) -> bool:
    return (
        seg.is_two_sided
        and seg.back_floor == seg.front_floor
        and seg.back_ceiling == seg.front_ceiling
        and not _has_texture(seg.middle_texture_name)
        and not _has_texture(seg.upper_texture_name)
        and not _has_texture(seg.lower_texture_name)
    )


@dataclass(frozen=True)
class _PlaneKey:
    height: float
    flat_id: int
    light: int
    is_sky: int


@dataclass(frozen=True)
class _ActivePlaneContext:
    floor_plane_id: int | None
    ceiling_plane_id: int | None
    ceiling_is_sky: bool


@dataclass(frozen=True)
class _SubsectorPlaneInfo:
    floor_plane_id: int | None
    ceiling_plane_id: int | None
    floor_height: float
    ceiling_height: float
    ceiling_is_sky: bool


@dataclass(frozen=True)
class _PlaneTables:
    planes: list[PlaneDefRecord]
    subsectors: list[_SubsectorPlaneInfo | None]

    def context_for_subsector(self, s: int, viewz: float) -> _ActivePlaneContext:
        info = self.subsectors[s] if 0 <= s < len(self.subsectors) else None
        if info is None:
            return _ActivePlaneContext(None, None, False)
        floor_plane_id = info.floor_plane_id if info.floor_height < viewz else None
        ceiling_plane_id = (
            info.ceiling_plane_id
            if info.ceiling_height > viewz or info.ceiling_is_sky
            else None
        )
        return _ActivePlaneContext(
            floor_plane_id=floor_plane_id,
            ceiling_plane_id=ceiling_plane_id,
            ceiling_is_sky=info.ceiling_is_sky,
        )


def _build_plane_tables(md: MapData) -> _PlaneTables:
    _assert_required_flats_compiled(md)
    flat_ids = FLAT_ID_BY_NAME
    keys = {
        _plane_key(sector.floor_h, sector.floor_tex, sector.light, flat_ids)
        for sector in md.sectors
    }
    keys.update(
        _plane_key(sector.ceiling_h, sector.ceiling_tex, sector.light, flat_ids)
        for sector in md.sectors
    )
    sorted_keys = sorted(keys, key=lambda k: (k.is_sky, k.height, k.flat_id, k.light))
    plane_id_by_key = {key: idx for idx, key in enumerate(sorted_keys)}
    planes = [
        PlaneDefRecord(
            plane_id=idx,
            height=key.height,
            flat_id=key.flat_id,
            light=key.light,
            is_sky=key.is_sky,
        )
        for idx, key in enumerate(sorted_keys)
    ]

    subsectors: list[_SubsectorPlaneInfo | None] = []
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
            _SubsectorPlaneInfo(
                floor_plane_id=plane_id_by_key[floor_key],
                ceiling_plane_id=plane_id_by_key[ceiling_key],
                floor_height=float(sector.floor_h),
                ceiling_height=float(sector.ceiling_h),
                ceiling_is_sky=_is_sky_flat(sector.ceiling_tex),
            )
        )
    return _PlaneTables(planes=planes, subsectors=subsectors)


def _plane_key(
    height: float, flat_name: str, light: int, flat_ids: dict[str, int]
) -> _PlaneKey:
    is_sky = _is_sky_flat(flat_name)
    return _PlaneKey(
        height=0.0 if is_sky else float(height),
        flat_id=flat_ids[flat_name],
        light=0 if is_sky else int(light),
        is_sky=1 if is_sky else 0,
    )


def _subsector_front_sector(md: MapData, s: int) -> Sector | None:
    sub = md.subsectors[s]
    if sub.seg_count <= 0:
        return None
    raw_seg = md.segs[sub.first_seg]
    linedef = md.linedefs[raw_seg.linedef]
    sidedef_idx = linedef.front_sidedef if raw_seg.side == 0 else linedef.back_sidedef
    if sidedef_idx < 0:
        return None
    return md.sectors[md.sidedefs[sidedef_idx].sector]


def _is_sky_flat(flat_name: str) -> bool:
    name = flat_name.upper()
    return name == "F_SKY1" or name.startswith("F_SKY")


def _wall_kind(seg: Segment) -> str:
    if not seg.is_two_sided:
        return "solid"
    if _is_closed(seg):
        return "closed"
    return "portal"


def _drawseg_meta(
    seg: Segment,
    raw_seg: Seg,
    record: StoreWallRangeRecord,
    state: GameState,
    viewz: float,
) -> DrawsegMetaRecord:
    scale_context = _doom_scale_context(seg, raw_seg, state)
    scale1 = _doom_scale_from_global_angle(record.x1, scale_context)
    scale2 = _doom_scale_from_global_angle(record.x2, scale_context)
    scalestep = (
        (scale2 - scale1) / (record.x2 - record.x1) if record.x2 > record.x1 else 0.0
    )
    wall_kind = _wall_kind(seg)
    silhouette = SIL_NONE
    bsilheight = 0.0
    tsilheight = 0.0

    if wall_kind in {"solid", "closed"}:
        silhouette = SIL_BOTH
        bsilheight = SIL_HEIGHT_MAX
        tsilheight = SIL_HEIGHT_MIN
    else:
        assert seg.back_floor is not None
        assert seg.back_ceiling is not None
        if seg.front_floor > seg.back_floor:
            silhouette |= SIL_BOTTOM
            bsilheight = seg.front_floor
        elif seg.back_floor > viewz:
            silhouette |= SIL_BOTTOM
            bsilheight = SIL_HEIGHT_MAX

        if seg.front_ceiling < seg.back_ceiling:
            silhouette |= SIL_TOP
            tsilheight = seg.front_ceiling
        elif seg.back_ceiling < viewz:
            silhouette |= SIL_TOP
            tsilheight = SIL_HEIGHT_MIN

    return DrawsegMetaRecord(
        seg_idx=record.seg_idx,
        x1=record.x1,
        x2=record.x2,
        wall_kind=wall_kind,
        scale1=scale1,
        scale2=scale2,
        scalestep=scalestep,
        silhouette=silhouette,
        bsilheight=bsilheight,
        tsilheight=tsilheight,
    )


def _render_wall_columns(
    *,
    seg: Segment,
    record: StoreWallRangeRecord,
    meta: DrawsegMetaRecord,
    viewz: float,
    ceilingclip: list[int],
    floorclip: list[int],
    spans: list[WallSpanRecord],
    clip_updates: list[ClipUpdateRecord],
    plane_marks: list[PlaneColumnMarkRecord] | None = None,
    runtime_visplanes: _RuntimeVisplanes | None = None,
    ceiling_plane_id: int | None = None,
    ceiling_vp: int | None = None,
    floor_plane_id: int | None = None,
    floor_vp: int | None = None,
    ceiling_is_sky: bool = False,
) -> None:
    worldtop = seg.front_ceiling - viewz
    worldbottom = seg.front_floor - viewz
    worldhigh = (seg.back_ceiling - viewz) if seg.back_ceiling is not None else worldtop
    worldlow = (seg.back_floor - viewz) if seg.back_floor is not None else worldbottom

    upper_texture = _has_texture(seg.upper_texture_name)
    lower_texture = _has_texture(seg.lower_texture_name)
    if not seg.is_two_sided:
        markceiling = True
        markfloor = True
    else:
        markceiling = seg.back_ceiling != seg.front_ceiling
        markfloor = seg.back_floor != seg.front_floor
    if meta.wall_kind == "closed":
        markceiling = True
        markfloor = True
    if seg.front_floor >= viewz:
        markfloor = False
    if seg.front_ceiling <= viewz and not ceiling_is_sky:
        markceiling = False

    for x in range(record.x1, record.x2 + 1):
        scale_x = meta.scale1 + (x - record.x1) * meta.scalestep
        top_y_raw = CENTER_Y - worldtop * scale_x
        bot_y_raw = CENTER_Y - worldbottom * scale_x
        yl = max(math.ceil(top_y_raw), ceilingclip[x] + 1)
        yh = min(math.floor(bot_y_raw), floorclip[x] - 1)

        if plane_marks is not None:
            _append_plane_marks(
                plane_marks=plane_marks,
                x=x,
                yl=yl,
                yh=yh,
                ceilingclip=ceilingclip[x],
                floorclip=floorclip[x],
                markceiling=markceiling,
                markfloor=markfloor,
                ceiling_plane_id=ceiling_plane_id,
                ceiling_vp=ceiling_vp,
                floor_plane_id=floor_plane_id,
                floor_vp=floor_vp,
                runtime_visplanes=runtime_visplanes,
            )

        if meta.wall_kind in {"solid", "closed"}:
            _append_span(spans, record.seg_idx, x, "middle", yl, yh)
            ceilingclip[x] = VIEW_HEIGHT
            floorclip[x] = -1
            clip_updates.append(ClipUpdateRecord(x, ceilingclip[x], floorclip[x]))
            continue

        if worldhigh < worldtop:
            mid = min(math.floor(CENTER_Y - worldhigh * scale_x), floorclip[x] - 1)
            if upper_texture:
                _append_span(spans, record.seg_idx, x, "upper", yl, mid)
                if mid >= yl:
                    ceilingclip[x] = _clamp_ceilingclip(mid)
                else:
                    ceilingclip[x] = _clamp_ceilingclip(yl - 1)
            elif markceiling:
                ceilingclip[x] = _clamp_ceilingclip(yl - 1)
        elif markceiling:
            ceilingclip[x] = _clamp_ceilingclip(yl - 1)

        if worldlow > worldbottom:
            mid = math.ceil(CENTER_Y - worldlow * scale_x)
            if mid <= ceilingclip[x]:
                mid = ceilingclip[x] + 1
            if lower_texture:
                _append_span(spans, record.seg_idx, x, "lower", mid, yh)
                if mid <= yh:
                    floorclip[x] = _clamp_floorclip(mid)
                else:
                    floorclip[x] = _clamp_floorclip(yh + 1)
            elif markfloor:
                floorclip[x] = _clamp_floorclip(yh + 1)
        elif markfloor:
            floorclip[x] = _clamp_floorclip(yh + 1)

        clip_updates.append(ClipUpdateRecord(x, ceilingclip[x], floorclip[x]))


def _append_plane_marks(
    *,
    plane_marks: list[PlaneColumnMarkRecord],
    x: int,
    yl: int,
    yh: int,
    ceilingclip: int,
    floorclip: int,
    markceiling: bool,
    markfloor: bool,
    ceiling_plane_id: int | None,
    ceiling_vp: int | None,
    floor_plane_id: int | None,
    floor_vp: int | None,
    runtime_visplanes: _RuntimeVisplanes | None,
) -> None:
    if markceiling and ceiling_plane_id is not None and ceiling_vp is not None:
        top = ceilingclip + 1
        bottom = yl - 1
        if bottom >= floorclip:
            bottom = floorclip - 1
        _append_plane_mark(
            plane_marks,
            plane_id=ceiling_plane_id,
            vp=ceiling_vp,
            plane_kind="ceiling",
            x=x,
            y1=top,
            y2=bottom,
            runtime_visplanes=runtime_visplanes,
        )

    if markfloor and floor_plane_id is not None and floor_vp is not None:
        top = yh + 1
        bottom = floorclip - 1
        if top <= ceilingclip:
            top = ceilingclip + 1
        _append_plane_mark(
            plane_marks,
            plane_id=floor_plane_id,
            vp=floor_vp,
            plane_kind="floor",
            x=x,
            y1=top,
            y2=bottom,
            runtime_visplanes=runtime_visplanes,
        )


# DOOM: visplane top/bottom marking (r_segs.c:R_RenderSegLoop lines 228-259)
def _append_plane_mark(
    plane_marks: list[PlaneColumnMarkRecord],
    *,
    plane_id: int,
    vp: int,
    plane_kind: str,
    x: int,
    y1: int,
    y2: int,
    runtime_visplanes: _RuntimeVisplanes | None,
) -> None:
    y1 = max(0, y1)
    y2 = min(VIEW_HEIGHT - 1, y2)
    if y1 <= y2:
        if runtime_visplanes is not None:
            runtime_visplanes.publish_occupancy(plane_id, vp, x)
        plane_marks.append(
            PlaneColumnMarkRecord(
                plane_id=plane_id,
                vp=vp,
                plane_kind=plane_kind,
                x=x,
                y1=y1,
                y2=y2,
            )
        )


def _append_span(
    spans: list[WallSpanRecord],
    seg_idx: int,
    x: int,
    part: str,
    y1: int,
    y2: int,
) -> None:
    y1 = max(0, y1)
    y2 = min(VIEW_HEIGHT - 1, y2)
    if y1 <= y2:
        spans.append(WallSpanRecord(seg_idx=seg_idx, x=x, part=part, y1=y1, y2=y2))


def _clamp_ceilingclip(value: int) -> int:
    return max(-1, min(VIEW_HEIGHT, value))


def _clamp_floorclip(value: int) -> int:
    return max(-1, min(VIEW_HEIGHT, value))


@dataclass(frozen=True)
class _DoomScaleContext:
    view_angle: float
    rw_normalangle: float
    rw_distance: float


def _xtoviewangle_rad(x: int) -> float:
    if COLUMN_COUNT <= 1:
        return 0.0
    tangent = _TAN_FOV_HALF * (1.0 - 2.0 * x / (COLUMN_COUNT - 1))
    return math.atan(tangent)


def _doom_scale_context(
    seg: Segment, raw_seg: Seg, state: GameState
) -> _DoomScaleContext:
    """Compute Doom's ``R_StoreWallRange`` scale setup values.

    This is a real-valued port of the Doom setup:

    - ``rw_normalangle = curline->angle + ANG90``
    - ``rw_angle1 = R_PointToAngle(curline->v1)``
    - ``rw_distance = R_PointToDist(v1) * sin(ANG90 - offsetangle)``
    """

    view_angle = state.angle * 2.0 * math.pi / 256.0
    rw_angle1 = math.atan2(seg.ay - state.y, seg.ax - state.x)
    rw_normalangle = _normalize_angle(_seg_angle_rad(raw_seg) + math.pi / 2.0)
    offsetangle = abs(_angle_diff(rw_normalangle, rw_angle1))
    if offsetangle > math.pi / 2.0:
        offsetangle = math.pi / 2.0
    distangle = math.pi / 2.0 - offsetangle
    hyp = math.hypot(seg.ax - state.x, seg.ay - state.y)
    rw_distance = hyp * math.sin(distangle)
    return _DoomScaleContext(
        view_angle=view_angle,
        rw_normalangle=rw_normalangle,
        rw_distance=max(rw_distance, _NEAR_DEPTH),
    )


def _doom_scale_from_global_angle(x: int, ctx: _DoomScaleContext) -> float:
    """Real-valued port of Doom's ``R_ScaleFromGlobalAngle``."""

    visangle = _normalize_angle(ctx.view_angle + _xtoviewangle_rad(x))
    anglea = math.pi / 2.0 + _angle_diff(visangle, ctx.view_angle)
    angleb = math.pi / 2.0 + _angle_diff(visangle, ctx.rw_normalangle)
    sinea = max(math.sin(anglea), 1e-6)
    sineb = max(math.sin(angleb), 1e-6)
    num = _PROJECTION * sineb
    den = ctx.rw_distance * sinea
    if den <= num / 65536.0:
        return _MAX_SCALE
    return max(_MIN_SCALE, min(_MAX_SCALE, num / den))


def _seg_angle_rad(raw_seg: Seg) -> float:
    # `Seg.angle` is stored as 16-bit BAM (Doom's native WAD SEGS encoding —
    # one full turn = 65536). Values are int16, so e.g. -32768 = 180° (west).
    return raw_seg.angle * 2.0 * math.pi / 65536.0


def _normalize_angle(angle: float) -> float:
    return angle % (2.0 * math.pi)


def _angle_diff(a: float, b: float) -> float:
    """Return signed angular difference ``a - b`` in ``[-pi, pi)``."""

    return (a - b + math.pi) % (2.0 * math.pi) - math.pi
