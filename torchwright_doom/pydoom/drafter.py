"""History-conditioned AR drafter for the implementation.

State-machine drafter for AR-phase token prediction. The drafter holds
explicit protocol state, advances it on each ``consume(actual)``, and
returns the canonical next emission via ``next_draft()``. There is no
precomputed plan — every prediction is conditioned on the history fed
in via ``consume``, so a structural divergence (e.g. an FOV-edge bbox
the implementation visits but exact math culls) costs one mispredict
instead of cascading through the rollout.

Public contract:

- ``ARDrafter(scene, state)`` — constructor.
- ``next_draft() -> Token | None`` — peek the canonical next emission.
- ``consume(actual)`` — advance state by what the impl actually emitted.
- ``snapshot()`` / ``rollback(snap)`` — for the runtime's draft-decode
  K-batch lookahead. O(state-size); no consume-history replay.

The main consumers are ``test_pixel_render`` (fresh-rollout validation)
and the runtime's ``CacheWithReferenceFallback`` (consume-aware fallback
after warm-cache exhaustion / divergence).

**Numerical-divergence philosophy.** The implementation uses PWL
approximations — the drafter does not try to mirror them. ``consume``
adopts the actual emitted value into state, so subsequent predictions
ride on the implementation's rollout. Numeric drift (PWL-rounded
yl/yh/scale) costs one mispredict per drift event; no compounding.
Structural divergence (where the impl's PWL-derived branch decision
disagrees with exact math) is handled by the same mechanism: each
state's ``consume`` knows how to transition for any token that could
legally follow.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

from ..tokens import Token
from ._bsp import decode_child, make_plane, side_P
from ..prompt.geometry import bake_segments
from ._scene import GameState, Scene
from . import renderer as ref
from ..protocol_tokens import (
    ADVANCE_SEG,
    BBOX_BOXPOS,
    BBOX_CORNER_X_MARK_A,
    BBOX_CORNER_X_MARK_B,
    BBOX_CORNER_Y_MARK_A,
    BBOX_CORNER_Y_MARK_B,
    BBOX_SCAN,
    BBOX_THETA_MARK_A,
    BBOX_THETA_MARK_B,
    BBOX_WORLD_ANGLE_MARK_A,
    BBOX_WORLD_ANGLE_MARK_B,
    CLIP_UPDATE,
    DRAWSEG_BSILHEIGHT,
    DRAWSEG_META,
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE1_DEN,
    DRAWSEG_SCALE2,
    DRAWSEG_SCALE2_DEN,
    DRAWSEG_SCALESTEP,
    DRAWSEG_SCALESTEP_DEN,
    DRAWSEG_TSILHEIGHT,
    DRAWSEG_U_PHASE,
    EMIT_X2,
    FIND_RUN,
    FLAT_NEXT_PLANE,
    FLAT_NEXT_VP,
    FLAT_VISPLANE_BEGIN,
    MAKE_SPANS_COL,
    PIXEL,
    PLANE_MARK,
    PROCESS_SEG,
    R_CHECK_PLANE,
    R_CHECK_PLANE_RESULT,
    R_STORE_WALL_RANGE,
    SCREEN_RANGE,
    SCREEN_Y_VALUE,
    SEG_DC_TMID_LOWER,
    SEG_DC_TMID_MID,
    SEG_DC_TMID_UPPER,
    SET_CURSOR_X,
    SET_CURSOR_Y,
    SIDE_RECORD,
    SPAN_CLOSE_SLOT,
    SPAN_ROW,
    THETA_MARK_A,
    THETA_MARK_B,
    THINK_SIDE,
    TRAVERSE_BETWEEN,
    TRAVERSE_ENTER,
    TRAVERSE_RETURN,
    VALUE,
    VISIT_SUBSECTOR,
    WALL_COL_U,
    WALL_SPAN_META,
    WORLD_ANGLE_MARK_A,
    WORLD_ANGLE_MARK_B,
)
from ..vocab import (
    ANGLE_BAM,
    ANGLE_VALUE,
    DONE,
    DRAW_PLANES_BEGIN,
    N_NODES_MAX,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SEG_KPART,
    SET_CURSOR_DIRECTION_X,
    SET_CURSOR_DIRECTION_Y,
    _K_PART_TABLES,
)
from ..value_ranges import (
    ValueRange,
    decode_float,
    encode_float,
)

_FIXTURE_TO_BAM = ANGLE_BAM // 256
_BAM_HALF = ANGLE_BAM // 2

_WALL_KIND_CODE = {"solid": 0, "closed": 1, "portal": 2}
_WALL_KIND_BY_CODE = {v: k for k, v in _WALL_KIND_CODE.items()}
_SPAN_PART_CODE = {"middle": 0, "upper": 1, "lower": 2}
_SPAN_PART_BY_CODE = {v: k for k, v in _SPAN_PART_CODE.items()}
_PLANE_KIND_CODE = {"ceiling": 0, "floor": 1}
_PLANE_KIND_BY_CODE = {v: k for k, v in _PLANE_KIND_CODE.items()}

_BBOX_TYPES = {
    BBOX_BOXPOS,
    BBOX_CORNER_X_MARK_A,
    BBOX_CORNER_Y_MARK_A,
    BBOX_WORLD_ANGLE_MARK_A,
    BBOX_THETA_MARK_A,
    BBOX_CORNER_X_MARK_B,
    BBOX_CORNER_Y_MARK_B,
    BBOX_WORLD_ANGLE_MARK_B,
    BBOX_THETA_MARK_B,
    BBOX_SCAN,
    VALUE,
    ANGLE_VALUE,
}

_SEG_TYPES = {
    WORLD_ANGLE_MARK_A,
    THETA_MARK_A,
    WORLD_ANGLE_MARK_B,
    THETA_MARK_B,
    ANGLE_VALUE,
    FIND_RUN,
    EMIT_X2,
    R_STORE_WALL_RANGE,
    SEG_KPART,
    SEG_DC_TMID_MID,
    SEG_DC_TMID_UPPER,
    SEG_DC_TMID_LOWER,
    DRAWSEG_META,
    DRAWSEG_SCALE1_DEN,
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE2_DEN,
    DRAWSEG_SCALE2,
    DRAWSEG_SCALESTEP_DEN,
    DRAWSEG_SCALESTEP,
    DRAWSEG_BSILHEIGHT,
    DRAWSEG_TSILHEIGHT,
    DRAWSEG_U_PHASE,
    R_CHECK_PLANE,
    R_CHECK_PLANE_RESULT,
    VALUE,
    SET_CURSOR_X,
    WALL_COL_U,
    SCREEN_Y_VALUE,
    SCREEN_RANGE,
    PLANE_MARK,
    WALL_SPAN_META,
    SET_CURSOR_Y,
    PIXEL,
    CLIP_UPDATE,
}

_WALL_RANGE_TYPES = {
    R_STORE_WALL_RANGE,
    SEG_KPART,
    SEG_DC_TMID_MID,
    SEG_DC_TMID_UPPER,
    SEG_DC_TMID_LOWER,
    DRAWSEG_META,
    DRAWSEG_SCALE1_DEN,
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE2_DEN,
    DRAWSEG_SCALE2,
    DRAWSEG_SCALESTEP_DEN,
    DRAWSEG_SCALESTEP,
    DRAWSEG_BSILHEIGHT,
    DRAWSEG_TSILHEIGHT,
    DRAWSEG_U_PHASE,
    R_CHECK_PLANE,
    R_CHECK_PLANE_RESULT,
    VALUE,
    ANGLE_VALUE,
    SET_CURSOR_X,
    WALL_COL_U,
    SCREEN_Y_VALUE,
    SCREEN_RANGE,
    PLANE_MARK,
    WALL_SPAN_META,
    SET_CURSOR_Y,
    PIXEL,
    CLIP_UPDATE,
}

_WALL_COLUMN_TYPES = {
    SET_CURSOR_X,
    WALL_COL_U,
    VALUE,
    SCREEN_Y_VALUE,
    SCREEN_RANGE,
    PLANE_MARK,
    WALL_SPAN_META,
    SET_CURSOR_Y,
    PIXEL,
    CLIP_UPDATE,
}

_FLAT_SCAN_TYPES = {
    DRAW_PLANES_BEGIN,
    SET_CURSOR_DIRECTION_X,
    FLAT_NEXT_PLANE,
    FLAT_NEXT_VP,
    FLAT_VISPLANE_BEGIN,
    MAKE_SPANS_COL,
    SPAN_CLOSE_SLOT,
    SPAN_ROW,
    SET_CURSOR_X,
    SET_CURSOR_Y,
    PIXEL,
}

# Per-column span-emission tokens within a single visplane (everything a
# MAKE_SPANS_COL column transition can emit, minus the column marker and
# the visplane terminator). A `_VisplaneSpanState` absorbs these as
# in-visplane mispredicts rather than ending the visplane early.
_SPAN_INTERIOR_TYPES = {
    SPAN_CLOSE_SLOT,
    SPAN_ROW,
    SET_CURSOR_Y,
    SET_CURSOR_X,
    PIXEL,
}


@dataclass
class _ClipRange:
    first: int
    last: int


@dataclass(frozen=True)
class _SpanEmit:
    part: str
    y1: int
    y2: int
    span_ordinal: int = 0
    dc_iscale: float = 0.0
    dc_tmid: float = 0.0
    h_scaled: int = 0
    tex_id: int = 0


@dataclass(frozen=True)
class _PlaneMarkEmit:
    plane_id: int
    vp: int
    plane_kind: str
    y1: int
    y2: int


# =============================================================================
# VALUE range helpers mirroring `value_ranges.py`.
#
# The impl emits raw values through `make_value(range, value)`, which
# normalizes into the shared VALUE.v slot. The drafter predicts the same
# normalized slot value here and decodes actual feedback with the range
# selected by the preceding marker.
# =============================================================================


def _silheight_token_value(height: float) -> float:
    if height >= ref.SIL_HEIGHT_MAX * 0.5:
        return 256.0
    if height <= ref.SIL_HEIGHT_MIN * 0.5:
        return -256.0
    return height


def _scale_denominator(seg, raw_seg, state: GameState, x: int) -> float:
    ctx = ref._doom_scale_context(seg, raw_seg, state)
    visangle = ref._normalize_angle(ctx.view_angle + ref._xtoviewangle_rad(x))
    anglea = math.pi / 2.0 + ref._angle_diff(visangle, ctx.view_angle)
    sinea = max(math.sin(anglea), 1e-6)
    raw_den = ctx.rw_distance * sinea
    if ctx.rw_distance < 1.0:
        return 1024.0 * max(0.0007, min(1.0, raw_den))
    return max(0.7, min(1500.0, raw_den))


def _quantize_value(raw: float) -> float:
    slot = VALUE.slots["v"]
    span = slot.hi - slot.lo
    t = (raw - slot.lo) / span
    t = max(0.0, min(1.0, t))
    idx = round(t * (slot.levels - 1))
    return slot.lo + (idx / (slot.levels - 1)) * span


def _signed_bam(angle: int) -> int:
    wrapped = int(angle) % ANGLE_BAM
    return wrapped if wrapped < _BAM_HALF else wrapped - ANGLE_BAM


def _world_angle_bam(vx: float, vy: float, px: float, py: float) -> int:
    angle_world = math.atan2(vy - py, vx - px)
    unsigned = round(angle_world * ANGLE_BAM / (2 * math.pi)) % ANGLE_BAM
    return _signed_bam(unsigned)


def _seg_normal_angle_bam(raw_angle: int) -> int:
    seg_angle = round(raw_angle * ANGLE_BAM / 65536)
    return _signed_bam(seg_angle + ANGLE_BAM // 4)


def _u_phase_angle_bam(raw_angle: int, view_angle_bam: int) -> int:
    return _signed_bam(view_angle_bam - _seg_normal_angle_bam(raw_angle))


def _covering_range(ranges: list[_ClipRange], x: int) -> _ClipRange | None:
    for entry in ranges:
        if entry.first <= x <= entry.last:
            return entry
    return None


def _next_solid_start(ranges: list[_ClipRange], x: int) -> int:
    for entry in sorted(ranges, key=lambda r: r.first):
        if entry.first > x:
            return entry.first
    return SCREEN_WIDTH


def _add_solid_range(ranges: list[_ClipRange], first: int, last: int) -> None:
    ranges.append(_ClipRange(first, last))
    ranges.sort(key=lambda r: r.first)


def _clamp_span_y(value: int) -> int:
    return max(0, min(SCREEN_HEIGHT - 1, int(value)))


def _append_plane_mark_emit(
    out: list[_PlaneMarkEmit],
    *,
    plane_id: int,
    vp: int,
    plane_kind: str,
    y1: int,
    y2: int,
) -> None:
    y1 = max(0, y1)
    y2 = min(SCREEN_HEIGHT - 1, y2)
    if y1 <= y2:
        out.append(
            _PlaneMarkEmit(
                plane_id=plane_id,
                vp=vp,
                plane_kind=plane_kind,
                y1=y1,
                y2=y2,
            )
        )


def _plane_mark_emits(
    *,
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
) -> list[_PlaneMarkEmit]:
    out: list[_PlaneMarkEmit] = []
    if markceiling and ceiling_plane_id is not None and ceiling_vp is not None:
        top = ceilingclip + 1
        bottom = yl - 1
        if bottom >= floorclip:
            bottom = floorclip - 1
        _append_plane_mark_emit(
            out,
            plane_id=ceiling_plane_id,
            vp=ceiling_vp,
            plane_kind="ceiling",
            y1=top,
            y2=bottom,
        )
    if markfloor and floor_plane_id is not None and floor_vp is not None:
        top = yh + 1
        bottom = floorclip - 1
        if top <= ceilingclip:
            top = ceilingclip + 1
        _append_plane_mark_emit(
            out,
            plane_id=floor_plane_id,
            vp=floor_vp,
            plane_kind="floor",
            y1=top,
            y2=bottom,
        )
    return out


def _check_plane_tokens(plane_kind: str, selected_vp: int) -> list[Token]:
    kind_code = _PLANE_KIND_CODE[plane_kind]
    out = [
        Token(R_CHECK_PLANE, {"kind": kind_code, "vp": vp})
        for vp in range(int(selected_vp) + 1)
    ]
    out.append(
        Token(
            R_CHECK_PLANE_RESULT,
            {
                "kind": kind_code,
                "vp": int(selected_vp),
            },
        )
    )
    return out


def _emit_angle(angle: int) -> Token:
    return Token(ANGLE_VALUE, {"angle": _signed_bam(angle)})


def _emit_value(range_id: ValueRange, value: float) -> Token:
    return Token(VALUE, {"v": _quantize_value(encode_float(range_id, value))})


def _project_from_theta_values(
    theta1: int,
    theta2: int,
    *,
    on_span_wrap: str = "cull",
) -> tuple[int, int] | None:
    clipped = ref._fov_clip(
        int(theta1) % ANGLE_BAM,
        int(theta2) % ANGLE_BAM,
        on_span_wrap=on_span_wrap,
    )
    if clipped is None:
        return None
    a1, a2 = clipped
    a1_signed = a1 - ANGLE_BAM if a1 >= ANGLE_BAM // 2 else a1
    a2_signed = a2 - ANGLE_BAM if a2 >= ANGLE_BAM // 2 else a2
    sx1 = ref.viewangletox(a1_signed)
    sx2 = ref.viewangletox(a2_signed)
    if sx1 == sx2:
        return None
    return (sx1, sx2) if sx1 <= sx2 else (sx2, sx1)


@dataclass
class _Context:
    scene: Scene
    state: GameState
    md: object
    segments: list
    viewz: float
    view_angle_bam: int
    plane_tables: object
    runtime_visplanes: ref._RuntimeVisplanes
    solidsegs: list[_ClipRange]
    ceilingclip: list[int]
    floorclip: list[int]
    side_table: dict[int, int]
    stack: list

    def __deepcopy__(self, memo):
        """Copy only mutable rollout state; share immutable scene data.

        Draft-decode snapshots are taken once per batch. The scene, map,
        baked segments, and plane tables are read-only across drafter
        operation, and copying them pulls in large pydantic object graphs.
        The mutable state is the horizontal/vertical clip state, side table,
        and traversal stack.
        """
        copied = _Context(
            scene=self.scene,
            state=self.state,
            md=self.md,
            segments=self.segments,
            viewz=self.viewz,
            view_angle_bam=self.view_angle_bam,
            plane_tables=self.plane_tables,
            runtime_visplanes=copy.deepcopy(self.runtime_visplanes, memo),
            solidsegs=[_ClipRange(r.first, r.last) for r in self.solidsegs],
            ceilingclip=list(self.ceilingclip),
            floorclip=list(self.floorclip),
            side_table=dict(self.side_table),
            stack=[],
        )
        memo[id(self)] = copied
        copied.stack = copy.deepcopy(self.stack, memo)
        return copied


def _child_first_token(is_subsector: bool, idx: int, depth: int) -> Token:
    if is_subsector:
        return Token(VISIT_SUBSECTOR, {"s": idx, "depth": depth})
    return Token(TRAVERSE_ENTER, {"node": idx, "depth": depth})


def _push_child(ctx: _Context, is_subsector: bool, idx: int, depth: int) -> None:
    if is_subsector:
        ctx.stack.append(_SubsectorFrame(idx, depth))
    else:
        ctx.stack.append(_NodeFrame(idx, depth))


class _BBoxState:
    def __init__(self, ctx: _Context, node_idx: int) -> None:
        self.node_idx = node_idx
        self.phase = "boxpos"
        self.boxpos: int | None = None
        self.region: tuple[int, int] | None = None
        self.coords: dict[str, float] = {}
        self.proj_first: int | None = None
        self.proj_last: int | None = None
        self.scan_x: int | None = None
        self.completion: str | None = None  # "open" or "closed"
        self.theta1: int | None = None
        self.theta2: int | None = None
        self._init_bbox(ctx)

    def _init_bbox(self, ctx: _Context) -> None:
        node = ctx.md.nodes[self.node_idx]
        side = ctx.side_table.get(
            self.node_idx, side_P(make_plane(node), ctx.state.x, ctx.state.y)
        )
        bbox = node.back_bbox if side == 1 else node.front_bbox
        top, bottom, left, right = bbox
        region = ref._viewport_region(
            ctx.state.x, ctx.state.y, top, bottom, left, right
        )
        self.region = region
        self.boxpos = region[0] * 4 + region[1]
        bspcoord = (top, bottom, left, right)
        if region != (1, 1):
            x1_idx, y1_idx, x2_idx, y2_idx = ref._CHECKCOORD[region]
            self.coords = {
                "cx1": float(bspcoord[x1_idx]),
                "cy1": float(bspcoord[y1_idx]),
                "cx2": float(bspcoord[x2_idx]),
                "cy2": float(bspcoord[y2_idx]),
            }

    def next_token(self, ctx: _Context) -> Token | None:
        if self.phase == "boxpos":
            return Token(BBOX_BOXPOS, {"boxpos": int(self.boxpos or 0)})
        if self.phase == "after_boxpos":
            if self.region == (1, 1):
                self.completion = "open"
                return None
            self.phase = "corner_x1"
            return self.next_token(ctx)
        if self.phase == "corner_x1":
            return Token(BBOX_CORNER_X_MARK_A, {"boxpos": int(self.boxpos or 0)})
        if self.phase == "value_x1":
            return _emit_value(ValueRange.R0, self.coords.get("cx1", 0.0))
        if self.phase == "corner_y1":
            return Token(BBOX_CORNER_Y_MARK_A, {"boxpos": int(self.boxpos or 0)})
        if self.phase == "value_y1":
            return _emit_value(ValueRange.R0, self.coords.get("cy1", 0.0))
        if self.phase == "angle1":
            return Token(BBOX_WORLD_ANGLE_MARK_A)
        if self.phase == "angle1_value":
            return _emit_angle(
                _world_angle_bam(
                    self.coords.get("cx1", 0.0),
                    self.coords.get("cy1", 0.0),
                    ctx.state.x,
                    ctx.state.y,
                )
            )
        if self.phase == "theta1":
            return Token(BBOX_THETA_MARK_A)
        if self.phase == "theta1_value":
            return _emit_angle(
                ref._theta_bam(
                    self.coords.get("cx1", 0.0),
                    self.coords.get("cy1", 0.0),
                    ctx.state.x,
                    ctx.state.y,
                    ctx.view_angle_bam,
                )
            )
        if self.phase == "corner_x2":
            return Token(BBOX_CORNER_X_MARK_B, {"boxpos": int(self.boxpos or 0)})
        if self.phase == "value_x2":
            return _emit_value(ValueRange.R0, self.coords.get("cx2", 0.0))
        if self.phase == "corner_y2":
            return Token(BBOX_CORNER_Y_MARK_B, {"boxpos": int(self.boxpos or 0)})
        if self.phase == "value_y2":
            return _emit_value(ValueRange.R0, self.coords.get("cy2", 0.0))
        if self.phase == "angle2":
            return Token(BBOX_WORLD_ANGLE_MARK_B)
        if self.phase == "angle2_value":
            return _emit_angle(
                _world_angle_bam(
                    self.coords.get("cx2", 0.0),
                    self.coords.get("cy2", 0.0),
                    ctx.state.x,
                    ctx.state.y,
                )
            )
        if self.phase == "theta2":
            return Token(BBOX_THETA_MARK_B)
        if self.phase == "theta2_value":
            return _emit_angle(
                ref._theta_bam(
                    self.coords.get("cx2", 0.0),
                    self.coords.get("cy2", 0.0),
                    ctx.state.x,
                    ctx.state.y,
                    ctx.view_angle_bam,
                )
            )
        if self.phase == "decision":
            self._prepare_scan(ctx)
            return self.next_token(ctx)
        if self.phase == "scan":
            return Token(BBOX_SCAN, {"x": int(self.scan_x or 0)})
        if self.phase == "complete":
            return None
        raise AssertionError(f"unknown bbox phase {self.phase}")

    def _prepare_scan(self, ctx: _Context) -> None:
        if self.theta1 is not None and self.theta2 is not None:
            projected = _project_from_theta_values(
                self.theta1,
                self.theta2,
                on_span_wrap="keep",
            )
        else:
            projected = ref._project_seg_endpoints(
                self.coords.get("cx1", 0.0),
                self.coords.get("cy1", 0.0),
                self.coords.get("cx2", 0.0),
                self.coords.get("cy2", 0.0),
                ctx.state.x,
                ctx.state.y,
                ctx.view_angle_bam,
                on_span_wrap="keep",
            )
        if projected is None or projected[0] == projected[1]:
            if (
                self.theta1 is not None
                and self.theta2 is not None
                and (
                    (
                        self.theta1 <= -ref.FOV_HALF_BAM
                        and self.theta2 >= ref.FOV_HALF_BAM
                    )
                    or (
                        self.theta2 <= -ref.FOV_HALF_BAM
                        and self.theta1 >= ref.FOV_HALF_BAM
                    )
                )
            ):
                projected = (0, SCREEN_WIDTH - 1)
            else:
                self.completion = "closed"
                self.phase = "complete"
                return
        if projected[0] == projected[1]:
            self.completion = "closed"
            self.phase = "complete"
            return
        self.proj_first, self.proj_last = projected
        self.scan_x = self.proj_first
        self.phase = "scan"

    def consume(self, ctx: _Context, actual: Token) -> bool:
        t = actual.type
        if self.phase == "complete" and t == BBOX_SCAN:
            self.phase = "scan"
            self.completion = None
            self.scan_x = int(actual.values.get("x", self.scan_x or 0))

        if self.phase == "boxpos" and t == BBOX_BOXPOS:
            self.boxpos = int(actual.values["boxpos"])
            self.region = (self.boxpos // 4, self.boxpos % 4)
            if self.region != (1, 1) and not self.coords:
                self._init_bbox(ctx)
            self.phase = "after_boxpos"
            return True
        if self.phase == "after_boxpos":
            if t == BBOX_CORNER_X_MARK_A:
                self.phase = "value_x1"
                return True
            if self.region == (1, 1):
                self.completion = "open"
                self.phase = "complete"
                return False
            self.phase = "corner_x1"
            return self.consume(ctx, actual)
        if self.phase == "corner_x1" and t == BBOX_CORNER_X_MARK_A:
            self.phase = "value_x1"
            return True
        if self.phase == "value_x1" and t == VALUE:
            self.coords["cx1"] = decode_float(ValueRange.R0, float(actual.values["v"]))
            self.phase = "corner_y1"
            return True
        if self.phase == "corner_y1" and t == BBOX_CORNER_Y_MARK_A:
            self.phase = "value_y1"
            return True
        if self.phase == "value_y1" and t == VALUE:
            self.coords["cy1"] = decode_float(ValueRange.R0, float(actual.values["v"]))
            self.phase = "angle1"
            return True
        if self.phase == "angle1" and t == BBOX_WORLD_ANGLE_MARK_A:
            self.phase = "angle1_value"
            return True
        if self.phase == "angle1_value" and t == ANGLE_VALUE:
            self.phase = "theta1"
            return True
        if self.phase == "theta1" and t == BBOX_THETA_MARK_A:
            self.phase = "theta1_value"
            return True
        if self.phase == "theta1_value" and t == ANGLE_VALUE:
            self.theta1 = int(actual.values["angle"])
            self.phase = "corner_x2"
            return True
        if self.phase == "corner_x2" and t == BBOX_CORNER_X_MARK_B:
            self.phase = "value_x2"
            return True
        if self.phase == "value_x2" and t == VALUE:
            self.coords["cx2"] = decode_float(ValueRange.R0, float(actual.values["v"]))
            self.phase = "corner_y2"
            return True
        if self.phase == "corner_y2" and t == BBOX_CORNER_Y_MARK_B:
            self.phase = "value_y2"
            return True
        if self.phase == "value_y2" and t == VALUE:
            self.coords["cy2"] = decode_float(ValueRange.R0, float(actual.values["v"]))
            self.phase = "angle2"
            return True
        if self.phase == "angle2" and t == BBOX_WORLD_ANGLE_MARK_B:
            self.phase = "angle2_value"
            return True
        if self.phase == "angle2_value" and t == ANGLE_VALUE:
            self.phase = "theta2"
            return True
        if self.phase == "theta2" and t == BBOX_THETA_MARK_B:
            self.phase = "theta2_value"
            return True
        if self.phase == "theta2_value" and t == ANGLE_VALUE:
            self.theta2 = int(actual.values["angle"])
            self.phase = "decision"
            return True
        if self.phase == "decision":
            if t == BBOX_SCAN:
                self._prepare_scan(ctx)
                if self.phase == "complete":
                    self.phase = "scan"
                    self.completion = None
                return self.consume(ctx, actual)
            self._prepare_scan(ctx)
            return False
        if self.phase == "scan" and t == BBOX_SCAN:
            x = int(actual.values["x"])
            self.scan_x = x
            last = self.proj_last if self.proj_last is not None else SCREEN_WIDTH - 1
            if x > last:
                self.completion = "closed"
                self.phase = "complete"
                return True
            covered = _covering_range(ctx.solidsegs, x)
            if covered is None:
                self.completion = "open"
                self.phase = "complete"
                return True
            x = covered.last + 1
            if x > last:
                if self.scan_x != min(x, SCREEN_WIDTH):
                    self.scan_x = min(x, SCREEN_WIDTH)
                    return True
                self.completion = "closed"
                self.phase = "complete"
                return True
            self.scan_x = x
            return True
        return False


@dataclass
class _NodeFrame:
    node_idx: int
    depth: int
    phase: str = "enter"
    bbox: _BBoxState | None = None

    def _side(self, ctx: _Context) -> int:
        node = ctx.md.nodes[self.node_idx]
        return ctx.side_table.get(
            self.node_idx, side_P(make_plane(node), ctx.state.x, ctx.state.y)
        )

    def _children(self, ctx: _Context):
        node = ctx.md.nodes[self.node_idx]
        side = self._side(ctx)
        if side == 1:
            return decode_child(node.front_child), decode_child(node.back_child)
        return decode_child(node.back_child), decode_child(node.front_child)

    def next_token(self, ctx: _Context) -> Token | None:
        if self.phase == "enter":
            return Token(TRAVERSE_ENTER, {"node": self.node_idx, "depth": self.depth})
        if self.phase == "between":
            return Token(TRAVERSE_BETWEEN, {"node": self.node_idx, "depth": self.depth})
        if self.phase == "bbox":
            if self.bbox is None:
                self.bbox = _BBoxState(ctx, self.node_idx)
            tok = self.bbox.next_token(ctx)
            if tok is not None:
                return tok
            if self.bbox.completion == "open":
                _, back = self._children(ctx)
                return _child_first_token(back[0], back[1], self.depth + 1)
            return Token(
                TRAVERSE_RETURN,
                {
                    "entity_u": self.node_idx,
                    "depth": self.depth,
                },
            )
        if self.phase == "return":
            return Token(
                TRAVERSE_RETURN,
                {
                    "entity_u": self.node_idx,
                    "depth": self.depth,
                },
            )
        raise AssertionError(f"unknown node phase {self.phase}")

    def consume(self, ctx: _Context, actual: Token) -> bool:
        if self.phase == "enter":
            if actual.type == TRAVERSE_ENTER:
                self.phase = "between"
                front, _ = self._children(ctx)
                _push_child(ctx, front[0], front[1], self.depth + 1)
                return True
            return True

        if self.phase == "between":
            if actual.type == TRAVERSE_BETWEEN:
                self.phase = "bbox"
                self.bbox = _BBoxState(ctx, self.node_idx)
                return True
            return True

        if self.phase == "bbox":
            if self.bbox is None:
                self.bbox = _BBoxState(ctx, self.node_idx)
            if actual.type in _BBOX_TYPES and self.bbox.consume(ctx, actual):
                return True
            if actual.type in {TRAVERSE_ENTER, VISIT_SUBSECTOR}:
                self.phase = "return"
                _, back = self._children(ctx)
                _push_child(ctx, back[0], back[1], self.depth + 1)
                return False
            if actual.type == TRAVERSE_RETURN:
                ctx.stack.pop()
                return True
            if self.bbox.completion == "open":
                self.phase = "return"
                _, back = self._children(ctx)
                _push_child(ctx, back[0], back[1], self.depth + 1)
                return False
            if self.bbox.completion == "closed":
                self.phase = "return"
                return False
            return True

        if self.phase == "return":
            if actual.type == TRAVERSE_RETURN:
                ctx.stack.pop()
                return True
            return True

        return True


class _SegState:
    def __init__(self, ctx: _Context, seg_idx: int, subsector_idx: int) -> None:
        self.seg_idx = seg_idx
        self.subsector_idx = subsector_idx
        self.phase = "projection"
        self.proj_index = 0
        self.scan: _HorizontalScanState | None = None
        self.completion: str | None = None
        self.theta1: int | None = None
        self.theta2: int | None = None
        if self._skip_before_projection(ctx):
            self.phase = "complete"
            self.completion = "skip_before_projection"

    def _skip_before_projection(self, ctx: _Context) -> bool:
        seg = ctx.segments[self.seg_idx]
        cross_z = (seg.bx - seg.ax) * (ctx.state.y - seg.ay) - (seg.by - seg.ay) * (
            ctx.state.x - seg.ax
        )
        return cross_z >= 0 or ref._is_empty_line(seg)

    def _projection_sequence(self, ctx: _Context) -> list[Token]:
        seg = ctx.segments[self.seg_idx]
        return [
            Token(WORLD_ANGLE_MARK_A),
            _emit_angle(_world_angle_bam(seg.ax, seg.ay, ctx.state.x, ctx.state.y)),
            Token(THETA_MARK_A),
            _emit_angle(
                ref._theta_bam(
                    seg.ax,
                    seg.ay,
                    ctx.state.x,
                    ctx.state.y,
                    ctx.view_angle_bam,
                )
            ),
            Token(WORLD_ANGLE_MARK_B),
            _emit_angle(_world_angle_bam(seg.bx, seg.by, ctx.state.x, ctx.state.y)),
            Token(THETA_MARK_B),
            _emit_angle(
                ref._theta_bam(
                    seg.bx,
                    seg.by,
                    ctx.state.x,
                    ctx.state.y,
                    ctx.view_angle_bam,
                )
            ),
        ]

    def _prepare_scan(self, ctx: _Context, first_override: int | None = None) -> None:
        seg = ctx.segments[self.seg_idx]
        if self.theta1 is not None and self.theta2 is not None:
            projected = _project_from_theta_values(self.theta1, self.theta2)
        else:
            projected = ref._project_seg_endpoints(
                seg.ax,
                seg.ay,
                seg.bx,
                seg.by,
                ctx.state.x,
                ctx.state.y,
                ctx.view_angle_bam,
            )
        if projected is None or projected[0] == projected[1]:
            if first_override is None:
                self.phase = "complete"
                self.completion = "advance"
                return
            first = first_override
            last = SCREEN_WIDTH - 1
        else:
            first, last = projected
            if first_override is not None:
                first = first_override
        solid_wall = not (seg.is_two_sided and not ref._is_closed(seg))
        self.scan = _HorizontalScanState(
            seg_idx=self.seg_idx,
            subsector_idx=self.subsector_idx,
            first=first,
            last=last,
            solid_wall=solid_wall,
        )
        self.phase = "scan"
        self.completion = None

    def next_token(self, ctx: _Context) -> Token | None:
        if self.phase == "complete":
            return None
        if self.phase == "projection":
            seq = self._projection_sequence(ctx)
            if self.proj_index < len(seq):
                return seq[self.proj_index]
            self._prepare_scan(ctx)
            return self.next_token(ctx)
        if self.phase == "scan":
            if self.scan is None:
                self._prepare_scan(ctx)
            if self.scan is None:
                return None
            tok = self.scan.next_token(ctx)
            if tok is not None:
                return tok
            self.completion = self.scan.completion
            self.phase = "complete"
            return None
        raise AssertionError(f"unknown seg phase {self.phase}")

    def consume(self, ctx: _Context, actual: Token) -> bool:
        if self.phase == "complete":
            if actual.type in {
                WORLD_ANGLE_MARK_A,
                THETA_MARK_A,
                WORLD_ANGLE_MARK_B,
                THETA_MARK_B,
                ANGLE_VALUE,
            }:
                self.phase = "projection"
                self.proj_index = 0
                self.completion = None
            elif actual.type == FIND_RUN:
                self._prepare_scan(ctx, int(actual.values["x"]))
                return self.consume(ctx, actual)
            elif actual.type in _WALL_RANGE_TYPES:
                x = int(actual.values.get("x", 0))
                self._prepare_scan(ctx, x)
                return self.consume(ctx, actual)
            else:
                return False

        if self.phase == "projection":
            seq = self._projection_sequence(ctx)
            if self.proj_index < len(seq) and actual.type == seq[self.proj_index].type:
                if actual.type == ANGLE_VALUE and self.proj_index == 3:
                    self.theta1 = int(actual.values["angle"])
                elif actual.type == ANGLE_VALUE and self.proj_index == 7:
                    self.theta2 = int(actual.values["angle"])
                self.proj_index += 1
                if self.proj_index >= len(seq):
                    self._prepare_scan(ctx)
                return True
            if actual.type == FIND_RUN:
                self._prepare_scan(ctx, int(actual.values["x"]))
                return self.consume(ctx, actual)
            if actual.type in {ADVANCE_SEG, PROCESS_SEG, TRAVERSE_RETURN}:
                self._prepare_scan(ctx)
                return False
            self.proj_index = min(self.proj_index + 1, len(seq))
            if self.proj_index >= len(seq):
                self._prepare_scan(ctx)
            return True

        if self.phase == "scan":
            if self.scan is None:
                self._prepare_scan(ctx)
            if self.scan is not None and actual.type in _SEG_TYPES:
                consumed = self.scan.consume(ctx, actual)
                if consumed:
                    if self.scan.completion is not None:
                        self.completion = self.scan.completion
                        self.phase = "complete"
                    return True
            if self.scan is not None and self.scan.completion is not None:
                self.completion = self.scan.completion
                self.phase = "complete"
            return False

        return False


@dataclass
class _HorizontalScanState:
    seg_idx: int
    subsector_idx: int
    first: int
    last: int
    solid_wall: bool
    phase: str = "find"
    x: int | None = None
    next_solid: int | None = None
    run_stop: int | None = None
    emitted_range: bool = False
    wall_range: object | None = None
    completion: str | None = None

    def __post_init__(self) -> None:
        self.x = self.first

    def next_token(self, ctx: _Context) -> Token | None:
        if self.completion is not None:
            return None
        if self.phase == "find":
            return Token(FIND_RUN, {"x": min(int(self.x or 0), SCREEN_WIDTH)})
        if self.phase == "final_find":
            return Token(FIND_RUN, {"x": min(int(self.x or 0), SCREEN_WIDTH)})
        if self.phase == "x2":
            return Token(
                EMIT_X2,
                {"x": int(self.run_stop if self.run_stop is not None else self.x or 0)},
            )
        if self.phase == "wall_range":
            assert self.wall_range is not None
            tok = self.wall_range.next_token(ctx)
            if tok is not None:
                return tok
            self._finish_wall_range(ctx)
            return self.next_token(ctx)
        return None

    def _start_run(self, ctx: _Context, x: int) -> None:
        self.x = x
        self.next_solid = _next_solid_start(ctx.solidsegs, x)
        self.run_stop = min(self.last, self.next_solid - 1)
        self.phase = "x2"

    def _finish_wall_range(self, ctx: _Context) -> None:
        if self.wall_range is None:
            return
        wr = self.wall_range
        if self.solid_wall:
            _add_solid_range(ctx.solidsegs, wr.record.x1, wr.record.x2)
        self.emitted_range = True
        self.x = wr.record.x2 + 1
        self.wall_range = None
        if self.x <= self.last:
            self.phase = "find"
        else:
            self.completion = "none"
            self.phase = "complete"

    def consume(self, ctx: _Context, actual: Token) -> bool:
        if self.completion is not None:
            return False
        if self.phase == "find" and actual.type == FIND_RUN:
            x = int(actual.values["x"])
            self.x = x
            covered = _covering_range(ctx.solidsegs, x)
            if covered is not None:
                self.x = covered.last + 1
                if self.x > self.last:
                    self.phase = "final_find"
                return True
            self._start_run(ctx, x)
            return True
        if self.phase == "final_find" and actual.type == FIND_RUN:
            self.completion = "advance"
            self.phase = "complete"
            return True
        if self.phase == "x2" and actual.type == EMIT_X2:
            self.run_stop = int(actual.values["x"])
            self.wall_range = _WallRangeState(
                ctx,
                self.seg_idx,
                self.subsector_idx,
                int(self.x or 0),
                int(self.run_stop),
            )
            self.phase = "wall_range"
            return True
        if self.phase == "wall_range":
            assert self.wall_range is not None
            if actual.type in _WALL_RANGE_TYPES and self.wall_range.consume(
                ctx, actual
            ):
                if self.wall_range.complete:
                    self._finish_wall_range(ctx)
                return True
            if self.wall_range.complete:
                self._finish_wall_range(ctx)
                return False
        if actual.type == R_STORE_WALL_RANGE:
            x = int(actual.values.get("x", self.x or 0))
            self.run_stop = self.run_stop if self.run_stop is not None else x
            self.wall_range = _WallRangeState(
                ctx, self.seg_idx, self.subsector_idx, x, int(self.run_stop)
            )
            self.phase = "wall_range"
            return self.consume(ctx, actual)
        if actual.type in {ADVANCE_SEG, PROCESS_SEG, TRAVERSE_RETURN}:
            self.completion = "advance" if not self.emitted_range else "none"
            self.phase = "complete"
            return False
        return True


class _WallRangeState:
    _PAYLOAD_MARKERS = [
        DRAWSEG_SCALE1_DEN,
        DRAWSEG_SCALE1,
        DRAWSEG_SCALE2_DEN,
        DRAWSEG_SCALE2,
        DRAWSEG_SCALESTEP_DEN,
        DRAWSEG_SCALESTEP,
        DRAWSEG_BSILHEIGHT,
        DRAWSEG_TSILHEIGHT,
    ]

    _TMID_PARTS = ("mid", "upper", "lower")
    _TMID_MARKERS = {
        "mid": SEG_DC_TMID_MID,
        "upper": SEG_DC_TMID_UPPER,
        "lower": SEG_DC_TMID_LOWER,
    }
    _TMID_RANGES = {
        "mid": ValueRange.R3,
        "upper": ValueRange.R4,
        "lower": ValueRange.R4,
    }
    _PAYLOAD_RANGES = {
        DRAWSEG_SCALE1_DEN: ValueRange.R6,
        DRAWSEG_SCALE1: ValueRange.R5,
        DRAWSEG_SCALE2_DEN: ValueRange.R6,
        DRAWSEG_SCALE2: ValueRange.R5,
        DRAWSEG_SCALESTEP_DEN: ValueRange.R7,
        DRAWSEG_SCALESTEP: ValueRange.R8,
        DRAWSEG_BSILHEIGHT: ValueRange.R9,
        DRAWSEG_TSILHEIGHT: ValueRange.R9,
    }

    def __init__(
        self, ctx: _Context, seg_idx: int, subsector_idx: int, x1: int, x2: int
    ) -> None:
        self.seg_idx = seg_idx
        self.subsector_idx = subsector_idx
        self.record = ref.StoreWallRangeRecord(seg_idx=seg_idx, x1=x1, x2=x2)
        self.meta = self._compute_meta(ctx)
        self.phase = "store"
        self.payload_index = 0
        self.last_marker = None
        self.columns: _WallColumnsState | None = None
        self.check_tokens: list[Token] = []
        self.check_index = 0
        self.complete = False
        self.pat = 0
        self.h_scaled_by_part: dict[str, int] = {"mid": 0, "upper": 0, "lower": 0}
        self.tex_id_by_part: dict[str, int] = {"mid": 0, "upper": 0, "lower": 0}
        self.tmid: dict[str, float] = {"mid": 0.0, "upper": 0.0, "lower": 0.0}
        self._init_kpart_and_tmid(ctx)
        self._tmid_index = 0

    def _init_kpart_and_tmid(self, ctx: _Context) -> None:
        seg = ctx.segments[self.seg_idx]
        flags, rowoffset = ref._seg_flags_and_rowoffset(ctx.md, self.seg_idx)
        name_to_id = ref._build_uv_texture_id_map()
        mid_id, upper_id, lower_id = ref._seg_part_tex_ids(
            ctx.md,
            self.seg_idx,
            name_to_id,
        )
        self.tex_id_by_part = {
            "mid": mid_id,
            "upper": upper_id,
            "lower": lower_id,
        }
        h_by_id = ref._texture_height_by_id()
        self.h_scaled_by_part = {
            "mid": h_by_id.get(mid_id, 0),
            "upper": h_by_id.get(upper_id, 0),
            "lower": h_by_id.get(lower_id, 0),
        }

        wall_kind = self.meta.wall_kind
        is_portal = wall_kind == "portal"
        solid_or_closed = not is_portal
        has_mid = solid_or_closed and mid_id != 0
        upper_geom = (
            is_portal
            and seg.back_ceiling is not None
            and seg.front_ceiling > seg.back_ceiling
        )
        lower_geom = (
            is_portal
            and seg.back_floor is not None
            and seg.back_floor > seg.front_floor
        )
        has_upper = is_portal and upper_id != 0 and upper_geom
        has_lower = is_portal and lower_id != 0 and lower_geom
        self.pat = (
            (4 if has_mid else 0) + (2 if has_upper else 0) + (1 if has_lower else 0)
        )

        # When a back sector is missing, the impl carries the sentinel
        # `BACK_HEIGHT_SENTINEL = -4096.0` through prefill; emit-side math
        # uses that sentinel instead of skipping the part. Mirror that here
        # so SEG_DC_TMID_{UPPER,LOWER} VALUEs match the impl's emission
        # (which then clamps to the FloatSlot bound).
        _BACK_SENTINEL = -4096.0
        for part in self._TMID_PARTS:
            h_scaled = self.h_scaled_by_part[part]
            patched_seg = seg
            if part == "upper" and seg.back_ceiling is None:
                patched_seg = replace(seg, back_ceiling=_BACK_SENTINEL)
            elif part == "lower" and seg.back_floor is None:
                patched_seg = replace(seg, back_floor=_BACK_SENTINEL)
            self.tmid[part] = ref._dc_texturemid_for(
                part=part,
                seg=patched_seg,
                flags=flags,
                rowoffset=rowoffset,
                h_scaled=h_scaled,
                viewz=ctx.viewz,
            )

    def _compute_meta(self, ctx: _Context) -> ref.DrawsegMetaRecord:
        seg = ctx.segments[self.seg_idx]
        raw_seg = ctx.md.segs[self.seg_idx]
        return ref._drawseg_meta(seg, raw_seg, self.record, ctx.state, ctx.viewz)

    def _payload_value(self, ctx: _Context, marker) -> float:
        seg = ctx.segments[self.seg_idx]
        raw_seg = ctx.md.segs[self.seg_idx]
        if marker == DRAWSEG_SCALE1_DEN:
            return _scale_denominator(seg, raw_seg, ctx.state, self.record.x1)
        if marker == DRAWSEG_SCALE1:
            return self.meta.scale1
        if marker == DRAWSEG_SCALE2_DEN:
            return _scale_denominator(seg, raw_seg, ctx.state, self.record.x2)
        if marker == DRAWSEG_SCALE2:
            return self.meta.scale2
        if marker == DRAWSEG_SCALESTEP_DEN:
            return float(self.record.x2 - self.record.x1)
        if marker == DRAWSEG_SCALESTEP:
            return self.meta.scalestep
        if marker == DRAWSEG_BSILHEIGHT:
            return _silheight_token_value(self.meta.bsilheight)
        if marker == DRAWSEG_TSILHEIGHT:
            return _silheight_token_value(self.meta.tsilheight)
        return 0.0

    def _payload_range(self, marker) -> ValueRange:
        return self._PAYLOAD_RANGES.get(marker, ValueRange.R3)

    def next_token(self, ctx: _Context) -> Token | None:
        if self.complete:
            return None
        if self.phase == "store":
            return Token(R_STORE_WALL_RANGE, {"i": self.seg_idx})
        if self.phase == "seg_kpart":
            return Token(SEG_KPART, {"pat": int(self.pat)})
        if self.phase == "tmid_marker":
            part = self._TMID_PARTS[self._tmid_index]
            return Token(self._TMID_MARKERS[part])
        if self.phase == "tmid_value":
            part = self._TMID_PARTS[self._tmid_index]
            return _emit_value(self._TMID_RANGES[part], self.tmid[part])
        if self.phase == "meta":
            return Token(
                DRAWSEG_META,
                {
                    "i": self.seg_idx,
                    "wall_kind": _WALL_KIND_CODE[self.meta.wall_kind],
                    "silhouette": self.meta.silhouette,
                },
            )
        if self.phase == "payload_marker":
            return Token(self._PAYLOAD_MARKERS[self.payload_index])
        if self.phase == "payload_value":
            range_id = self._payload_range(self.last_marker)
            return _emit_value(range_id, self._payload_value(ctx, self.last_marker))
        if self.phase == "u_phase_marker":
            return Token(DRAWSEG_U_PHASE)
        if self.phase == "u_phase_angle":
            return _emit_angle(
                _u_phase_angle_bam(
                    ctx.md.segs[self.seg_idx].angle,
                    ctx.view_angle_bam,
                )
            )
        if self.phase == "checks":
            if self.check_index < len(self.check_tokens):
                return self.check_tokens[self.check_index]
            self.phase = "columns"
            return self.next_token(ctx)
        if self.phase == "columns":
            if self.columns is None:
                self.columns = _WallColumnsState(
                    ctx,
                    self.record,
                    self.meta,
                    self.subsector_idx,
                    pat=self.pat,
                    tmid=self.tmid,
                    h_scaled_by_part=self.h_scaled_by_part,
                    tex_id_by_part=self.tex_id_by_part,
                )
            tok = self.columns.next_token(ctx)
            if tok is not None:
                return tok
            self.complete = True
            return None
        return None

    def _adopt_payload(self, marker, value: float) -> None:
        if marker == DRAWSEG_SCALE1:
            self.meta = replace(self.meta, scale1=decode_float(ValueRange.R5, value))
        elif marker == DRAWSEG_SCALE2:
            self.meta = replace(self.meta, scale2=decode_float(ValueRange.R5, value))
        elif marker == DRAWSEG_SCALESTEP:
            self.meta = replace(
                self.meta,
                scalestep=decode_float(ValueRange.R8, value),
            )

    def consume(self, ctx: _Context, actual: Token) -> bool:
        if self.complete:
            return False
        t = actual.type
        if self.phase == "store" and t == R_STORE_WALL_RANGE:
            self.seg_idx = int(actual.values.get("i", self.seg_idx))
            self.record = replace(self.record, seg_idx=self.seg_idx)
            self.meta = self._compute_meta(ctx)
            self._init_kpart_and_tmid(ctx)
            self.phase = "seg_kpart"
            return True
        if self.phase == "seg_kpart" and t == SEG_KPART:
            self.pat = int(actual.values.get("pat", self.pat))
            self._tmid_index = 0
            self.phase = "tmid_marker"
            return True
        if self.phase == "tmid_marker":
            expected = self._TMID_MARKERS[self._TMID_PARTS[self._tmid_index]]
            if t == expected:
                self.phase = "tmid_value"
                return True
            # Allow out-of-order: if it's still one of the three markers,
            # re-anchor to that one.
            for i, part in enumerate(self._TMID_PARTS):
                if t == self._TMID_MARKERS[part]:
                    self._tmid_index = i
                    self.phase = "tmid_value"
                    return True
        if self.phase == "tmid_value" and t == VALUE:
            part = self._TMID_PARTS[self._tmid_index]
            range_id = self._TMID_RANGES[part]
            self.tmid[part] = decode_float(
                range_id,
                float(actual.values.get("v", encode_float(range_id, self.tmid[part]))),
            )
            self._tmid_index += 1
            if self._tmid_index >= len(self._TMID_PARTS):
                self.phase = "meta"
            else:
                self.phase = "tmid_marker"
            return True
        if self.phase == "meta" and t == DRAWSEG_META:
            kind_code = int(
                actual.values.get("wall_kind", _WALL_KIND_CODE[self.meta.wall_kind])
            )
            self.meta = replace(
                self.meta,
                wall_kind=_WALL_KIND_BY_CODE.get(kind_code, self.meta.wall_kind),
                silhouette=int(actual.values.get("silhouette", self.meta.silhouette)),
            )
            self.phase = "payload_marker"
            return True
        if self.phase == "payload_marker" and t in self._PAYLOAD_MARKERS:
            self.last_marker = t
            try:
                self.payload_index = self._PAYLOAD_MARKERS.index(t)
            except ValueError:
                pass
            self.phase = "payload_value"
            return True
        if self.phase == "payload_value" and t == VALUE:
            if self.last_marker is not None:
                self._adopt_payload(self.last_marker, float(actual.values["v"]))
            self.payload_index += 1
            if self.payload_index >= len(self._PAYLOAD_MARKERS):
                self.phase = "u_phase_marker"
            else:
                self.phase = "payload_marker"
            return True
        if self.phase == "u_phase_marker" and t == DRAWSEG_U_PHASE:
            self.phase = "u_phase_angle"
            return True
        if self.phase == "u_phase_angle" and t == ANGLE_VALUE:
            self.columns = _WallColumnsState(
                ctx,
                self.record,
                self.meta,
                self.subsector_idx,
                pat=self.pat,
                tmid=self.tmid,
                h_scaled_by_part=self.h_scaled_by_part,
                tex_id_by_part=self.tex_id_by_part,
            )
            self.check_tokens = self.columns.check_tokens()
            self.check_index = 0
            self.phase = "checks" if self.check_tokens else "columns"
            return True
        if self.phase == "checks" and t in {R_CHECK_PLANE, R_CHECK_PLANE_RESULT}:
            if t == R_CHECK_PLANE_RESULT and self.columns is not None:
                kind = _PLANE_KIND_BY_CODE.get(
                    int(actual.values.get("kind", 0)), "ceiling"
                )
                vp = int(actual.values.get("vp", 0))
                if kind == "ceiling":
                    self.columns.ceiling_vp = vp
                else:
                    self.columns.floor_vp = vp
            self.check_index += 1
            if self.check_index >= len(self.check_tokens):
                self.phase = "columns"
            return True
        if self.phase == "columns":
            if self.columns is None:
                self.columns = _WallColumnsState(
                    ctx,
                    self.record,
                    self.meta,
                    self.subsector_idx,
                    pat=self.pat,
                    tmid=self.tmid,
                    h_scaled_by_part=self.h_scaled_by_part,
                    tex_id_by_part=self.tex_id_by_part,
                )
            if t in _WALL_COLUMN_TYPES and self.columns.consume(ctx, actual):
                if self.columns.complete:
                    self.complete = True
                return True
            if self.columns.complete:
                self.complete = True
                return False
        if t == SET_CURSOR_X:
            self.phase = "columns"
            self.columns = _WallColumnsState(
                ctx,
                self.record,
                self.meta,
                self.subsector_idx,
                pat=self.pat,
                tmid=self.tmid,
                h_scaled_by_part=self.h_scaled_by_part,
                tex_id_by_part=self.tex_id_by_part,
            )
            return self.consume(ctx, actual)
        return False


class _WallColumnsState:
    def __init__(
        self,
        ctx: _Context,
        record: ref.StoreWallRangeRecord,
        meta: ref.DrawsegMetaRecord,
        subsector_idx: int,
        *,
        pat: int = 0,
        tmid: dict[str, float] | None = None,
        h_scaled_by_part: dict[str, int] | None = None,
        tex_id_by_part: dict[str, int] | None = None,
    ) -> None:
        self.record = record
        self.meta = meta
        self.subsector_idx = subsector_idx
        self.ctx_scene = ctx.scene
        self.seg = ctx.segments[record.seg_idx]
        self.raw_seg = ctx.md.segs[record.seg_idx]
        self.scale_ctx = ref._doom_scale_context(self.seg, self.raw_seg, ctx.state)
        self.x = record.x1
        self.phase = "column"
        self.complete = record.x1 > record.x2
        self.scale_x = meta.scale1
        self.u_idx = 0
        self.pat = int(pat)
        self.tmid = dict(tmid) if tmid else {"mid": 0.0, "upper": 0.0, "lower": 0.0}
        self.h_scaled_by_part = (
            dict(h_scaled_by_part)
            if h_scaled_by_part
            else {"mid": 0, "upper": 0, "lower": 0}
        )
        self.tex_id_by_part = (
            dict(tex_id_by_part)
            if tex_id_by_part
            else {"mid": 0, "upper": 0, "lower": 0}
        )
        self.new_ceiling = -1
        self.new_floor = SCREEN_HEIGHT
        self.plane_marks: list[_PlaneMarkEmit] = []
        self.spans: list[_SpanEmit] = []
        self.plane_idx = 0
        self.span_idx = 0
        self.pending_y: tuple[int, int] = (0, 0)
        self.pending_y_pos = 0
        self.active_span: _SpanEmit | None = None
        self.pixel_y = 0
        self.pixel_y_end = -1
        self._init_wall_flags(ctx)

    def _init_wall_flags(self, ctx: _Context) -> None:
        seg = self.seg
        self.worldtop = seg.front_ceiling - ctx.viewz
        self.worldbottom = seg.front_floor - ctx.viewz
        self.worldhigh = (
            seg.back_ceiling - ctx.viewz
            if seg.back_ceiling is not None
            else self.worldtop
        )
        self.worldlow = (
            seg.back_floor - ctx.viewz
            if seg.back_floor is not None
            else self.worldbottom
        )
        self.upper_texture = ref._has_texture(seg.upper_texture_name)
        self.lower_texture = ref._has_texture(seg.lower_texture_name)
        if not seg.is_two_sided:
            self.markceiling = True
            self.markfloor = True
        else:
            self.markceiling = seg.back_ceiling != seg.front_ceiling
            self.markfloor = seg.back_floor != seg.front_floor
        if self.meta.wall_kind == "closed":
            self.markceiling = True
            self.markfloor = True
        if seg.front_floor >= ctx.viewz:
            self.markfloor = False
        plane_context = ctx.plane_tables.context_for_subsector(
            self.subsector_idx, ctx.viewz
        )
        if seg.front_ceiling <= ctx.viewz and not plane_context.ceiling_is_sky:
            self.markceiling = False
        self.ceiling_plane_id = plane_context.ceiling_plane_id
        self.floor_plane_id = plane_context.floor_plane_id
        self.ceiling_vp = (
            ref._check_plane(
                ctx.runtime_visplanes,
                plane_id=self.ceiling_plane_id,
                x1=self.record.x1,
                x2=self.record.x2,
            )
            if self.ceiling_plane_id is not None
            else None
        )
        self.floor_vp = (
            ref._check_plane(
                ctx.runtime_visplanes,
                plane_id=self.floor_plane_id,
                x1=self.record.x1,
                x2=self.record.x2,
            )
            if self.floor_plane_id is not None
            else None
        )

    def check_tokens(self) -> list[Token]:
        out: list[Token] = []
        if (
            self.markceiling
            and self.ceiling_plane_id is not None
            and self.ceiling_vp is not None
        ):
            out.extend(_check_plane_tokens("ceiling", self.ceiling_vp))
        if (
            self.markfloor
            and self.floor_plane_id is not None
            and self.floor_vp is not None
        ):
            out.extend(_check_plane_tokens("floor", self.floor_vp))
        return out

    def next_token(self, ctx: _Context) -> Token | None:
        if self.complete:
            return None
        if self.phase == "column":
            return Token(SET_CURSOR_X, {"x": self.x})
        if self.phase == "col_u":
            u_native = ref._texturecolumn_native(self.x, self.scale_ctx)
            self.u_idx = u_native
            return Token(WALL_COL_U, {"u_idx": int(self.u_idx)})
        if self.phase == "scale":
            scale_x = self.meta.scale1 + (self.x - self.record.x1) * self.meta.scalestep
            return _emit_value(ValueRange.R5, scale_x)
        if self.phase == "staged":
            return Token(SCREEN_Y_VALUE, {"y": int(self.new_ceiling)})
        if self.phase == "body":
            if self.plane_idx < len(self.plane_marks):
                mark = self.plane_marks[self.plane_idx]
                return Token(
                    PLANE_MARK,
                    {
                        "p": mark.plane_id,
                        "kind": _PLANE_KIND_CODE[mark.plane_kind],
                        "vp": mark.vp,
                    },
                )
            if self.span_idx < len(self.spans):
                span = self.spans[self.span_idx]
                return Token(
                    WALL_SPAN_META,
                    {
                        "y": int(span.y1),
                        "ordinal": int(span.span_ordinal),
                    },
                )
            return Token(CLIP_UPDATE)
        if self.phase == "body_y":
            return Token(
                SCREEN_RANGE,
                {
                    "y1": self.pending_y[0],
                    "y2": self.pending_y[1],
                },
            )
        if self.phase == "span_cursor_y":
            span = self.active_span
            return Token(SET_CURSOR_Y, {"y": int(span.y1 if span is not None else 0)})
        if self.phase == "span_v0":
            span = self.active_span
            if span is None:
                return _emit_value(ValueRange.R3, 0.0)
            v0_at_top = span.dc_tmid + (span.y1 - ref.CENTER_Y) * span.dc_iscale
            return _emit_value(ValueRange.R3, v0_at_top)
        if self.phase == "span_pixel":
            return self._emit_pixel_color()
        if self.phase == "clip_range":
            return Token(
                SCREEN_RANGE,
                {
                    "y1": int(self.new_ceiling),
                    "y2": int(self.new_floor),
                },
            )
        return None

    def _emit_pixel_color(self) -> Token:
        span = self.active_span
        if span is None or span.h_scaled <= 0 or span.tex_id <= 0:
            return Token(PIXEL, {"color": 0, "w": 1})
        tex = ref.ASSET_BOOK.wall_textures[span.tex_id - 1]
        src_x = self.u_idx % tex.width
        v_native = span.dc_tmid + (self.pixel_y - ref.CENTER_Y) * span.dc_iscale
        v_scaled = math.floor(v_native)
        src_y = v_scaled % span.h_scaled
        raw_idx = int(tex.pixels[src_x][src_y])
        colormap_row = ref._wall_colormap_row(
            md=self.ctx_scene.map_data,
            seg=self.seg,
            raw_seg=self.raw_seg,
            scale=self.scale_x,
        )
        lit_idx = ref.apply_doom_colormap(
            ref.COLORMAP_ROWS,
            colormap_row,
            raw_idx,
        )
        return Token(PIXEL, {"color": lit_idx, "w": 1})

    def _enter_active_span(self, span: _SpanEmit) -> None:
        self.active_span = span
        self.pixel_y = int(span.y1)
        self.pixel_y_end = int(span.y2)

    def _compute_column_body(self, ctx: _Context) -> None:
        old_ceiling = ctx.ceilingclip[self.x]
        old_floor = ctx.floorclip[self.x]
        top_y_raw = ref.CENTER_Y - self.worldtop * self.scale_x
        bot_y_raw = ref.CENTER_Y - self.worldbottom * self.scale_x
        yl = max(math.ceil(top_y_raw), old_ceiling + 1)
        yh = min(math.floor(bot_y_raw), old_floor - 1)
        self.plane_marks = _plane_mark_emits(
            yl=yl,
            yh=yh,
            ceilingclip=old_ceiling,
            floorclip=old_floor,
            markceiling=self.markceiling,
            markfloor=self.markfloor,
            ceiling_plane_id=self.ceiling_plane_id,
            ceiling_vp=self.ceiling_vp,
            floor_plane_id=self.floor_plane_id,
            floor_vp=self.floor_vp,
        )
        part_candidates: dict[str, _SpanEmit] = {}
        part_visible: dict[str, bool] = {}
        new_ceiling = old_ceiling
        new_floor = old_floor
        dc_iscale = 1.0 / self.scale_x if self.scale_x else 0.0

        def _record_part(part: str, y1: int, y2: int) -> None:
            # Map internal "middle" naming to dc_tmid/h_scaled lookup key
            # "mid" (the part identity the impl uses).
            lookup_key = "mid" if part == "middle" else part
            part_candidates[lookup_key] = _SpanEmit(
                part=part,
                y1=_clamp_span_y(y1),
                y2=_clamp_span_y(y2),
                span_ordinal=0,
                dc_iscale=dc_iscale,
                dc_tmid=self.tmid.get(lookup_key, 0.0),
                h_scaled=self.h_scaled_by_part.get(lookup_key, 0),
                tex_id=self.tex_id_by_part.get(lookup_key, 0),
            )
            part_visible[lookup_key] = y1 <= y2

        if self.meta.wall_kind in {"solid", "closed"}:
            _record_part("middle", yl, yh)
            new_ceiling = SCREEN_HEIGHT
            new_floor = -1
        else:
            if self.worldhigh < self.worldtop:
                mid = min(
                    math.floor(ref.CENTER_Y - self.worldhigh * self.scale_x),
                    old_floor - 1,
                )
                if self.upper_texture:
                    _record_part("upper", yl, mid)
                    if mid >= yl:
                        new_ceiling = ref._clamp_ceilingclip(mid)
                    else:
                        new_ceiling = ref._clamp_ceilingclip(yl - 1)
                elif self.markceiling:
                    new_ceiling = ref._clamp_ceilingclip(yl - 1)
            elif self.markceiling:
                new_ceiling = ref._clamp_ceilingclip(yl - 1)

            if self.worldlow > self.worldbottom:
                mid = math.ceil(ref.CENTER_Y - self.worldlow * self.scale_x)
                if mid <= new_ceiling:
                    mid = new_ceiling + 1
                if self.lower_texture:
                    _record_part("lower", mid, yh)
                    if mid <= yh:
                        new_floor = ref._clamp_floorclip(mid)
                    else:
                        new_floor = ref._clamp_floorclip(yh + 1)
                elif self.markfloor:
                    new_floor = ref._clamp_floorclip(yh + 1)
            elif self.markfloor:
                new_floor = ref._clamp_floorclip(yh + 1)
        # The implementation emits SET_CURSOR_Y(y) in K-order over existing
        # parts, skipping parts whose column span is empty.
        part_by_id = {0: "mid", 1: "upper", 2: "lower"}
        spans: list[_SpanEmit] = []
        for k in range(3):
            part_id = _K_PART_TABLES[k][self.pat]
            if part_id == 3:
                break
            part_key = part_by_id[part_id]
            if not part_visible.get(part_key, False):
                continue
            candidate = part_candidates.get(part_key)
            if candidate is None:
                continue
            spans.append(replace(candidate, span_ordinal=k))

        self.spans = spans
        self.new_ceiling = new_ceiling
        self.new_floor = new_floor
        self.plane_idx = 0
        self.span_idx = 0
        self.active_span = None
        self.pixel_y = 0
        self.pixel_y_end = -1

    def _start_body_y(self, y1: int, y2: int) -> None:
        self.pending_y = (int(y1), int(y2))
        self.pending_y_pos = 0
        self.phase = "body_y"

    def _consume_plane_mark(self, ctx: _Context, actual: Token) -> None:
        p = int(actual.values.get("p", -1))
        vp = int(actual.values.get("vp", 0))
        kind = _PLANE_KIND_BY_CODE.get(int(actual.values.get("kind", 0)), "ceiling")
        match_idx = None
        for i in range(self.plane_idx, len(self.plane_marks)):
            mark = self.plane_marks[i]
            if mark.plane_id == p and mark.vp == vp and mark.plane_kind == kind:
                match_idx = i
                break
        if match_idx is None:
            for i in range(self.plane_idx, len(self.plane_marks)):
                mark = self.plane_marks[i]
                if mark.plane_id == p and mark.plane_kind == kind:
                    match_idx = i
                    break
        if match_idx is None and self.plane_idx < len(self.plane_marks):
            match_idx = self.plane_idx
        if match_idx is not None:
            mark = self.plane_marks[match_idx]
            ctx.runtime_visplanes.publish_occupancy(p, vp, self.x)
            self.plane_idx = match_idx + 1
            self._start_body_y(mark.y1, mark.y2)
        else:
            self._start_body_y(0, 0)

    def _consume_wall_span_meta(self, actual: Token) -> None:
        """Adopt span metadata before the host-visible SET_CURSOR_Y."""
        y = int(actual.values.get("y", 0))
        ordinal = int(actual.values.get("ordinal", 0))
        self.plane_idx = len(self.plane_marks)
        match_idx = None
        for i in range(self.span_idx, len(self.spans)):
            span = self.spans[i]
            if span.span_ordinal == ordinal and span.y1 == y:
                match_idx = i
                break
        if match_idx is None:
            for i in range(self.span_idx, len(self.spans)):
                if self.spans[i].span_ordinal == ordinal:
                    match_idx = i
                    break
        if match_idx is None and self.span_idx < len(self.spans):
            match_idx = self.span_idx
        if match_idx is not None:
            span = self.spans[match_idx]
            self.span_idx = match_idx + 1
            self._enter_active_span(span)
        else:
            self.active_span = _SpanEmit(
                part="middle",
                y1=0,
                y2=-1,
            )
            self.pixel_y = 0
            self.pixel_y_end = -1
        self.phase = "span_cursor_y"

    def _consume_set_cursor_y(self, actual: Token) -> None:
        """Adopt the impl's SET_CURSOR_Y(y) and prep pixel emission.

        The drafter pre-built `self.spans` in the order parts append; the
        impl's y identifies the chosen span. On a missing span (drafter built
        a different set than the impl is emitting), fall back to the current
        `span_idx` so the protocol still advances.
        """
        y = int(actual.values.get("y", 0))
        self.plane_idx = len(self.plane_marks)
        match_idx = None
        for i in range(self.span_idx, len(self.spans)):
            if self.spans[i].y1 == y:
                match_idx = i
                break
        if match_idx is None and self.span_idx < len(self.spans):
            match_idx = self.span_idx
        if match_idx is not None:
            span = self.spans[match_idx]
            self.span_idx = match_idx + 1
            self._enter_active_span(span)
            self.phase = "span_v0"
        else:
            # Degenerate fallback — no matching span in the drafter's set.
            # Synthesize a zero-height span so the protocol still advances.
            self.active_span = _SpanEmit(
                part="middle",
                y1=0,
                y2=-1,
            )
            self.pixel_y = 0
            self.pixel_y_end = -1
            self.phase = "span_v0"

    def _finish_column(self, ctx: _Context) -> None:
        ctx.ceilingclip[self.x] = int(self.new_ceiling)
        ctx.floorclip[self.x] = int(self.new_floor)
        self.x += 1
        if self.x > self.record.x2:
            self.complete = True
            self.phase = "complete"
        else:
            self.phase = "column"

    def consume(self, ctx: _Context, actual: Token) -> bool:
        t = actual.type
        if self.complete:
            if t == SET_CURSOR_X:
                self.complete = False
                self.x = int(actual.values["x"])
                self.phase = "col_u"
                return True
            return False
        if self.phase == "column" and t == SET_CURSOR_X:
            self.x = int(actual.values["x"])
            self.phase = "col_u"
            return True
        if self.phase == "col_u" and t == WALL_COL_U:
            self.u_idx = int(actual.values.get("u_idx", self.u_idx))
            self.phase = "scale"
            return True
        if self.phase == "scale" and t == VALUE:
            self.scale_x = decode_float(ValueRange.R5, float(actual.values["v"]))
            self._compute_column_body(ctx)
            self.phase = "staged"
            return True
        if self.phase == "staged" and t == SCREEN_Y_VALUE:
            self.new_ceiling = int(actual.values["y"])
            self.phase = "body"
            return True
        if self.phase == "body":
            if t == PLANE_MARK:
                self._consume_plane_mark(ctx, actual)
                return True
            if t == WALL_SPAN_META:
                self._consume_wall_span_meta(actual)
                return True
            if t == SET_CURSOR_Y:
                self._consume_set_cursor_y(actual)
                return True
            if t == CLIP_UPDATE:
                self.plane_idx = len(self.plane_marks)
                self.span_idx = len(self.spans)
                self.phase = "clip_range"
                return True
            if t == SET_CURSOR_X:
                self._finish_column(ctx)
                return False
        if self.phase == "body_y" and t == SCREEN_RANGE:
            self.phase = "body"
            return True
        if self.phase == "span_cursor_y" and t == SET_CURSOR_Y:
            self.phase = "span_v0"
            return True
        if self.phase == "span_v0" and t == VALUE:
            self.phase = "span_pixel"
            return True
        if self.phase == "span_pixel" and t == PIXEL:
            self.pixel_y += 1
            if self.pixel_y > self.pixel_y_end:
                # End of span. Decide next branch on the next consumed
                # token (next SET_CURSOR_Y, CLIP_UPDATE, or SET_CURSOR_X
                # advancing to the next column).
                self.phase = "body"
            return True
        if self.phase == "clip_range" and t == SCREEN_RANGE:
            self.new_ceiling = int(actual.values["y1"])
            self.new_floor = int(actual.values["y2"])
            self._finish_column(ctx)
            return True
        return False


@dataclass
class _SubsectorFrame:
    subsector_idx: int
    depth: int
    phase: str = "visit"
    seg_idx: int | None = None
    seg_state: _SegState | None = None

    def _first_seg(self, ctx: _Context) -> int:
        return ctx.md.subsectors[self.subsector_idx].first_seg

    def _last_seg(self, ctx: _Context) -> int:
        ss = ctx.md.subsectors[self.subsector_idx]
        return ss.first_seg + ss.seg_count - 1

    def next_token(self, ctx: _Context) -> Token | None:
        if self.phase == "visit":
            return Token(
                VISIT_SUBSECTOR, {"s": self.subsector_idx, "depth": self.depth}
            )
        if self.phase == "process":
            return Token(PROCESS_SEG, {"i": int(self.seg_idx or 0)})
        if self.phase == "seg":
            assert self.seg_state is not None
            tok = self.seg_state.next_token(ctx)
            if tok is not None:
                return tok
            return self._after_seg_token(ctx)
        if self.phase == "after_seg":
            return self._after_seg_token(ctx)
        if self.phase == "return":
            return Token(
                TRAVERSE_RETURN,
                {
                    "entity_u": N_NODES_MAX + self.subsector_idx,
                    "depth": self.depth,
                },
            )
        raise AssertionError(f"unknown subsector phase {self.phase}")

    def _after_seg_token(self, ctx: _Context) -> Token:
        assert self.seg_idx is not None and self.seg_state is not None
        mode = self.seg_state.completion
        last = self._last_seg(ctx)
        if mode == "advance":
            return Token(ADVANCE_SEG, {"i": self.seg_idx})
        if self.seg_idx < last:
            return Token(PROCESS_SEG, {"i": self.seg_idx + 1})
        return Token(
            TRAVERSE_RETURN,
            {
                "entity_u": N_NODES_MAX + self.subsector_idx,
                "depth": self.depth,
            },
        )

    def _advance_to_next_seg(self, ctx: _Context) -> None:
        assert self.seg_idx is not None
        if self.seg_idx < self._last_seg(ctx):
            self.seg_idx += 1
            self.phase = "process"
            self.seg_state = None
        else:
            self.phase = "return"
            self.seg_state = None

    def consume(self, ctx: _Context, actual: Token) -> bool:
        if self.phase == "visit":
            if actual.type == VISIT_SUBSECTOR:
                ss = ctx.md.subsectors[self.subsector_idx]
                if ss.seg_count <= 0:
                    self.phase = "return"
                else:
                    self.seg_idx = ss.first_seg
                    self.phase = "process"
                return True
            return True
        if self.phase == "process":
            if actual.type == PROCESS_SEG:
                self.seg_idx = int(actual.values["i"])
                self.seg_state = _SegState(ctx, self.seg_idx, self.subsector_idx)
                self.phase = "seg"
                return True
            if actual.type == TRAVERSE_RETURN:
                self.phase = "return"
                return False
            return True
        if self.phase == "seg":
            assert self.seg_state is not None
            if actual.type in _SEG_TYPES and self.seg_state.consume(ctx, actual):
                return True
            if self.seg_state.completion is None:
                self.seg_state.next_token(ctx)
            self.phase = "after_seg"
            return False
        if self.phase == "after_seg":
            if actual.type == ADVANCE_SEG:
                self._advance_to_next_seg(ctx)
                return True
            if actual.type in _SEG_TYPES:
                self.phase = "seg"
                if self.seg_state is None:
                    assert self.seg_idx is not None
                    self.seg_state = _SegState(ctx, self.seg_idx, self.subsector_idx)
                return self.seg_state.consume(ctx, actual)
            self._advance_to_next_seg(ctx)
            return False
        if self.phase == "return":
            if actual.type == TRAVERSE_RETURN:
                ctx.stack.pop()
                return True
            return True
        return True


def _flat_pass_tokens(scene: Scene, state: GameState) -> list[Token]:
    wall_pass = ref.expected_wall_plane_mark_pass(scene, state)
    planes_by_id = {plane.plane_id: plane for plane in wall_pass.planes}
    columns = ref._runtime_visplane_columns(wall_pass.plane_marks)
    used_by_plane: dict[int, list[int]] = {}
    for plane_id, vp in sorted(columns):
        if any(top <= bottom for top, bottom in columns[(plane_id, vp)]):
            used_by_plane.setdefault(int(plane_id), []).append(int(vp))

    flat_atlas = ref._build_flat_atlas(scene)
    view_x, view_y, view_z, view_angle_rad = ref._state_view(state)

    tokens: list[Token] = [
        Token(DRAW_PLANES_BEGIN),
        Token(SET_CURSOR_DIRECTION_X),
        Token(FLAT_NEXT_PLANE, {"p": -1}),
    ]
    for plane_id in sorted(used_by_plane):
        tokens.append(Token(FLAT_NEXT_VP, {"p": plane_id, "vp": -1}))
        plane = planes_by_id.get(plane_id)
        for vp in sorted(used_by_plane[plane_id]):
            tokens.append(Token(FLAT_VISPLANE_BEGIN, {"p": plane_id, "vp": vp}))
            if plane is None or plane.is_sky:
                tokens.append(Token(FLAT_NEXT_VP, {"p": plane_id, "vp": vp}))
                continue
            table = columns[(plane_id, vp)]
            xs = [x for x, (top, bottom) in enumerate(table) if top <= bottom]
            if xs:
                tokens.extend(
                    _make_spans_tokens(
                        plane_id,
                        vp,
                        table,
                        min(xs),
                        max(xs),
                        scene=scene,
                        plane=plane,
                        flat_atlas_by_id=flat_atlas,
                        view_x=view_x,
                        view_y=view_y,
                        view_z=view_z,
                        view_angle_rad=view_angle_rad,
                    )
                )
            else:
                tokens.append(Token(FLAT_NEXT_VP, {"p": plane_id, "vp": vp}))
        tokens.append(Token(FLAT_NEXT_PLANE, {"p": plane_id}))
    tokens.append(Token(DONE))
    return tokens


@dataclass(frozen=True)
class _VisplaneSpanCtx:
    """Per-visplane data the span emitter needs (geometry + lighting)."""

    plane_id: int
    vp: int
    table: list[tuple[int, int]]
    minx: int
    maxx: int
    plane: ref.PlaneDefRecord
    atlas: list[list[int]]
    view_x: float
    view_y: float
    view_z: float
    view_angle_rad: float


def _make_visplane_span_ctx(
    plane_id: int,
    vp: int,
    table: list[tuple[int, int]],
    minx: int,
    maxx: int,
    *,
    plane: ref.PlaneDefRecord,
    flat_atlas_by_id: dict[int, list[list[int]]],
    view_x: float,
    view_y: float,
    view_z: float,
    view_angle_rad: float,
) -> _VisplaneSpanCtx:
    return _VisplaneSpanCtx(
        plane_id=plane_id,
        vp=vp,
        table=table,
        minx=minx,
        maxx=maxx,
        plane=plane,
        atlas=flat_atlas_by_id[plane.flat_id],
        view_x=view_x,
        view_y=view_y,
        view_z=view_z,
        view_angle_rad=view_angle_rad,
    )


def _column_coverage(table: list[tuple[int, int]], x: int) -> tuple[int, int]:
    """Coverage `(top, bottom)` of column `x`; sentinel for off-table x.

    `table` already holds the sentinel `(SCREEN_HEIGHT, -1)` for every
    column R_MakeSpans never marked — including the terminal `maxx+1`
    column — so this is just a bounds-guarded read.
    """
    if 0 <= x < len(table):
        return table[x]
    return (SCREEN_HEIGHT, -1)


def _column_transition_tokens(
    ctx: _VisplaneSpanCtx,
    prev_x: int | None,
    cur_x: int,
    open_x_by_y: dict[int, int],
) -> list[Token]:
    """Tokens emitted when R_MakeSpans steps onto column `cur_x`.

    This is the literal body of `_make_spans`/`_make_spans_tokens`'s
    per-column loop, but driven by *actual* adopted columns rather than
    the canonical `range(minx, maxx + 2)`. The closes happen at the
    previous column `prev_x` (its coverage is the left side `t1,b1`);
    the opens happen at `cur_x` (its coverage is the right side
    `t2,b2`). `open_x_by_y` carries the open-span left edges across
    columns and is mutated in place, exactly as the canonical loop's
    closure does.

    With `prev_x is None` (the visplane's first column) the left side
    is the sentinel `(SCREEN_HEIGHT, -1)`, matching the canonical
    `x == minx` branch. Because the model walks columns consecutively
    (R_MakeSpans iterates `minx..maxx+1`), an interior `prev_x` equals
    `cur_x - 1`, so `table[prev_x]` reproduces the canonical
    `table[x - 1]` even when the model's `minx` differs from canonical.
    """
    tokens: list[Token] = []

    def emit_close(slot: int, lo: int, hi: int, x_close: int) -> None:
        if lo > hi:
            return
        tokens.append(Token(SPAN_CLOSE_SLOT, {"slot": slot}))
        for y in range(lo, hi + 1):
            x1 = open_x_by_y.pop(y, ctx.minx)
            tokens.append(Token(SPAN_ROW, {"y": y}))
            tokens.append(Token(SET_CURSOR_Y, {"y": y}))
            tokens.append(Token(SET_CURSOR_X, {"x": x1}))
            xfrac0, yfrac0, xstep, ystep = ref._map_plane_setup(
                plane_height=ctx.plane.height,
                view_x=ctx.view_x,
                view_y=ctx.view_y,
                view_z=ctx.view_z,
                view_angle_rad=ctx.view_angle_rad,
                y=y,
                x1=x1,
            )
            colormap_row = ref._flat_colormap_row_for(ctx.plane, ctx.view_z, y)
            for raw in ref._draw_span(
                ctx.atlas, xfrac0, yfrac0, xstep, ystep, x_close - x1
            ):
                lit = ref.apply_doom_colormap(
                    ref.COLORMAP_ROWS,
                    colormap_row,
                    raw,
                )
                tokens.append(Token(PIXEL, {"color": lit, "w": 1}))

    def open_rows(lo: int, hi: int, x_open: int) -> None:
        if lo > hi:
            return
        for y in range(lo, hi + 1):
            open_x_by_y[y] = x_open

    if prev_x is None:
        t1, b1 = SCREEN_HEIGHT, -1
    else:
        t1, b1 = _column_coverage(ctx.table, prev_x)
    t2, b2 = _column_coverage(ctx.table, cur_x)
    # x_close is the previous adopted column (where the closing span ends).
    x_close = cur_x - 1 if prev_x is None else prev_x

    close_top_lo = t1
    close_top_hi = min(t2 - 1, b1)
    if close_top_lo <= close_top_hi:
        emit_close(0, close_top_lo, close_top_hi, x_close)
        t1_after = min(t2, b1 + 1)
    else:
        t1_after = t1

    close_bottom_lo = max(b2 + 1, t1_after)
    close_bottom_hi = b1
    if close_bottom_lo <= close_bottom_hi:
        emit_close(1, close_bottom_lo, close_bottom_hi, x_close)
        b1_after = max(b2, t1_after - 1)
    else:
        b1_after = b1

    open_top_lo = t2
    open_top_hi = min(t1_after - 1, b2)
    if open_top_lo <= open_top_hi:
        open_rows(open_top_lo, open_top_hi, cur_x)
        t2_after = min(t1_after, b2 + 1)
    else:
        t2_after = t2

    open_bottom_lo = max(b1_after + 1, t2_after)
    open_bottom_hi = b2
    if open_bottom_lo <= open_bottom_hi:
        open_rows(open_bottom_lo, open_bottom_hi, cur_x)

    return tokens


def _make_spans_tokens(
    plane_id: int,
    vp: int,
    table: list[tuple[int, int]],
    minx: int,
    maxx: int,
    *,
    scene: Scene,
    plane: ref.PlaneDefRecord,
    flat_atlas_by_id: dict[int, list[list[int]]],
    view_x: float,
    view_y: float,
    view_z: float,
    view_angle_rad: float,
) -> list[Token]:
    """Canonical flat-pass token list for one visplane (whole-stream tests).

    Kept as the ground-truth generator used by fresh-rollout validation;
    the AR drafter drives the same per-column emission consume-style via
    `_column_transition_tokens` so it can re-anchor on the model's
    actual column run.
    """
    ctx = _make_visplane_span_ctx(
        plane_id,
        vp,
        table,
        minx,
        maxx,
        plane=plane,
        flat_atlas_by_id=flat_atlas_by_id,
        view_x=view_x,
        view_y=view_y,
        view_z=view_z,
        view_angle_rad=view_angle_rad,
    )
    tokens: list[Token] = []
    open_x_by_y: dict[int, int] = {}
    prev_x: int | None = None
    for x in range(minx, maxx + 2):
        tokens.append(Token(MAKE_SPANS_COL, {"x": x}))
        tokens.extend(_column_transition_tokens(ctx, prev_x, x, open_x_by_y))
        prev_x = x
    tokens.append(Token(FLAT_NEXT_VP, {"p": plane_id, "vp": vp}))
    return tokens


class _VisplaneSpanState:
    """Consume-driven span emitter for one visplane.

    Mirrors `_HorizontalScanState`: on `consume(MAKE_SPANS_COL(x))` it
    adopts the model's actual column `x` (instead of a precomputed
    canonical column), re-derives that column's close/open transition
    from the visplane geometry, and predicts the rest of *that* span.
    A divergent `minx` (or any column shift) costs a single mispredict
    at the differing structural token and then re-syncs within the same
    visplane, instead of cascading every interior token until the next
    `FLAT_VISPLANE_BEGIN`.

    The visplane completes when `FLAT_NEXT_VP` is consumed (the canonical
    terminator). `next_token` returns `None` once complete so the parent
    flat scan advances to the next scaffold token.
    """

    def __init__(self, ctx: _VisplaneSpanCtx) -> None:
        self._ctx = ctx
        self._open_x_by_y: dict[int, int] = {}
        self._prev_x: int | None = None
        # Pending tokens after the current MAKE_SPANS_COL (close/open
        # events for the adopted column), then the next canonical
        # MAKE_SPANS_COL, drained one per next_token/consume.
        self._pending: list[Token] = []
        self._started = False
        self._complete = False

    @property
    def complete(self) -> bool:
        return self._complete

    def _canonical_next_col(self) -> int:
        """The column the canonical plan would emit next (for the draft)."""
        if self._prev_x is None:
            return self._ctx.minx
        return self._prev_x + 1

    def _next_make_spans_or_done(self) -> Token:
        nxt = self._canonical_next_col()
        if nxt <= self._ctx.maxx + 1:
            return Token(MAKE_SPANS_COL, {"x": nxt})
        return Token(FLAT_NEXT_VP, {"p": self._ctx.plane_id, "vp": self._ctx.vp})

    def next_token(self) -> Token | None:
        if self._complete:
            return None
        if self._pending:
            return self._pending[0]
        if not self._started:
            return Token(MAKE_SPANS_COL, {"x": self._ctx.minx})
        return self._next_make_spans_or_done()

    def consume(self, actual: Token) -> bool:
        if self._complete:
            return False
        if actual.type == MAKE_SPANS_COL:
            # Adopt the model's actual column and re-derive its
            # transition from the previously-adopted column.
            self._started = True
            cur_x = int(actual.values["x"])
            self._pending = _column_transition_tokens(
                self._ctx, self._prev_x, cur_x, self._open_x_by_y
            )
            self._prev_x = cur_x
            return True
        if actual.type == FLAT_NEXT_VP:
            # Canonical visplane terminator — done with this visplane.
            self._complete = True
            return True
        if self._pending and actual.type == self._pending[0].type:
            # In-phase structural/pixel token of the current column.
            self._pending.pop(0)
            return True
        if actual.type in _SPAN_INTERIOR_TYPES:
            # A span-interior token (pixel-color mismatch, an extra pixel
            # from a longer model span, a close/row the re-derivation
            # didn't line up on). Absorb it as a single mispredict and
            # stay inside this visplane: dropping the stale pending head
            # (if any) of a different interior kind keeps us aligned for
            # the next adopted column without ending the visplane early.
            if self._pending and self._pending[0].type in _SPAN_INTERIOR_TYPES:
                self._pending.pop(0)
            return True
        # A scaffold-boundary token (FLAT_VISPLANE_BEGIN, FLAT_NEXT_PLANE,
        # DONE) the model emitted without the canonical FLAT_NEXT_VP:
        # this visplane is over — let the parent re-route.
        return False


def _build_flat_plan(
    scene: Scene, state: GameState
) -> list[Token | _VisplaneSpanState]:
    """Flat-pass plan: scaffold Tokens interleaved with span emitters.

    Each list entry is either a literal scaffold `Token` (DRAW_PLANES_BEGIN,
    FLAT_NEXT_PLANE, FLAT_VISPLANE_BEGIN, the sky/empty FLAT_NEXT_VP, DONE)
    or a `_VisplaneSpanState` that drives one visplane's span emission
    consume-style (re-anchoring on the model's actual columns). The
    closing FLAT_NEXT_VP of a real visplane is owned by its
    `_VisplaneSpanState`, not the scaffold.
    """
    wall_pass = ref.expected_wall_plane_mark_pass(scene, state)
    planes_by_id = {plane.plane_id: plane for plane in wall_pass.planes}
    columns = ref._runtime_visplane_columns(wall_pass.plane_marks)
    used_by_plane: dict[int, list[int]] = {}
    for plane_id, vp in sorted(columns):
        if any(top <= bottom for top, bottom in columns[(plane_id, vp)]):
            used_by_plane.setdefault(int(plane_id), []).append(int(vp))

    flat_atlas = ref._build_flat_atlas(scene)
    view_x, view_y, view_z, view_angle_rad = ref._state_view(state)

    plan: list[Token | _VisplaneSpanState] = [
        Token(DRAW_PLANES_BEGIN),
        Token(SET_CURSOR_DIRECTION_X),
        Token(FLAT_NEXT_PLANE, {"p": -1}),
    ]
    for plane_id in sorted(used_by_plane):
        plan.append(Token(FLAT_NEXT_VP, {"p": plane_id, "vp": -1}))
        plane = planes_by_id.get(plane_id)
        for vp in sorted(used_by_plane[plane_id]):
            plan.append(Token(FLAT_VISPLANE_BEGIN, {"p": plane_id, "vp": vp}))
            if plane is None or plane.is_sky:
                plan.append(Token(FLAT_NEXT_VP, {"p": plane_id, "vp": vp}))
                continue
            table = columns[(plane_id, vp)]
            xs = [x for x, (top, bottom) in enumerate(table) if top <= bottom]
            if xs:
                ctx = _make_visplane_span_ctx(
                    plane_id,
                    vp,
                    table,
                    min(xs),
                    max(xs),
                    plane=plane,
                    flat_atlas_by_id=flat_atlas,
                    view_x=view_x,
                    view_y=view_y,
                    view_z=view_z,
                    view_angle_rad=view_angle_rad,
                )
                plan.append(_VisplaneSpanState(ctx))
            else:
                plan.append(Token(FLAT_NEXT_VP, {"p": plane_id, "vp": vp}))
        plan.append(Token(FLAT_NEXT_PLANE, {"p": plane_id}))
    plan.append(Token(DONE))
    return plan


class _FlatScanState:
    """Consume-driven flat-pass state machine.

    Walks `_build_flat_plan`'s step list. Literal scaffold tokens are
    emitted/consumed verbatim and advance the cursor; a
    `_VisplaneSpanState` step is delegated to until it completes (its
    closing FLAT_NEXT_VP is consumed), at which point the cursor
    advances. This replaces the old precomputed `(_flat_tokens,
    _flat_idx)` plan so a divergent span decomposition re-syncs within
    the visplane instead of cascading.

    snapshot/rollback for the runtime's K-batch lookahead is handled by
    the parent `ARDrafter` via `copy.deepcopy`, which captures this
    object's cursor and the per-visplane open-span bookkeeping exactly.
    """

    def __init__(self, scene: Scene, state: GameState) -> None:
        self._plan = _build_flat_plan(scene, state)
        self._idx = 0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def _current(self) -> Token | _VisplaneSpanState | None:
        if self._idx >= len(self._plan):
            return None
        return self._plan[self._idx]

    def next_token(self) -> Token | None:
        step = self._current()
        if step is None:
            return None
        if isinstance(step, _VisplaneSpanState):
            tok = step.next_token()
            if tok is not None:
                return tok
            # Span emitter drained without seeing its terminator; fall
            # through to the next scaffold step.
            self._idx += 1
            return self.next_token()
        return step

    def consume(self, actual: Token) -> None:
        step = self._current()
        if step is None:
            if actual.type == DONE:
                self._done = True
            return
        if isinstance(step, _VisplaneSpanState):
            if step.consume(actual):
                if step.complete:
                    self._idx += 1
                return
            # The span emitter rejected this token: it's done with its
            # visplane (or never started). Advance and re-route.
            self._idx += 1
            self.consume(actual)
            return
        # Literal scaffold token: advance past it (consume-to-advance,
        # like the old _flat_idx bump) and pick up DONE termination.
        self._idx += 1
        if actual.type == DONE or step.type == DONE:
            self._done = True


class ARDrafter:
    """State-machine drafter with O(1)-style snapshot/rollback."""

    def __init__(self, scene: Scene, state: GameState) -> None:
        md = scene.map_data
        self._ctx = _Context(
            scene=scene,
            state=state,
            md=md,
            segments=bake_segments(md),
            viewz=ref._state_viewz(state),
            view_angle_bam=int(state.angle) * _FIXTURE_TO_BAM,
            plane_tables=ref._build_plane_tables(md),
            runtime_visplanes=ref._RuntimeVisplanes(),
            solidsegs=[_ClipRange(-(10**9), -1), _ClipRange(SCREEN_WIDTH, 10**9)],
            ceilingclip=[-1] * SCREEN_WIDTH,
            floorclip=[SCREEN_HEIGHT] * SCREEN_WIDTH,
            side_table={},
            stack=[],
        )
        self._side_idx = 0
        self._side_phase = "think"
        self._direction_y_emitted = False
        self._root_started = False
        self._flat_scan: _FlatScanState | None = None
        self._done_consumed = False

    def _exact_side(self, node_idx: int) -> int:
        node = self._ctx.md.nodes[node_idx]
        return side_P(make_plane(node), self._ctx.state.x, self._ctx.state.y)

    def _start_root(self) -> None:
        if self._root_started:
            return
        self._root_started = True
        if self._ctx.md.nodes:
            self._ctx.stack.append(_NodeFrame(len(self._ctx.md.nodes) - 1, 0))
        elif self._ctx.md.subsectors:
            self._ctx.stack.append(_SubsectorFrame(0, 0))

    def next_draft(self) -> Token | None:
        if self._done_consumed:
            return None
        if not self._direction_y_emitted:
            return Token(SET_CURSOR_DIRECTION_Y)
        if self._side_idx < len(self._ctx.md.nodes):
            if self._side_phase == "think":
                return Token(THINK_SIDE, {"node": self._side_idx})
            return Token(
                SIDE_RECORD,
                {
                    "node": self._side_idx,
                    "side": self._ctx.side_table.get(
                        self._side_idx, self._exact_side(self._side_idx)
                    ),
                },
            )
        self._start_root()
        if self._ctx.stack:
            return self._ctx.stack[-1].next_token(self._ctx)
        return self._ensure_flat_scan().next_token()

    def consume(self, actual: Token) -> None:
        if self._done_consumed:
            return
        if not self._direction_y_emitted:
            if actual.type == SET_CURSOR_DIRECTION_Y:
                self._direction_y_emitted = True
                return
            self._direction_y_emitted = True
        if self._side_idx < len(self._ctx.md.nodes):
            if self._side_phase == "think" and actual.type == THINK_SIDE:
                self._side_phase = "record"
                return
            if actual.type == SIDE_RECORD:
                node_idx = int(actual.values.get("node", self._side_idx))
                self._ctx.side_table[node_idx] = int(actual.values["side"])
                if node_idx == self._side_idx:
                    self._side_idx += 1
                else:
                    self._side_idx = max(self._side_idx + 1, node_idx + 1)
                self._side_phase = "think"
                return
            self._side_phase = "record" if self._side_phase == "think" else "think"
            return

        if not self._ctx.stack and actual.type in _FLAT_SCAN_TYPES | {DONE}:
            self._consume_flat(actual)
            return

        # Bubble the token up the frame stack: each frame either
        # handles the token (returns True) or signals "I'm done with
        # this token, try whoever I leave behind" (returns False, after
        # advancing its own phase or popping itself off the stack).
        # Progress = stack length changes OR top frame's phase changes.
        # If neither happens between iterations, the state machine is
        # spinning — that's a drafter bug, fail loudly.
        self._start_root()
        prev_progress = None
        while self._ctx.stack:
            frame = self._ctx.stack[-1]
            if frame.consume(self._ctx, actual):
                return
            progress = (
                len(self._ctx.stack),
                id(self._ctx.stack[-1]) if self._ctx.stack else None,
                (
                    getattr(self._ctx.stack[-1], "phase", None)
                    if self._ctx.stack
                    else None
                ),
            )
            assert progress != prev_progress, (
                f"ARDrafter.consume: no progress consuming {actual} at frame "
                f"{type(frame).__name__}(phase={getattr(frame, 'phase', None)!r})"
            )
            prev_progress = progress
        if actual.type == DONE:
            self._done_consumed = True

    def _ensure_flat_scan(self) -> _FlatScanState:
        if self._flat_scan is None:
            self._flat_scan = _FlatScanState(self._ctx.scene, self._ctx.state)
        return self._flat_scan

    def _consume_flat(self, actual: Token) -> None:
        flat_scan = self._ensure_flat_scan()
        flat_scan.consume(actual)
        if actual.type == DONE or flat_scan.done:
            self._done_consumed = True

    def snapshot(self):
        return copy.deepcopy(
            (
                self._ctx,
                self._side_idx,
                self._side_phase,
                self._direction_y_emitted,
                self._root_started,
                self._flat_scan,
                self._done_consumed,
            )
        )

    def rollback(self, snap) -> None:
        (
            self._ctx,
            self._side_idx,
            self._side_phase,
            self._direction_y_emitted,
            self._root_started,
            self._flat_scan,
            self._done_consumed,
        ) = snap


# ---- Public entry points ----------------------------------------------------


def expected_ar_tokens(scene: Scene, state: GameState) -> list[Token]:
    """Return the canonical AR-phase Token sequence (no consume feedback).

    The reference's exact-math predictions are used at every position;
    no resync. Suitable for the no-feedback comparison path. For
    consume-aware iteration use `ARDrafter` directly.
    """
    drafter = ARDrafter(scene, state)
    out: list[Token] = []
    while True:
        tok = drafter.next_draft()
        if tok is None:
            break
        out.append(tok)
        # Advance with the prediction itself as feedback (no resync).
        drafter.consume(tok)
    return out


class CacheWithReferenceFallback:
    """Drafter starting from a cached token list, swapping to a
    consume-aware fallback drafter on the first cache-vs-actual
    divergence (or when the cache exhausts before the rollout ends).

    Closes ``StaticDrafter``'s wedge case: a one-token misaligned cache
    rejects every subsequent K-batch, committing only the bonus row,
    which at K=48 runs (K+1)x slower than per-token AR. This wrapper
    detects the mismatch on the first divergent ``consume(actual)``,
    constructs / aligns the fallback drafter by replaying the consume
    history into it, and from that point on uses the fallback's
    consume-driven resync (with its own per-batch K) for the rest of
    the rollout.

    The cached fast path runs at ``k_cached`` (default 48) — the
    byte-exact-cache regime where every draft accepts. After a swap,
    the fallback runs at ``k_fallback`` (default 16) — the state-machine
    fallback keeps structural predictions aligned, so a moderate window
    reduces Python batch overhead without wasting too much rejected
    tail. The wrapper enforces the per-mode K by returning ``None``
    from ``next_draft`` once the in-batch limit hits; the runtime
    treats that as "no more drafts" and proceeds with however many it
    got.

    **This silently overrides the ``draft_window`` passed to ``run()``.**
    ``draft_window`` is only an upper bound (see ``api.forward.run``); the
    realized batch width is ``min(draft_window, k_cached_or_k_fallback)``.
    So ``run(..., draft_window=48)`` wrapped in this drafter still batches
    only ``k_fallback`` (=16) fresh drafts on the cold path — the cap that
    actually governs cold-path throughput lives *here*, not at the call
    site.

    Cursor convention: ``_idx`` is the absolute position-in-rollout
    that both ``cached`` and the fallback's predictions index by.
    Swapping mid-rollout doesn't require re-aligning the cursor —
    the same counter advances through either source.

    The fallback is fed only once the swap fires (lazily). Until
    then the wrapper accumulates the consume history independently;
    at swap time the history is replayed into the fallback so its
    next prediction rides on the implementation's actual rollout.
    Snapshots always capture the fallback's state too, so a
    spec-phase swap (``next_draft`` notices cache exhaustion mid-
    batch) is correctly undone by the runtime's rollback even when
    the snapshot was taken in cached mode.
    """

    def __init__(
        self,
        cached: list[Token] | None,
        fallback,  # any Drafter — typically ARDrafter
        *,
        k_cached: int = 48,
        k_fallback: int = 16,
    ) -> None:
        self._cached = list(cached) if cached else []
        self._fallback = fallback
        self._using_fallback = not self._cached
        self._idx = 0
        self._in_batch = 0
        self._consume_history: list[Token] = []
        self._k_cached = k_cached
        self._k_fallback = k_fallback
        self._fallback_start_idx = 0 if self._using_fallback else None
        self._fallback_reason = "no_cache" if self._using_fallback else None

    def _swap_to_fallback(self, reason: str) -> None:
        """Replay consume history into the fallback drafter and switch.

        Idempotent; calling after the swap has already fired is a no-op.
        """
        if self._using_fallback:
            return
        for tok in self._consume_history:
            self._fallback.consume(tok)
        self._using_fallback = True
        self._fallback_start_idx = self._idx
        self._fallback_reason = reason

    def next_draft(self) -> Token | None:
        if not self._using_fallback and self._idx >= len(self._cached):
            self._swap_to_fallback("cache_exhausted")
        k_limit = self._k_fallback if self._using_fallback else self._k_cached
        if self._in_batch >= k_limit:
            return None
        if self._using_fallback:
            d = self._fallback.next_draft()
            if d is None:
                return None
            self._in_batch += 1
            return d
        self._in_batch += 1
        return self._cached[self._idx]

    def consume(self, actual: Token) -> None:
        self._consume_history.append(actual)
        if not self._using_fallback:
            if self._idx >= len(self._cached) or self._cached[self._idx] != actual:
                self._swap_to_fallback("cache_mismatch")
        else:
            self._fallback.consume(actual)
        self._idx += 1

    def stats(self) -> dict[str, object]:
        fallback_start = self._fallback_start_idx
        if fallback_start is None:
            cache_committed = self._idx
            fallback_committed = 0
        else:
            cache_committed = min(self._idx, fallback_start)
            fallback_committed = max(0, self._idx - fallback_start)
        return {
            "cached_tokens": len(self._cached),
            "committed_tokens": self._idx,
            "cache_committed_tokens": cache_committed,
            "fallback_committed_tokens": fallback_committed,
            "using_fallback": self._using_fallback,
            "fallback_start_idx": fallback_start,
            "fallback_reason": self._fallback_reason,
            "k_cached": self._k_cached,
            "k_fallback": self._k_fallback,
        }

    def snapshot(self) -> tuple[int, bool, int, object, int | None, str | None]:
        self._in_batch = 0
        return (
            self._idx,
            self._using_fallback,
            len(self._consume_history),
            self._fallback.snapshot(),
            self._fallback_start_idx,
            self._fallback_reason,
        )

    def rollback(
        self, snap: tuple[int, bool, int, object, int | None, str | None]
    ) -> None:
        idx, using_fb, hist_len, fb_snap, fb_start_idx, fb_reason = snap
        self._idx = idx
        self._using_fallback = using_fb
        del self._consume_history[hist_len:]
        self._fallback.rollback(fb_snap)
        self._fallback_start_idx = fb_start_idx
        self._fallback_reason = fb_reason
        self._in_batch = 0
