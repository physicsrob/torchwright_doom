"""Forward/render math ops for the read-side + write-side subset, ported from
the original plain-Python implementation.

The file opens with the read-side helpers (``MARKER_PRESENT`` / ``and_`` /
``one_minus`` / ``or_`` / ``snap_bool`` / ``SCREEN_X_CLAMP``), then nine
``# ---`` banner sections, in the order they appear:

1. Affine glue + the ``R_PointOnSide`` cross product — ``sub`` / ``neg`` /
   ``add_const`` plus ``mul_side`` and the ``SIDE_POSITIVE`` / ``IS_SUBSECTOR``
   / ``DEPTH_NONZERO`` comparators.
2. ``R_PointToAngle`` BAM-atan2 octant + one-turn angle wrap —
   ``signed_world_angle`` / ``wrap_signed_angle`` and their helpers.
3. Screen-coordinate product (the solid-interval coverage key, ``MUL_SCREEN``).
4. Screen comparators, integer-equality, and the backface cross product
   (``gt_screen`` / ``same_int`` / ``MUL_CROSS`` / ``is_negative_cross`` …).
5. ``R_CheckBBox`` region classifier (``COORD_GT_ZERO``).
6. The drawseg perspective-scale chain — the scale-denominator / scale products
   and their clamps and comparators.
7. Wall-column rasterizer screen-y ops — the ``CEIL_Y`` / ``FLOOR_Y`` clamp
   staircases and the per-column projection products.
8. The pixel pass — the per-pixel texture-coordinate products and the
   native-coordinate floor.
9. Screen-column radix key (``radix_col_key``, shared across the
   solid-interval / visplane / wall-column column-equality tests).

Per-op piecewise-linear noise bounds live in torchwright's
``docs/op_noise_data.json``; the inline notes here explain why each op's
breakpoint grid / sharpness fits its measured bound.
"""

from __future__ import annotations

import math

import torch

from torchwright.graph import Linear, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.const import scale
from torchwright.ops.swiglu.arithmetic_ops import (
    ceil_int,
    clamp,
    compare,
    floor_int,
    multiply,
    piecewise_linear,
    thermometer_floor_div,
)
from torchwright.ops.swiglu.logic_ops import bool_all_true, bool_any_true, bool_not
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn

from .constants import COLUMN_COUNT, PIXEL_WIDTH, SCREEN_WIDTH, VIEW_HEIGHT
from .std import concat, constant, linear, one_hot, select
from .vocab import _TAN_FOV_HALF, ANGLE_BAM, N_NODES_MAX


# Marker and integer-slot ids are exact integers, so ``> 0.5`` cleanly
# means "different / present / nonzero". The threshold sits at 0.5 over the
# ``[-0.5, 1.5]`` input range at the default ``step_sharpness`` (deadband 0.1,
# far wider than any marker noise).
def MARKER_PRESENT(marker: Node) -> Node:
    """±1 boolean: was a marker (≈1) published at the picked row?"""
    return compare(marker, 0.5)


def and_(a: Node, b: Node) -> Node:
    """Boolean conjunction over two ±1 predicates."""
    return bool_all_true([a, b])


def one_minus(a: Node) -> Node:
    """Boolean negation of a ±1 predicate."""
    return bool_not(a)


def or_(a: Node, b: Node) -> Node:
    """Boolean disjunction over two ±1 predicates."""
    return bool_any_true([a, b])


def snap_bool(x: Node) -> Node:
    """Snap a recovered near-±1 boolean back to a clean ±1 by its sign.

    A ``pick_argmax`` over ±1 keys (e.g. ``SolidIntervals``' ``solid_emit``)
    recovers the matched value as ≈±1 but with fp32 softmax noise (e.g.
    ``-1.0000080``). When that value is then used as a ``select`` *cond*, the
    noise scales the kept branch and leaks a sliver of the discarded branch's
    emit *head* into the result — and against the renderer's razor-thin
    integer-slot argmax margins (a descend ``VISIT_SUBSECTOR`` head beats its
    depth±1 neighbour by ~0.97 out of ~46000) that sliver is enough to flip the
    next token. The float64 reference renderer (pydoom) keeps the cond exact, so
    this snap is the fp32-fidelity counterpart (cf. the float64->fp32 sharp-step
    discipline)."""
    return compare(x, 0.0)


def SCREEN_X_CLAMP(x: Node) -> Node:
    """Clamp a rendered column into ``[0, COLUMN_COUNT-1]``.

    Guards an ``IntSlot(0, COLUMN_COUNT)`` deembed when a pick falls through to
    pure recency (no FIND_RUN match in past) and would otherwise return the
    COLUMN_COUNT sentinel."""
    return clamp(x, 0.0, float(COLUMN_COUNT - 1))


def screen_x_from_column(col: Node) -> Node:
    """Column index (0..COLUMN_COUNT-1) -> screen paint coordinate
    (col * PIXEL_WIDTH). The ONE place the graph maps internal column space to
    the host's screen coordinate (emitted into ``setCursorX.x``, read literally
    by the dumb host). Identity in high-detail (PIXEL_WIDTH=1), adding no op."""
    if PIXEL_WIDTH == 1:
        return col
    return linear(col, [[float(PIXEL_WIDTH)]])


def column_from_screen_x(screen_x: Node) -> Node:
    """Screen coordinate (a multiple of PIXEL_WIDTH) -> column index
    (screen_x / PIXEL_WIDTH). Used where a recovered/extracted ``setCursorX``
    value indexes a per-column table. Exact (no floor) because the graph only
    ever emits col*PIXEL_WIDTH. Identity in high-detail, adding no op."""
    if PIXEL_WIDTH == 1:
        return screen_x
    return linear(screen_x, [[1.0 / float(PIXEL_WIDTH)]])


# ---------------------------------------------------------------------------
# Affine glue + the R_PointOnSide cross product
# ---------------------------------------------------------------------------


def sub(a: Node, b: Node) -> Node:
    """``a - b`` as ``linear(concat(a, b), [[1],[-1]])``."""
    return Linear(
        concat(a, b),
        torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
        name="sub",
    )


def neg(a: Node) -> Node:
    """``-a``."""
    return Linear(a, torch.tensor([[-1.0]], dtype=torch.float32), name="neg")


def add_const(a: Node, c: float) -> Node:
    """``a + c`` as one fused ``Linear``."""
    return Linear(
        a,
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.tensor([float(c)], dtype=torch.float32),
        name="add_const",
    )


# DOOM: R_PointOnSide cross-product sign (r_main.c). Only the SIGN feeds
# SIDE_POSITIVE. The original keeps a magnitude split (MUL_SIDE_SMALL/LARGE)
# for precision near unit-normal coefficients; the swiglu ``multiply`` is
# exact to ~2 ulp of the product at any magnitude (no grid, no range limit,
# no extrapolation regime), so one op serves every operand class. Node deltas
# are recovered within R2 = [-512, 512]; rel = view - p with both in
# R1 = [-1152, 1152], so |rel| <= 2304 and |product| <= ~1.2e6 — the ~2-ulp
# error (~0.3 abs) is invisible to the sign test, which tolerated ~75 of
# grid product noise before the cutover.
def mul_side(coef: Node, rel: Node) -> Node:
    """Product of a partition coefficient and a view-relative coordinate,
    resolved to the sign by ``SIDE_POSITIVE``."""
    return multiply(coef, rel)


def SIDE_POSITIVE(cross_z: Node) -> Node:
    """±1: is the R_PointOnSide cross product > 0?"""
    return compare(cross_z, 0.0)


def IS_SUBSECTOR(child_u: Node) -> Node:
    """±1: is a unified child id a subsector (>= N_NODES_MAX) vs a node?"""
    return compare(child_u, float(N_NODES_MAX) - 0.5)


def DEPTH_NONZERO(depth: Node) -> Node:
    """±1: is the BSP tree depth nonzero?"""
    return compare(depth, 0.5)


# ---------------------------------------------------------------------------
# R_PointToAngle BAM-atan2 octant + one-turn angle wrap
# ---------------------------------------------------------------------------
#
# The original keeps two near-duplicate octant builders —
# ``signed_world_angle`` (clamp 2048, seg endpoints) and
# ``_signed_world_angle_bbox`` (clamp 3072, bbox corners). They are unified
# here on the **wider 3072 clamp**:
# e1m1_subset bbox corners reach |dx|,|dy| ≈ 2752 (vs ≈ 308 for seg endpoints),
# which overruns 2048 and would clip to a wrong BAM angle, but sits comfortably
# inside 3072. One helper serves both the seg-projection (F) and bbox (G) sides.
#
# The angle of (|dx|, |dy|) in the first octant-pair [0, ANGLE_BAM/4) is read
# as a *count* of ray-threshold crossings: for each candidate BAM angle k the
# ray test ``|dy| - tan(k)·|dx| > 0`` (low half, 0..1024) or
# ``cot(k)·|dy| - |dx| > 0`` (high half, 1024..2048) is +1 while the true angle
# exceeds k, so the booleans form a thermometer whose sum is the first-quadrant
# angle. Quadrant placement then folds in the signs of dx, dy.
# (ANGLE_BAM itself is canonical in vocab.py and imported above.)

# Coordinate deltas entering the ray classifier clamp to this square before the
# abs PWL. 3072 covers every committed-fixture corner with headroom (see above).
_ATAN_ABS_RANGE = 3072.0
_ABS_BREAKPOINTS = [-_ATAN_ABS_RANGE, 0.0, _ATAN_ABS_RANGE]

# Ray thresholds at the half-integer BAM angles k + 0.5 (so an integer-valued
# true angle never lands inside a ramp): low half spans the first 1/8 turn
# [0, ANGLE_BAM/8); high half the second [ANGLE_BAM/8, ANGLE_BAM/4).
_N8 = ANGLE_BAM // 8
_LOW_THRESHOLDS = [float(k) + 0.5 for k in range(_N8)]
_HIGH_THRESHOLDS = [float(k) + 0.5 for k in range(_N8, ANGLE_BAM // 4)]
_LOW_SLOPES = [math.tan(t * 2.0 * math.pi / ANGLE_BAM) for t in _LOW_THRESHOLDS]
_HIGH_COTS = [1.0 / math.tan(t * 2.0 * math.pi / ANGLE_BAM) for t in _HIGH_THRESHOLDS]
# Applied to ``concat(abs_dy, abs_dx)`` (row 0 weights abs_dy, row 1 abs_dx):
#   low  col k = abs_dy - slope_k·abs_dx
#   high col k = cot_k·abs_dy - abs_dx
_LOW_RAY_MATRIX = [[1.0] * len(_LOW_SLOPES), [-slope for slope in _LOW_SLOPES]]
_HIGH_RAY_MATRIX = [list(_HIGH_COTS), [-1.0] * len(_HIGH_COTS)]

# Q2/Q3 reflections of the first-quadrant base angle, as affine maps over
# ``concat(base, 1)``.
_HALF_BAM = float(ANGLE_BAM // 2)
_Q2_AFFINE = [[-1.0], [_HALF_BAM]]  # ANGLE_BAM/2 - base
_Q3_AFFINE = [[1.0], [-_HALF_BAM]]  # -ANGLE_BAM/2 + base

# The minimum non-zero ray magnitude is bounded below by the BAM granularity
# (2π/8192) times the slope; the ray step's unsaturated band — |v| <
# 17/(scale·s) ≈ 4.2e-6 input units, where the swish hinge in ``_ray_count``
# is not yet exactly 0-or-identity — sits well inside that bound, so an
# integer-angle ray never lands in the band (the smallest nonzero
# committed-fixture ray, ~1.5e-4, clears it by ~36x).
_RAY_SHARPNESS = 32000.0

# Player-relative angles normalize to the signed BAM interval [-4096, 4096);
# one-turn deltas outside it wrap by ANGLE_BAM.
_SIGNED_ANGLE_ABOVE_MAX = 4095.5
_SIGNED_ANGLE_ABOVE_MIN = -4096.5


def _abs_coord(x: Node) -> Node:
    """``|x|`` clamped to the atan2 square.

    3 breakpoints with the kink at 0 make ``abs`` exact in-range; out-of-range
    inputs clamp to ±``_ATAN_ABS_RANGE`` (``piecewise_linear``'s default).
    """
    return piecewise_linear(x, _ABS_BREAKPOINTS, lambda v: abs(v), name="abs_coord")


def _ray_count(rays: Node) -> Node:
    """Count the +1 (positive) components of a ray vector.

    The reference renderer spells this ``reduce_sum(bool_to_01(...))`` — an
    elementwise ±1 ``compare`` (sharpness 32000), a 0/1 cast, then a sum — and
    runs it in float64. The real graph is float32 and ``compare`` is
    scalar-only, so the elementwise 0/1 step is built directly on the swish
    hinge ``hinge(z) = Swish(scale·z)/scale``, which equals ``relu(z)``
    exactly once ``|scale·z| >= 17`` (fp32 sigmoid saturation; the machine
    ``scale`` = 128 and the ·scale / /scale shifts are exact powers of two).
    The construction ports the relu-era min-form hinge-for-hinge and keeps
    its **cancellation-free** property:

    - Stage 1, lane i: ``a_i = hinge(s·v_i)`` — gate row ``scale·s·v_i``,
      degenerate up lane, output ``1/scale``. Exactly 0 for a nonpositive
      ray at or below the saturation band and exactly ``s·v_i`` for a
      positive ray above it; the band (|v| < 17/(scale·s) ≈ 4.2e-6) is
      cleared ~36x by the smallest nonzero fixture ray (see the
      ``_RAY_SHARPNESS`` note above).
    - Stage 2, lane i: gate row ``scale·(1 − a_i)``, output ``−1/scale``,
      summed with bias ``n``: ``count = n − Σ_i hinge(1 − a_i)``. For a
      saturated positive ray ``a_i >= ~4.8`` so the gate argument is <=
      ~−486 and the hinge is exactly 0 — no subtraction of two large
      near-equal numbers anywhere (the property that motivated the
      original min-form). For a nonpositive ray ``a_i = 0`` and
      ``hinge(1) = Swish(128)/128 = 1.0`` exactly on every deployed kernel
      (σ(128) = 1.0; ·128 and /128 exact), so each term is exactly 1 or 0
      and the count — a sum of <= 1024 exact 0/1 terms, an integer far
      below 2^24 — is exact in fp32.

    The exact-integer claim is pinned at this layer by
    ``tests/scene/test_ray_count.py`` (fixture-extreme ray magnitudes,
    including the smallest nonzero ray).
    """
    n = len(rays)
    s = _RAY_SHARPNESS

    # Stage 1: a_i = hinge(s·v_i).
    a = swiglu_ffn(
        rays,
        scale * s * torch.eye(n, dtype=torch.float32),
        torch.zeros(n, dtype=torch.float32),
        (1.0 / scale) * torch.eye(n, dtype=torch.float32),
        torch.zeros(n, dtype=torch.float32),
        name="ray_scaled",
    )
    # Stage 2: count = n − Σ_i hinge(1 − a_i). hinge(1 − a_i) is 0 for a
    # saturated ray (hinge of a large negative), 1 for a nonpositive ray —
    # so count adds exactly 1 per positive ray.
    count = swiglu_ffn(
        a,
        -scale * torch.eye(n, dtype=torch.float32),  # gate_i = scale·(1 − a_i)
        scale * torch.ones(n, dtype=torch.float32),
        (-1.0 / scale) * torch.ones((n, 1), dtype=torch.float32),  # −Σ hinge_i
        torch.tensor([float(n)], dtype=torch.float32),  # + n
        name="ray_count",
    )
    # Pin the true [0, n] range so the downstream quadrant ``select`` sees a
    # bounded gated input (stage 1's hinge range is otherwise ~s·rays).
    return assert_matches_value_type(
        count, NodeValueType(value_range=Range(0.0, float(n)))
    )


# ---------------------------------------------------------------------------
# Screen-coordinate product (solid-interval coverage key)
# ---------------------------------------------------------------------------


def MUL_SCREEN(a: Node, b: Node) -> Node:
    """Product of two screen-column operands.

    ~2 ulp relative (<= ~1e-3 abs at products <= ~4400). One regression in
    kind vs the relu-era grid, far inside margin: the grid was exactly 0
    error at integer grid points, while the swiglu ± lane pair leaves both
    lanes unsaturated for |a| < 17, so integer screen columns now carry the
    ~2-ulp noise too. The consumers compare against half-integer thresholds
    with the default 0.1 deadband — a >= 100x margin.
    """
    return multiply(a, b)


# DOOM: R_PointToAngle (r_main.c) — BAM angle of (dx, dy) via atan2 octant count.
def signed_world_angle(dx: Node, dy: Node) -> Node:
    """Signed BAM angle in [-ANGLE_BAM/2, ANGLE_BAM/2) of the vector (dx, dy)
    (unified on the 3072 clamp)."""
    abs_dx = _abs_coord(dx)
    abs_dy = _abs_coord(dy)
    abs_pair = concat(abs_dy, abs_dx)
    low_count = _ray_count(linear(abs_pair, _LOW_RAY_MATRIX))
    high_count = _ray_count(linear(abs_pair, _HIGH_RAY_MATRIX))
    base = Linear(
        concat(low_count, high_count),
        torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        name="atan_base",
    )

    one = constant(1.0)
    q1 = base
    q2 = linear(concat(base, one), _Q2_AFFINE)  # ANGLE_BAM/2 - base
    q3 = linear(concat(base, one), _Q3_AFFINE)  # -ANGLE_BAM/2 + base
    q4 = neg(base)

    dx_positive = compare(dx, 0.0)
    dy_positive = compare(dy, 0.0)
    upper = select(dx_positive, q1, q2)
    lower = select(dx_positive, q4, q3)
    return select(dy_positive, upper, lower)


# DOOM: angle wrapping / FOV clipping (r_bsp.c:R_AddLine — clip to ±clipangle).
def wrap_signed_angle(delta: Node) -> Node:
    """Wrap one-turn signed BAM deltas back into [-4096, 4096)."""
    minus_full = add_const(delta, -float(ANGLE_BAM))
    plus_full = add_const(delta, float(ANGLE_BAM))
    over = compare(delta, _SIGNED_ANGLE_ABOVE_MAX)
    under = one_minus(compare(delta, _SIGNED_ANGLE_ABOVE_MIN))
    return select(over, minus_full, select(under, plus_full, delta))


# ---------------------------------------------------------------------------
# Screen comparators, integer-equality, and the backface cross product
# ---------------------------------------------------------------------------
#
# The original's ``compare_const(c, input_range=...)`` drops to the real
# ``compare(node, c)`` (default sharpness 10, deadband 0.1 — the ``input_range``
# has no real-graph counterpart). The thresholds are half-integers, so
# integer-valued inputs land squarely in a flat zone.


def ABS_SMALL_INT(x: Node) -> Node:
    """``|x|`` over the small-integer range.

    Only used for equality / pixel-width decisions, where clamping large
    differences is fine (zero vs nonzero is all that matters)."""
    return piecewise_linear(
        x, [-64.0, 0.0, 64.0], lambda v: abs(v), name="abs_small_int"
    )


def HAS_PIXEL_WIDTH(x_diff: Node) -> Node:
    """±1: does an absolute screen-x difference exceed 0.5 (>= 1 pixel wide)?"""
    return compare(x_diff, 0.5)


def gt_screen(a: Node, b: Node) -> Node:
    """±1: is integer screen column ``a`` greater than ``b``?"""
    return compare(sub(a, b), 0.5)


def min_screen(a: Node, b: Node) -> Node:
    """Smaller of two screen columns."""
    return select(gt_screen(a, b), b, a)


def max_screen(a: Node, b: Node) -> Node:
    """Larger of two screen columns."""
    return select(gt_screen(a, b), a, b)


def same_int(a: Node, b: Node) -> Node:
    """±1: are two integer-valued nodes equal?

    One hat-shaped PWL over the difference: +1 where ``|a-b| <= 0.4``, -1
    where ``|a-b| >= 0.6``, linear ramp between. The ``sub`` fuses into the
    PWL's input projection, so integer equality costs one sublayer instead
    of the previous abs -> compare -> not three (it sat on the compiled
    depth floor via the clip-presence check; paint_cascade_plan.md).
    Integer inputs land squarely in the flats; the 0.4 flat margin covers
    the recovered-value noise the old compare-threshold-at-0.5 form
    absorbed (attention recoveries here carry ~0.02). Differences beyond
    +-64 stay in the outer flats, matching ABS_SMALL_INT's clamp range.
    """

    def hat(v: float) -> float:
        av = abs(v)
        if av <= 0.4:
            return 1.0
        if av >= 0.6:
            return -1.0
        return 1.0 - 2.0 * (av - 0.4) / 0.2

    return piecewise_linear(
        sub(a, b),
        [-64.0, -0.6, -0.4, 0.4, 0.6, 64.0],
        hat,
        name="same_int",
    )


# ---------------------------------------------------------------------------
# R_CheckBBox region classifier
# ---------------------------------------------------------------------------
#
# DOOM: R_CheckBBox boxx/boxy region computation (r_bsp.c lines 404-418) — the
# player-vs-bbox-edge sign tests that index into the 9-region ``checkcoord``
# table. The original spells this ``compare_const(0.0, input_range=(-3000, 3000))``;
# the real-side default-sharpness ``compare`` has the same 0.1 deadband at the
# zero threshold (mirrors the ``dx``/``dy`` sign test inside ``signed_world_angle``).
def COORD_GT_ZERO(x: Node) -> Node:
    """±1: is a map-coordinate / player-vs-bbox-edge delta > 0?"""
    return compare(x, 0.0)


# DOOM: backface-cull sign test (r_bsp.c:R_AddLine, r_main.c:R_PointOnSegSide)
def is_negative_cross(cross_z: Node) -> Node:
    """±1: is the seg-vs-player cross product < 0?"""
    return compare(neg(cross_z), 0.0)


def MUL_CROSS(a: Node, b: Node) -> Node:
    """Cross-product term for the seg backface cull; only the sign feeds
    ``is_negative_cross``.

    e1m1 seg vectors reach |256| and relative coords |308|, so |product| <=
    ~1.5e5 and the ~2-ulp error (~0.03 abs) is invisible to the sign test
    (the old grid budgeted 37.5 of product noise there).
    """
    return multiply(a, b)


# ---------------------------------------------------------------------------
# The drawseg perspective-scale chain — products, clamps, and
# comparators feeding the DRAWSEG_SCALE* / DRAWSEG_*SILHEIGHT VALUE carriers.
# ---------------------------------------------------------------------------
#
# Projection / scale constants mirror ``reference.py``. FOV_HALF_BAM =
# ANGLE_BAM/8 = 45°, so the focal length is (SCREEN_WIDTH-1)/(2·tan 45°).
# ``_TAN_FOV_HALF`` is canonical in vocab.py (imported above); the scale
# ranges declared in value_ranges.py must track this module's scale math.
# (The relu-era grids also sized their breakpoints from value_ranges'
# ``_PROJ_RATIO``; the swiglu ``multiply`` has no grid, so that coupling is
# gone from this file.)
_PROJECTION = (SCREEN_WIDTH - 1) / (2.0 * _TAN_FOV_HALF)
_MIN_SCALE = 1.0 / 256.0
_MAX_SCALE = 64.0
NEAR_DEN_SCALE_FACTOR = 1024.0


# The drawseg / pixel-pass products below all lower to the swiglu
# ``multiply`` — exact to ~2 ulp of the product at any magnitude, no grid,
# no range limit, no extrapolation regime. The named wrappers survive the
# cutover so each J-file call site keeps its semantic name and the operand
# provenance notes; every relu-era grid parameter is gone (the grid builder
# ``_mul_grid`` died with them, and with it the extrapolation trap its
# callers had to size around).


def mul_normal_coord(coef: Node, rel: Node) -> Node:
    """Unit-normal coefficient × view-relative coordinate."""
    return multiply(coef, rel)


def MUL_UNIT(a: Node, b: Node) -> Node:
    """Unit × unit product."""
    return multiply(a, b)


def MUL_FAR_DEN(distance: Node, xtova_cos: Node) -> Node:
    """Far scale denominator: distance × view-angle cosine."""
    return multiply(distance, xtova_cos)


def MUL_NEAR_DEN(distance: Node, xtova_cos: Node) -> Node:
    """Near scale denominator: distance × view-angle cosine, near regime."""
    return multiply(distance, xtova_cos)


def mul_far_scale(numerator: Node, inverse_denominator: Node) -> Node:
    """Far scale = numerator × (1/denominator)."""
    return multiply(numerator, inverse_denominator)


def MUL_NEAR_FLOOR_SCALE(numerator: Node, inverse_denominator: Node) -> Node:
    """Near-floor scale = numerator × (1/denominator)."""
    return multiply(numerator, inverse_denominator)


def mul_scalestep(diff: Node, inverse_width: Node) -> Node:
    """Per-column scale step = (scale2-scale1) × (1/width)."""
    return multiply(diff, inverse_width)


# ---------------------------------------------------------------------------
# Wall-column rasterizer screen-y ops
# ---------------------------------------------------------------------------
# Per-column wall projection: ``top_y_raw = CENTER_Y − worldheight × scale``,
# then round to an integer scanline. The ``mul_height_scale`` product feeds
# the CEIL_Y/FLOOR_Y staircases across a ±0.4-unit deadband; the swiglu
# multiply's ~2-ulp product noise (<= ~3e-3 abs at the maximal ~12800
# screen-y offset) sits far inside it, and the staircases use
# ``sharpness=10000`` (a 1e-4 ramp) so the floored value rounds to an exact
# integer instead of interpolating across a scanline boundary.


def mul_height_scale(height: Node, wall_scale: Node) -> Node:
    """World-relative height × wall scale → screen-y offset."""
    return multiply(height, wall_scale)


def mul_column_scalestep(x_offset: Node, scalestep: Node) -> Node:
    """Column offset × per-column scale step."""
    return multiply(x_offset, scalestep)


def CEIL_Y(x: Node) -> Node:
    """Integer ceil over the 3D-view y range ``[0, VIEW_HEIGHT-1]``;
    ``sharpness=10000`` → 1e-4 ramp. (VIEW_HEIGHT == SCREEN_HEIGHT when the
    status bar is off.)

    Built as ``-floor(-x)`` via ``floor_int(output_map=-k)`` instead of
    ``ceil_int``: ceil_int's output affine (add_const + negate after the
    saturate stage) occupies a scheduled layer on the paint spine, while
    ``output_map`` folds the same per-step constants into the saturate
    stage's weights (the FLOOR_MOD64 precedent) — one layer shallower,
    identical step count and lane/column cost, same saturation clamp at
    the range edges."""
    neg = linear(x, [[-1.0]])
    return floor_int(
        neg,
        -(VIEW_HEIGHT - 1),
        0,
        sharpness=10_000.0,
        output_map=lambda k: float(-k),
    )


def FLOOR_Y(x: Node) -> Node:
    """Integer floor over ``[0, VIEW_HEIGHT-1]``."""
    return floor_int(x, 0, VIEW_HEIGHT - 1, sharpness=10_000.0)


def FLOOR_Y_WIDE(x: Node) -> Node:
    """Integer floor over ``[-128, VIEW_HEIGHT-1]``.
    Keeps negative results so an above-horizon upper region yields a negative
    ``mid`` and the integer ``le_span_y`` visibility check marks the empty upper
    span empty (narrow ``FLOOR_Y`` would clamp it to 0 and hide that case)."""
    return floor_int(x, -128, VIEW_HEIGHT - 1, sharpness=10_000.0)


def CEIL_Y_WIDE(x: Node) -> Node:
    """Integer ceil over ``[0, 128]``. Keeps the large
    positive result for a below-screen lower region so ``le_span_y`` treats the
    lower-empty case correctly (narrow ``CEIL_Y`` would clamp to SCREEN_HEIGHT-1
    and read visible).

    Same ``floor_int(output_map=-k)`` form as :func:`CEIL_Y` (see there) —
    deletes ceil_int's scheduled output-affine layer from the paint spine."""
    neg = linear(x, [[-1.0]])
    return floor_int(
        neg,
        -128,
        0,
        sharpness=10_000.0,
        output_map=lambda k: float(-k),
    )


def gt_y_ceil_boundary(raw_y: Node, boundary: Node) -> Node:
    """±1: ``raw_y > boundary - 0.4``. "ceil" names the reference renderer's
    rounding direction, not a ceiling plane; the -0.4
    deadband sits just below an integer scanline so multiply noise cannot flip
    the test at a boundary."""
    return compare(sub(raw_y, boundary), -0.4)


def gt_y_floor_boundary(raw_y: Node, boundary: Node) -> Node:
    """±1: ``raw_y > boundary + 0.4``."""
    return compare(sub(raw_y, boundary), 0.4)


def le_span_y(y1: Node, y2: Node) -> Node:
    """±1: integer span non-empty test ``y1 <= y2`` as ``y2 - y1 > -0.5``.

    One fused sub+compare sublayer (the old ``not (y1 - y2 > 0.5)`` form
    spent a second sublayer on the negation, and sat on the compiled depth
    floor via the span-ok flags; paint_cascade_plan.md). Same half-integer
    threshold contract on integer y values."""
    return compare(sub(y2, y1), -0.5)


def SPAN_Y_CLAMP(x: Node) -> Node:
    """Clamp a span y to the 3D view ``[0, VIEW_HEIGHT-1]`` (== SCREEN_HEIGHT-1
    when the status bar is off)."""
    return clamp(x, 0.0, float(VIEW_HEIGHT - 1))


def CLIP_Y_CLAMP(x: Node) -> Node:
    """Clamp a clip-array y to ``[-1, VIEW_HEIGHT]`` (the open-floor bound is the
    view bottom; == SCREEN_HEIGHT when the status bar is off)."""
    return clamp(x, -1.0, float(VIEW_HEIGHT))


def FAR_DEN_CLAMP(x: Node) -> Node:
    """Clamp the far denominator to ``[0.7, 1500]``."""
    return clamp(x, 0.7, 1500.0)


def NEAR_DEN_CLAMP(x: Node) -> Node:
    """Clamp the near denominator to ``[0.0007, 1]``."""
    return clamp(x, 0.0007, 1.0)


def RW_DISTANCE_CLAMP(x: Node) -> Node:
    """Clamp the perpendicular seg distance to ``[0.001, 1500]`` (covers e1m1
    far walls)."""
    return clamp(x, 0.001, 1500.0)


def SCALE_CLAMP(x: Node) -> Node:
    """Clamp wall scale to ``[1/256, 64]``."""
    return clamp(x, _MIN_SCALE, _MAX_SCALE)


def SINEB_CLAMP(x: Node) -> Node:
    """Clamp the sine-basis to ``[1e-6, 1]``."""
    return clamp(x, 1.0e-6, 1.0)


def DIST_GT_ONE(distance: Node) -> Node:
    """±1: is the distance past the near plane (> 1)?"""
    return compare(distance, 1.0)


def SINEB_ABOVE_FLOOR(sineb: Node) -> Node:
    """±1: is the sine-basis above the near-floor threshold 1/256? (sharpness
    32000 for a ~0.001 deadband)."""
    return compare(sineb, 1.0 / 256.0, sharpness=32000.0)


def gt_height(a: Node, b: Node) -> Node:
    """±1: is height ``a`` greater than ``b``?"""
    return compare(sub(a, b), 0.5)


def NEAR_DEN_SCALE_UP(near_den: Node) -> Node:
    """Scale the near denominator up by 1024 before reciprocal."""
    return linear(near_den, [[NEAR_DEN_SCALE_FACTOR]])


def PROJECT_SCALE(sineb: Node) -> Node:
    """Multiply the sine-basis by the projection focal length."""
    return linear(sineb, [[_PROJECTION]])


def NEAR_FLOOR_NUMERATOR(_: Node) -> Node:
    """The constant near-floor scale numerator."""
    return constant(_PROJECTION * 1.0e-6 * NEAR_DEN_SCALE_FACTOR)


def MAX_SCALE_VALUE(_: Node) -> Node:
    """The max-scale sentinel."""
    return constant(_MAX_SCALE)


# ---------------------------------------------------------------------------
# The pixel pass: per-pixel texture-coordinate products + the native
# coordinate floor. Each is a ``multiply(...)`` / ``floor_int(...)`` from the
# pixel pipeline (``uv_compute`` / ``pixel_dispatcher`` / ``flat_state``),
# lowered here so the J files stay node-free at import (the multiply / floor
# node is built only on call).
# ---------------------------------------------------------------------------


def FLOOR_NATIVE(x: Node) -> Node:
    """Integer floor over the native texture-coordinate range ``[-1023, 1023]``
    (``floor_int(-1023, 1023, sharpness=10_000.0)``, shared by the wall-u,
    wall-v, and flat-frac coordinate floors). The 1e-4 ramp keeps the floored
    value an exact integer that lands on a sawtooth/mod grid line."""
    return floor_int(x, -1023, 1023, sharpness=10_000.0)


def FLAT_DIST_INDEX_FLOOR(x: Node) -> Node:
    """Integer floor of the flat distance-light index over ``[0, MAXLIGHTZ]``
    (``floor_int(0, MAXLIGHTZ, sharpness=10_000.0)``); clamps to ``[0, 128]`` so
    the distance-light table pick never overruns its 128 entries."""
    from .doom_lighting import MAXLIGHTZ

    return floor_int(x, 0, MAXLIGHTZ, sharpness=10_000.0)


def mul_u_native(rw_distance: Node, tan_rel: Node) -> Node:
    """rw_distance × per-column tangent → native wall u."""
    return multiply(rw_distance, tan_rel)


def mul_pixel_dc_iscale(pixel_index: Node, dc_iscale: Node) -> Node:
    """pixel/screen-y offset × dc_iscale → native-v offset."""
    return multiply(pixel_index, dc_iscale)


def mul_k_step(pixel_index: Node, step: Node) -> Node:
    """flat pixel index × affine cursor step → texture-coordinate delta."""
    return multiply(pixel_index, step)


def mul_ph_yslope(planeheight: Node, yslope: Node) -> Node:
    """planeheight × per-scanline yslope → ray distance."""
    return multiply(planeheight, yslope)


def mul_dist_distscale(distance: Node, distscale: Node) -> Node:
    """ray distance × column distscale → ray length."""
    return multiply(distance, distscale)


def mul_len_trig(length: Node, trig: Node) -> Node:
    """ray length × view-ray cos/sin → world-space frac offset."""
    return multiply(length, trig)


def mul_dist_base(distance: Node, basescale: Node) -> Node:
    """ray distance × base x/y scale → per-screen-x texture step."""
    return multiply(distance, basescale)


# ---------------------------------------------------------------------------
# Screen-column radix key (shared by solid_intervals / visplane_state /
# wall_column_state — was three private copies under two naming schemes)
# ---------------------------------------------------------------------------

# Columns split into a high *bucket* digit and a low digit so a column-
# equality key is two one-hots of total width N_COL_BUCKETS + COL_RADIX_BASE
# (16 at SCREEN_WIDTH=60, 26 at 160) instead of a width-SCREEN_WIDTH one-hot.
COL_RADIX_BASE = math.ceil(math.sqrt(COLUMN_COUNT + 1))  # 8 at CC=60, 13 at 160
N_COL_BUCKETS = COLUMN_COUNT // COL_RADIX_BASE + 1


def mod_sawtooth(scalar: Node, base: int, max_value: int) -> Node:
    """``scalar % base`` as ONE sawtooth PWL — the depth-parallel form of
    ``mod_const``'s serial chain (thermometer -> scale -> subtract), for
    every site that builds a radix (bucket, digit) pair: the digit runs in
    the same layer as the bucket thermometer instead of two layers after it.

    Exact at integer inputs: breakpoint pairs bracket each ``k*base - 0.5``
    jump by ±0.05 — the same transition placement as
    ``thermometer_floor_div``, so the pair stays bucket-consistent (a soft
    ~0.02-leaked input near an edge lands on the same (bucket, digit) pair
    as the thermometer's bucket), and integers sit 0.45 from any fillet.
    A digit read just below a bucket edge comes out slightly negative
    (e.g. -0.02 at a leaked ``k*base``) — inside ``one_hot`` slot 0's
    in_range window, exactly matching ``mod_const``.

    The grid extends down to -1.45 holding the identity below the k=0
    edge (``v - base*max(0, floor((v+0.5)/base))``): ``mod_const`` maps a
    ``threshold == -1`` (bucket 0) to digit -1 — ``next_plane_after``'s
    find-first query needs that — and the PWL's default clamp at the
    first breakpoint would pin it to -0.45 instead. The extension keeps
    slope 1, so the extra grid point costs no lanes (equal-slope segments
    are free)."""
    grid: list[float] = [-1.45]
    n_buckets = max_value // base + 1
    for k in range(1, n_buckets + 1):
        edge = float(k * base)
        grid += [edge - 0.55, edge - 0.45]
    grid.append(float(n_buckets * base) + float(base) - 0.55)
    return piecewise_linear(
        scalar,
        grid,
        lambda v: v - base * max(0.0, math.floor((v + 0.5) / base)),
        name="mod_sawtooth",
    )


def radix_col_key(col_scalar: Node) -> Node:
    """Exact screen-column-equality key: ``concat(one_hot(c // B),
    one_hot(c % B))`` over the column radix base.  The dot of two such keys
    is ``bucket_match + digit_match`` — 2 on an exact column match, <= 1
    otherwise (cancellation-free one-hot dot).

    The column may be the soft (~0.02 leak) output of a ``pick_most_recent``
    recovery: ``thermometer_floor_div`` and ``mod_sawtooth`` place their
    ramps at the same ``k*B - 0.5`` transitions (bucket-consistent, and
    0.45 away from any integer, far beyond the leak), so a leaked value
    stays on the right (bucket, digit) pair, and ``one_hot`` rounds it to
    a clean integer one-hot — no pre-snap is needed and a matched key's
    dot is exactly 2 (paint_cascade_plan.md, step 5)."""
    b = COL_RADIX_BASE
    hi = thermometer_floor_div(col_scalar, b, COLUMN_COUNT)
    lo = mod_sawtooth(col_scalar, b, COLUMN_COUNT)
    return concat(one_hot(hi, N_COL_BUCKETS), one_hot(lo, COL_RADIX_BASE))
