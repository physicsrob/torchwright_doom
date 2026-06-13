"""Forward/render math ops, mirroring the read-side + write-side subset of
``doom_sandbox/implementation/forward/constants.py`` / ``forward/ops.py``.

Grouped by what they compute:

- Read side: ``MARKER_PRESENT`` / ``and_`` / ``one_minus``.
- BSP traversal: ``sub`` / ``neg`` / ``add_const`` (affine glue), ``mul_side``
  (the ``R_PointOnSide`` cross-product multiply), and the ``SIDE_POSITIVE`` /
  ``IS_SUBSECTOR`` / ``DEPTH_NONZERO`` comparators.
- Seg projection: ``signed_world_angle`` / ``wrap_signed_angle`` (the
  ``R_PointToAngle`` BAM-atan2 octant builder).
- Wall-column rasterization: the screen-y clamp staircases (``CEIL_Y`` /
  ``FLOOR_Y``) and the perspective-scale products.
- Pixel pass: the per-pixel texture-coordinate products and native-coordinate
  rounding.
"""

from __future__ import annotations

import math

import torch

from torchwright.graph import Linear, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.arithmetic_ops import (
    ceil_int,
    clamp,
    compare,
    floor_int,
    mod_const,
    multiply_2d,
    piecewise_linear,
    thermometer_floor_div,
)
from torchwright.ops.linear_relu_linear import linear_relu_linear
from torchwright.ops.logic_ops import bool_all_true, bool_any_true, bool_not

from .constants import SCREEN_HEIGHT, SCREEN_WIDTH
from .std import concat, constant, linear, one_hot, select
from .value_ranges import _PROJ_RATIO
from .vocab import _TAN_FOV_HALF, ANGLE_BAM, N_NODES_MAX


# Marker and integer-slot ids are exact integers, so ``> 0.5`` cleanly
# means "different / present / nonzero". Mirrors the sandbox
# ``compare_const(0.5, input_range=(-0.5, 1.5))`` at the default
# ``step_sharpness`` (deadband 0.1, far wider than any marker noise).
def MARKER_PRESENT(marker: Node) -> Node:
    """±1 boolean: was a marker (≈1) published at the picked row?"""
    return compare(marker, 0.5)


def and_(a: Node, b: Node) -> Node:
    """Boolean conjunction over two ±1 predicates (sandbox ``and_``)."""
    return bool_all_true([a, b])


def one_minus(a: Node) -> Node:
    """Boolean negation of a ±1 predicate (sandbox ``one_minus``)."""
    return bool_not(a)


def or_(a: Node, b: Node) -> Node:
    """Boolean disjunction over two ±1 predicates (sandbox ``or_``)."""
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
    next token. The float64 sandbox kept the cond exact, so this snap is the
    fp32-fidelity counterpart (cf. the float64->fp32 sharp-step discipline)."""
    return compare(x, 0.0)


def SCREEN_X_CLAMP(x: Node) -> Node:
    """Clamp a screen column into ``[0, SCREEN_WIDTH-1]`` (sandbox
    ``SCREEN_X_CLAMP``).

    Guards an ``IntSlot(0, SCREEN_WIDTH)`` deembed when a pick falls through to
    pure recency (no FIND_RUN match in past) and would otherwise return the
    SCREEN_WIDTH sentinel."""
    return clamp(x, 0.0, float(SCREEN_WIDTH - 1))


# ---------------------------------------------------------------------------
# Affine glue + the R_PointOnSide cross product
# ---------------------------------------------------------------------------


def sub(a: Node, b: Node) -> Node:
    """``a - b`` (sandbox ``sub`` -> ``linear(concat(a, b), [[1],[-1]])``)."""
    return Linear(
        concat(a, b),
        torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
        name="sub",
    )


def neg(a: Node) -> Node:
    """``-a`` (sandbox ``neg``)."""
    return Linear(a, torch.tensor([[-1.0]], dtype=torch.float32), name="neg")


def add_const(a: Node, c: float) -> Node:
    """``a + c`` as one fused ``Linear`` (sandbox ``add_const``)."""
    return Linear(
        a,
        torch.tensor([[1.0]], dtype=torch.float32),
        torch.tensor([float(c)], dtype=torch.float32),
        name="add_const",
    )


# DOOM: R_PointOnSide cross-product sign (r_main.c). Only the SIGN feeds
# SIDE_POSITIVE. The sandbox keeps a magnitude split (MUL_SIDE_SMALL/LARGE)
# for precision near unit-normal coefficients, but (1) the BSP side test
# always feeds large node deltas (R2 = [-512, 512]) and (2) the real-side
# ``multiply_2d`` *extrapolates* out-of-range inputs (the sandbox ``multiply``
# clamps), so feeding a large coef to a small grid yields a degenerate bound.
# A single grid over the full coefficient / relative-coordinate ranges is
# therefore both correct (the sign is robust to the ~step1*step2/4 product
# noise for poses off the partition planes) and avoids the extrapolation trap.
# Node deltas are recovered within R2 = [-512, 512]; rel = view - p with both
# in R1 = [-1152, 1152], so |rel| <= 2304 — all inputs stay in-grid.
def mul_side(coef: Node, rel: Node) -> Node:
    """Product of a partition coefficient and a view-relative coordinate
    (sandbox ``mul_side``), resolved to the sign by ``SIDE_POSITIVE``."""
    return multiply_2d(
        coef,
        rel,
        max_abs1=512.0,
        max_abs2=2400.0,
        step1=8.0,
        step2=37.5,
        name="mul_side",
    )


def SIDE_POSITIVE(cross_z: Node) -> Node:
    """±1: is the R_PointOnSide cross product > 0? (sandbox ``SIDE_POSITIVE``)."""
    return compare(cross_z, 0.0)


def IS_SUBSECTOR(child_u: Node) -> Node:
    """±1: is a unified child id a subsector (>= N_NODES_MAX) vs a node?"""
    return compare(child_u, float(N_NODES_MAX) - 0.5)


def DEPTH_NONZERO(depth: Node) -> Node:
    """±1: is the BSP tree depth nonzero? (sandbox ``DEPTH_NONZERO``)."""
    return compare(depth, 0.5)


# ---------------------------------------------------------------------------
# R_PointToAngle BAM-atan2 octant + one-turn angle wrap
# ---------------------------------------------------------------------------
#
# Mirrors ``forward/ops.py:signed_world_angle`` / ``wrap_signed_angle``. The
# sandbox keeps two near-duplicate octant builders — ``ops.signed_world_angle``
# (clamp 2048, seg endpoints) and ``bbox_pruning._signed_world_angle_bbox``
# (clamp 3072, bbox corners). They are unified here on the **wider 3072 clamp**:
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
# (2π/8192) times the slope; sharpness 32000 (ramp width 3.125e-5 input units)
# sits well inside that bound, so an integer-angle ray never lands in a ramp.
# Mirrors the sandbox ``RAY_GT_ZERO`` deadband.
_RAY_SHARPNESS = 32000.0

# Player-relative angles normalize to the signed BAM interval [-4096, 4096);
# one-turn deltas outside it wrap by ANGLE_BAM (sandbox SIGNED_ANGLE_ABOVE_*).
_SIGNED_ANGLE_ABOVE_MAX = 4095.5
_SIGNED_ANGLE_ABOVE_MIN = -4096.5


def _abs_coord(x: Node) -> Node:
    """``|x|`` clamped to the atan2 square (sandbox ``ABS_COORD``).

    3 breakpoints with the kink at 0 make ``abs`` exact in-range; out-of-range
    inputs clamp to ±``_ATAN_ABS_RANGE`` (``piecewise_linear``'s default).
    """
    return piecewise_linear(x, _ABS_BREAKPOINTS, lambda v: abs(v), name="abs_coord")


def _ray_count(rays: Node) -> Node:
    """Count the +1 (positive) components of a ray vector.

    The sandbox spells this ``reduce_sum(bool_to_01(RAY_GT_ZERO(rays)))`` — an
    elementwise ±1 ``compare`` (sharpness 32000), a 0/1 cast, then a sum — and
    runs it in float64. The real graph is float32 and ``compare`` is
    scalar-only, so the elementwise 0/1 step is built directly, in the
    **cancellation-free** form ``step(v) = min(relu(s·v), 1) = 1 − relu(1 −
    relu(s·v))``. The naive ``relu(s·v) − relu(s·v − 1)`` would subtract two
    near-equal large numbers — a ray reaches ~6000, ``s·v`` ~2e8, and the
    ``−1`` is lost to fp32 ulp (~16 at that magnitude; even ~0.004 at the
    saturated ~32000 scale accumulates to ~0.2 over 1024 rays). The ``min``
    form instead computes ``relu(1 − big) = 0`` exactly for a saturated ray
    (no subtraction of two big numbers), so each step is exactly 1 or 0 and the
    sum is an exact integer — matching the sandbox on any ray outside the ±1/s
    ramp (the smallest nonzero fixture ray ≈1.5e-4 clears the 3.1e-5 ramp).

    Two sublayers: ``a = relu(s·v)`` then ``count = n − Σ_i relu(1 − a_i)``.
    """
    n = len(rays)
    s = _RAY_SHARPNESS

    # Stage 1: a_i = relu(s·v_i).
    a = linear_relu_linear(
        rays,
        s * torch.eye(n, dtype=torch.float32),
        torch.zeros(n, dtype=torch.float32),
        torch.eye(n, dtype=torch.float32),
        torch.zeros(n, dtype=torch.float32),
        name="ray_scaled",
    )
    # Stage 2: count = n − Σ_i relu(1 − a_i). relu(1 − a_i) is 0 for a
    # saturated ray (relu of a large negative), 1 − a_i in the ramp, 1 for a
    # negative ray — so count adds exactly 1 per positive ray.
    count = linear_relu_linear(
        a,
        -torch.eye(n, dtype=torch.float32),  # hidden_i = relu(−a_i + 1)
        torch.ones(n, dtype=torch.float32),
        -torch.ones((n, 1), dtype=torch.float32),  # −Σ hidden_i
        torch.tensor([float(n)], dtype=torch.float32),  # + n
        name="ray_count",
    )
    # Pin the true [0, n] range so the downstream quadrant ``select`` sees a
    # bounded gated input (stage 1's relu range is otherwise ~s·rays).
    return assert_matches_value_type(
        count, NodeValueType(value_range=Range(0.0, float(n)))
    )


# ---------------------------------------------------------------------------
# Screen-coordinate product (solid-interval coverage key)
# ---------------------------------------------------------------------------


def MUL_SCREEN(a: Node, b: Node) -> Node:
    """Product of two screen-column operands (sandbox ``MUL_SCREEN``).

    Sandbox grid: ``multiply(input_range=((-2, SW+2), (-2, SW+2)),
    breakpoints=65)`` — step 1.0, so integer columns land on grid lines and the
    product is exact there. The operands (``x1-1`` / ``x2+1``) stay within
    ``[-2, SW+2]``; a symmetric ``max_abs = SW+2`` over-covers the sandbox's
    asymmetric range at the same step (real ``multiply_2d`` extrapolates
    out-of-grid, so sizing to the full operand range is the safe choice — same
    extrapolation-trap note as ``mul_side``).
    """
    bound = float(SCREEN_WIDTH + 2)
    return multiply_2d(
        a, b, max_abs1=bound, max_abs2=bound, step1=1.0, step2=1.0, name="mul_screen"
    )


# DOOM: R_PointToAngle (r_main.c) — BAM angle of (dx, dy) via atan2 octant count.
def signed_world_angle(dx: Node, dy: Node) -> Node:
    """Signed BAM angle in [-ANGLE_BAM/2, ANGLE_BAM/2) of the vector (dx, dy)
    (sandbox ``signed_world_angle``, unified on the 3072 clamp)."""
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
    """Wrap one-turn signed BAM deltas back into [-4096, 4096)
    (sandbox ``wrap_signed_angle``)."""
    minus_full = add_const(delta, -float(ANGLE_BAM))
    plus_full = add_const(delta, float(ANGLE_BAM))
    over = compare(delta, _SIGNED_ANGLE_ABOVE_MAX)
    under = one_minus(compare(delta, _SIGNED_ANGLE_ABOVE_MIN))
    return select(over, minus_full, select(under, plus_full, delta))


# ---------------------------------------------------------------------------
# Screen comparators, integer-equality, and the backface cross product
# ---------------------------------------------------------------------------
#
# Mirror of the corresponding ``forward/ops.py`` helpers. The sandbox
# ``compare_const(c, input_range=...)`` drops to the real ``compare(node, c)``
# (default sharpness 10, deadband 0.1 — the ``input_range`` has no real-graph
# counterpart). The thresholds are half-integers, so integer-valued inputs land
# squarely in a flat zone.


def ABS_SMALL_INT(x: Node) -> Node:
    """``|x|`` over the small-integer range (sandbox ``ABS_SMALL_INT``).

    Only used for equality / pixel-width decisions, where clamping large
    differences is fine (zero vs nonzero is all that matters)."""
    return piecewise_linear(
        x, [-64.0, 0.0, 64.0], lambda v: abs(v), name="abs_small_int"
    )


def HAS_PIXEL_WIDTH(x_diff: Node) -> Node:
    """±1: does an absolute screen-x difference exceed 0.5 (>= 1 pixel wide)?
    (sandbox ``HAS_PIXEL_WIDTH``)."""
    return compare(x_diff, 0.5)


def gt_screen(a: Node, b: Node) -> Node:
    """±1: is integer screen column ``a`` greater than ``b``? (sandbox
    ``gt_screen`` -> ``SCREEN_GT_ZERO(sub(a, b))``)."""
    return compare(sub(a, b), 0.5)


def min_screen(a: Node, b: Node) -> Node:
    """Smaller of two screen columns (sandbox ``min_screen``)."""
    return select(gt_screen(a, b), b, a)


def max_screen(a: Node, b: Node) -> Node:
    """Larger of two screen columns (sandbox ``max_screen``)."""
    return select(gt_screen(a, b), a, b)


def same_int(a: Node, b: Node) -> Node:
    """±1: are two integer-valued nodes equal? (sandbox ``same_int``)."""
    diff_abs = ABS_SMALL_INT(sub(a, b))
    return one_minus(compare(diff_abs, 0.5))  # not (|a-b| > 0.5)


# ---------------------------------------------------------------------------
# R_CheckBBox region classifier
# ---------------------------------------------------------------------------
#
# DOOM: R_CheckBBox boxx/boxy region computation (r_bsp.c lines 404-418) — the
# player-vs-bbox-edge sign tests that index into the 9-region ``checkcoord``
# table. The sandbox spells this ``compare_const(0.0, input_range=(-3000, 3000))``;
# the real-side default-sharpness ``compare`` has the same 0.1 deadband at the
# zero threshold (mirrors the ``dx``/``dy`` sign test inside ``signed_world_angle``).
def COORD_GT_ZERO(x: Node) -> Node:
    """±1: is a map-coordinate / player-vs-bbox-edge delta > 0? (sandbox
    ``COORD_GT_ZERO``)."""
    return compare(x, 0.0)


# DOOM: backface-cull sign test (r_bsp.c:R_AddLine, r_main.c:R_PointOnSegSide)
def is_negative_cross(cross_z: Node) -> Node:
    """±1: is the seg-vs-player cross product < 0? (sandbox ``is_negative_cross``
    -> ``CROSS_GT_ZERO(neg(cross_z))``)."""
    return compare(neg(cross_z), 0.0)


def MUL_CROSS(a: Node, b: Node) -> Node:
    """Cross-product term for the seg backface cull (sandbox ``MUL_CROSS``).

    Sandbox grid ``multiply(input_range=((-256, 256), (-600, 600)),
    breakpoints=65)`` — step 8.0 on the seg-vector axis, 18.75 on the
    view-relative axis. e1m1 seg vectors reach |256| (a grid vertex) and
    relative coords |308| (< 600), so the operands stay in-grid; only the sign
    feeds ``is_negative_cross``.
    """
    return multiply_2d(
        a, b, max_abs1=256.0, max_abs2=600.0, step1=8.0, step2=18.75, name="mul_cross"
    )


# ---------------------------------------------------------------------------
# The drawseg perspective-scale chain — products, clamps, and
# comparators feeding the DRAWSEG_SCALE* / DRAWSEG_*SILHEIGHT VALUE carriers.
# ---------------------------------------------------------------------------
#
# Projection / scale constants mirror ``reference.py``. FOV_HALF_BAM =
# ANGLE_BAM/8 = 45°, so the focal length is (SCREEN_WIDTH-1)/(2·tan 45°).
# ``_TAN_FOV_HALF`` is canonical in vocab.py and ``_PROJ_RATIO`` (the focal
# length normalised to the width-60 tuning, = 1.0 at 60×50) in
# value_ranges.py — both imported above; the ranges in value_ranges and
# this module's scale math must track the same values.
_PROJECTION = (SCREEN_WIDTH - 1) / (2.0 * _TAN_FOV_HALF)
_MIN_SCALE = 1.0 / 256.0
_MAX_SCALE = 64.0
NEAR_DEN_SCALE_FACTOR = 1024.0


def _mul_grid(
    a: Node,
    b: Node,
    *,
    lo1: float,
    hi1: float,
    lo2: float,
    hi2: float,
    n: int,
    name: str,
) -> Node:
    """Sandbox ``multiply(input_range=((lo1,hi1),(lo2,hi2)), breakpoints=n)``.

    Lowers to ``multiply_2d`` with explicit (asymmetric) breakpoints. Now that
    ``multiply_2d`` builds in O(n) (the quarter-square fast path, torchwright
    2066416) the prior ``low_rank_2d(rank=1)`` build-cost stopgap is gone. The
    rank-1 form was *not* more precise: its SVD truncation is lossless for a
    product, but it then multiplies the two (exact, linear) singular-vector
    interpolants through an inner ``multiply_2d`` on a coarse 20-step grid, so it
    carries a larger residual (~0.1 abs on these grids) than a direct
    ``multiply_2d`` over the full 257-bp grid (~1e-3 abs). One behavioural note:
    ``multiply_2d`` *extrapolates* out-of-grid operands (the sandbox ``multiply``
    and ``low_rank_2d`` clamp) — fine here because every grid spans the operand
    range and any wrong-regime product is discarded by ``select(near, …)``
    downstream. Both validated by the projection gate.
    """
    bp1 = [lo1 + i * (hi1 - lo1) / (n - 1) for i in range(n)]
    bp2 = [lo2 + i * (hi2 - lo2) / (n - 1) for i in range(n)]
    return multiply_2d(
        a, b, max_abs1=hi1, max_abs2=hi2, breakpoints1=bp1, breakpoints2=bp2, name=name
    )


def mul_normal_coord(coef: Node, rel: Node) -> Node:
    """Unit-normal coefficient × view-relative coordinate (sandbox
    ``MUL_NORMAL_COORD``, ``((-1,1),(-1200,1200))``, 65 bp)."""
    return _mul_grid(
        coef,
        rel,
        lo1=-1.0,
        hi1=1.0,
        lo2=-1200.0,
        hi2=1200.0,
        n=65,
        name="mul_normal_coord",
    )


def MUL_UNIT(a: Node, b: Node) -> Node:
    """Unit × unit product (sandbox ``MUL_UNIT``, ``((-1,1),(-1,1))``, 33 bp)."""
    return _mul_grid(a, b, lo1=-1.0, hi1=1.0, lo2=-1.0, hi2=1.0, n=33, name="mul_unit")


def MUL_FAR_DEN(distance: Node, xtova_cos: Node) -> Node:
    """Far scale denominator: distance × view-angle cosine (sandbox
    ``MUL_FAR_DEN``, ``((1,1500),(0.7,1.01))``, 257 bp)."""
    return _mul_grid(
        distance,
        xtova_cos,
        lo1=1.0,
        hi1=1500.0,
        lo2=0.7,
        hi2=1.01,
        n=257,
        name="mul_far_den",
    )


def MUL_NEAR_DEN(distance: Node, xtova_cos: Node) -> Node:
    """Near scale denominator (sandbox ``MUL_NEAR_DEN``, ``((0.001,1),(0.7,1.01))``,
    257 bp)."""
    return _mul_grid(
        distance,
        xtova_cos,
        lo1=0.001,
        hi1=1.0,
        lo2=0.7,
        hi2=1.01,
        n=257,
        name="mul_near_den",
    )


def mul_far_scale(numerator: Node, inverse_denominator: Node) -> Node:
    """Far scale = numerator × (1/denominator) (sandbox ``MUL_FAR_SCALE``,
    ``((0, 32·ratio),(0, 0.1))``, 257 bp)."""
    return _mul_grid(
        numerator,
        inverse_denominator,
        lo1=0.0,
        hi1=32.0 * _PROJ_RATIO,
        lo2=0.0,
        hi2=0.1,
        n=257,
        name="mul_far_scale",
    )


def MUL_NEAR_FLOOR_SCALE(numerator: Node, inverse_denominator: Node) -> Node:
    """Near-floor scale (sandbox ``MUL_NEAR_FLOOR_SCALE``, ``((0, 0.1·ratio),(0, 2))``,
    257 bp)."""
    return _mul_grid(
        numerator,
        inverse_denominator,
        lo1=0.0,
        hi1=0.1 * _PROJ_RATIO,
        lo2=0.0,
        hi2=2.0,
        n=257,
        name="mul_near_floor_scale",
    )


def mul_scalestep(diff: Node, inverse_width: Node) -> Node:
    """Per-column scale step = (scale2-scale1) × (1/width) (sandbox
    ``MUL_SCALESTEP``, ``((-2.5·ratio, 2.5·ratio),(0, 1))``, 257 bp)."""
    return _mul_grid(
        diff,
        inverse_width,
        lo1=-2.5 * _PROJ_RATIO,
        hi1=2.5 * _PROJ_RATIO,
        lo2=0.0,
        hi2=1.0,
        n=257,
        name="mul_scalestep",
    )


# ---------------------------------------------------------------------------
# Wall-column rasterizer screen-y ops
# ---------------------------------------------------------------------------
# Per-column wall projection: ``top_y_raw = CENTER_Y − worldheight × scale``,
# then round to an integer scanline. The ``mul_height_scale`` product feeds the
# CEIL_Y/FLOOR_Y staircases across a ±0.4-unit deadband, so it rides a wide
# 1024-point grid (the comparator-sensitive projection product) and the
# staircases use ``sharpness=10000`` (a 1e-4 ramp) — narrow enough that the
# product's piecewise-linear noise (~0.077) lands in the flat zone and rounds
# to an exact integer instead of interpolating across a scanline boundary.


def mul_height_scale(height: Node, scale: Node) -> Node:
    """World-relative height × wall scale → screen-y offset (sandbox
    ``MUL_HEIGHT_SCALE``, ``((-200, 200), (0, 64))``, 1024 bp). The height axis
    spans the full e1m1 ``|sector_h − viewz|`` range (~175) so a far wall does
    not extrapolate off the grid."""
    return _mul_grid(
        height,
        scale,
        lo1=-200.0,
        hi1=200.0,
        lo2=0.0,
        hi2=_MAX_SCALE,
        n=1024,
        name="mul_height_scale",
    )


def mul_column_scalestep(x_offset: Node, scalestep: Node) -> Node:
    """Column offset × per-column scale step (sandbox ``MUL_COLUMN_SCALESTEP``,
    ``((0, SCREEN_WIDTH), (-1, 1))``, ``SCREEN_WIDTH+1`` bp). The offset axis is
    an integer column, so ``SCREEN_WIDTH+1`` breakpoints land each column
    exactly on a grid line."""
    return _mul_grid(
        x_offset,
        scalestep,
        lo1=0.0,
        hi1=float(SCREEN_WIDTH),
        lo2=-1.0,
        hi2=1.0,
        n=SCREEN_WIDTH + 1,
        name="mul_column_scalestep",
    )


def CEIL_Y(x: Node) -> Node:
    """Integer ceil over the screen-y range ``[0, SCREEN_HEIGHT-1]`` (sandbox
    ``CEIL_Y``); ``sharpness=10000`` → 1e-4 ramp."""
    return ceil_int(x, 0, SCREEN_HEIGHT - 1, sharpness=10_000.0)


def FLOOR_Y(x: Node) -> Node:
    """Integer floor over ``[0, SCREEN_HEIGHT-1]`` (sandbox ``FLOOR_Y``)."""
    return floor_int(x, 0, SCREEN_HEIGHT - 1, sharpness=10_000.0)


def FLOOR_Y_WIDE(x: Node) -> Node:
    """Integer floor over ``[-128, SCREEN_HEIGHT-1]`` (sandbox ``FLOOR_Y_WIDE``).
    Keeps negative results so an above-horizon upper region yields a negative
    ``mid`` and the integer ``le_span_y`` visibility check marks the empty upper
    span empty (narrow ``FLOOR_Y`` would clamp it to 0 and hide that case)."""
    return floor_int(x, -128, SCREEN_HEIGHT - 1, sharpness=10_000.0)


def CEIL_Y_WIDE(x: Node) -> Node:
    """Integer ceil over ``[0, 128]`` (sandbox ``CEIL_Y_WIDE``). Keeps the large
    positive result for a below-screen lower region so ``le_span_y`` treats the
    lower-empty case correctly (narrow ``CEIL_Y`` would clamp to SCREEN_HEIGHT-1
    and read visible)."""
    return ceil_int(x, 0, 128, sharpness=10_000.0)


def gt_y_ceil_boundary(raw_y: Node, boundary: Node) -> Node:
    """±1: ``raw_y > boundary - 0.4`` (sandbox ``gt_y_ceil_boundary``). "ceil"
    names the reference's rounding direction, not a ceiling plane; the -0.4
    deadband sits just below an integer scanline so multiply noise cannot flip
    the test at a boundary."""
    return compare(sub(raw_y, boundary), -0.4)


def gt_y_floor_boundary(raw_y: Node, boundary: Node) -> Node:
    """±1: ``raw_y > boundary + 0.4`` (sandbox ``gt_y_floor_boundary``)."""
    return compare(sub(raw_y, boundary), 0.4)


def le_span_y(y1: Node, y2: Node) -> Node:
    """±1: integer span non-empty test ``y1 <= y2`` (sandbox ``le_span_y`` =
    ``not (y1 - y2 > 0.5)``)."""
    return one_minus(compare(sub(y1, y2), 0.5))


def SPAN_Y_CLAMP(x: Node) -> Node:
    """Clamp a span y to ``[0, SCREEN_HEIGHT-1]`` (sandbox ``SPAN_Y_CLAMP``)."""
    return clamp(x, 0.0, float(SCREEN_HEIGHT - 1))


def CLIP_Y_CLAMP(x: Node) -> Node:
    """Clamp a clip-array y to ``[-1, SCREEN_HEIGHT]`` (sandbox ``CLIP_Y_CLAMP``)."""
    return clamp(x, -1.0, float(SCREEN_HEIGHT))


def FAR_DEN_CLAMP(x: Node) -> Node:
    """Clamp the far denominator to ``[0.7, 1500]`` (sandbox ``FAR_DEN_CLAMP``)."""
    return clamp(x, 0.7, 1500.0)


def NEAR_DEN_CLAMP(x: Node) -> Node:
    """Clamp the near denominator to ``[0.0007, 1]`` (sandbox ``NEAR_DEN_CLAMP``)."""
    return clamp(x, 0.0007, 1.0)


def RW_DISTANCE_CLAMP(x: Node) -> Node:
    """Clamp the perpendicular seg distance to ``[0.001, 1500]`` (sandbox
    ``RW_DISTANCE_CLAMP`` — covers e1m1 far walls)."""
    return clamp(x, 0.001, 1500.0)


def SCALE_CLAMP(x: Node) -> Node:
    """Clamp wall scale to ``[1/256, 64]`` (sandbox ``SCALE_CLAMP``)."""
    return clamp(x, _MIN_SCALE, _MAX_SCALE)


def SINEB_CLAMP(x: Node) -> Node:
    """Clamp the sine-basis to ``[1e-6, 1]`` (sandbox ``SINEB_CLAMP``)."""
    return clamp(x, 1.0e-6, 1.0)


def DIST_GT_ONE(distance: Node) -> Node:
    """±1: is the distance past the near plane (> 1)? (sandbox ``DIST_GT_ONE``)."""
    return compare(distance, 1.0)


def SINEB_ABOVE_FLOOR(sineb: Node) -> Node:
    """±1: is the sine-basis above the near-floor threshold 1/256? (sandbox
    ``SINEB_ABOVE_FLOOR`` — sharpness 32000 for a ~0.001 deadband)."""
    return compare(sineb, 1.0 / 256.0, sharpness=32000.0)


def gt_height(a: Node, b: Node) -> Node:
    """±1: is height ``a`` greater than ``b``? (sandbox ``gt_height`` ->
    ``HEIGHT_GT_HALF(sub(a, b))``)."""
    return compare(sub(a, b), 0.5)


def NEAR_DEN_SCALE_UP(near_den: Node) -> Node:
    """Scale the near denominator up by 1024 before reciprocal (sandbox
    ``NEAR_DEN_SCALE_UP``)."""
    return linear(near_den, [[NEAR_DEN_SCALE_FACTOR]])


def PROJECT_SCALE(sineb: Node) -> Node:
    """Multiply the sine-basis by the projection focal length (sandbox
    ``PROJECT_SCALE``)."""
    return linear(sineb, [[_PROJECTION]])


def NEAR_FLOOR_NUMERATOR(_: Node) -> Node:
    """The constant near-floor scale numerator (sandbox ``NEAR_FLOOR_NUMERATOR``)."""
    return constant(_PROJECTION * 1.0e-6 * NEAR_DEN_SCALE_FACTOR)


def MAX_SCALE_VALUE(_: Node) -> Node:
    """The max-scale sentinel (sandbox ``MAX_SCALE_VALUE``)."""
    return constant(_MAX_SCALE)


# ---------------------------------------------------------------------------
# The pixel pass: per-pixel texture-coordinate products + the native
# coordinate floor. Each mirrors a sandbox module-level ``multiply(...)`` /
# ``floor_int(...)`` definition (``uv_compute`` / ``pixel_dispatcher`` /
# ``flat_state``), lowered to ``_mul_grid`` / ``floor_int`` here so the J files
# stay node-free at import (the multiply / floor node is built only on call).
# ---------------------------------------------------------------------------


def FLOOR_NATIVE(x: Node) -> Node:
    """Integer floor over the native texture-coordinate range ``[-1023, 1023]``
    (sandbox ``_FLOOR_U_NATIVE`` / ``_FLOOR_V_NATIVE`` / ``_FLOOR_FLAT_FRAC``,
    all ``floor_int(-1023, 1023, sharpness=10_000.0)``). The 1e-4 ramp keeps the
    floored value an exact integer that lands on a sawtooth/mod grid line."""
    return floor_int(x, -1023, 1023, sharpness=10_000.0)


def FLAT_DIST_INDEX_FLOOR(x: Node) -> Node:
    """Integer floor of the flat distance-light index over ``[0, MAXLIGHTZ]``
    (sandbox ``_FLAT_DIST_INDEX_FLOOR`` = ``floor_int(0, MAXLIGHTZ,
    sharpness=10_000.0)``); clamps to ``[0, 128]`` so the distance-light table
    pick never overruns its 128 entries."""
    from .doom_lighting import MAXLIGHTZ

    return floor_int(x, 0, MAXLIGHTZ, sharpness=10_000.0)


def mul_u_native(rw_distance: Node, tan_rel: Node) -> Node:
    """rw_distance × per-column tangent → native wall u (sandbox
    ``_MUL_U_NATIVE``, ``((0, 800), (-10.5, 10.5))``, 1024 bp)."""
    return _mul_grid(
        rw_distance,
        tan_rel,
        lo1=0.0,
        hi1=800.0,
        lo2=-10.5,
        hi2=10.5,
        n=1024,
        name="mul_u_native",
    )


def mul_pixel_dc_iscale(pixel_index: Node, dc_iscale: Node) -> Node:
    """pixel/screen-y offset × dc_iscale → native-v offset (sandbox
    ``MUL_PIXEL_DC_ISCALE``, ``((-32, 64), (-64, 64))``, 97 bp). Exact on the
    pixel_index axis because that operand is always an integer (lands on a grid
    line, step 1.0), so the dc_iscale axis cell precision does not matter."""
    return _mul_grid(
        pixel_index,
        dc_iscale,
        lo1=-32.0,
        hi1=64.0,
        lo2=-64.0,
        hi2=64.0,
        n=97,
        name="mul_pixel_dc_iscale",
    )


def mul_k_step(pixel_index: Node, step: Node) -> Node:
    """flat pixel index × affine cursor step → texture-coordinate delta (sandbox
    ``_MUL_K_STEP``, ``((-2, 320), (-16, 16))``, 512 bp)."""
    return _mul_grid(
        pixel_index,
        step,
        lo1=-2.0,
        hi1=320.0,
        lo2=-16.0,
        hi2=16.0,
        n=512,
        name="mul_k_step",
    )


def mul_ph_yslope(planeheight: Node, yslope: Node) -> Node:
    """planeheight × per-scanline yslope → ray distance (sandbox
    ``_MUL_PH_YSLOPE``, ``((-2, 128), (0, 64))``, 512 bp)."""
    return _mul_grid(
        planeheight,
        yslope,
        lo1=-2.0,
        hi1=128.0,
        lo2=0.0,
        hi2=64.0,
        n=512,
        name="mul_ph_yslope",
    )


def mul_dist_distscale(distance: Node, distscale: Node) -> Node:
    """ray distance × column distscale → ray length (sandbox
    ``_MUL_DIST_DISTSCALE``, ``((0, 1024), (0.5, 5))``, 512 bp)."""
    return _mul_grid(
        distance,
        distscale,
        lo1=0.0,
        hi1=1024.0,
        lo2=0.5,
        hi2=5.0,
        n=512,
        name="mul_dist_distscale",
    )


def mul_len_trig(length: Node, trig: Node) -> Node:
    """ray length × view-ray cos/sin → world-space frac offset (sandbox
    ``_MUL_LEN_TRIG``, ``((-4096, 4096), (-1.5, 1.5))``, 512 bp)."""
    return _mul_grid(
        length,
        trig,
        lo1=-4096.0,
        hi1=4096.0,
        lo2=-1.5,
        hi2=1.5,
        n=512,
        name="mul_len_trig",
    )


def mul_dist_base(distance: Node, basescale: Node) -> Node:
    """ray distance × base x/y scale → per-screen-x texture step (sandbox
    ``_MUL_DIST_BASE``, ``((0, 1024), (-0.05, 0.05))``, 512 bp)."""
    return _mul_grid(
        distance,
        basescale,
        lo1=0.0,
        hi1=1024.0,
        lo2=-0.05,
        hi2=0.05,
        n=512,
        name="mul_dist_base",
    )


# ---------------------------------------------------------------------------
# Screen-column radix key (shared by solid_intervals / visplane_state /
# wall_column_state — was three private copies under two naming schemes)
# ---------------------------------------------------------------------------

# Columns split into a high *bucket* digit and a low digit so a column-
# equality key is two one-hots of total width N_COL_BUCKETS + COL_RADIX_BASE
# (16 at SCREEN_WIDTH=60, 26 at 160) instead of a width-SCREEN_WIDTH one-hot.
COL_RADIX_BASE = math.ceil(math.sqrt(SCREEN_WIDTH + 1))  # 8 at SW=60, 13 at 160
N_COL_BUCKETS = SCREEN_WIDTH // COL_RADIX_BASE + 1


def radix_col_key(col_scalar: Node) -> Node:
    """Exact screen-column-equality key: ``concat(one_hot(c // B),
    one_hot(c % B))`` over the column radix base.  The dot of two such keys
    is ``bucket_match + digit_match`` — 2 on an exact column match, <= 1
    otherwise (cancellation-free one-hot dot).

    The column may be the soft (~0.02 leak) output of a ``pick_most_recent``
    recovery: ``thermometer_floor_div`` (ramps at ``k*B - 0.5``) and the
    bucket-consistent ``mod_const`` keep the leaked value on the right digit,
    and ``one_hot`` rounds it to a clean integer one-hot — so no pre-snap is
    needed and a matched key's dot is exactly 2."""
    hi = thermometer_floor_div(col_scalar, COL_RADIX_BASE, SCREEN_WIDTH)
    lo = mod_const(col_scalar, COL_RADIX_BASE, SCREEN_WIDTH)
    return concat(one_hot(hi, N_COL_BUCKETS), one_hot(lo, COL_RADIX_BASE))
