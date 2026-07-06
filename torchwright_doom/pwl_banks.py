"""Module-level texture-coordinate wrap builders for the pixel pass.

The two floor-then-wrap forms the wall texel pass and the flat span pass
compute:

* ``build_v_mod_bank`` — one ``v_floor mod H`` PWL per distinct wall-texture
  height (aligned to ``WALL_HEIGHT_BANK``); each wall bank reads its own
  height's entry directly (banks are single-height, see ``asset_banks``).
  The closures consume the FLOORED coordinate (``uv_compute`` floors once,
  shared): folding the wrap into per-height ``floor_int(output_map=…)`` ops
  was measured and REVERTED — floor_int's bounded-step stage materializes
  ~2046 residual columns, and four of them live at once exceed the d=4096
  compile gate (the depth oracle doesn't model column capacity; the gate
  does).  One shared floor + K single-FFN mod PWLs is the width-safe form
  until torchwright grows a multi-``output_map`` floor_int that shares the
  step stage (stage 2 is hidden lanes, not residual columns).
* ``FLOOR_MOD64`` — the ``floor(frac) mod 64`` wrap for the 64×64 flat tiles,
  as ONE op via ``output_map`` (``g(floor(x))`` shares floor's integer
  breakpoints; see torchwright ``arithmetic_ops.floor_int``).  A 1:1 swap for
  the old floor→PWL pair — no duplication, so no width cost.  Consumes the
  RAW native coordinate — callers must NOT pre-floor.

These are texture-coordinate wraps for the pixel pass — NOT the numeric
digit-quadratic encoding PWL in ``emit.py`` (which splits an integer step
index into base-256 digits to address an embedding row).

The builders return *closures* (they build no graph node until called), so
module-level state is node-free at import — ``global_node_id`` stays ``0`` —
exactly like ``assets._U_MOD_BY_BANK`` / ``lighting._COLORMAP_ROW_PWLS``.
The original ``INV_COS_PWL`` is **not** ported: ``u_native`` rides the
already-landed ``u_tan_by_column`` derived column, not an in-graph
inverse-cosine.
"""

from __future__ import annotations

import math

from torchwright.graph import Node
from torchwright.ops.swiglu.arithmetic_ops import floor_int

from .asset_banks import WALL_HEIGHT_BANK
from .std import pwl_def

# The input contract carried over from the shared FLOOR_NATIVE: the raw
# native coordinate lives in [-1023, 1023] (integer grid; the mod-H wrap is
# exact at every integer for every H in the bank).
_V_NATIVE_LO = -1023
_V_NATIVE_HI = 1023
_FLOOR_SHARPNESS = 10_000.0

# The mod PWLs' breakpoint grid spans [-1024, 1024] with 2049 breakpoints —
# exactly 2048 unit-wide intervals, so the step is 1.0 and every integer
# floored value lands on a grid line, keeping the mod-H wrap exact for every
# H in the bank.
_V_SCALED_LO = -1024.0
_V_SCALED_HI = 1024.0
_V_SCALED_BREAKPOINTS = 2049


def _python_floor_mod(v: float, h: int) -> float:
    iv = math.floor(v)
    return float(iv - h * math.floor(iv / h))


def build_v_mod_bank(wall_height_bank: tuple[int, ...]) -> tuple:
    """Build per-height ``v_floor mod H`` PWLs for one wall-height bank."""
    return tuple(
        pwl_def(
            (lambda v, h=h: _python_floor_mod(v, h)),
            breakpoints=_V_SCALED_BREAKPOINTS,
            input_range=(_V_SCALED_LO, _V_SCALED_HI),
            name="sawtooth",
        )
        for h in wall_height_bank
    )


# Per-H ``v_floor mod H`` PWLs over the (shared) floored coordinate. Bank
# entries are aligned to ``WALL_HEIGHT_BANK`` (the sorted distinct
# wall-texture heights; (16, 56, 72, 128) for the committed wad); each wall
# bank consumes the entry matching its own height. Wall U instead uses the
# per-bank u_mod PWLs in ``assets.py`` because native wall widths vary and
# the published u is already floored.
V_MOD_BANK = build_v_mod_bank(WALL_HEIGHT_BANK)


def FLOOR_MOD64(frac_native: Node) -> Node:
    """Native flat x/y -> ``floor(frac) mod 64`` as ONE two-stage floor_int
    (the wrap folded into the saturating stage via ``output_map``)."""
    return floor_int(
        frac_native,
        _V_NATIVE_LO,
        _V_NATIVE_HI,
        sharpness=_FLOOR_SHARPNESS,
        output_map=lambda k: float(k % 64),
    )
