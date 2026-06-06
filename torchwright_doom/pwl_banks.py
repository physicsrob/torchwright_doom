"""Module-level PWL banks for the pixel pass.

Real-side port of ``doom_sandbox/implementation/pwl_banks.py`` — the two PWL
banks the wall texel pass and the flat span pass read:

* ``SAWTOOTH_BANK`` — one ``v mod H`` sawtooth per distinct wall-texture height
  (aligned to ``WALL_HEIGHT_BANK``), selected per span by ``h_idx_oh``.
* ``MOD64_PWL`` — the ``frac mod 64`` wrap for the 64×64 flat tiles.

``pwl_def`` returns a *closure* (it builds no graph node until the closure is
called), so these module-level tuples are node-free at import — ``global_node_id``
stays ``0`` — exactly like ``assets._U_MOD_BY_BANK`` /
``lighting._COLORMAP_ROW_PWLS``. The sandbox's ``INV_COS_PWL`` is **not** ported:
``u_native`` rides the already-landed ``u_tan_by_column`` derived column, not
an in-graph inverse-cosine.
"""

from __future__ import annotations

import math

from .asset_banks import WALL_HEIGHT_BANK
from .std import pwl_def

# v_scaled / frac is the floored native coordinate. The breakpoint grid spans
# [-1024, 1024] with 2049 breakpoints — exactly 2048 unit-wide intervals, so the
# step is 1.0 and every integer floored value lands on a grid line, keeping the
# mod-H wrap exact for every H in the bank.
_V_SCALED_LO = -1024.0
_V_SCALED_HI = 1024.0
_V_SCALED_BREAKPOINTS = 2049


def _python_floor_mod(v: float, h: int) -> float:
    iv = math.floor(v)
    return float(iv - h * math.floor(iv / h))


# Per-H sawtooth PWLs for ``v_scaled mod H``. Bank entries are aligned to
# ``WALL_HEIGHT_BANK`` (= the sandbox ``H_BANK``, the sorted distinct wall-texture
# heights; (16, 56, 72, 128) for the committed wad), so a ``pick_by_one_hot`` over
# the per-texture ``h_idx_oh`` selects the matching height's mod.
SAWTOOTH_BANK = tuple(
    pwl_def(
        (lambda v, h=h: _python_floor_mod(v, h)),
        breakpoints=_V_SCALED_BREAKPOINTS,
        input_range=(_V_SCALED_LO, _V_SCALED_HI),
        name="sawtooth",
    )
    for h in WALL_HEIGHT_BANK
)


# Native flat x/y -> floor -> mod 64. Wall U instead uses the per-bank sawtooth
# PWLs in ``assets.py`` because native wall widths vary.
MOD64_PWL = pwl_def(
    lambda v: _python_floor_mod(v, 64),
    breakpoints=_V_SCALED_BREAKPOINTS,
    input_range=(_V_SCALED_LO, _V_SCALED_HI),
    name="mod64",
)
