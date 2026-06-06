"""Per-pixel v coordinate decode for texture sampling.

Real-side port of ``doom_sandbox/implementation/forward/uv_compute.py``. The
chain mirrors DOOM ``R_DrawColumn`` texture stepping (r_draw.c:132-145)::

    v_native = dc_texturemid + (y - centery) * dc_iscale
    v_floor  = floor(v_native)
    v_mod_h  = v_floor % texture_height

``SAWTOOTH_BANK`` is one ``mod H`` PWL per ``WALL_HEIGHT_BANK`` height;
``h_idx_oh`` selects which mod to return via ``pick_by_one_hot``.

Changes from the sandbox: ``Vec`` -> ``Node``; the module-level ``multiply`` /
``floor_int`` / ``constant(CENTER_Y)`` nodes become the ``render_ops`` shims
(``mul_pixel_dc_iscale`` / ``FLOOR_NATIVE`` / ``add_const``) applied inside the
functions (no import-time graph nodes). The dead ``compute_v_at_screen_y`` (the
combined native+floor+sawtooth form, imported nowhere) is **not** ported.

``mul_pixel_dc_iscale`` is exact on its first axis because ``pixel_index`` /
``screen_y - CENTER_Y`` are always integers (CENTER_Y = 25.0), so they land on
the grid's unit step and the dc_iscale axis cell precision does not matter.
"""

from __future__ import annotations

from torchwright.graph import Node

from .constants import CENTER_Y
from .pwl_banks import SAWTOOTH_BANK
from .render_ops import FLOOR_NATIVE, add_const, mul_pixel_dc_iscale
from .std import concat, pick_by_one_hot
from .std import sum as vec_sum


def compute_v_at_pixel(
    *,
    pixel_index_vec: Node,
    dc_iscale: Node,
    v_0_at_top: Node,
    h_idx_oh: Node,
) -> Node:
    """Return ``v_scaled_mod_H`` for one pixel."""
    v_offset = mul_pixel_dc_iscale(pixel_index_vec, dc_iscale)
    v_native = vec_sum(v_offset, v_0_at_top)
    v_scaled = FLOOR_NATIVE(v_native)
    bank = concat(*[pwl(v_scaled) for pwl in SAWTOOTH_BANK])
    return pick_by_one_hot(h_idx_oh, bank)


def compute_v_native_at_screen_y(
    *,
    screen_y: Node,
    dc_iscale: Node,
    dc_texturemid: Node,
) -> Node:
    """Return native texture v at an absolute screen y before floor + mod H."""
    y_offset = add_const(screen_y, -float(CENTER_Y))
    v_offset = mul_pixel_dc_iscale(y_offset, dc_iscale)
    return vec_sum(dc_texturemid, v_offset)
