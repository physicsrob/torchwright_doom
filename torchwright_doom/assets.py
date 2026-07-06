"""Compiled texture and flat lookup helpers.

The
``WallAssets`` / ``FlatAssets`` metadata accessors and full-resolution
``palette_index`` lookups, wired onto the zero-``past.publish`` ``AssetIndex``
shell that ``scene_index.py`` constructs.

``Vec`` -> ``Node`` and the original ``...api`` imports become ``.std``; the
compiled banks come from :mod:`.asset_banks` (data-source **B**). Per the
no-import-time-nodes rule, module-level state is **raw data only** — numpy
``table`` reshapes, row-address lists, and ``pwl_def`` closures (which build no
node until called). The original module-level ``constant(...)`` selection
tables are instead wrapped *inside* the accessor methods, so ``global_node_id``
stays ``0`` after import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from torchwright.graph import Node
from torchwright.graph import annotated

from .std import (
    concat,
    indicator_to_bool,
    linear,
    one_hot,
    pick_by_one_hot,
    pick_const_by_index,
    pwl_def,
    table_lookup_2d,
)
from .asset_banks import DEFAULT_ASSET_BANKS, AssetBanks
from .pwl_banks import build_v_mod_bank
from .render_ops import add_const

# Width of the per-texture height-index one-hot ``h_idx_oh`` returns
# (= number of distinct wall-texture heights in the WAD: sorted{16, 56, 72,
# 128}). Consumed by ``wall_column_state`` for the span h_idx_oh payload width.
_H_IDX_OH_WIDTH = len(DEFAULT_ASSET_BANKS.wall_height_bank)


def _python_floor_mod(v: float, h: int) -> float:
    iv = math.floor(v)
    return float(iv - h * math.floor(iv / h))


# ``pwl_def`` returns a closure, so this tuple builds no graph node at import.
_U_MOD_BY_BANK = tuple(
    pwl_def(
        (lambda v, width=bank.width: _python_floor_mod(v, width)),
        breakpoints=2049,
        input_range=(-1024.0, 1024.0),
    )
    for bank in DEFAULT_ASSET_BANKS.wall_banks
)


def _snap_index(index: Node, n: int, values: list[float]) -> Node:
    """Round ``index`` to its exact integer value via ``one_hot`` -> lookup.

    ``values`` is raw data baked into the ``pick_const_by_index`` selection
    Linear — no ``constant`` node, no gated table copy on the residual stream.
    """
    return pick_const_by_index(index, values, n)


@dataclass(frozen=True)
class WallAssets:
    """Wall texture metadata and full-resolution palette-index lookup."""

    banks: AssetBanks = DEFAULT_ASSET_BANKS
    u_mod_by_bank: tuple = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "u_mod_by_bank",
            tuple(
                pwl_def(
                    (lambda v, width=bank.width: _python_floor_mod(v, width)),
                    breakpoints=2049,
                    input_range=(-1024.0, 1024.0),
                )
                for bank in self.banks.wall_banks
            ),
        )

    @annotated("tex")
    def bank_id(self, tex_id: Node) -> Node:
        return pick_const_by_index(
            tex_id,
            list(self.banks.wall_tex_bank_id),
            self.banks.n_wall_textures + 1,
        )

    @annotated("tex")
    def local_id(self, tex_id: Node) -> Node:
        return pick_const_by_index(
            tex_id,
            list(self.banks.wall_tex_local_id),
            self.banks.n_wall_textures + 1,
        )

    @annotated("tex")
    def width(self, tex_id: Node) -> Node:
        return pick_const_by_index(
            tex_id,
            list(self.banks.wall_tex_width),
            self.banks.n_wall_textures + 1,
        )

    @annotated("tex")
    def height(self, tex_id: Node) -> Node:
        return pick_const_by_index(
            tex_id,
            list(self.banks.wall_tex_height),
            self.banks.n_wall_textures + 1,
        )

    @annotated("tex")
    def h_idx_oh(self, tex_id: Node) -> Node:
        return pick_const_by_index(
            tex_id,
            list(self.banks.wall_tex_h_idx_oh),
            self.banks.n_wall_textures + 1,
            d_fill=len(self.banks.wall_height_bank),
        )

    def _v_mod_for_bank(self, bank, v_mods: tuple) -> Node:
        """The bank's own-height ``floor(v) mod H`` entry.

        Banks are single-height, so no ``h_idx_oh`` pick sits on the
        row-address path.  A bank whose height is not in
        ``wall_height_bank`` (the 1x1 missing-texture bank) gets entry 0:
        its table has a single row, so any bounded input addresses it.
        """
        heights = self.banks.wall_height_bank
        idx = heights.index(bank.height) if bank.height in heights else 0
        return v_mods[idx]

    @annotated("tex")
    def palette_index(self, tex_id: Node, u_native: Node, v_mods: tuple) -> Node:
        bank_id = self.bank_id(tex_id)
        local_id = self.local_id(tex_id)
        bank_mask = one_hot(bank_id, len(self.banks.wall_banks))
        candidates = concat(
            *(
                table_lookup_2d(
                    linear(
                        concat(
                            _snap_index(
                                local_id,
                                len(bank.global_ids),
                                self.banks.wall_local_id_values_by_bank[bank.bank_id],
                            ),
                            self._v_mod_for_bank(bank, v_mods),
                        ),
                        self.banks.wall_bank_row_addr[bank.bank_id],
                    ),
                    self.u_mod_by_bank[bank.bank_id](u_native),
                    self.banks.wall_bank_table_2d[bank.bank_id],
                    sharpness=1000.0,
                )
                for bank in self.banks.wall_banks
            )
        )
        return pick_by_one_hot(bank_mask, candidates)


@dataclass(frozen=True)
class FlatAssets:
    """Flat metadata and full-resolution palette-index lookup."""

    banks: AssetBanks = DEFAULT_ASSET_BANKS

    @annotated("tex")
    def is_sky(self, flat_id: Node) -> Node:
        return indicator_to_bool(
            pick_const_by_index(
                flat_id, list(self.banks.flat_is_sky), self.banks.n_flats
            )
        )

    @annotated("tex")
    def palette_index(self, flat_id: Node, u: Node, v: Node) -> Node:
        row = linear(
            concat(
                _snap_index(flat_id, self.banks.n_flats, self.banks.flat_id_values),
                v,
            ),
            self.banks.flat_row_addr,
        )
        return table_lookup_2d(row, u, self.banks.flat_table_2d, sharpness=1000.0)


@dataclass(frozen=True)
class WeaponAssets:
    """Player-weapon (R_DrawPlayerSprites) baked-picture lookup.

    One masked bitmap indexed by the screen cursor, encoded exactly like a flat
    texture: a ``table_lookup_2d`` over the baked ``(bbox_h, bbox_w)`` weight. The
    value is a *lit* palette index, or ``WEAPON_TRANSPARENT`` (256) for a
    transparent cell — the render loop reads it at the cursor and emits a pixel
    (opaque) or a ``setCursorY`` skip (transparent).
    """

    banks: AssetBanks = DEFAULT_ASSET_BANKS

    @annotated("tex/R_DrawPlayerSprites")
    def color_or_transparent(self, col: Node, row: Node) -> Node:
        # col = rendered column, row = screen row; offset into the bbox table.
        local_col = add_const(col, -float(self.banks.weapon_min_col))
        local_row = add_const(row, -float(self.banks.weapon_top))
        return table_lookup_2d(
            local_row, local_col, self.banks.weapon_table_2d, sharpness=1000.0
        )


@dataclass(frozen=True)
class HudAssets:
    """Status-bar (ST_Drawer) patch-bank lookup.

    Banked like the flats (one stacked table, a per-patch base row) but returning
    color-or-transparent like the weapon: ``table[base_rows[patch_id] + v, u]`` is
    a raw (unlit) palette index, or ``HUD_TRANSPARENT`` (256) for a transparent or
    padding cell. The draw-list spine supplies ``patch_id`` and the local ``(u, v)``
    (cursor minus the draw-item origin); the loop emits a pixel (opaque) or a
    ``setCursorY`` skip (transparent), exactly as the weapon does.
    """

    banks: AssetBanks = DEFAULT_ASSET_BANKS

    @annotated("tex/ST_Drawer")
    def color_or_transparent(self, patch_id: Node, u: Node, v: Node) -> Node:
        # row = base_rows[patch_id] + v; column = u (local to the patch).
        base = pick_const_by_index(
            patch_id, self.banks.hud_base_rows, self.banks.n_hud_patches
        )
        row = linear(concat(base, v), [[1.0], [1.0]])
        return table_lookup_2d(row, u, self.banks.hud_table_2d, sharpness=1000.0)


@dataclass(frozen=True)
class AssetIndex:
    """Weight-side asset lookups; constructed with zero ``past.publish``."""

    banks: AssetBanks = DEFAULT_ASSET_BANKS
    walls: WallAssets = field(init=False)
    flats: FlatAssets = field(init=False)
    weapon: WeaponAssets = field(init=False)
    hud: HudAssets = field(init=False)
    v_mod_bank: tuple = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "walls", WallAssets(self.banks))
        object.__setattr__(self, "flats", FlatAssets(self.banks))
        object.__setattr__(self, "weapon", WeaponAssets(self.banks))
        object.__setattr__(self, "hud", HudAssets(self.banks))
        object.__setattr__(
            self, "v_mod_bank", build_v_mod_bank(self.banks.wall_height_bank)
        )
