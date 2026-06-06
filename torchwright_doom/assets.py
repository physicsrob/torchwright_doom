"""Compiled texture and flat lookup helpers.

Real-side port of ``doom_sandbox/implementation/forward/assets.py``: the
``WallAssets`` / ``FlatAssets`` metadata accessors and full-resolution
``palette_index`` lookups, wired onto the zero-``past.publish`` ``AssetIndex``
shell that ``scene_index.py`` constructs.

``Vec`` -> ``Node`` and the sandbox ``...api`` imports become ``.std``; the
compiled banks come from :mod:`.asset_banks` (data-source **B**). Per the
no-import-time-nodes rule, module-level state is **raw data only** — numpy
``table`` reshapes, row-address lists, and ``pwl_def`` closures (which build no
node until called). The sandbox's module-level ``constant(...)`` selection
tables are instead wrapped *inside* the accessor methods, so ``global_node_id``
stays ``0`` after import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from torchwright.graph import Node

from .std import (
    concat,
    constant,
    indicator_to_bool,
    linear,
    one_hot,
    pick_by_index,
    pick_by_one_hot,
    pwl_def,
    table_lookup_2d,
)
from .asset_banks import (
    FLAT_IS_SKY,
    FLAT_TABLE,
    N_FLATS,
    N_WALL_TEXTURES,
    WALL_BANKS,
    WALL_HEIGHT_BANK,
    WALL_TEX_BANK_ID,
    WALL_TEX_H_IDX_OH,
    WALL_TEX_HEIGHT,
    WALL_TEX_LOCAL_ID,
    WALL_TEX_WIDTH,
)

_N_TEX_ID_SLOTS = N_WALL_TEXTURES + 1

# Width of the per-texture height-index one-hot ``h_idx_oh`` returns
# (= number of distinct wall-texture heights in the WAD: sorted{16, 56, 72,
# 128}). Consumed by ``wall_column_state`` for the span h_idx_oh payload width.
_H_IDX_OH_WIDTH = len(WALL_HEIGHT_BANK)

# Module-level RAW data only (no graph nodes — see module docstring). The
# sandbox wraps these in ``constant(...)`` at module scope; the real side keeps
# them as plain Python/numpy and wraps inside the accessors below.
_WALL_LOCAL_ID_VALUES_BY_BANK = tuple(
    [float(i) for i in range(len(bank.global_ids))] for bank in WALL_BANKS
)
_WALL_BANK_TABLE_2D = tuple(
    bank.table.reshape(len(bank.global_ids) * bank.height, bank.width)
    for bank in WALL_BANKS
)
_WALL_BANK_ROW_ADDR = tuple([[float(bank.height)], [1.0]] for bank in WALL_BANKS)
_FLAT_ID_VALUES = [float(i) for i in range(N_FLATS)]
_FLAT_TABLE_2D = FLAT_TABLE.reshape(N_FLATS * 64, 64)
_FLAT_ROW_ADDR = [[64.0], [1.0]]


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
    for bank in WALL_BANKS
)


def _snap_index(index: Node, n: int, values: list[float]) -> Node:
    """Round ``index`` to its exact integer value via ``one_hot`` -> pick.

    ``values`` is raw data (the sandbox passes a ``constant(...)`` node); the
    ``constant`` is built here, inside the call, not at module level.
    """
    return pick_by_one_hot(one_hot(index, n), constant(values))


@dataclass(frozen=True)
class WallAssets:
    """Wall texture metadata and full-resolution palette-index lookup."""

    def bank_id(self, tex_id: Node) -> Node:
        return pick_by_index(tex_id, constant(list(WALL_TEX_BANK_ID)), _N_TEX_ID_SLOTS)

    def local_id(self, tex_id: Node) -> Node:
        return pick_by_index(tex_id, constant(list(WALL_TEX_LOCAL_ID)), _N_TEX_ID_SLOTS)

    def width(self, tex_id: Node) -> Node:
        return pick_by_index(tex_id, constant(list(WALL_TEX_WIDTH)), _N_TEX_ID_SLOTS)

    def height(self, tex_id: Node) -> Node:
        return pick_by_index(tex_id, constant(list(WALL_TEX_HEIGHT)), _N_TEX_ID_SLOTS)

    def h_idx_oh(self, tex_id: Node) -> Node:
        return pick_by_index(
            tex_id,
            constant(list(WALL_TEX_H_IDX_OH)),
            _N_TEX_ID_SLOTS,
            d_fill=len(WALL_HEIGHT_BANK),
        )

    def palette_index(self, tex_id: Node, u_native: Node, v_mod_h: Node) -> Node:
        bank_id = self.bank_id(tex_id)
        local_id = self.local_id(tex_id)
        bank_mask = one_hot(bank_id, len(WALL_BANKS))
        candidates = concat(
            *(
                table_lookup_2d(
                    linear(
                        concat(
                            _snap_index(
                                local_id,
                                len(bank.global_ids),
                                _WALL_LOCAL_ID_VALUES_BY_BANK[bank.bank_id],
                            ),
                            v_mod_h,
                        ),
                        _WALL_BANK_ROW_ADDR[bank.bank_id],
                    ),
                    _U_MOD_BY_BANK[bank.bank_id](u_native),
                    _WALL_BANK_TABLE_2D[bank.bank_id],
                    sharpness=1000.0,
                )
                for bank in WALL_BANKS
            )
        )
        return pick_by_one_hot(bank_mask, candidates)


@dataclass(frozen=True)
class FlatAssets:
    """Flat metadata and full-resolution palette-index lookup."""

    def is_sky(self, flat_id: Node) -> Node:
        return indicator_to_bool(
            pick_by_index(flat_id, constant(list(FLAT_IS_SKY)), N_FLATS)
        )

    def palette_index(self, flat_id: Node, u: Node, v: Node) -> Node:
        row = linear(
            concat(
                _snap_index(flat_id, N_FLATS, _FLAT_ID_VALUES),
                v,
            ),
            _FLAT_ROW_ADDR,
        )
        return table_lookup_2d(row, u, _FLAT_TABLE_2D, sharpness=1000.0)


@dataclass(frozen=True)
class AssetIndex:
    """Weight-side asset lookups; constructed with zero ``past.publish``."""

    walls: WallAssets = field(default_factory=WallAssets)
    flats: FlatAssets = field(default_factory=FlatAssets)
