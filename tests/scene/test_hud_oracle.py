"""Status-bar patch bank — graph oracle (reference_eval value-match vs the bake).

Teacher-forced direct probe (the texture-oracle pattern): drive
``HudAssets.color_or_transparent`` with literal ``(patch_id, u, v)`` and assert
the channel equals the baked patch pixel — a palette index for an opaque cell,
``HUD_TRANSPARENT`` for a transparent or padding cell. This pins the graph-side
lookup against the same bake the reference ``V_DrawPatch`` blit composites.
"""

from __future__ import annotations

import dataclasses

from torchwright.debug.probe import reference_eval

from torchwright_doom import asset_banks as ab
from torchwright_doom.assets import HudAssets
from torchwright_doom.hud_assets import (
    HUD_TRANSPARENT,
    bake_hud_bank,
    load_hud_patches,
)
from torchwright_doom.std import constant


def _hud_on_banks(scale: int = 1):
    """A HUD-on AssetBanks (the default build is HUD-off / placeholder)."""
    bake = bake_hud_bank(scale)
    banks = dataclasses.replace(
        ab.DEFAULT_ASSET_BANKS,
        hud_table_2d=bake.table,
        hud_base_rows=[float(r) for r in bake.base_rows],
        hud_patch_widths=[float(w) for w in bake.widths],
        hud_patch_heights=[float(h) for h in bake.heights],
        n_hud_patches=len(bake.base_rows),
    )
    return banks, bake


def _scalar(node) -> float:
    return float(reference_eval(node, {}, 1)[node].reshape(-1)[0])


def test_hud_accessor_matches_bake_full_res():
    banks, bake = _hud_on_banks(1)
    hud = HudAssets(banks)
    patches = load_hud_patches()

    # A spread of patches: the wide plate, tall digits (incl. the -1-offset "1"),
    # the small arms number, the face. For each, sample corners + centre +
    # known-transparent cells.
    for name in ["STBAR", "STTNUM0", "STTNUM1", "STYSNUM2", "STGNUM3", "STFST01"]:
        pid = bake.patch_ids[name]
        patch = patches[name]
        samples = {
            (0, 0),
            (patch.width - 1, 0),
            (0, patch.height - 1),
            (patch.width - 1, patch.height - 1),
            (patch.width // 2, patch.height // 2),
        }
        for u, v in samples:
            got = _scalar(
                hud.color_or_transparent(
                    constant(float(pid)), constant(float(u)), constant(float(v))
                )
            )
            cell = patch.pixels[u][v]
            expected = HUD_TRANSPARENT if cell is None else float(cell)
            # Palette indices and the 256 sentinel are >=1 apart, so a 0.5 band
            # confirms the exact cell was selected.
            assert abs(got - expected) < 0.5, f"{name} ({u},{v}): {got} != {expected}"


def test_hud_accessor_padding_reads_transparent():
    # A glyph narrower than the 320-wide table: columns past the glyph width are
    # padding and must read the sentinel (so the spine never paints padding).
    banks, bake = _hud_on_banks(1)
    hud = HudAssets(banks)
    pid = bake.patch_ids["STTNUM0"]  # 14 wide in a 320-wide table
    got = _scalar(
        hud.color_or_transparent(constant(float(pid)), constant(100.0), constant(0.0))
    )
    assert abs(got - HUD_TRANSPARENT) < 0.5


def test_hud_accessor_matches_bake_half_res():
    banks, bake = _hud_on_banks(2)
    hud = HudAssets(banks)
    # Decimated patches; just re-derive expected from the bake table directly.
    for name in ["STBAR", "STTNUM5", "STFST01"]:
        pid = bake.patch_ids[name]
        base = bake.base_rows[pid]
        for u, v in [(0, 0), (bake.widths[pid] - 1, bake.heights[pid] - 1)]:
            got = _scalar(
                hud.color_or_transparent(
                    constant(float(pid)), constant(float(u)), constant(float(v))
                )
            )
            expected = float(bake.table[base + v, u])
            assert abs(got - expected) < 0.5, f"{name} ({u},{v})"
