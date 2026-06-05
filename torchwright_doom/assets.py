"""Asset index — Plan D scope boundary (D5).

``SceneIndex.build`` constructs ``AssetIndex()`` as a field so the real build
matches the sandbox build shape, but the weight-side texture/flat palette
lookup internals (``table_lookup_2d`` / ``table_lookup_3d`` over the compiled
``WALL_BANKS`` / ``FLAT_TABLE``, with **zero** ``past.publish``) are the
**lookup track's** deliverable (``plan_lookup3d.md`` / ``torchwright_lookup``),
not ported by Plan D. The read-side oracle gate (``view``/``nodes``/
``subsectors``/``segs``/``planes``) does not exercise assets.

The placeholder ``WallAssets`` / ``FlatAssets`` carry no lookup methods yet;
touching one from the read-side raises ``AttributeError`` loudly rather than
silently returning a wrong value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torchwright.graph import Node


# Number of distinct wall-texture heights in the WAD asset book
# (``len(WALL_HEIGHT_BANK)`` = sorted{16, 56, 72, 128}); the width of the
# per-texture height-index one-hot ``h_idx_oh`` returns.
_H_IDX_OH_WIDTH = 4


@dataclass(frozen=True)
class WallAssets:
    """Wall texture metadata + palette lookup (lookup-track deliverable)."""

    def h_idx_oh(self, tex_id: Node) -> Node:
        """Per-texture height-index one-hot (sandbox ``WallAssets.h_idx_oh`` ->
        ``pick_by_index(tex_id, WALL_TEX_H_IDX_OH, ..., d_fill=len(WALL_HEIGHT_BANK))``).

        Stubbed to a width-``_H_IDX_OH_WIDTH`` zero vector for Phase H. The real
        per-texture height-index table is the lookup track's deliverable
        (``plan_lookup3d.md``); the real-side WAD loader does not parse texture
        lumps yet. Safe on the geometry fixtures: ``h_idx_oh`` only feeds the
        wall-span height / texel sawtooth consumed by the Phase-J pixel pass — it
        never lands on a compared Phase-H next-token (the H gate caps before the
        first PIXEL), so the value is published-for-coherence and discarded.
        """
        from .std import constant

        return constant([0.0] * _H_IDX_OH_WIDTH)

    def height(self, tex_id: Node) -> Node:
        """Native texture height for ``dc_texturemid`` pegging (sandbox
        ``WallAssets.height`` -> ``pick_by_index(tex_id, WALL_TEX_HEIGHT, ...)``).

        Stubbed to 0 for Phase F. The full per-texture height table is the
        lookup track's deliverable (``plan_lookup3d.md``) and the real-side WAD
        loader does not parse texture lumps yet. This is safe on the geometry
        fixtures: ``dc_tmid_compute`` only consumes ``height`` inside a
        ``select(dontpegtop/dontpegbottom, ...)`` arm, and every textured seg in
        ``e1m1_subset`` sets its pegging flag in the height-bypassing direction
        — so the value built from this stub is computed but discarded. The Phase
        F gate compares the SEG_DC_TMID next-tokens, which verifies this.
        """
        from .std import constant

        return constant(0.0)


@dataclass(frozen=True)
class FlatAssets:
    """Flat metadata + palette lookup (lookup-track deliverable)."""


@dataclass(frozen=True)
class AssetIndex:
    """Weight-side asset lookups; constructed with zero ``past.publish``."""

    walls: WallAssets = field(default_factory=WallAssets)
    flats: FlatAssets = field(default_factory=FlatAssets)
