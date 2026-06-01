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


@dataclass(frozen=True)
class WallAssets:
    """Wall texture metadata + palette lookup (lookup-track deliverable)."""


@dataclass(frozen=True)
class FlatAssets:
    """Flat metadata + palette lookup (lookup-track deliverable)."""


@dataclass(frozen=True)
class AssetIndex:
    """Weight-side asset lookups; constructed with zero ``past.publish``."""

    walls: WallAssets = field(default_factory=WallAssets)
    flats: FlatAssets = field(default_factory=FlatAssets)
