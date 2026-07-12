"""WAD-backed render scene: the input side's production entry point.

``load_render_scene`` opens the job's WAD, subsets the map to the config
region, and loads the asset book; ``pose_from_world`` shifts a world-space
pose into the subset frame; ``prefill_rows_for`` emits the prompt row ids the
transformer reads before autoregression. This completes the prefill chain
``wad.py`` -> ``subset.py`` -> ``build.py`` -> ``scene.py`` -> rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..model.asset_config import AssetConfig
from ..config import RenderConfig, resolve_wad_path
from ..model.assets.wad_assets import AssetBook, load_asset_book
from .build import build_prompt
from .subset import subset_by_bbox
from .types import GameState, MapData
from .wad import WADReader


@dataclass(frozen=True)
class LoadedRenderScene:
    config: RenderConfig
    wad_path: Path
    map_data: MapData
    asset_config: AssetConfig
    asset_book: AssetBook

    @property
    def origin(self) -> tuple[float, float]:
        return self.map_data.scene_origin


def load_render_scene(
    config: RenderConfig, *, base_dir: str | Path | None = None
) -> LoadedRenderScene:
    wad_path = resolve_wad_path(config, base_dir=base_dir)
    raw = WADReader(wad_path).get_map(config.map)
    md = subset_by_bbox(raw, *config.region.bbox)
    asset_config = config.asset_config()
    asset_book = load_asset_book(
        wad_path,
        wall_texture_names=asset_config.wall_names,
        flat_names=asset_config.flat_names,
    )
    return LoadedRenderScene(
        config=config,
        wad_path=wad_path,
        map_data=md,
        asset_config=asset_config,
        asset_book=asset_book,
    )


def default_pose_world(config: RenderConfig) -> tuple[float, float, int, float]:
    """The config's default pose (``run.pose``), in world coordinates."""
    pose = config.run.pose
    return (pose.x, pose.y, pose.angle, pose.viewz)


def pose_from_world(
    scene: LoadedRenderScene,
    *,
    x: float | None = None,
    y: float | None = None,
    angle: int | None = None,
    viewz: float | None = None,
) -> GameState:
    dx, dy, default_angle, default_viewz = default_pose_world(scene.config)
    x = dx if x is None else x
    y = dy if y is None else y
    angle = default_angle if angle is None else angle
    viewz = default_viewz if viewz is None else viewz
    ox, oy = scene.origin
    return GameState(
        x=float(x) - ox, y=float(y) - oy, angle=int(angle), viewz=float(viewz)
    )


def prefill_rows_for(scene: LoadedRenderScene, pose: GameState) -> list[int]:
    from ..tokenizer.rows import row_index

    tokens = build_prompt(scene.map_data, pose, asset_config=scene.asset_config)
    return [row_index(t.type, dict(t.values)) for t in tokens]
