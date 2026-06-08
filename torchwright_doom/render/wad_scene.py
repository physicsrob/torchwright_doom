"""WAD-backed prompt scene and sandbox adapter for render jobs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..asset_config import AssetConfig
from ..prompt.build import build_prompt
from ..prompt.subset import subset_by_bbox
from ..prompt.types import GameState, MapData
from ..prompt.wad import WADReader
from ..wad_assets import AssetBook, load_asset_book
from .config import RenderConfig, resolve_wad_path


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


def load_render_scene(config: RenderConfig, *, base_dir: str | Path | None = None) -> LoadedRenderScene:
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
    # The plan's default scene is the E1M1 start room. Keeping this local avoids
    # importing prompt.scenes in CLI setup paths that only need config parsing.
    return (1056.0, -3616.0, 64, 41.0)


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
    return GameState(x=float(x) - ox, y=float(y) - oy, angle=int(angle), viewz=float(viewz))


def prefill_rows_for(scene: LoadedRenderScene, pose: GameState) -> list[int]:
    from .tokens_bridge import row_index

    tokens = build_prompt(scene.map_data, pose, asset_config=scene.asset_config)
    return [row_index(t.type, dict(t.values)) for t in tokens]


def sandbox_scene_for(scene: LoadedRenderScene, pose: GameState):
    _ensure_doom_sandbox()
    from doom_sandbox.types import GameState as SandboxGameState
    from doom_sandbox.types import Scene as SandboxScene

    sb_scene = SandboxScene(
        map_data=scene.map_data.model_dump(),
        test_poses=[SandboxGameState(**pose.model_dump())],
        wall_textures=[_texture_dict(t) for t in scene.asset_book.wall_textures],
        flat_textures=[_texture_dict(t) for t in scene.asset_book.flat_textures],
        palette=[tuple(int(c) for c in rgb) for rgb in scene.asset_book.palette],
        colormap=[list(int(v) for v in row) for row in scene.asset_book.colormap],
    )
    patch_sandbox_assets(sb_scene, scene.asset_config)
    return sb_scene


def patch_sandbox_assets(sb_scene, asset_config: AssetConfig) -> None:
    """Patch doom_sandbox reference globals to the YAML asset ordering.

    The sandbox reference renderer historically reads module-level asset globals.
    The WAD adapter supplies an in-memory Scene, so the corresponding globals are
    updated before the drafter/reference render runs. This keeps host work at
    reference/draft generation only; the compiled renderer still owns rendering.
    """
    _ensure_doom_sandbox()
    import doom_sandbox.implementation.reference as ref

    book = _SandboxAssetBook(
        wall_textures=tuple(sb_scene.wall_textures),
        flat_textures=tuple(sb_scene.flat_textures),
        palette=tuple(tuple(rgb) for rgb in sb_scene.palette),
        colormap=tuple(tuple(row) for row in sb_scene.colormap),
    )
    ref.ASSET_BOOK = book
    ref.PLAYPAL = book.palette
    ref.COLORMAP_ROWS = tuple(tuple(row) for row in book.colormap[: len(ref.COLORMAP_ROWS)])
    ref.WALL_TEXTURE_ID_BY_NAME = asset_config.wall_id_by_name
    ref.FLAT_ID_BY_NAME = asset_config.flat_id_by_name


@dataclass(frozen=True)
class _SandboxAssetBook:
    wall_textures: tuple[Any, ...]
    flat_textures: tuple[Any, ...]
    palette: tuple[tuple[int, int, int], ...]
    colormap: tuple[tuple[int, ...], ...]


def _texture_dict(texture) -> dict[str, Any]:
    return {
        "name": texture.name,
        "width": int(texture.width),
        "height": int(texture.height),
        "pixels": texture.pixels,
    }


def _ensure_doom_sandbox() -> None:
    try:
        import doom_sandbox  # noqa: F401

        return
    except ImportError:
        pass
    umbrella = Path(__file__).resolve().parents[3]
    if (umbrella / "doom_sandbox").is_dir():
        sys.path.insert(0, str(umbrella))
    import doom_sandbox  # noqa: F401  - raise clearly if still missing
