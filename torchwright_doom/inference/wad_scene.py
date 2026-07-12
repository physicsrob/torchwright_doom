"""WAD-backed prompt scene and the in-tree pydoom adapter for render jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..asset_config import AssetConfig
from ..prompt.build import build_prompt
from ..prompt.subset import subset_by_bbox
from ..prompt.types import GameState, MapData
from ..prompt.wad import WADReader
from ..config import RenderConfig, resolve_wad_path
from ..wad_assets import AssetBook, load_asset_book


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


def pydoom_scene_for(scene: LoadedRenderScene, pose: GameState):
    """Build the in-tree :class:`torchwright_doom.pydoom.Scene` the renderer and
    drafter consume, from the WAD-loaded render scene and pose."""
    from ..pydoom import Scene as PyScene

    # model_validate over a plain dict: pydantic coerces map_data / textures /
    # palette into the typed pydoom models (the native<->pydoom-shape round-trip
    # the adapter has always relied on).
    py_scene = PyScene.model_validate(
        {
            "map_data": scene.map_data.model_dump(),
            "test_poses": [pose.model_dump()],
            "wall_textures": [_texture_dict(t) for t in scene.asset_book.wall_textures],
            "flat_textures": [_texture_dict(t) for t in scene.asset_book.flat_textures],
            "palette": [tuple(int(c) for c in rgb) for rgb in scene.asset_book.palette],
            "colormap": [
                list(int(v) for v in row) for row in scene.asset_book.colormap
            ],
        }
    )
    patch_pydoom_assets(py_scene, scene.asset_config)
    return py_scene


def patch_pydoom_assets(py_scene, asset_config: AssetConfig) -> None:
    """Patch the pydoom renderer's module-level asset globals to the YAML asset
    ordering.

    The vendored renderer reads module-level asset globals (``ASSET_BOOK``,
    ``PLAYPAL``, ...) and the drafter reads them through that same module. The WAD
    adapter supplies an in-memory scene, so those globals are updated before the
    drafter / reference render runs. This keeps host work at reference/draft
    generation only; the compiled renderer still owns rendering.
    """
    from ..pydoom import renderer

    # Deliberate monkey-patch of the renderer's module-level asset globals; cast
    # to Any so the (intentionally) type-incompatible rebind is explicit, not a
    # silent error.
    ref = cast(Any, renderer)

    book = _PydoomAssetBook(
        wall_textures=tuple(py_scene.wall_textures),
        flat_textures=tuple(py_scene.flat_textures),
        palette=tuple(tuple(rgb) for rgb in py_scene.palette),
        colormap=tuple(tuple(row) for row in py_scene.colormap),
    )
    ref.ASSET_BOOK = book
    ref.PLAYPAL = book.palette
    ref.COLORMAP_ROWS = tuple(
        tuple(row) for row in book.colormap[: len(ref.COLORMAP_ROWS)]
    )
    ref.WALL_TEXTURE_ID_BY_NAME = asset_config.wall_id_by_name
    ref.FLAT_ID_BY_NAME = asset_config.flat_id_by_name


@dataclass(frozen=True)
class _PydoomAssetBook:
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
