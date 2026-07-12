"""In-tree pydoom adapter: build the reference oracle's scene for scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ..model.asset_config import AssetConfig
from ..prompt.scene import LoadedRenderScene
from ..prompt.types import GameState


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
