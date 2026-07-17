"""Scene declarations — WAD path, subset box, initial player pose.

Fixture-style scene declarations used only by the oracle/prompt tests;
the production input path is ``scene.py`` (:func:`scene.load_render_scene`).

A :class:`Scene` ties together everything needed to run the renderer
on a specific slice of a DOOM map: which WAD, which map marker, which
world-space region to subset, and where the player starts. The
:func:`load` helper opens the WAD, subsets the map, and shifts the
initial pose into subset frame so it pairs with the renumbered
:class:`MapData` returned alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .subset import subset_by_bbox
from .types import GameState, MapData
from .wad import WADReader

# Submodule-root doom1.wad. ``WAD_PATH`` is resolved at import time so
# scene declarations can reference a single canonical location.
WAD_PATH: Path = Path(__file__).resolve().parents[2] / "doom1.wad"


@dataclass(frozen=True)
class Scene:
    name: str
    wad_path: Path
    map_name: str
    bbox: tuple[float, float, float, float]
    """``(left, bottom, right, top)`` world-space subset box."""

    initial_pose_world: tuple[float, float, int]
    """``(x, y, angle_256)`` in world (raw WAD) coordinates."""


E1M1_START_ROOM = Scene(
    name="e1m1_start_room",
    wad_path=WAD_PATH,
    map_name="E1M1",
    bbox=(627.2, -3760.0, 1395.2, -2800.0),
    initial_pose_world=(1056.0, -3616.0, 64),
)


def load(scene: Scene) -> tuple[MapData, GameState]:
    wad = WADReader(scene.wad_path)
    raw = wad.get_map(scene.map_name)
    md = subset_by_bbox(raw, *scene.bbox)
    ox, oy = md.scene_origin
    px, py, ang = scene.initial_pose_world
    state = GameState(x=px - ox, y=py - oy, angle=ang)
    return md, state
