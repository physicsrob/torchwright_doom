"""Reference renderer (DOOM-faithful, perspective-correct).

Top-level entry: :func:`R_RenderPlayerView` from
:mod:`torchwright_doom.reference_renderer.doom_render`.

The old ray-cast renderer (``render.py``, ``render_frame``) was
deleted: it used a linear-angle horizontal projection that produced
fisheye distortion on flat screens, inconsistent with its own
linear-X vertical projection and with DOOM's actual algorithm.
:func:`mapdata_from_segments` is the migration adapter for legacy
``List[Segment]`` scenes.
"""

from torchwright_doom.reference_renderer.doom_render import (
    R_RenderPlayerView,
    mapdata_from_segments,
    save_png,
    single_sector_map,
)
from torchwright_doom.reference_renderer.geometry import intersect_ray_segment
from torchwright_doom.reference_renderer.scenes import box_room, box_room_textured
from torchwright_doom.reference_renderer.trig import generate_trig_table
from torchwright_doom.reference_renderer.types import RenderConfig, Segment

__all__ = [
    "R_RenderPlayerView",
    "RenderConfig",
    "Segment",
    "box_room",
    "box_room_textured",
    "generate_trig_table",
    "intersect_ray_segment",
    "mapdata_from_segments",
    "save_png",
    "single_sector_map",
]
