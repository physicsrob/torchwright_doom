"""Plan D / D2 + D3 + D5: SceneIndex.build end-to-end on a hand-built prefill.

Exercises the full scene read-side stack — SceneTokenView interpretation,
HeaderContext recency recovery, the lifted/one-hot keyed lookups, and the
marker/value association — by recovering known facts from a compact prompt.
Channels are evaluated through the graph oracle (``node.compute``) at the final
(BEGIN) position, where the whole prefill has been seen.
"""

from __future__ import annotations

import math

import pytest

from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom import std
from torchwright_doom.embedding import TOKEN_VOCAB
from torchwright_doom.past import GraphPast
from torchwright_doom.scene_index import SceneIndex
from torchwright_doom.value_ranges import ValueRange
from torchwright_doom.vocab import (
    ANGLE_VALUE,
    BEGIN,
    NODE,
    NODE_PX,
    NODE_PY,
    PLAYER_ANGLE_MARK,
    PLAYER_X_MARK,
    PLAYER_Y_MARK,
    SEG,
    SEG_AX,
    SS,
)

from ..prefill_fixture import tokens_to_input, value

_D_EMBED = TOKEN_VOCAB.layout.d_embed
_ANGLE = 512


def _scene_and_inputs():
    seq = [
        (PLAYER_X_MARK, {}),
        value(ValueRange.R1, 100.0),
        (PLAYER_Y_MARK, {}),
        value(ValueRange.R1, -30.0),
        (PLAYER_ANGLE_MARK, {}),
        (ANGLE_VALUE, {"angle": _ANGLE}),
        (NODE, {"j": 0}),
        (NODE_PX, {}),
        value(ValueRange.R1, 50.0),
        (NODE_PY, {}),
        value(ValueRange.R1, -20.0),
        (NODE, {"j": 1}),
        (NODE_PX, {}),
        value(ValueRange.R1, 200.0),
        (SS, {"s": 0}),
        (SEG, {"i": 0, "is_first_of_ss": 1}),
        (SEG_AX, {}),
        value(ValueRange.R1, 10.0),
        (BEGIN, {}),
    ]
    n = len(seq)
    inputs = {"iv": tokens_to_input(seq)}
    past = GraphPast(
        input_vec=create_input("iv", _D_EMBED), pos_encoding=create_pos_encoding()
    )
    scene = SceneIndex.build(create_input("iv", _D_EMBED), past, create_pos_encoding())
    return scene, n, inputs


def _last(node, n, inputs) -> float:
    return node.compute(n, inputs)[-1].item()


def test_player_view_pose() -> None:
    scene, n, inputs = _scene_and_inputs()
    # R1 round-trip leaves < 0.05 of quantization error.
    assert _last(scene.view.x, n, inputs) == pytest.approx(100.0, abs=0.1)
    assert _last(scene.view.y, n, inputs) == pytest.approx(-30.0, abs=0.1)
    assert _last(scene.view.angle, n, inputs) == pytest.approx(float(_ANGLE), abs=1e-2)
    assert _last(scene.view.angle_sin, n, inputs) == pytest.approx(
        math.sin(_ANGLE * 2 * math.pi / 8192), abs=1e-4
    )
    assert _last(scene.view.angle_cos, n, inputs) == pytest.approx(
        math.cos(_ANGLE * 2 * math.pi / 8192), abs=1e-4
    )


def test_player_view_ray_widths() -> None:
    scene, _n, _inputs = _scene_and_inputs()
    from torchwright_doom.constants import SCREEN_WIDTH

    assert len(scene.view.ray_x_by_screen) == SCREEN_WIDTH
    assert len(scene.view.ray_y_by_screen) == SCREEN_WIDTH


def test_node_index_root_existence_coords() -> None:
    scene, n, inputs = _scene_and_inputs()
    # root = last NODE header = j=1.
    assert _last(scene.nodes.root, n, inputs) == pytest.approx(1.0, abs=1e-2)
    assert _last(scene.nodes.has_any, n, inputs) == pytest.approx(1.0, abs=1e-2)
    assert _last(scene.nodes.exists(std.constant(0.0)), n, inputs) == pytest.approx(
        1.0, abs=1e-2
    )
    assert _last(scene.nodes.exists(std.constant(1.0)), n, inputs) == pytest.approx(
        1.0, abs=1e-2
    )
    # absent node id -> no-match path -> below presence threshold.
    assert _last(scene.nodes.exists(std.constant(5.0)), n, inputs) == pytest.approx(
        -1.0, abs=1e-2
    )
    assert _last(scene.nodes.px(std.constant(0.0)), n, inputs) == pytest.approx(
        50.0, abs=0.1
    )
    assert _last(scene.nodes.px(std.constant(1.0)), n, inputs) == pytest.approx(
        200.0, abs=0.1
    )
    assert _last(scene.nodes.py(std.constant(0.0)), n, inputs) == pytest.approx(
        -20.0, abs=0.1
    )


def test_seg_and_subsector_index() -> None:
    scene, n, inputs = _scene_and_inputs()
    assert _last(scene.segs.exists(std.constant(0.0)), n, inputs) == pytest.approx(
        1.0, abs=1e-2
    )
    # absent seg id -> no-match path.
    assert _last(scene.segs.exists(std.constant(9.0)), n, inputs) == pytest.approx(
        -1.0, abs=1e-2
    )
    ax, _ay = scene.segs.endpoint_a(std.constant(0.0))
    assert _last(ax, n, inputs) == pytest.approx(10.0, abs=0.1)
    # subsector 0's first seg is seg 0.
    assert _last(
        scene.subsectors.first_seg(std.constant(0.0)), n, inputs
    ) == pytest.approx(0.0, abs=1e-2)
    assert _last(
        scene.subsectors.has_first_seg(std.constant(0.0)), n, inputs
    ) == pytest.approx(1.0, abs=1e-2)


def test_scene_index_publishes_expected_channels() -> None:
    """Smoke: the six published groups exist with the right callable shapes."""
    scene, _n, _inputs = _scene_and_inputs()
    assert hasattr(scene, "view") and hasattr(scene, "nodes")
    assert hasattr(scene, "subsectors") and hasattr(scene, "segs")
    assert hasattr(scene, "assets") and hasattr(scene, "planes")
    # PlaneIndex exposes height/flat_id/light_static — and no `light`.
    assert hasattr(scene.planes, "height")
    assert hasattr(scene.planes, "flat_id")
    assert hasattr(scene.planes, "light_static")
    assert not hasattr(scene.planes, "light")
