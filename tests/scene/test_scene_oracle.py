"""Real-fixture scene-index round-trip correctness.

This is the hard gate the other scene tests cannot replace: it drives
``SceneIndex.build`` over a *real* map's prompt (E1M1 start room, 41 BSP
nodes / 81 segs) and confirms every probed scene-index channel recovers the
value the prompt encoded.

Two comparisons are kept separate:

1. **Port-correctness (this gate).** Each channel node is evaluated through the
   graph oracle (``reference_eval`` -> memoised ``node.compute``) and compared
   against the map data the prompt builder encoded. This proves the ported
   graph computes the *right* values.
2. **Compile fidelity.** ``test_scene_compiled_probe`` compares the compiled
   transformer against this same graph oracle.

Tolerance: scene coordinate reads are one affine ``Linear`` over a
``FloatSlot`` (65536 levels) — quantization error is < 0.1 in the R0/R1 ranges,
far below the DOOM-scale ``atol ~= 500`` floor documented in CLAUDE.md (that
floor is for the deep renderer-write chains, not these shallow reads). We use
``atol=0.5`` for coordinates and exact ``+/-1`` for booleans, and record the
rationale here per the acceptance criteria.

The no-match branch (absent node/seg id -> presence below threshold) is probed
explicitly so a single-pose run does not leave it for the write-side port.
"""

from __future__ import annotations

import pytest

from torchwright.graph import Concatenate
from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import create_input, create_rope_config

from torchwright_doom import std
from torchwright_doom.embedding import TOKEN_VOCAB
from torchwright_doom.past import GraphPast
from torchwright_doom.prompt.build import _player_angle_signed, build_prompt
from torchwright_doom.prompt.geometry import bake_segments
from torchwright_doom.prompt.scenes import E1M1_START_ROOM, load
from torchwright_doom.scene_index import SceneIndex

from ..prefill_fixture import tokens_to_input

_D_EMBED = TOKEN_VOCAB.layout.d_embed
_COORD_ATOL = 0.5
_BOOL_ATOL = 1e-2


def _build_scene_oracle():
    md, state = load(E1M1_START_ROOM)
    tokens = build_prompt(md, state)
    n_pos = len(tokens)
    inputs = {"iv": tokens_to_input(tokens)}

    past = GraphPast(
        input_vec=create_input("iv", _D_EMBED),
        rope=create_rope_config(d_head=128, max_positions=65536, d_rot=64),
    )
    scene = SceneIndex.build(create_input("iv", _D_EMBED), past)

    segments = bake_segments(md)
    n_nodes = len(md.nodes)
    n_segs = len(md.segs)

    # Curate a representative set of channels; absent ids exercise the no-match
    # branch. One memoised reference_eval pass covers them all.
    probes = {
        "view_x": (scene.view.x, state.x),
        "view_y": (scene.view.y, state.y),
        "view_angle": (scene.view.angle, float(_player_angle_signed(state.angle))),
        "root": (scene.nodes.root, float(n_nodes - 1)),
        "node_exists_0": (scene.nodes.exists(std.constant(0.0)), 1.0),
        "node_exists_absent": (scene.nodes.exists(std.constant(float(n_nodes))), -1.0),
        "node_px_0": (scene.nodes.px(std.constant(0.0)), md.nodes[0].px),
        "node_py_0": (scene.nodes.py(std.constant(0.0)), md.nodes[0].py),
        "node_px_mid": (
            scene.nodes.px(std.constant(float(n_nodes // 2))),
            md.nodes[n_nodes // 2].px,
        ),
        "seg_exists_0": (scene.segs.exists(std.constant(0.0)), 1.0),
        "seg_exists_absent": (scene.segs.exists(std.constant(float(n_segs))), -1.0),
        "seg_ax_0": (scene.segs.ax(std.constant(0.0)), segments[0].ax),
        "seg_ay_0": (scene.segs.ay(std.constant(0.0)), segments[0].ay),
    }
    output = Concatenate([node for node, _ in probes.values()])
    oracle = reference_eval(output, inputs, n_pos)
    last = {name: oracle[node][-1].item() for name, (node, _) in probes.items()}
    expected = {name: exp for name, (_, exp) in probes.items()}
    return last, expected


def test_scene_index_oracle_matches_map_data() -> None:
    last, expected = _build_scene_oracle()

    boolean = {
        "node_exists_0",
        "node_exists_absent",
        "seg_exists_0",
        "seg_exists_absent",
    }
    for name, got in last.items():
        atol = _BOOL_ATOL if name in boolean else _COORD_ATOL
        assert got == pytest.approx(expected[name], abs=atol), (
            f"channel {name}: graph oracle {got} != map-data {expected[name]} "
            f"(atol={atol})"
        )
