"""End-to-end smoke test: WAD → subset → prompt token sequence."""

from __future__ import annotations

from collections import Counter

from torchwright_doom.prompt.build import build_prompt
from torchwright_doom.prompt.scenes import E1M1_START_ROOM, load
from torchwright_doom.vocab import (
    BEGIN,
    BOS,
    NODE,
    PLAYER_X_MARK,
    PROMPT_TYPES,
    SEG,
    SS,
)


def test_e1m1_start_room_subset_shape() -> None:
    md, _state = load(E1M1_START_ROOM)
    assert md.name == "E1M1"
    # README pins these counts for the (627.2, -3760.0, 1395.2, -2800.0) box:
    # 81 segs, 29 real subsectors + 1 synthetic empty, 41 BSP nodes.
    assert len(md.segs) == 81
    assert len(md.subsectors) == 30
    assert len(md.nodes) == 41


def test_prompt_endpoints_and_counts() -> None:
    md, state = load(E1M1_START_ROOM)
    tokens = build_prompt(md, state)

    # Position 0 is the BOS anchor; the player-x marker follows it. BEGIN still
    # closes the prompt as the prompt->AR boundary.
    assert tokens[0].type is BOS
    assert tokens[1].type is PLAYER_X_MARK
    assert tokens[-1].type is BEGIN

    type_counts = Counter(t.type for t in tokens)

    assert type_counts[NODE] == len(md.nodes)
    assert type_counts[SS] == len(md.subsectors)
    assert type_counts[SEG] == len(md.segs)

    # BOS is emitted by build_prompt but deliberately excluded from PROMPT_TYPES
    # (it is appended last in VOCAB_TYPES for E8 append-safety; see vocab.py).
    declared = {BOS, *PROMPT_TYPES}
    for typ in type_counts:
        assert typ in declared, f"unexpected token type: {typ.name}"


def test_initial_pose_in_subset_frame() -> None:
    md, state = load(E1M1_START_ROOM)
    ox, oy = md.scene_origin
    world_x = state.x + ox
    world_y = state.y + oy
    assert world_x == E1M1_START_ROOM.initial_pose_world[0]
    assert world_y == E1M1_START_ROOM.initial_pose_world[1]
    assert state.angle == E1M1_START_ROOM.initial_pose_world[2]
