"""token_dump.json shape (viewer-compatible) (no GPU)."""

from __future__ import annotations

import json

from torchwright_doom.interpret import artifacts
from torchwright_doom.tokenizer.rows import row_index
from torchwright_doom.vocab import BEGIN, PIXEL, SET_CURSOR_DIRECTION_Y


def test_build_and_write_token_dump(tmp_path):
    prefill_rows = [row_index(BEGIN, {})]
    emitted_rows = [
        row_index(SET_CURSOR_DIRECTION_Y, {}),
        row_index(PIXEL, {"color": 3, "w": 1}),
    ]
    dump = artifacts.build_token_dump(
        fixture="e1m1_subset_textured",
        pose_index=0,
        pose={"x": 1.0, "y": 2.0, "angle": 0, "viewz": 41.0},
        prefill_rows=prefill_rows,
        emitted_rows=emitted_rows,
        mode="pure_ar",
    )
    assert dump["schema_version"] == 1
    assert dump["implementation"] == "torchwright_doom_compiled"
    assert set(dump["screen"]) == {"width", "height"}
    case = dump["cases"][0]
    assert case["counts"]["rollout_positions"] == 2
    rollout = case["predicted_next_tokens"]
    assert [e["type"] for e in rollout] == ["setCursorDirectionY", "pixel"]
    assert all(e["phase"] == "rollout" for e in rollout)
    assert rollout[1]["values"] == {"color": 3, "w": 1}

    path = artifacts.write_token_dump(tmp_path / "token_dump.json", dump)
    assert json.loads(path.read_text())["cases"][0]["fixture"] == "e1m1_subset_textured"
