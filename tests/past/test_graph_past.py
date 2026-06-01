"""Plan B: GraphPast facade tests."""

from __future__ import annotations

import pytest
import torch

from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import TOKEN_VOCAB, W_EMBED
from torchwright_doom.extract import type_code
from torchwright_doom.past import GraphPast, PastHandleScope
from torchwright_doom.vocab import NODE, SEG, VALUE


def _dummy_past() -> GraphPast:
    return GraphPast(
        input_vec=create_input("iv", TOKEN_VOCAB.layout.d_embed),
        pos_encoding=create_pos_encoding(),
    )


def _token_row(token_type) -> torch.Tensor:
    start, _end = TOKEN_VOCAB.type_to_row_range[token_type]
    return W_EMBED[start : start + 1].clone()


def test_duplicate_publish_rejected() -> None:
    past = _dummy_past()
    value = create_input("value", 1)
    past.publish("v", value)

    with pytest.raises(RuntimeError, match="already published"):
        past.publish("v", value)


def test_reserved_input_name_rejected() -> None:
    past = _dummy_past()

    with pytest.raises(ValueError, match="reserved input channel"):
        past.publish("input.foo", create_input("value", 1))


def test_non_node_value_rejected() -> None:
    past = _dummy_past()

    with pytest.raises(TypeError, match="expected a torchwright graph Node"):
        past.publish("v", torch.tensor([1.0]))


def test_foreign_handle_rejected_before_lowering() -> None:
    past_a = _dummy_past()
    past_b = _dummy_past()
    key = past_a.publish("key", create_input("key", 1))
    value = past_b.publish("value", create_input("value", 1))
    query = create_input("query", 1)

    with pytest.raises(RuntimeError, match="different GraphPast instance"):
        past_a.pick_argmax(query, key, value)


def test_exclude_self_rejected_on_dense_dot_picks() -> None:
    past = _dummy_past()
    key = past.publish("key", create_input("key", 1))
    value = past.publish("value", create_input("value", 1))
    query = create_input("query", 1)

    with pytest.raises(NotImplementedError, match="exclude_self=True"):
        past.pick_argmax(query, key, value, exclude_self=True)
    with pytest.raises(NotImplementedError, match="exclude_self=True"):
        past.pick_argmin(query, key, value, exclude_self=True)


def test_input_slot_raises_with_typed_extract_guidance() -> None:
    past = _dummy_past()

    with pytest.raises(NotImplementedError, match="TOKEN.extract"):
        past.input_slot("x")


def test_past_handle_scope_input_and_missing_name_behaviour() -> None:
    past = _dummy_past()
    scope = PastHandleScope(past)
    value = create_input("value", 1)

    handle = scope.publish("v", value)
    assert scope["v"] is handle
    assert scope["input.type"].source == "input"

    with pytest.raises(NotImplementedError, match="input_slot"):
        _ = scope["input.x"]
    with pytest.raises(RuntimeError, match="no published channel"):
        _ = scope["missing"]


def test_pick_argmax_by_one_hot_key() -> None:
    past = _dummy_past()
    key = past.publish("key", create_input("key", 3))
    value = past.publish("value", create_input("value", 1))
    query = create_input("query", 3)
    out = past.pick_argmax(query, key, value)

    n_pos = 4
    result = out.compute(
        n_pos,
        {
            "query": torch.tensor([[0.0, 1.0, 0.0]]).expand(n_pos, -1),
            "key": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "value": torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        },
    )

    assert result[1].item() == pytest.approx(20.0, abs=1e-3)
    assert result[2].item() == pytest.approx(20.0, abs=1e-3)
    assert result[3].item() == pytest.approx(20.0, abs=1e-3)


def test_pick_argmin_dense_key_smoke() -> None:
    past = _dummy_past()
    key = past.publish("key", create_input("key", 1))
    value = past.publish("value", create_input("value", 1))
    query = create_input("query", 1)
    out = past.pick_argmin(query, key, value)

    result = out.compute(
        3,
        {
            "query": torch.ones(3, 1),
            "key": torch.tensor([[2.0], [1.0], [3.0]]),
            "value": torch.tensor([[10.0], [20.0], [30.0]]),
        },
    )

    assert result[0].item() == pytest.approx(10.0, abs=1e-3)
    assert result[1].item() == pytest.approx(20.0, abs=1e-3)
    assert result[2].item() == pytest.approx(20.0, abs=1e-3)


def test_lifted_id_key_lookup() -> None:
    past = _dummy_past()
    key = past.publish("key", create_input("key", 3))
    value = past.publish("value", create_input("value", 1))
    query = create_input("query", 3)
    out = past.pick_argmax(query, key, value)

    result = out.compute(
        4,
        {
            "query": torch.tensor([[4.0, 1.0, 1.0]]).expand(4, -1),
            "key": torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [2.0, -4.0, 1.0],
                    [3.0, -9.0, 1.0],
                ]
            ),
            "value": torch.tensor([[0.0], [10.0], [20.0], [30.0]]),
        },
    )

    assert result[2].item() == pytest.approx(20.0, abs=1e-3)
    assert result[3].item() == pytest.approx(20.0, abs=1e-3)


def test_pick_most_recent_short_span() -> None:
    past = _dummy_past()
    key = past.publish("key", create_input("key", 1))
    value = past.publish("value", create_input("value", 1))
    query = create_input("query", 1)
    out = past.pick_most_recent(query, key, value)

    result = out.compute(
        5,
        {
            "query": torch.ones(5, 1),
            "key": torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]]),
            "value": torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]]),
        },
    )

    expected = [10.0, 10.0, 10.0, 40.0, 40.0]
    for pos, expected_value in enumerate(expected):
        assert result[pos].item() == pytest.approx(expected_value, abs=1e-2)


def test_pick_most_recent_forwards_match_gain() -> None:
    n_pos = 51
    past = _dummy_past()
    key = past.publish("key", create_input("key", 1))
    value = past.publish("value", create_input("value", 1))
    query = create_input("query", 1)
    out_default = past.pick_most_recent(query, key, value)
    out_high_gain = past.pick_most_recent(query, key, value, match_gain=800.0)

    inputs = {
        "query": torch.ones(n_pos, 1),
        "key": torch.cat([torch.ones(1, 1), torch.zeros(n_pos - 1, 1)]),
        "value": torch.cat([torch.tensor([[1.0]]), torch.full((n_pos - 1, 1), 99.0)]),
    }

    default_result = out_default.compute(n_pos, inputs)
    high_gain_result = out_high_gain.compute(n_pos, inputs)

    assert abs(default_result[-1].item() - 1.0) > 50.0
    assert high_gain_result[-1].item() == pytest.approx(1.0, abs=1e-3)


def test_mean_where_with_explicit_validity_mask() -> None:
    past = _dummy_past()
    validity = past.publish("validity", create_input("validity", 1))
    value = past.publish("value", create_input("value", 1))
    out = past.mean_where(validity, value)

    result = out.compute(
        4,
        {
            "validity": torch.tensor([[1.0], [-1.0], [1.0], [-1.0]]),
            "value": torch.tensor([[2.0], [20.0], [6.0], [40.0]]),
        },
    )

    assert result[0].item() == pytest.approx(2.0, abs=1e-3)
    assert result[1].item() == pytest.approx(2.0, abs=1e-3)
    assert result[2].item() == pytest.approx(4.0, abs=1e-3)
    assert result[3].item() == pytest.approx(4.0, abs=1e-3)


def test_pick_argmin_above_synthetic_threshold() -> None:
    past = _dummy_past()
    score = past.publish("score", create_input("score", 1))
    indicators = past.publish("indicators", create_input("indicators", 3))
    value = past.publish("value", create_input("value", 1))
    threshold = create_input("threshold", 3)
    out = past.pick_argmin_above(score, indicators, threshold, value)

    result = out.compute(
        4,
        {
            "score": torch.tensor([[1.0], [3.0], [2.0], [4.0]]),
            "indicators": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            "threshold": torch.tensor([[0.0, 1.0, 0.0]]).expand(4, -1),
            "value": torch.tensor([[10.0], [30.0], [20.0], [40.0]]),
        },
    )

    assert result[1].item() == pytest.approx(30.0, abs=1e-3)
    assert result[2].item() == pytest.approx(20.0, abs=5e-3)
    assert result[3].item() == pytest.approx(20.0, abs=5e-3)


def test_wide_attend_to_offset_direct_path() -> None:
    pos_encoding = create_pos_encoding()
    width = 2 * pos_encoding.trig_width + 3
    past = GraphPast(input_vec=create_input("iv", 1), pos_encoding=pos_encoding)
    handle = past.publish("value", create_input("value", width))
    out = past.attend_to_offset(handle, delta_pos=-2)

    value = torch.arange(5 * width, dtype=torch.float32).reshape(5, width)
    result = out.compute(5, {"value": value})

    assert torch.allclose(result[2:], value[:-2], atol=1e-4)


def test_attend_to_offset_rejects_positive_delta() -> None:
    past = _dummy_past()
    handle = past.publish("value", create_input("value", 1))

    with pytest.raises(NotImplementedError, match="causal-only"):
        past.attend_to_offset(handle, delta_pos=1)


def test_input_type_e8_round_trip_through_attend_to_offset() -> None:
    input_vec = create_input("iv", TOKEN_VOCAB.layout.d_embed)
    past = GraphPast(input_vec=input_vec, pos_encoding=create_pos_encoding())
    out = past.attend_to_offset(past.input_type(), delta_pos=-1)

    rows = torch.cat(
        [
            _token_row(NODE),
            _token_row(SEG),
            _token_row(VALUE),
        ],
        dim=0,
    )
    result = out.compute(3, {"iv": rows})
    expected_node_code = type_code(NODE).compute(1, {})[0]
    expected_seg_code = type_code(SEG).compute(1, {})[0]

    assert torch.allclose(result[1], expected_node_code, atol=1e-4)
    assert torch.allclose(result[2], expected_seg_code, atol=1e-4)
