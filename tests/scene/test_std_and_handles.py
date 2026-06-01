"""Plan D / D0 + D1: std shim and attention-handle correctness.

All checks evaluate the graph oracle (``node.compute``) — the same recursive
exact-math evaluation ``probe_compiled`` compares the compiled transformer
against. No compile step is needed to prove the read-side graph math is right.
"""

from __future__ import annotations

import pytest
import torch

from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom import render_ops, std
from torchwright_doom.attention_handles import (
    KeyMarkerHandle,
    KeyPresenceLookup,
    KeyValueHandle,
    LiftedKeyValueHandle,
    OptionalKeyValueHandle,
    OptionalKeyValueLookup,
    lifted_id_query,
)
from torchwright_doom.embedding import TOKEN_VOCAB
from torchwright_doom.past import GraphPast

_D_EMBED = TOKEN_VOCAB.layout.d_embed


def _fresh_past() -> GraphPast:
    return GraphPast(
        input_vec=create_input("iv", _D_EMBED),
        pos_encoding=create_pos_encoding(),
    )


def _val(node, n_pos: int, inputs: dict) -> torch.Tensor:
    return node.compute(n_pos, inputs)


# ---------------------------------------------------------------------------
# D0 std shim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_one_hot_integers(k: int) -> None:
    out = _val(std.one_hot(std.constant(float(k)), 5), 1, {})[0]
    expected = [1.0 if i == k else 0.0 for i in range(5)]
    assert out.tolist() == pytest.approx(expected, abs=1e-3)


def test_gate_passes_and_zeros() -> None:
    assert _val(std.gate(std.constant(1.0), std.constant([5.0, 6.0])), 1, {})[
        0
    ].tolist() == pytest.approx([5.0, 6.0])
    assert _val(std.gate(std.constant(-1.0), std.constant([5.0, 6.0])), 1, {})[
        0
    ].tolist() == pytest.approx([0.0, 0.0])


def test_split_and_concat_roundtrip() -> None:
    packed = std.concat(std.constant([10.0, 20.0, 30.0]), std.constant([40.0]))
    assert _val(packed, 1, {})[0].tolist() == pytest.approx([10.0, 20.0, 30.0, 40.0])
    a, b = std.split(packed, [3, 1])
    assert _val(a, 1, {})[0].tolist() == pytest.approx([10.0, 20.0, 30.0])
    assert _val(b, 1, {})[0].tolist() == pytest.approx([40.0])


def test_linear_builds_lifted_query() -> None:
    # [[2,0,0],[0,1,1]] applied to [q, 1] -> [2q, 1, 1].
    out = _val(
        std.linear(std.constant([4.0, 1.0]), [[2.0, 0.0, 0.0], [0.0, 1.0, 1.0]]), 1, {}
    )[0]
    assert out.tolist() == pytest.approx([8.0, 1.0, 1.0])


def test_render_ops_booleans() -> None:
    assert _val(render_ops.MARKER_PRESENT(std.constant(1.0)), 1, {})[
        0
    ].item() == pytest.approx(1.0)
    assert _val(render_ops.MARKER_PRESENT(std.constant(-1.0)), 1, {})[
        0
    ].item() == pytest.approx(-1.0)
    assert _val(render_ops.and_(std.constant(1.0), std.constant(1.0)), 1, {})[
        0
    ].item() == pytest.approx(1.0)
    assert _val(render_ops.and_(std.constant(1.0), std.constant(-1.0)), 1, {})[
        0
    ].item() == pytest.approx(-1.0)
    assert _val(render_ops.one_minus(std.constant(1.0)), 1, {})[
        0
    ].item() == pytest.approx(-1.0)


def test_lifted_id_query_shape() -> None:
    out = _val(lifted_id_query(std.constant(7.0)), 1, {})[0]
    assert out.tolist() == pytest.approx([14.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# D1 attention handles
# ---------------------------------------------------------------------------


def test_key_value_handle_picks_active_match() -> None:
    past = _fresh_past()
    active = create_input("active", 1)
    idn = create_input("id", 1)
    value = create_input("value", 1)
    handle = KeyValueHandle.publish(past, "kv", active, std.one_hot(idn, 5), value)
    out = handle.pick(past, std.constant(2.0), 5)
    res = _val(
        out,
        4,
        {
            # pos 2 also has id=2 but is inactive -> gated key 0, must lose.
            "active": torch.tensor([[1.0], [1.0], [-1.0], [1.0]]),
            "id": torch.tensor([[2.0], [0.0], [2.0], [3.0]]),
            "value": torch.tensor([[20.0], [10.0], [999.0], [30.0]]),
        },
    )
    assert res[3].item() == pytest.approx(20.0, abs=1e-3)


def test_lifted_key_value_handle_exact_equality() -> None:
    past = _fresh_past()
    active = create_input("active", 1)
    key = create_input("lkey", 3)
    value = create_input("value", 1)
    handle = LiftedKeyValueHandle.publish(past, "lk", active, key, value)
    out = handle.pick(past, std.constant(2.0))
    res = _val(
        out,
        4,
        {
            "active": torch.ones(4, 1),
            "lkey": torch.tensor(
                [[0.0, 0.0, 1.0], [1.0, -1.0, 1.0], [2.0, -4.0, 1.0], [3.0, -9.0, 1.0]]
            ),
            "value": torch.tensor([[0.0], [10.0], [20.0], [30.0]]),
        },
    )
    assert res[3].item() == pytest.approx(20.0, abs=1e-3)


def test_presence_lookup_detects_absent_id_no_match_path() -> None:
    """The missing-id / no-match branch: with a realistic ±1 marker mix
    (most rows inactive), an absent-id probe falls below the presence
    threshold while a present id reads +1. Exercising this here keeps the
    deferred write-side port from inheriting an untested no-match branch."""
    past = _fresh_past()
    active = create_input("active", 1)
    idn = create_input("id", 1)
    handle = KeyMarkerHandle.publish(past, "km", active, std.one_hot(idn, 5))
    lookup = KeyPresenceLookup(past, handle, 5)
    present_2 = lookup(std.constant(2.0))
    present_4 = lookup(std.constant(4.0))
    n = 8
    inputs = {
        "active": torch.tensor(
            [[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0]]
        ),
        "id": torch.tensor([[0.0], [1.0], [2.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
    }
    assert _val(present_2, n, inputs)[-1].item() == pytest.approx(1.0, abs=1e-2)
    assert _val(present_4, n, inputs)[-1].item() == pytest.approx(-1.0, abs=1e-2)


def test_optional_lookup_value_and_presence() -> None:
    past = _fresh_past()
    active = create_input("active", 1)
    idn = create_input("id", 1)
    value = create_input("value", 1)
    handle = OptionalKeyValueHandle.publish(
        past, "opt", active, std.one_hot(idn, 5), value
    )
    lookup = OptionalKeyValueLookup(past, handle, 5)
    n = 8
    inputs = {
        "active": torch.tensor(
            [[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0]]
        ),
        "id": torch.tensor([[0.0], [1.0], [2.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
        "value": torch.tensor(
            [[20.0], [10.0], [200.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        ),
    }
    assert _val(lookup(std.constant(2.0)), n, inputs)[-1].item() == pytest.approx(
        200.0, abs=1e-2
    )
    assert _val(lookup.present(std.constant(2.0)), n, inputs)[
        -1
    ].item() == pytest.approx(1.0, abs=1e-2)
    assert _val(lookup.present(std.constant(4.0)), n, inputs)[
        -1
    ].item() == pytest.approx(-1.0, abs=1e-2)
