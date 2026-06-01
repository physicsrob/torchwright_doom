"""Plan D / D7: ProtocolTokenView predicates, prev-type chains, accessors."""

from __future__ import annotations

import pytest
import torch

from torchwright.ops.inout_nodes import create_input

from torchwright_doom.embedding import TOKEN_VOCAB
from torchwright_doom.extract import type_code
from torchwright_doom.protocol_tokens import ProtocolTokenView, screen_column_one_hot
from torchwright_doom.vocab import (
    DRAWSEG_SCALE1,
    DRAWSEG_SCALE2,
    DONE,
    NODE,
    PIXEL,
    PROCESS_SEG,
    SCREEN_WIDTH,
    SCREEN_Y_VALUE,
    SET_CURSOR_X,
    VALUE,
    WALL_COL_U,
)

from ..prefill_fixture import token_row

_D_EMBED = TOKEN_VOCAB.layout.d_embed


def _view() -> ProtocolTokenView:
    return ProtocolTokenView(
        create_input("iv", _D_EMBED),
        create_input("pv", 8),
        create_input("ppv", 8),
    )


def _e8(token_type) -> torch.Tensor:
    return type_code(token_type).compute(1, {})[0]


def _eval(node, iv_row, prev=None, prev_prev=None) -> torch.Tensor:
    inputs = {
        "iv": iv_row.unsqueeze(0),
        "pv": (_e8(prev) if prev else torch.zeros(8)).unsqueeze(0),
        "ppv": (_e8(prev_prev) if prev_prev else torch.zeros(8)).unsqueeze(0),
    }
    return node.compute(1, inputs)[0]


def test_installed_one_token_predicates() -> None:
    view = _view()
    assert _eval(view.is_value, token_row(VALUE, {"v": 0.3})).item() == pytest.approx(
        1.0, abs=1e-2
    )
    assert _eval(
        view.is_value, token_row(PROCESS_SEG, {"i": 3})
    ).item() == pytest.approx(-1.0, abs=1e-2)
    assert _eval(
        view.is_process_seg, token_row(PROCESS_SEG, {"i": 3})
    ).item() == pytest.approx(1.0, abs=1e-2)
    assert _eval(view.is_done, token_row(DONE)).item() == pytest.approx(1.0, abs=1e-2)


def test_slot_accessor() -> None:
    view = _view()
    assert _eval(
        view.process_i, token_row(PROCESS_SEG, {"i": 3})
    ).item() == pytest.approx(3.0, abs=1e-2)


def test_two_token_value_after_marker_chain() -> None:
    view = _view()
    row = token_row(VALUE, {"v": 0.1})
    assert _eval(
        view.is_value_after_drawseg_scale1, row, DRAWSEG_SCALE1
    ).item() == pytest.approx(1.0, abs=1e-2)
    assert _eval(
        view.is_value_after_drawseg_scale1, row, DRAWSEG_SCALE2
    ).item() == pytest.approx(-1.0, abs=1e-2)


def test_three_token_screen_y_chain() -> None:
    view = _view()
    row = token_row(SCREEN_Y_VALUE, {"y": 5})
    # SCREEN_Y after VALUE after WALL_COL_U
    assert _eval(
        view.screen_y_after_wall_column_scale, row, VALUE, WALL_COL_U
    ).item() == pytest.approx(1.0, abs=1e-2)
    # wrong prev_prev token breaks the chain.
    assert _eval(
        view.screen_y_after_wall_column_scale, row, VALUE, PROCESS_SEG
    ).item() == pytest.approx(-1.0, abs=1e-2)


def test_is_inert_non_payload_excludes_carriers() -> None:
    view = _view()
    assert _eval(
        view.is_inert_non_payload, token_row(NODE, {"j": 1})
    ).item() == pytest.approx(1.0, abs=1e-2)
    # VALUE is a carrier, not inert — its meaning depends on the preceding marker.
    assert _eval(
        view.is_inert_non_payload, token_row(VALUE, {"v": 0.1})
    ).item() == pytest.approx(-1.0, abs=1e-2)


def test_screen_column_one_hot_addresses_column() -> None:
    iv = create_input("iv", _D_EMBED)
    out = _eval(screen_column_one_hot(iv), token_row(SET_CURSOR_X, {"x": 5}))
    assert len(out) == SCREEN_WIDTH
    assert int(torch.argmax(out).item()) == 5


def test_pixel_rgb_derived_accessors() -> None:
    view = _view()
    # PLAYPAL channel reads are 0..255 floats; just confirm shape-correct and bounded.
    for chan in (view.pixel_r, view.pixel_g, view.pixel_b):
        val = _eval(chan, token_row(PIXEL, {"color": 10})).item()
        assert 0.0 <= val <= 255.0
