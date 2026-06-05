"""Regression: the 2-byte digit-quad carrier must not amplify a near-byte-
boundary step index into a whole-bucket miss.

The bug (pos 1251 ``drawseg.scalestep.den`` in the projection carrier gate): a
range-encoded *integer* carrier — here a 33-column drawseg width, encoded to the
scalar ``0.03125`` — has a continuous step index ``q = 32767.5·(v+1) = 33791.48``
that lands just below the byte boundary ``132·256 = 33792``. The old ``+0.5``
round-to-nearest high-byte floor put its ``floor_int`` ramp at ``m·256 − 0.5`` —
exactly where this ``q`` sits — so the high byte came out fractional (131.5) and
the low-byte recovery amplified that by 256 into a ~128-step miss (row 33919
instead of 33791).

The carry-free single-stage split (``floor(q/256)`` with the ramp AT the
boundary, sharpness 32768) keeps the high byte a clean integer just below a
boundary and degrades to ±1 step (round-down, no carry) only within a hair of
the boundary itself. This pins both the exact-math behavior and compiled fp32.
"""

from __future__ import annotations

import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.graph import fresh_graph_session
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright_doom.embedding import W_EMBED
from torchwright_doom.emit import emit_token
from torchwright_doom.tokens import FloatSlot
from torchwright_doom.vocab import VALUE


def _slot() -> FloatSlot:
    slot = VALUE.slots["v"]
    assert isinstance(slot, FloatSlot)
    return slot


def _q_of_v(v: float) -> float:
    slot = _slot()
    return (slot.levels - 1) / (slot.hi - slot.lo) * (v - slot.lo)


def _v_of_q(q: float) -> float:
    slot = _slot()
    return slot.lo + q / (slot.levels - 1) * (slot.hi - slot.lo)


def _emit_value_rows(vs: list[float]):
    """Return (graph_output, inputs, decoded_rows) for VALUE emitted at each v."""
    slot = _slot()
    with fresh_graph_session():
        v_in = create_input("v", 1, value_range=(slot.lo - 1.0, slot.hi + 1.0))
        out = emit_token(VALUE, include_derived=True, v=v_in)
        inputs = {"v": torch.tensor([[v] for v in vs], dtype=torch.float32)}
        emitted = reference_eval(out, inputs, len(vs))[out]
        rows = [int((emitted[i] @ W_EMBED.T).argmax().item()) for i in range(len(vs))]
        return out, v_in, inputs, rows


def test_int33_carrier_decodes_without_bucket_jump():
    """The exact bug value: drawseg width 33 → v=0.03125 → row 33791, not 33919."""
    v33 = 2.0 * 33.0 / 64.0 - 1.0  # = 0.03125, the R7 encoding of width 33
    assert _q_of_v(v33) == pytest.approx(33791.48, abs=0.1)
    _out, _v, _inp, rows = _emit_value_rows([v33])
    assert rows[0] == 33791, rows[0]  # was 33919 (a 128-step jump)


def test_no_whole_bucket_miss_across_byte_boundaries():
    """Step indices straddling byte boundaries decode within ±1 of round(q):
    the carry-free split degrades gracefully, never by a whole bucket."""
    boundaries = [256, 512, 4096, 33792, 65280]  # multiples of 256, incl. 16·256
    deltas = [-0.6, -0.4, -0.1, 0.0, 0.1, 0.4, 0.6]
    qs = [
        b + d
        for b in boundaries
        for d in deltas
        if 0.0 <= b + d <= 65535.0
    ]
    _out, _v, _inp, rows = _emit_value_rows([_v_of_q(q) for q in qs])
    bad = [(q, r, round(q)) for q, r in zip(qs, rows) if abs(r - round(q)) > 1]
    assert not bad, bad


def test_digit_quad_compiled_matches_oracle():
    """Compiled fp32 reproduces the exact-math digit-quad at sharpness 32768
    (the steep-ramp concern): no node diverges from the oracle."""
    qs = [33791.48, 33791.9, 33792.1, 1023.98, 32767.9, 100.5, 60000.3]
    out, _v, inputs, _rows = _emit_value_rows([_v_of_q(q) for q in qs])
    report = probe_graph(
        out, create_pos_encoding(), inputs, len(qs), d=1024, d_head=16, atol=0.1
    )
    assert report.first_divergent is None, report.format_short()
