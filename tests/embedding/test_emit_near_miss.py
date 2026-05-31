"""Near-miss tests for the digit-quadratic emit.

The emit helper takes a slot value that, in production, may be
slightly off-integer (FloatSlot quantization residue; near-integer
arithmetic results from an upstream PWL). This test sweeps non-integer
``q`` (step-index) values around every flavor of corner — interior k,
``k ± 0.4``, byte-boundary midpoints — and verifies the host argmax
through ``W_EMBED.T`` resolves to the right row.

The digit-quad encoder has two known soft regions:

* **fp32 noise floor at ``d ≈ ±0.5``**. The score gap between row k
  and row k±1 at ``q = k + d`` is ``1 − 2|d|``, so at ``d = ±0.499``
  the gap is ~0.002. Score magnitudes here run ~32,500, where fp32's
  absolute precision is ~0.004 — gap below noise floor. We accept
  either ``k`` or its near neighbor inside the noise-floor band.
* **byte-boundary transition zone**. The high-byte staircase
  (``thermometer_floor_div``) ramps over ``±0.05`` around each
  half-integer threshold. Inside this ramp, ``hi_q`` is fractional,
  ``lo_q`` lands near 127.5, and argmax may pick *any* row in the
  active byte rather than the two byte-end neighbors. Outside the
  ramp (≥ 0.06 from midpoint) the split is clean.

VALUE is the FloatSlot under test; ANGLE_VALUE doubles as a 2-digit
IntSlot stress (its 8192-step cardinality also crosses the 256-byte
boundary, so it exercises the same staircase path).
"""

from __future__ import annotations

import torch

from torchwright.debug.probe import reference_eval
from torchwright.graph import fresh_graph_session
from torchwright.ops.inout_nodes import create_input

from torchwright_doom.embedding import BASE, TOKEN_VOCAB, W_EMBED
from torchwright_doom.emit import emit_float_slot_token, emit_int_slot_token
from torchwright_doom.tokens import FloatSlot
from torchwright_doom.vocab import ANGLE_VALUE, VALUE


def _project_argmax(emit_value: torch.Tensor) -> int:
    scores = emit_value @ W_EMBED.T
    return int(scores.argmax(dim=1).item())


def _value_row(t, values: dict) -> int:
    start, end = TOKEN_VOCAB.type_to_row_range[t]
    for offset, (_t, v) in enumerate(TOKEN_VOCAB.row_to_token[start:end]):
        if v == values:
            return start + offset
    raise KeyError(f"No row for {t.name}{values!r}")


def _value_for_q(q: float) -> float:
    slot = _value_slot()
    span = slot.hi - slot.lo
    return slot.lo + q / (slot.levels - 1) * span


def _value_slot() -> FloatSlot:
    slot = VALUE.slots["v"]
    assert isinstance(slot, FloatSlot)
    return slot


def _emit_value_for_q(q: float) -> torch.Tensor:
    """Emit residual for ``VALUE`` at a (possibly non-integer) step
    index ``q`` via ``emit_float_slot_token``."""
    slot = _value_slot()
    v = _value_for_q(q)
    with fresh_graph_session():
        v_in = create_input(
            "v",
            1,
            value_range=(float(slot.lo) - 1.0, float(slot.hi) + 1.0),
        )
        out = emit_float_slot_token(VALUE, v=v_in)
        cache = reference_eval(out, {"v": torch.tensor([[float(v)]])}, n_pos=1)
        return cache[out]


def _quantized_v(k: int) -> float:
    slot = _value_slot()
    span = slot.hi - slot.lo
    return slot.lo + (k / (slot.levels - 1)) * span


def test_value_near_integer_step_strict() -> None:
    """At ``q = k + d`` for ``|d| ≤ 0.4``, argmax → row k strictly.

    ``d = ±0.4`` keeps the score gap at 0.36 and ``q`` strictly outside
    the half-integer ramp for every byte-boundary-adjacent k.
    """
    deltas = [-0.4, -0.1, 0.0, 0.1, 0.4]
    # Cover the full range plus byte-boundary neighbors.
    test_ks = [0, 1, 5, 100, 255, 256, 257, 511, 512, 32767, 32768, 65534, 65535]
    for k in test_ks:
        for d in deltas:
            q = k + d
            if not (0.0 <= q <= 65535.0):
                continue
            emit = _emit_value_for_q(q)
            argmax = _project_argmax(emit)
            expected = _value_row(VALUE, {"v": _quantized_v(k)})
            assert argmax == expected, (
                f"VALUE q={q} (k={k}, d={d}): argmax {argmax} != "
                f"expected {expected}"
            )


def test_value_near_half_step_accepts_either_neighbor() -> None:
    """At ``q = k ± 0.499`` for ``k`` interior to its byte, the score
    gap (0.002) sits below the fp32 noise floor at score-magnitude
    32,500 (~0.004 absolute precision). Either ``k`` or its near
    neighbor is acceptable.

    ``k`` at a byte boundary is covered by
    :func:`test_value_byte_boundary_outside_ramp` — at ``k = m·BASE``,
    ``q = k - 0.499 = midpoint + 0.001`` lands inside the staircase
    ramp, where any row in the active byte may win.
    """
    # All test ks are interior to their byte (no byte boundary within 1
    # of them).
    for k in [100, 1000, 16000, 50000]:
        for d in (-0.499, 0.499):
            q = k + d
            emit = _emit_value_for_q(q)
            argmax = _project_argmax(emit)
            neighbor = k - 1 if d < 0 else k + 1
            allowed = {
                _value_row(VALUE, {"v": _quantized_v(k)}),
                _value_row(VALUE, {"v": _quantized_v(neighbor)}),
            }
            assert argmax in allowed, (
                f"VALUE q={q} (k={k}, d={d}): argmax {argmax} not "
                f"in {{k={k}, neighbor={neighbor}}} = {sorted(allowed)}"
            )


def test_value_byte_boundary_outside_ramp() -> None:
    """Outside the high-byte staircase ramp the byte boundary splits
    cleanly: ``q = midpoint − δ`` picks the lower neighbor's byte,
    ``q = midpoint + δ`` picks the upper neighbor's byte (δ ≥ 0.06,
    one step past the 0.05-wide ramp).

    "Picks the byte" — not necessarily the nearest integer. ``q = m·BASE
    − 0.5 + 0.06 = m·BASE − 0.44`` rounds to integer ``m·BASE``: hi_q =
    m (post-ramp), lo_q = -0.44, nearest row by (hi, lo) distance is
    ``(m, 0) = m·BASE``.
    """
    levels = _value_slot().levels
    test_ms = [1, 2, 32, 128, 200, 255]
    for m in test_ms:
        midpoint = m * BASE - 0.5
        if midpoint < 0.0 or midpoint > levels - 1:
            continue
        lower_k = m * BASE - 1  # last integer in byte m-1
        upper_k = m * BASE  # first integer in byte m

        for delta, expected_k, label in [
            (-0.06, lower_k, "midpoint-0.06 → byte m-1"),
            (+0.06, upper_k, "midpoint+0.06 → byte m"),
        ]:
            q = midpoint + delta
            if not (0.0 <= q <= levels - 1):
                continue
            emit = _emit_value_for_q(q)
            argmax = _project_argmax(emit)
            expected = _value_row(VALUE, {"v": _quantized_v(expected_k)})
            assert argmax == expected, (
                f"VALUE q={q} ({label}, m={m}): argmax {argmax} != "
                f"expected k={expected_k} (row {expected})"
            )


def test_int_slot_near_integer_step_strict() -> None:
    """ANGLE_VALUE.angle is an 8192-step IntSlot; the producer may
    feed near-integer angles when the upstream graph is approximate.

    Strict at ``|d| ≤ 0.4``."""
    slot = ANGLE_VALUE.slots["angle"]
    deltas = [-0.4, -0.1, 0.0, 0.1, 0.4]
    test_angles = [-4096, -4095, -3840, -3839, 0, 1, 1024, 1025, 4095]
    for angle in test_angles:
        for d in deltas:
            angle_real = angle + d
            if not (slot.lo <= angle_real < slot.hi):
                continue
            with fresh_graph_session():
                a_in = create_input(
                    "a",
                    1,
                    value_range=(float(slot.lo) - 1, float(slot.hi) + 1),
                )
                out = emit_int_slot_token(ANGLE_VALUE, angle=a_in)
                cache = reference_eval(
                    out,
                    {"a": torch.tensor([[float(angle_real)]])},
                    n_pos=1,
                )
                value = cache[out]
            argmax = _project_argmax(value)
            expected = _value_row(ANGLE_VALUE, {"angle": angle})
            assert argmax == expected, (
                f"ANGLE_VALUE angle={angle_real} (k={angle}, d={d}): "
                f"argmax {argmax} != expected {expected}"
            )
