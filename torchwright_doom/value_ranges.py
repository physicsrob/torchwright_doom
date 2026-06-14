"""Numeric range bank for the shared ``VALUE`` carrier.

The range id
is Python/control-flow metadata only — it is never embedded as a token
slot; the marker/context around a ``VALUE`` token chooses which range
interprets ``VALUE.v``.

Scope and layering:

* This module is **pure data + encode/decode**. It depends only on the
  primitive token layer (:mod:`.tokens` ``Derived`` / ``Token``) and on
  :mod:`.constants` (``SCREEN_WIDTH`` for the resolution-scaled R5/R7
  ranges) and the pure :mod:`.doom_lighting` math. It must **not** import
  :mod:`.vocab` or :mod:`.embedding`: ``vocab.py`` imports
  ``value_derived_columns`` from here, so the dependency runs one way
  (constants/tokens/doom_lighting -> value_ranges -> vocab). ``VALUE``
  itself stays declared in ``vocab.py`` (no ownership inversion).
* The **prefill** encoder (:func:`encode_float` / :func:`prefill_value`)
  produces the encoded float directly. The autoregressive **emit** path
  additionally quantizes onto the slot grid — that lives in ``emit.py``
  (``emit_float_slot_token``), matching the reference renderer (pydoom)
  drafter's value-emit quantization; it is not re-implemented here.
* The graph-construction helpers kept here (``encode_vec``
  building a graph ``Node`` via ``linear``; ``make_value`` via
  ``make_token``) are the forward-path runtime ``VALUE`` emitter. They turn
  a *computed* node — not an embed-time derived column — into
  an emitted ``VALUE`` row at run time. They lazy-import the graph helpers
  (``std`` / ``emit``) inside the function bodies so the module-load
  dependency stays one-way (``value_ranges`` is imported by ``vocab``, so it
  must not import ``vocab`` / ``embedding`` at module scope).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from .constants import SCREEN_WIDTH
from .doom_lighting import doom_wall_scale_diminish
from .tokens import Derived, Token, TokenType

if TYPE_CHECKING:  # avoid importing the graph layer at module load
    from torchwright.graph import Node


# Wall scale = projection / distance scales linearly with the projection
# focal length _PROJECTION = (SCREEN_WIDTH - 1) / (2 tan(fov/2)); _PROJ_RATIO
# is that focal length normalized to the historical width-60 tuning (= 1.0 at
# SCREEN_WIDTH=60). The scale (R5) and drawseg-width (R7, in screen columns)
# ranges track resolution through it, reproducing 60x50 exactly.
_PROJ_RATIO = (SCREEN_WIDTH - 1) / 59.0


class ValueRange(IntEnum):
    # The lo/hi bounds and their numeric rationale live on VALUE_RANGES below;
    # these trailing comments name what each range carries.
    R0 = 0  # wide map coordinates and bbox corners
    R1 = 1  # medium map coordinates
    R2 = 2  # node deltas
    R3 = 3  # normal sector heights and texture-mid/span values
    R4 = 4  # back-height / upper+lower dc_texturemid range (R3 widened down)
    R5 = 5  # wall scale (projection / distance)
    R6 = 6  # scale denominator (rw_distance x cosine)
    R7 = 7  # drawseg width in screen columns
    R8 = 8  # per-column scale step
    R9 = 9  # finite silheight values (sentinels via the range endpoints)


@dataclass(frozen=True)
class ValueRangeSpec:
    lo: float
    hi: float

    @property
    def scale(self) -> float:
        return 2.0 / (self.hi - self.lo)

    @property
    def bias(self) -> float:
        return -(self.hi + self.lo) / (self.hi - self.lo)

    def encode(self, value: float) -> float:
        return self.scale * float(value) + self.bias

    def decode(self, encoded: float) -> float:
        return self.lo + ((float(encoded) + 1.0) * 0.5) * (self.hi - self.lo)


VALUE_RANGES: dict[ValueRange, ValueRangeSpec] = {
    # R0: wide map coordinates and bbox corners.
    ValueRange.R0: ValueRangeSpec(-2048.0, 3072.0),
    # R1: medium map coordinates.
    ValueRange.R1: ValueRangeSpec(-1152.0, 1152.0),
    # R2: node deltas.
    ValueRange.R2: ValueRangeSpec(-512.0, 512.0),
    # R3: normal sector heights and texture-mid/span values.
    ValueRange.R3: ValueRangeSpec(-256.0, 256.0),
    # R4: back-height / upper+lower dc_texturemid range. Widens R3 downward so
    # it can carry the BACK_HEIGHT_SENTINEL = -4096.0 used for one-sided / back-
    # sector heights. The -4160.0 floor is that sentinel minus the largest
    # dc_texturemid viewz offset, with the FloatSlot clamping the tail.
    ValueRange.R4: ValueRangeSpec(-4160.0, 256.0),
    # R5: wall scale (= projection / distance, grows with _PROJECTION).
    ValueRange.R5: ValueRangeSpec(0.0, 2.5 * _PROJ_RATIO),
    # R6: scale denominator (rw_distance x cosine; resolution-independent).
    ValueRange.R6: ValueRangeSpec(0.0, 1500.0),
    # R7: drawseg width in screen columns (<= SCREEN_WIDTH; 64.0 at width 60).
    ValueRange.R7: ValueRangeSpec(0.0, float(SCREEN_WIDTH + 4)),
    # R8: per-column scale step.
    ValueRange.R8: ValueRangeSpec(-0.0625, 0.0625),
    # R9: finite silheight values use this range directly. ±sentinel values
    # are represented by the range endpoints and decoded by extract.
    ValueRange.R9: ValueRangeSpec(-256.0, 256.0),
}

_VALUE_INVERSE_LIMIT = 4096.0


def encode_float(range_id: ValueRange, value: float) -> float:
    """Prefill encoder: map a physical value into the range's [-1, 1] space."""
    return VALUE_RANGES[ValueRange(range_id)].encode(value)


def decode_float(range_id: ValueRange, encoded: float) -> float:
    """Inverse of :func:`encode_float`."""
    return VALUE_RANGES[ValueRange(range_id)].decode(encoded)


def prefill_value(value_type: TokenType, range_id: ValueRange, value: float) -> Token:
    """3-arg core: build a ``value_type`` Token carrying the encoded value.

    The prompt builder uses the 2-arg ``VALUE``-bound wrapper declared in
    ``vocab.py`` / the prompt layer (where ``VALUE`` is available); this
    core takes the type explicitly so it does not need to import
    ``vocab``.
    """
    return Token(value_type, {"v": encode_float(range_id, value)})


def encode_vec(range_id: ValueRange, value: "Node") -> "Node":
    """Runtime encoder: affine-map a computed node into the range's [-1, 1]
    space.

    ``encoded = scale·value + bias`` as one fused ``Linear`` over
    ``concat(value, 1)`` — the producer-side mirror of :func:`encode_float`.
    The ``1.0`` constant is built inside the call (no import-time graph nodes;
    see ``render_constants`` / the project node-id-reset rule).
    """
    from .std import concat, constant, linear

    spec = VALUE_RANGES[ValueRange(range_id)]
    return linear(concat(value, constant(1.0)), [[spec.scale], [spec.bias]])


def make_value(value_type: TokenType, range_id: ValueRange, value: "Node") -> "Node":
    """3-arg core: emit a ``value_type`` ``VALUE`` token carrying ``value``,
    range-encoded.

    Returns an emit **head** (``make_token_head``): the renderer's dispatch
    folds over heads and stamps one shared derived-zero tail after selecting
    the winning branch, so every owner ``after_*`` emitter — including the
    ``VALUE`` carriers this builds — must produce a head, not a full row. The
    2-arg ``VALUE``-bound wrapper lives in ``vocab.py`` next to the ``VALUE``
    type (where the type is available), mirroring :func:`prefill_value`.
    """
    from .std import make_token_head

    return make_token_head(value_type, v=encode_vec(range_id, value))


def derived_name(kind: str, range_id: ValueRange) -> str:
    return f"{kind}{int(ValueRange(range_id))}"


def value_derived(input_vec: "Node", range_id: ValueRange, kind: str = "v") -> "Node":
    """Read the ``VALUE`` carrier's derived column for ``(kind, range_id)``.

    Thin wrapper over ``extract.extract_derived`` — imported lazily
    because ``extract`` -> ``embedding`` -> ``vocab`` -> ``value_ranges``,
    so a module-level import would cycle. ``kind`` is ``"v"`` (decoded
    value) or ``"inv"`` (zero-guarded reciprocal); R5 also has
    ``"wall_scale_diminish"``.
    """
    from .extract import extract_derived

    return extract_derived(input_vec, derived_name(kind, range_id))


# Only the derived-column kinds the forward graph actually reads off a VALUE
# carrier are baked into the embedding:
#   "v{idx}"               raw decoded value      (value_derived's default kind)
#   "inv{idx}"             1/value, zero-guarded  (scale denominators / widths)
#   "wall_scale_diminish5" Doom wall light diminish, R5 only
def value_derived_columns() -> dict[str, Derived]:
    out: dict[str, Derived] = {}
    for range_id, spec in VALUE_RANGES.items():
        idx = int(range_id)

        def raw(encoded: float, spec: ValueRangeSpec = spec) -> float:
            return spec.decode(encoded)

        def inv(encoded: float, spec: ValueRangeSpec = spec) -> float:
            value = spec.decode(encoded)
            if value == 0.0:
                return _VALUE_INVERSE_LIMIT
            return 1.0 / value

        out[f"v{idx}"] = Derived(raw)
        out[f"inv{idx}"] = Derived(inv)

        if range_id == ValueRange.R5:

            def wall_scale_diminish(
                encoded: float, spec: ValueRangeSpec = spec
            ) -> float:
                return float(doom_wall_scale_diminish(max(0.0, spec.decode(encoded))))

            out[f"wall_scale_diminish{idx}"] = Derived(wall_scale_diminish)

    return out
