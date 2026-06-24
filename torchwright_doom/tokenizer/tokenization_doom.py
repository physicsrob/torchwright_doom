"""``DoomTokenizer`` — the **shipped, self-contained** HuggingFace tokenizer.

This is the file that rides into a saved tokenizer directory (copied verbatim by
``transformers``' ``custom_object_save``) and runs on a stranger's machine with
**only ``transformers`` installed** — no ``torch``, no ``torchwright_doom``. So it
imports *only* the standard library and ``transformers``; it must never grow a
``torch`` / ``torchwright_doom`` import or a two-dot relative import (those make
``check_imports`` raise on the loader's machine — see ``plan`` notes / the
hermetic test).

It reproduces the in-package readable surface (``tokenizer.surface``) without any
of the renderer machinery, by reading two frozen JSON artifacts written at save
time by ``tokenizer.freeze``:

* ``doom_vocab.json`` — the **pretty** ``{display_label: id}`` table (WAD texture
  names, decoded enums/bools/BSP-child-ids/bbox-region-codes already baked into
  the label strings) plus an identity card (screen config, fingerprint, n_rows).
* ``doom_tables.json`` — the carrier-fold data: each marker's value-range bounds,
  the angle markers, the back-height sentinel, the ``value`` / ``angleValue``
  carrier id ranges, and the x/y coordinate-marker sets.

Everything contextual collapses to **one rule**, transcribed verbatim from
``surface.decode`` / ``surface.encode``: a ``value`` / ``angleValue`` *carrier*
token folds into the *preceding marker's* parens as its trailing argument,
de-quantized through that marker's range (degrees for angle markers,
scene-relative coordinates baked at origin ``(0, 0)``). Per-id text is already
baked into the frozen labels, so the only thing this kernel computes is that fold
and its inverse. ``decode(ids)`` is byte-identical to the in-package
``surface.decode`` under the baked knobs; ``encode(decode(ids)) == ids``.

One saved bundle is valid for exactly the screen size it was frozen at (the
carrier id ranges are screen-parameterized). The identity card stamps the screen
dims, but a bundle cannot detect a mismatch on its own — there is no live vocab
to compare against here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {
    "vocab_file": "doom_vocab.json",
    "tables_file": "doom_tables.json",
}


# ---------------------------------------------------------------------------
# pure helpers (transcribed from tokenizer.surface — keep byte-identical)
# ---------------------------------------------------------------------------


def _scan(text: str) -> list[tuple[str, list[str] | None]]:
    """Split readable text into ``(type_name, args)`` units. ``args`` is the
    comma-separated list inside ``(...)`` (possibly empty), or ``None`` for a
    bare slotless type. Whitespace, newlines, and ``#`` comments are ignored — so
    the indented figure layout tokenizes the same as the flat stream. Verbatim
    copy of ``surface._scan``."""
    body = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    out: list[tuple[str, list[str] | None]] = []
    i, n = 0, len(body)
    while i < n:
        if body[i].isspace():
            i += 1
            continue
        start = i
        while i < n and not body[i].isspace() and body[i] != "(":
            i += 1
        name = body[start:i]
        args: list[str] | None = None
        if i < n and body[i] == "(":
            close = body.index(")", i)
            inner = body[i + 1 : close].strip()
            args = [a.strip() for a in inner.split(",")] if inner else []
            i = close + 1
        if name:
            out.append((name, args))
    return out


def _fmt_decimal(value: float, places: int) -> str:
    """Verbatim copy of ``surface._fmt_decimal``."""
    if places <= 0:
        return str(int(round(value)))
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _decode_float(lo: float, hi: float, encoded: float) -> float:
    """``value_ranges.ValueRangeSpec.decode`` for a ``[lo, hi]`` range."""
    return lo + (float(encoded) + 1.0) * 0.5 * (hi - lo)


def _encode_float(lo: float, hi: float, value: float) -> float:
    """``value_ranges.ValueRangeSpec.encode`` for a ``[lo, hi]`` range."""
    span = hi - lo
    return (2.0 / span) * float(value) - (hi + lo) / span


def _level(v: float, steps: int) -> int:
    """The ``value`` carrier's quantization level — exactly the step the row
    enumeration rounds it to (verbatim copy of ``surface._level``)."""
    return round((float(v) + 1.0) / 2.0 * steps)


def _label_of(name: str, args: list[str] | None) -> str:
    """Reconstruct a per-id label from ``(name, args)`` — the inverse of the
    ``display.token_label`` ``name`` / ``name(a, b, ...)`` formatting (and of
    ``_scan``). Empty / missing args render as a bare ``name``."""
    return name if not args else f"{name}({', '.join(args)})"


# ---------------------------------------------------------------------------
# the tokenizer
# ---------------------------------------------------------------------------


class DoomTokenizer(PreTrainedTokenizer):
    """Standalone slow ``PreTrainedTokenizer`` over the frozen DOOM surface.

    Core is a ``WordLevel`` ``{pretty_label: id}`` bijection; the readability and
    the one contextual carrier-fold rule live in ``_tokenize`` /
    ``convert_tokens_to_string`` (the pre-tokenizer / decoder boundary), reading
    the frozen tables. No renderer dependency — see the module docstring.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab_file: str | None = None,
        tables_file: str | None = None,
        *,
        eos_token: str = "done",
        bos_token: str = "begin",
        pad_token: str = "done",
        unk_token: str | None = None,
        **kwargs,
    ):
        if vocab_file is None or tables_file is None:
            raise ValueError(
                "DoomTokenizer needs both vocab_file (doom_vocab.json) and "
                "tables_file (doom_tables.json); load via from_pretrained on a "
                "saved bundle."
            )

        # --- pretty WordLevel bijection (built BEFORE super().__init__, which
        # processes special tokens and consults get_vocab()) -------------------
        self._vocab_blob = json.loads(Path(vocab_file).read_text())
        self._label_to_id: dict[str, int] = {
            label: int(i) for label, i in self._vocab_blob["vocab"].items()
        }
        self._n_rows = int(self._vocab_blob.get("n_rows", len(self._label_to_id)))
        # Injective by construction (freeze asserts it); invert to a dense list.
        self._id_to_label: list[str] = [""] * self._n_rows
        for label, row in self._label_to_id.items():
            self._id_to_label[row] = label

        # --- carrier-fold tables ----------------------------------------------
        self._tables_blob = json.loads(Path(tables_file).read_text())
        self._angle_bam: int = self._tables_blob["angle_bam"]
        self._sentinel_value: float = self._tables_blob["back_height_sentinel"]
        self._value_steps: int = self._tables_blob["value_steps"]
        carrier = self._tables_blob["carrier"]
        self._value_start: int = carrier["value"]["start"]
        self._value_size: int = carrier["value"]["size"]
        self._angle_start: int = carrier["angle"]["start"]
        self._angle_size: int = carrier["angle"]["size"]
        self._angle_lo: int = carrier["angle"]["lo"]
        self._marker_range: dict[str, tuple[float, float]] = {
            word: (float(lo), float(hi))
            for word, (lo, hi) in self._tables_blob["marker_range"].items()
        }
        self._angle_markers: set[str] = set(self._tables_blob["angle_markers"])
        self._sentinel_markers: set[str] = set(self._tables_blob["sentinel_markers"])
        self._x_coord_markers: set[str] = set(self._tables_blob["x_coord_markers"])
        self._y_coord_markers: set[str] = set(self._tables_blob["y_coord_markers"])
        # Baked rendering: scene-relative coordinates (origin 0,0) + degrees. The
        # origin shift is therefore always 0 here; the coord sets are kept so the
        # carrier path mirrors surface verbatim.
        self._origin: tuple[float, float] = (0.0, 0.0)

        # bos ("begin") is the prompt/gen boundary but is NOT auto-prepended (the
        # stream already carries it), so build_inputs_with_special_tokens stays
        # identity. No unk token (the stream is always in-vocab); pad == eos.
        # clean_up_tokenization_spaces=False keeps decode byte-exact — setdefault
        # (not a hardcoded kwarg) so a value round-tripped from tokenizer_config
        # on reload doesn't collide with an explicit one.
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        super().__init__(
            eos_token=eos_token,
            bos_token=bos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )

    # --- WordLevel id<->label bijection -----------------------------------

    @property
    def vocab_size(self) -> int:
        return self._n_rows

    def get_vocab(self) -> dict[str, int]:
        return {**self._label_to_id, **self.added_tokens_encoder}

    def _convert_token_to_id(self, token: str) -> int:
        return self._label_to_id[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_label[index]

    # --- carrier identity / value (carrier ids form two contiguous ranges) -

    def _carrier_kind(self, idx: int) -> str | None:
        if self._value_start <= idx < self._value_start + self._value_size:
            return "value"
        if self._angle_start <= idx < self._angle_start + self._angle_size:
            return "angle"
        return None

    def _carrier_value(self, idx: int) -> float:
        """A ``value`` carrier id back to its ``[-1, 1]`` number."""
        return -1.0 + (idx - self._value_start) / self._value_steps * 2.0

    def _carrier_bam(self, idx: int) -> int:
        """An ``angleValue`` carrier id back to its signed BAM integer."""
        return (idx - self._angle_start) + self._angle_lo

    def _origin_shift(self, marker_word: str) -> float:
        ox, oy = self._origin
        if marker_word in self._x_coord_markers:
            return ox
        if marker_word in self._y_coord_markers:
            return oy
        return 0.0

    # --- the contextual rule: de-quantize a carrier by its marker ---------

    def _shortest_value(
        self, lo: float, hi: float, carrier: float, shift: float
    ) -> str:
        """Shortest decimal of the de-quantized physical that re-encodes to the
        same carrier level (verbatim copy of ``surface._shortest_value``)."""
        target = _level(carrier, self._value_steps)
        physical = _decode_float(lo, hi, carrier) + shift
        for places in range(0, 10):
            candidate = round(physical, places)
            if (
                _level(_encode_float(lo, hi, candidate - shift), self._value_steps)
                == target
            ):
                return _fmt_decimal(candidate, places)
        return repr(physical)

    def _shortest_angle(self, bam: int) -> str:
        """Shortest decimal of the BAM angle in degrees (verbatim copy of
        ``surface._shortest_angle`` with ``degrees=True`` baked)."""
        target = int(bam)
        physical = target * 360.0 / self._angle_bam
        for places in range(0, 10):
            candidate = round(physical, places)
            if round(candidate * self._angle_bam / 360.0) == target:
                return _fmt_decimal(candidate, places)
        return repr(physical)

    def _render_carrier(self, marker_word: str, carrier_id: int) -> str:
        """The carrier's de-quantized string, folded into its marker's parens.
        Transcribed from ``surface._render_carrier`` (physical_values + degrees
        baked)."""
        if self._carrier_kind(carrier_id) == "value":
            rng = self._marker_range.get(marker_word)
            if rng is None:
                raise ValueError(f"value follows non-marker {marker_word!r}")
            lo, hi = rng
            carrier = self._carrier_value(carrier_id)
            shift = self._origin_shift(marker_word)
            physical = _decode_float(lo, hi, carrier) + shift
            if (
                marker_word in self._sentinel_markers
                and abs(physical - self._sentinel_value) < 0.5
            ):
                return "none"
            return self._shortest_value(lo, hi, carrier, shift)
        if marker_word not in self._angle_markers:
            raise ValueError(f"angleValue follows non-angle-marker {marker_word!r}")
        return self._shortest_angle(self._carrier_bam(carrier_id))

    def _encode_carrier(self, marker_word: str, carrier_word: str) -> int:
        """Inverse: a marker's trailing word back to its carrier id. Transcribed
        from the carrier branch of ``surface.encode``."""
        rng = self._marker_range.get(marker_word)
        if rng is not None:
            lo, hi = rng
            if marker_word in self._sentinel_markers and carrier_word == "none":
                carrier = _encode_float(lo, hi, self._sentinel_value)
            else:
                physical = float(carrier_word) - self._origin_shift(marker_word)
                carrier = _encode_float(lo, hi, physical)
            return self._value_start + _level(carrier, self._value_steps)
        if marker_word in self._angle_markers:
            bam = round(float(carrier_word) * self._angle_bam / 360.0)
            return self._angle_start + (int(bam) - self._angle_lo)
        raise ValueError(
            f"token {marker_word!r} has a value {carrier_word!r} but is not a marker"
        )

    @staticmethod
    def _fold(label: str, carrier_str: str | None) -> str:
        """Append the carrier as the marker's trailing positional arg."""
        if carrier_str is None:
            return label
        if "(" in label:  # marker already has args — extend them
            return label[:-1] + ", " + carrier_str + ")"
        return label + "(" + carrier_str + ")"

    # --- HF string layer ---------------------------------------------------

    def _decode_ids(self, ids: list[int]) -> str:
        """Row ids -> flat readable text. Each id becomes its frozen pretty label;
        a carrier id folds into the preceding marker. Transcribed from
        ``surface.decode``."""
        units: list[str] = []
        i, n = 0, len(ids)
        while i < n:
            idx = ids[i]
            if self._carrier_kind(idx) is not None:
                raise ValueError(f"carrier at {i} has no preceding marker")
            label = self._id_to_label[idx]
            carrier_str: str | None = None
            if i + 1 < n and self._carrier_kind(ids[i + 1]) is not None:
                marker_word = label.split("(", 1)[0]
                carrier_str = self._render_carrier(marker_word, ids[i + 1])
                i += 1
            units.append(self._fold(label, carrier_str))
            i += 1
        return " ".join(units)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Readable text -> the WordLevel label pieces. The leftover trailing
        positional arg of a marker is its carrier; everything else is a full
        per-id label looked up directly. Transcribed from ``surface.encode``."""
        labels: list[str] = []
        for name, args in _scan(text):
            full = _label_of(name, args)
            if full in self._label_to_id:
                labels.append(full)  # a complete per-id label, no carrier
                continue
            if not args:
                raise ValueError(f"unknown token keyword: {name!r}")
            *head, carrier_word = args
            base = _label_of(name, head)
            if base not in self._label_to_id:
                raise ValueError(f"unknown token: {full!r}")
            labels.append(base)
            carrier_id = self._encode_carrier(name, carrier_word)
            labels.append(self._id_to_label[carrier_id])
        return labels

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """WordLevel labels -> the flat readable surface (de-quantized)."""
        ids = [self._label_to_id[token] for token in tokens]
        return self._decode_ids(ids)

    # --- serialization -----------------------------------------------------

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str, str]:
        """Re-dump the two frozen artifacts verbatim (the bundle is the source of
        truth; nothing is recomputed here, so no renderer is needed)."""
        os.makedirs(save_directory, exist_ok=True)
        prefix = (filename_prefix + "-") if filename_prefix else ""
        vocab_path = os.path.join(
            save_directory, prefix + VOCAB_FILES_NAMES["vocab_file"]
        )
        tables_path = os.path.join(
            save_directory, prefix + VOCAB_FILES_NAMES["tables_file"]
        )
        Path(vocab_path).write_text(json.dumps(self._vocab_blob))
        Path(tables_path).write_text(json.dumps(self._tables_blob))
        return (vocab_path, tables_path)
