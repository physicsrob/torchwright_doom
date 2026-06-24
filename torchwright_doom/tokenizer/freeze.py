"""Save-time exporter: freeze the readable surface into a self-contained bundle.

This runs in the **full renderer environment** (it imports ``torchwright_doom``
freely) and writes the two JSON artifacts the shipped ``tokenization_doom``
kernel reads on a stranger's torch-free machine:

* ``doom_vocab.json`` — the **pretty** ``{display_label: id}`` table (WAD texture
  names + decoded enums/bools/BSP-child-ids/bbox-codes baked into the label
  strings via :func:`display.token_label`) plus an identity card (screen config,
  fingerprint, n_rows). The display knobs are baked into the labels here so the
  shipped kernel never ships the display layer.
* ``doom_tables.json`` — the one thing that combines two ids and so can't be
  pre-baked: the **carrier-fold rule** data (each marker's value-range bounds,
  the angle markers, the back-height sentinel, the carrier id ranges, the x/y
  coordinate-marker sets, ``ANGLE_BAM``).

:func:`export_bundle` then instantiates the shipped ``DoomTokenizer`` over those
artifacts and calls its ``save_pretrained`` — so ``custom_object_save`` copies the
*standalone* ``tokenization_doom.py`` (never this module or ``hf_tokenizer.py``,
which pull in torch) and writes ``auto_map`` pointing at it.

The frozen tables are the single point of drift between the model-side
``surface`` and the shipped kernel; the hermetic byte-exact test
(``tests/tokenizer/test_shipped_tokenizer_standalone.py``) is the gate.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from ..asset_config import DEFAULT_ASSET_CONFIG, AssetConfig
from ..embedding import TOKEN_VOCAB
from ..marker_ranges import ANGLE_MARKERS, MARKER_RANGE
from ..tokens import FloatSlot, IntSlot
from ..value_ranges import VALUE_RANGES
from ..vocab import ANGLE_BAM, ANGLE_VALUE, BACK_HEIGHT_SENTINEL, VALUE, VOCAB_TYPES
from . import display, surface
from .display import DISPLAY_NAME, token_label
from .hf_tokenizer import screen_config, vocab_fingerprint

_TYPE_BY_NAME = {t.name: t for t in VOCAB_TYPES}


def build_vocab(wall_names: Sequence[str], flat_names: Sequence[str]) -> dict[str, int]:
    """The frozen ``{pretty_label: id}`` table. Each row's label is its
    :func:`display.token_label` rendered **with** the asset name tables, so the
    pretty WAD-name / decoded-enum spellings ARE the stored labels. Must be
    injective (a ``WordLevel`` map is a bijection only if labels are unique)."""
    vocab: dict[str, int] = {}
    for row, (ttype, values) in enumerate(TOKEN_VOCAB.row_to_token):
        label = token_label(ttype, values, wall_names=wall_names, flat_names=flat_names)
        prior = vocab.get(label)
        if prior is not None:
            raise ValueError(
                f"pretty labels are not injective: rows {prior} and {row} both "
                f"render as {label!r}. The frozen WordLevel map needs unique "
                "labels — see tokenizer.display.token_label."
            )
        vocab[label] = row
    assert len(vocab) == TOKEN_VOCAB.n_rows
    return vocab


def build_tables() -> dict:
    """The carrier-fold rule, screen-baked. Everything keyed by a marker's
    **display word** (the leading word of its frozen label) so the shipped kernel
    can look it up from the label it already holds — markers are never
    value-in-name tagged, so the display word is a unique key for the type."""
    value_start, value_end = TOKEN_VOCAB.type_to_row_range[VALUE]
    angle_start, angle_end = TOKEN_VOCAB.type_to_row_range[ANGLE_VALUE]
    value_slot = VALUE.slots["v"]
    angle_slot = ANGLE_VALUE.slots["angle"]
    assert isinstance(value_slot, FloatSlot)  # the carrier grid (.levels)
    assert isinstance(angle_slot, IntSlot)  # signed BAM grid (.lo)

    marker_range: dict[str, list[float]] = {}
    for marker, range_id in MARKER_RANGE.items():
        spec = VALUE_RANGES[range_id]
        marker_range[DISPLAY_NAME[marker]] = [spec.lo, spec.hi]

    def coord_words(canon_names) -> list[str]:
        return sorted(
            {
                DISPLAY_NAME[_TYPE_BY_NAME[name]]
                for name in canon_names
                if name in _TYPE_BY_NAME
            }
        )

    return {
        "screen": screen_config(),
        "angle_bam": int(ANGLE_BAM),
        "back_height_sentinel": float(BACK_HEIGHT_SENTINEL),
        "value_steps": int(value_slot.levels) - 1,
        "carrier": {
            "value": {
                "start": int(value_start),
                "size": int(value_end - value_start),
                "lo": float(value_slot.lo),
                "hi": float(value_slot.hi),
                "levels": int(value_slot.levels),
            },
            "angle": {
                "start": int(angle_start),
                "size": int(angle_end - angle_start),
                "lo": int(angle_slot.lo),
            },
        },
        "marker_range": marker_range,
        "angle_markers": sorted(DISPLAY_NAME[m] for m in ANGLE_MARKERS),
        "sentinel_markers": sorted(
            DISPLAY_NAME[_TYPE_BY_NAME[name]] for name in display._SENTINEL_MARKERS
        ),
        "x_coord_markers": coord_words(surface._X_COORD_MARKERS),
        "y_coord_markers": coord_words(surface._Y_COORD_MARKERS),
    }


def build_vocab_blob(wall_names: Sequence[str], flat_names: Sequence[str]) -> dict:
    """``doom_vocab.json`` contents: the frozen pretty table + identity card."""
    return {
        "screen": screen_config(),
        "n_rows": int(TOKEN_VOCAB.n_rows),
        "fingerprint": vocab_fingerprint(),
        "vocab": build_vocab(wall_names, flat_names),
    }


def export_bundle(
    save_directory: str, *, asset_config: AssetConfig | None = None
) -> list[str]:
    """Write a self-contained, torch-free tokenizer directory at
    ``save_directory``: the two frozen JSONs, the standalone
    ``tokenization_doom.py`` (copied by ``custom_object_save``), and the
    ``tokenizer_config.json`` wiring ``AutoTokenizer`` to it.

    Returns the list of written file paths.
    """
    # Imported here (not at module scope) so this module stays importable for the
    # in-process freeze checks without dragging the shipped class in early.
    from .tokenization_doom import DoomTokenizer as ShippedDoomTokenizer

    config = asset_config or DEFAULT_ASSET_CONFIG
    vocab_blob = build_vocab_blob(config.wall_names, config.flat_names)
    tables = build_tables()

    os.makedirs(save_directory, exist_ok=True)
    # Stage the artifacts, then let the shipped tokenizer's own save_pretrained
    # emit the final bundle — this routes custom_object_save through the
    # standalone class so the copied .py is tokenization_doom.py, not this file.
    with tempfile.TemporaryDirectory() as staging:
        vocab_path = (
            Path(staging) / ShippedDoomTokenizer.vocab_files_names["vocab_file"]
        )
        tables_path = (
            Path(staging) / ShippedDoomTokenizer.vocab_files_names["tables_file"]
        )
        vocab_path.write_text(json.dumps(vocab_blob))
        tables_path.write_text(json.dumps(tables))

        ShippedDoomTokenizer.register_for_auto_class("AutoTokenizer")
        tokenizer = ShippedDoomTokenizer(
            vocab_file=str(vocab_path), tables_file=str(tables_path)
        )
        saved = tokenizer.save_pretrained(save_directory)
    return list(saved)
