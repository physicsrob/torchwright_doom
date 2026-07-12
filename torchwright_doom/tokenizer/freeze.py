"""Freeze tokenizer, formatter, and decoder identity into data-only JSON.

This runs in the full renderer environment and writes the JSON artifacts used
by the stock tokenizer and standalone post-processing examples:

* ``doom_vocab.json`` — the **pretty** ``{display_label: id}`` table (WAD texture
  names + decoded enums/bools/BSP-child-ids/bbox-codes baked into the label
  strings via :func:`display.token_label`) plus an identity card (screen config,
  fingerprint, n_rows). The display knobs are baked into the labels here so the
  shipped kernel never ships the display layer.
* ``doom_tables.json`` — the one thing that combines two ids and so can't be
  pre-baked: the **carrier-fold rule** data (each marker's value-range bounds,
  the angle markers, the back-height sentinel, the carrier id ranges, the x/y
  coordinate-marker sets, ``ANGLE_BAM``).

No Transformers auto class points at local Python code. The files are consumed
explicitly by ``DoomFormatter`` and ``txt_to_png.py`` after model inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..asset_config import DEFAULT_ASSET_CONFIG, AssetConfig
from ..embedding import TOKEN_VOCAB
from ..marker_ranges import ANGLE_MARKERS, MARKER_RANGE
from ..tokens import FloatSlot, IntSlot
from ..value_ranges import VALUE_RANGES
from ..vocab import ANGLE_BAM, ANGLE_VALUE, BACK_HEIGHT_SENTINEL, VALUE, VOCAB_TYPES
from . import display, surface
from .display import DISPLAY_NAME, token_label, token_word
from .identity import screen_config, vocab_fingerprint

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
    """``doom_vocab.json`` contents for both tokenizer and formatter identity.

    ``vocab`` remains during the migration for the retired custom tokenizer.
    New consumers use the ordered ``words`` / ``labels`` arrays and the explicit
    row records, whose positions are the model ids.
    """
    labels = [
        token_label(ttype, values, wall_names=wall_names, flat_names=flat_names)
        for ttype, values in TOKEN_VOCAB.row_to_token
    ]
    words = [
        token_word(ttype, values, wall_names=wall_names, flat_names=flat_names)
        for ttype, values in TOKEN_VOCAB.row_to_token
    ]
    if len(set(labels)) != len(labels) or len(set(words)) != len(words):
        raise ValueError("frozen Doom vocabulary labels and words must be injective")
    return {
        "format": "torchwright_doom.vocab.v2",
        "screen": screen_config(),
        "n_rows": int(TOKEN_VOCAB.n_rows),
        "fingerprint": vocab_fingerprint(),
        "vocab": {label: row for row, label in enumerate(labels)},
        "canonical_vocab": {word: row for row, word in enumerate(words)},
        "words": words,
        "labels": labels,
        "rows": [
            {
                "row": row,
                "type": ttype.name,
                "values": dict(values),
                "word": words[row],
                "label": labels[row],
            }
            for row, (ttype, values) in enumerate(TOKEN_VOCAB.row_to_token)
        ],
    }


def write_frozen_data(
    save_directory: str | Path,
    *,
    asset_config: AssetConfig | None = None,
    palette: Sequence[Sequence[int]],
    origin: tuple[float, float] = (0.0, 0.0),
) -> list[Path]:
    """Write the data-only formatter and frame-decoder inputs."""
    config = asset_config or DEFAULT_ASSET_CONFIG
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)

    vocab_blob = build_vocab_blob(config.wall_names, config.flat_names)
    tables = build_tables()
    tables.update(
        {
            "format": "torchwright_doom.tables.v2",
            "origin": [float(origin[0]), float(origin[1])],
            "wall_names": list(config.wall_names),
            "flat_names": list(config.flat_names),
            "header_levels": dict(surface.HEADER_DISPLAY_LEVEL),
            "layout": {"indent_unit": 2, "field_indent": 4},
        }
    )
    palette_blob = {
        "format": "torchwright_doom.palette.v1",
        "colors": [[int(c) for c in rgb] for rgb in palette],
    }
    if len(palette_blob["colors"]) != 256:
        raise ValueError("Doom palette must contain exactly 256 RGB entries")
    if any(len(rgb) != 3 for rgb in palette_blob["colors"]):
        raise ValueError("every Doom palette entry must be RGB")

    paths = [
        directory / "doom_vocab.json",
        directory / "doom_tables.json",
        directory / "doom_palette.json",
    ]
    for path, payload in zip(paths, (vocab_blob, tables, palette_blob), strict=True):
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return paths
