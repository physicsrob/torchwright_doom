"""Screen-sized Doom row-vocabulary identity helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..model.constants import COLUMN_COUNT, PIXEL_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH
from ..model.embedding import MODEL_VOCAB_SIZE
from ..model.tokens import FloatSlot, IntSlot
from ..model.vocab import VOCAB_TYPES


def screen_config() -> dict[str, int]:
    return {
        "width": int(SCREEN_WIDTH),
        "height": int(SCREEN_HEIGHT),
        "column_count": int(COLUMN_COUNT),
        "pixel_width": int(PIXEL_WIDTH),
    }


def vocab_fingerprint() -> str:
    signature: list[Any] = []
    for ttype in VOCAB_TYPES:
        slots: list[Any] = []
        for name, slot in ttype.slots.items():
            if isinstance(slot, IntSlot):
                slots.append([name, "int", slot.lo, slot.hi])
            elif isinstance(slot, FloatSlot):
                slots.append([name, "float", slot.lo, slot.hi, slot.levels])
        signature.append([ttype.name, slots])
    payload = json.dumps([signature, MODEL_VOCAB_SIZE], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
