"""Write the K output artifact: ``token_dump.json`` (the PNGs live in ``compare``).

The dump records the *generated* token stream — the single source of truth for
what the model rendered. Schema mirrors the ``doom_sandbox`` rollout dumps
(``rollout_cache.case_json_dump``) closely enough that ``viewer.html`` can
reconstruct the frame from it: ``predicted_next_tokens`` carries the rollout
entries (``phase == "rollout"``), each with its palette/cursor slots, and the
viewer walks them exactly as the host decode does. No precomputed pixel array —
the tokens are the source of truth (matching the sandbox dump convention).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..constants import SCREEN_HEIGHT, SCREEN_WIDTH
from ..embedding import TOKEN_VOCAB


def _row_entry(pos: int, row: int, phase: str) -> dict[str, Any]:
    rtype, values = TOKEN_VOCAB.row_to_token[row]
    vals = dict(values)
    if vals:
        parts = [
            f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in vals.items()
        ]
        text = f"{rtype.name}({','.join(parts)})"
    else:
        text = rtype.name
    return {"pos": pos, "type": rtype.name, "values": vals, "text": text, "phase": phase}


def build_token_dump(
    *,
    fixture: str,
    pose_index: int,
    pose: dict[str, Any],
    prefill_rows: list[int],
    emitted_rows: list[int],
    mode: str,
    spec_decode_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the K ``token_dump`` payload (one case)."""
    n_prefill = len(prefill_rows)
    prefill_entries = [_row_entry(i, r, "prefill") for i, r in enumerate(prefill_rows)]
    rollout_entries = [
        _row_entry(n_prefill + i, r, "rollout") for i, r in enumerate(emitted_rows)
    ]
    case: dict[str, Any] = {
        "label": f"{fixture}__pose{pose_index:02d}",
        "fixture": fixture,
        "pose_index": pose_index,
        "pose": pose,
        "mode": mode,
        "counts": {
            "prefill_positions": n_prefill,
            "rollout_positions": len(emitted_rows),
            "predicted_positions": n_prefill + len(emitted_rows),
        },
        "prefill_input_tokens": prefill_entries,
        "predicted_next_tokens": rollout_entries,
        "rollout_output_tokens": rollout_entries,
    }
    if spec_decode_stats is not None:
        case["spec_decode_stats"] = spec_decode_stats
    return {
        "schema_version": 1,
        "implementation": "torchwright_doom_compiled",
        "screen": {"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT},
        "cases": [case],
    }


def write_token_dump(path, dump: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
