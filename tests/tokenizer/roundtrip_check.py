"""Byte-exact round-trip + density report over a captured token trace.

Run as a module under the trace's screen config (the screen-sized vocab is
import-time, so the byte-exact check must run in a process configured for the
trace's resolution — the documented fresh-process pattern):

    TORCHWRIGHT_DOOM_SCREEN_WIDTH=320 TORCHWRIGHT_DOOM_SCREEN_HEIGHT=200 \\
    TORCHWRIGHT_DOOM_RENDER_SCALE=2 TORCHWRIGHT_DOOM_DETAIL=low \\
    python -m tests.tokenizer.roundtrip_check <trace.json.gz>

Prints a JSON report and exits non-zero if any leg fails to round-trip. The
property proven: reconstruct the integer ``W_EMBED`` rows from the trace, then
``parse(render(rows)) == rows`` — re-encoding the readable surface reproduces
the identical id stream, for the prompt *and* the full rollout, under every
display knob (none of which change the stream).
"""

from __future__ import annotations

import gzip
import json
import sys

from torchwright_doom.asset_config import DEFAULT_ASSET_CONFIG
from torchwright_doom.inference.tokens_bridge import row_to_token, token_to_row
from torchwright_doom.tokens import Token
from torchwright_doom.tokenizer import surface
from torchwright_doom.value_ranges import ValueRange, decode_float
from torchwright_doom.vocab import VOCAB_TYPES

_BY_NAME = {t.name: t for t in VOCAB_TYPES}


def _trace_rows(trace_tokens: list[dict]) -> list[int]:
    """Integer W_EMBED rows for a captured token list (``type`` strings are
    ``TokenType.name``; reconstruct the native Token, then encode)."""
    return [
        token_to_row(Token(_BY_NAME[t["type"]], dict(t["values"])))
        for t in trace_tokens
    ]


def _roundtrip(rows: list[int], **knobs) -> tuple[bool, str]:
    tokens = [row_to_token(r) for r in rows]
    text = surface.render(tokens, **knobs)
    back = surface.parse(text, **knobs)
    rows2 = [token_to_row(Token(ttype, dict(values))) for ttype, values in back]
    if rows2 == rows:
        return True, text
    for i, (a, b) in enumerate(zip(rows2, rows)):
        if a != b:
            return False, f"first mismatch at {i}: got row {a} want {b}"
    return False, f"length differs: got {len(rows2)} want {len(rows)}"


def _origin_from_pose(prompt_rows: list[int], player_world: tuple[float, float]):
    """Recover the subset centroid: raw_world = scene_relative + centroid."""
    toks = [row_to_token(r) for r in prompt_rows[:4]]
    rel_x = decode_float(ValueRange.R1, toks[1].values["v"])
    rel_y = decode_float(ValueRange.R1, toks[3].values["v"])
    return (player_world[0] - rel_x, player_world[1] - rel_y)


def _density(rows: list[int], **knobs) -> dict:
    text = surface.render([row_to_token(r) for r in rows], **knobs)
    n_lines = text.count("\n") + 1 if text else 0
    return {
        "tokens": len(rows),
        "lines": n_lines,
        "tokens_per_line": round(len(rows) / n_lines, 2) if n_lines else 0,
        "tokens_per_dozen_lines": round(12 * len(rows) / n_lines, 1) if n_lines else 0,
    }


def run(trace_path: str) -> dict:
    with gzip.open(trace_path) as fh:
        case = json.load(fh)["cases"][0]
    prompt_rows = _trace_rows(case["prefill_input_tokens"])
    rollout_rows = _trace_rows(case["rollout_output_tokens"])
    pose = case["pose"]
    origin = _origin_from_pose(prompt_rows, (pose["x"], pose["y"]))

    names = dict(
        wall_names=DEFAULT_ASSET_CONFIG.wall_names,
        flat_names=DEFAULT_ASSET_CONFIG.flat_names,
    )
    # The full human-facing figure config: stripped entity prefixes + decoded
    # values (enums, booleans, BSP child ids, bbox region code, sentinels) on
    # top of WAD names, degrees, and WAD coords. The hardest round-trip.
    friendly = {
        **names,
        "strip_prefixes": True,
        "decode_values": True,
        "angle_degrees": True,
    }
    legs = {
        "prompt_scene_relative": (prompt_rows, names),
        "prompt_wad_coords": (prompt_rows, {**names, "origin": origin}),
        "prompt_degrees": (prompt_rows, {**names, "angle_degrees": True}),
        "prompt_raw_carrier": (prompt_rows, {"physical_values": False}),
        "prompt_friendly": (prompt_rows, {**friendly, "origin": origin}),
        "rollout_scene_relative": (rollout_rows, names),
        "rollout_degrees": (rollout_rows, {**names, "angle_degrees": True}),
        "rollout_friendly": (rollout_rows, friendly),
    }
    results = {}
    ok = True
    for name, (rows, knobs) in legs.items():
        passed, detail = _roundtrip(rows, **knobs)
        ok = ok and passed
        results[name] = {"ok": passed, **({} if passed else {"detail": detail})}

    return {
        "ok": ok,
        "legs": results,
        "density": {
            "prompt": _density(prompt_rows, **names),
            "rollout": _density(rollout_rows, **names),
        },
        "origin": list(origin),
    }


def main() -> int:
    report = run(sys.argv[1])
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
