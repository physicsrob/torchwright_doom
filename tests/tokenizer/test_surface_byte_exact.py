"""Byte-exact round trip of the readable surface over the captured E1M1 trace.

The screen-sized vocab is built at import, so this runs in a fresh process
configured for the trace's resolution (the documented fresh-process pattern,
cf. ``tests/scene/test_drafter_resync.py``). The heavy lifting + density report
lives in ``roundtrip_check.py``; this asserts every leg round-trips and surfaces
the density numbers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SUBMODULE_ROOT = Path(__file__).resolve().parents[2]
_TRACES = _SUBMODULE_ROOT.parent / "blog" / "pieces" / "doom" / "vizkit" / "traces"

# The canonical production trace (configs/e1m1.yaml). The 160x100 trace is an
# older capture predating the PIXEL `w` slot, so it isn't byte-exact-reloadable.
_CONFIGS = {
    "320x200": ("320", "200"),
}


def _run(trace: Path, width: str, height: str) -> dict:
    env = dict(os.environ)
    env["TORCHWRIGHT_DOOM_SCREEN_WIDTH"] = width
    env["TORCHWRIGHT_DOOM_SCREEN_HEIGHT"] = height
    env["TORCHWRIGHT_DOOM_RENDER_SCALE"] = "2"
    env["TORCHWRIGHT_DOOM_DETAIL"] = "low"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_SUBMODULE_ROOT), env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-m", "tests.tokenizer.roundtrip_check", str(trace)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"surface round-trip failed at {width}x{height}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return json.loads(proc.stdout)


@pytest.mark.parametrize("label", list(_CONFIGS))
def test_surface_round_trips_trace(label: str) -> None:
    width, height = _CONFIGS[label]
    trace = _TRACES / f"e1m1_{label}.token_dump.json.gz"
    if not trace.exists():
        pytest.skip(f"trace not present: {trace}")
    report = _run(trace, width, height)

    assert report["ok"], json.dumps(report["legs"], indent=2)
    for name, leg in report["legs"].items():
        assert leg["ok"], f"{name}: {leg.get('detail')}"

    # Density is informational, but pin the shape so a regression that collapses
    # every token onto one line (or one-per-line) is visible.
    density = report["density"]
    assert density["prompt"]["tokens"] > 1000
    assert 1.0 < density["prompt"]["tokens_per_line"] < 60.0
    assert density["rollout"]["tokens_per_line"] > 1.0
    print(f"\n[{label}] density: {json.dumps(density)}")
