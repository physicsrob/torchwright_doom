"""``DoomTokenizer`` HF round-trip, run in a fresh 320x200 process.

The screen-sized vocab is import-time, so this mirrors the fresh-process pattern
of ``test_surface_byte_exact.py``. The assertions live in ``hf_check.py``; this
wrapper runs them at the trace's config and surfaces per-check failures.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SUBMODULE_ROOT = Path(__file__).resolve().parents[2]
_TRACE = (
    _SUBMODULE_ROOT.parent
    / "blog"
    / "pieces"
    / "doom"
    / "vizkit"
    / "traces"
    / "e1m1_320x200.token_dump.json.gz"
)

pytest.importorskip("transformers")


def test_doom_tokenizer_round_trip() -> None:
    if not _TRACE.exists():
        pytest.skip(f"trace not present: {_TRACE}")
    env = dict(os.environ)
    env["TORCHWRIGHT_DOOM_SCREEN_WIDTH"] = "320"
    env["TORCHWRIGHT_DOOM_SCREEN_HEIGHT"] = "200"
    env["TORCHWRIGHT_DOOM_RENDER_SCALE"] = "2"
    env["TORCHWRIGHT_DOOM_DETAIL"] = "low"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_SUBMODULE_ROOT), env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-m", "tests.tokenizer.hf_check", str(_TRACE)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"DoomTokenizer check failed\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    report = json.loads(proc.stdout)
    failed = [name for name, ok in report["checks"].items() if not ok]
    assert not failed, f"failed checks: {failed}\n{json.dumps(report, indent=2)}"
