"""Regression: the consume-driven drafter stays in sync through the
weapon + status-bar tail.

`ARDrafter.consume` advances the flat-pass plan pointer only for token types
in `_FLAT_SCAN_TYPES`. The weapon/bar scaffold tokens (`DRAW_PSPRITES_BEGIN`,
the weapon's re-asserted `SET_CURSOR_DIRECTION_Y`, `HUD_BEGIN`, every
`HUD_ITEM`) are literal steps in the flat plan's tail, so they must be in that
set too -- otherwise the plan pointer stalls and `next_draft` falls one step
further behind for each one, mispredicting almost everything from the weapon
onward (FINDINGS close-out #1: the weapon silently drafted at ~5% accept, the
bar at ~17%, until this was found).

The render stays correct regardless (spec-decode falls back to the model's own
token), so the symptom is a quiet speed regression that a token-match oracle
can't see. Here the self-feedback oracle is the probe: a stalled plan pointer
makes `next_draft` re-propose the same step forever, so **non-termination is
the failure**.

`HUD_ENABLED` is frozen at constants-import time from the env, so the check
runs in a fresh HUD-on process (the documented fresh-process pattern).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_SUBMODULE_ROOT = Path(__file__).resolve().parents[2]

_CHECK = textwrap.dedent("""
    import sys
    from pathlib import Path

    root = Path(sys.argv[1])
    from torchwright_doom.inference.config import load_render_config
    from torchwright_doom.inference.wad_scene import (
        load_render_scene, pose_from_world, pydoom_scene_for)
    from torchwright_doom.pydoom import drafter as D

    cfg = load_render_config(str(root / "configs" / "e1m1_lowres.yaml"))
    scene = load_render_scene(cfg, base_dir=root)
    pose = pose_from_world(scene)
    py_scene = pydoom_scene_for(scene, pose)
    py_pose = py_scene.test_poses[0]

    dr = D.ARDrafter(py_scene, py_pose)
    out = []
    CAP = 200000
    for _ in range(CAP):
        t = dr.next_draft()
        if t is None:
            break
        out.append(t)
        dr.consume(t)
    assert len(out) < CAP, "drafter did not terminate -- flat-scan desync loop"

    bar = D._statusbar_plan_tail()
    assert len(bar) > 1, "HUD must be on for this regression (bar tail empty)"
    tail = out[-len(bar):]
    assert all(
        a.type == b.type and a.values == b.values for a, b in zip(tail, bar)
    ), "drafter's bar tail diverged from _statusbar_plan_tail"
    print("OK tokens=%d bar=%d" % (len(out), len(bar)))
    """)


def test_drafter_resyncs_through_weapon_and_bar_hud_on():
    env = dict(os.environ)
    env["TORCHWRIGHT_DOOM_HUD"] = "1"
    env["TORCHWRIGHT_DOOM_RENDER_SCALE"] = "2"
    env["TORCHWRIGHT_DOOM_DETAIL"] = "low"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_SUBMODULE_ROOT), env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHECK, str(_SUBMODULE_ROOT)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"HUD-on drafter check failed\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "OK" in proc.stdout, proc.stdout
