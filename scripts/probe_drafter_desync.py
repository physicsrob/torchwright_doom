"""Measure the drafter's flat-scan re-sync on the weapon+bar tail.

The runtime feeds the MODEL's emitted token back into ``ARDrafter.consume``;
``next_draft`` is the spec-decode proposal, and a token is *accepted* when the
proposal equals the model's token. This script reproduces that loop on the
deterministic flat+weapon+bar tail and reports the accept rate per phase.

Ground truth (= what the model emits) is ``_build_flat_plan`` flattened with
its visplane spans drained correctly. The drafter under test is a fresh
``_FlatScanState`` walked with the *exact* routing condition from
``ARDrafter.consume`` (only advance the plan when the token's type is in
``_FLAT_SCAN_TYPES``). A scaffold token whose type is missing from that set
never advances the plan, so the proposal lags one step further behind from
that token onward -- visible here as a collapse in the per-phase accept rate.

Run (CPU, HUD on):

    TORCHWRIGHT_DOOM_HUD=1 TORCHWRIGHT_DOOM_RENDER_SCALE=2 \
    TORCHWRIGHT_DOOM_DETAIL=low PYTHONPATH=$PWD \
    /data/torchdoom/.venv/bin/python scripts/probe_drafter_desync.py
"""

from torchwright_doom.inference.config import load_render_config
from torchwright_doom.inference.wad_scene import (
    load_render_scene,
    pose_from_world,
    pydoom_scene_for,
)
from torchwright_doom.pydoom import drafter as D


def main() -> None:
    cfg = load_render_config("configs/e1m1_lowres.yaml")
    scene = load_render_scene(cfg)
    pose = pose_from_world(scene)
    py_scene = pydoom_scene_for(scene, pose)
    py_pose = py_scene.test_poses[0]

    # Ground truth: the plan flattened, spans drained the same way
    # _FlatScanState would drain them.
    plan = D._build_flat_plan(py_scene, py_pose)
    gt = []
    for step in plan:
        if isinstance(step, D._VisplaneSpanState):
            while True:
                t = step.next_token()
                if t is None:
                    break
                gt.append(t)
                step.consume(t)
        else:
            gt.append(step)

    # Simulate the runtime loop with ARDrafter.consume's exact routing.
    fs = D._FlatScanState(py_scene, py_pose)
    phase = "flat"
    stats = {"flat": [0, 0], "weapon": [0, 0], "bar": [0, 0]}
    for tok in gt:
        if tok.type == D.DRAW_PSPRITES_BEGIN:
            phase = "weapon"
        if tok.type == D.HUD_BEGIN:
            phase = "bar"
        prop = fs.next_token()
        match = prop is not None and prop.type == tok.type and prop.values == tok.values
        stats[phase][1] += 1
        if match:
            stats[phase][0] += 1
        # ARDrafter.consume line 2806: advance the flat scan only for these types.
        if tok.type in D._FLAT_SCAN_TYPES | {D.DONE}:
            fs.consume(tok)

    print(f"ground-truth tokens: {len(gt)}")
    print(
        f"  HUD_BEGIN x{sum(1 for t in gt if t.type == D.HUD_BEGIN)}, "
        f"HUD_ITEM x{sum(1 for t in gt if t.type == D.HUD_ITEM)}"
    )
    for ph, (a, n) in stats.items():
        pct = (100.0 * a / n) if n else 0.0
        print(f"  {ph:8s} accept {a:6d}/{n:<6d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
