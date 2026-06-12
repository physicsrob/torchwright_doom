"""Hook the canonical pass to capture exactly which marks land on plane (1,1),
and the yl/yh that produced them, by monkeypatching _append_plane_mark and
_append_plane_marks. Then reconstruct the raw projection for the boundary col.
"""

from __future__ import annotations

import math
from pathlib import Path

CONFIG = "configs/e1m1.yaml"
X, Y, ANGLE, VIEWZ = 1056.0, -3616.0, 64, 41.0
TP, TVP = 1, 1


def main() -> int:
    from torchwright_doom.inference.config import apply_screen_env, load_render_config
    from torchwright_doom.inference.wad_scene import (
        load_render_scene,
        pose_from_world,
        sandbox_scene_for,
    )

    cfg = load_render_config(CONFIG)
    apply_screen_env(cfg)
    scene = load_render_scene(cfg, base_dir=str(Path(CONFIG).parent))
    pose = pose_from_world(scene, x=X, y=Y, angle=ANGLE, viewz=VIEWZ)
    sb_scene = sandbox_scene_for(scene, pose)
    state = sb_scene.test_poses[0]

    import doom_sandbox.implementation.reference as ref

    captured = []  # (x, yl, yh, cc, fc, markceiling, markfloor, cpid, cvp, fpid, fvp)
    orig_marks = ref._append_plane_marks

    def patched_marks(
        *,
        plane_marks,
        x,
        yl,
        yh,
        ceilingclip,
        floorclip,
        markceiling,
        markfloor,
        ceiling_plane_id,
        ceiling_vp,
        floor_plane_id,
        floor_vp,
        runtime_visplanes,
    ):
        if (ceiling_plane_id == TP and ceiling_vp == TVP) or (
            floor_plane_id == TP and floor_vp == TVP
        ):
            captured.append(
                dict(
                    x=x,
                    yl=yl,
                    yh=yh,
                    cc=ceilingclip,
                    fc=floorclip,
                    markceiling=markceiling,
                    markfloor=markfloor,
                    cpid=ceiling_plane_id,
                    cvp=ceiling_vp,
                    fpid=floor_plane_id,
                    fvp=floor_vp,
                )
            )
        return orig_marks(
            plane_marks=plane_marks,
            x=x,
            yl=yl,
            yh=yh,
            ceilingclip=ceilingclip,
            floorclip=floorclip,
            markceiling=markceiling,
            markfloor=markfloor,
            ceiling_plane_id=ceiling_plane_id,
            ceiling_vp=ceiling_vp,
            floor_plane_id=floor_plane_id,
            floor_vp=floor_vp,
            runtime_visplanes=runtime_visplanes,
        )

    ref._append_plane_marks = patched_marks
    try:
        wall_pass = ref.expected_wall_plane_mark_pass(sb_scene, state)
    finally:
        ref._append_plane_marks = orig_marks

    columns = ref._runtime_visplane_columns(wall_pass.plane_marks)
    table = columns.get((TP, TVP))

    print(f"plane ({TP},{TVP}) capture: {len(captured)} columns marked")
    print("col-by-col yl/yh that fed this plane's marks:")
    for c in sorted(captured, key=lambda d: d["x"]):
        which = []
        if c["cpid"] == TP and c["cvp"] == TVP and c["markceiling"]:
            cm_top = max(0, c["cc"] + 1)
            cm_bot = c["yl"] - 1
            if cm_bot >= c["fc"]:
                cm_bot = c["fc"] - 1
            cm_bot = min(ref.SCREEN_HEIGHT - 1, cm_bot)
            which.append(f"CEIL(top={cm_top},bot={cm_bot},occ={cm_top<=cm_bot})")
        if c["fpid"] == TP and c["fvp"] == TVP and c["markfloor"]:
            fm_top = c["yh"] + 1
            if fm_top <= c["cc"]:
                fm_top = c["cc"] + 1
            fm_top = max(0, fm_top)
            fm_bot = min(ref.SCREEN_HEIGHT - 1, c["fc"] - 1)
            which.append(f"FLOOR(top={fm_top},bot={fm_bot},occ={fm_top<=fm_bot})")
        print(
            f"  x={c['x']:3d} yl={c['yl']:3d} yh={c['yh']:3d} cc={c['cc']:3d} fc={c['fc']:3d} "
            f"mc={int(c['markceiling'])} mf={int(c['markfloor'])}  {' '.join(which)}"
        )

    print(f"\nfinal table (occupied cols x: top..bottom):")
    if table:
        for x, (t, b) in enumerate(table):
            if t <= b:
                print(f"  x={x:3d} top={t:3d} bottom={b:3d}")
        occ_xs = [x for x, (t, b) in enumerate(table) if t <= b]
        print(f"  minx={min(occ_xs)} maxx={max(occ_xs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
