"""Capture the exact bot_y_raw feeding yh at the boundary columns (21..30) for
the seg that marks floor plane (1,1). The flip is floor(bot_y_raw): 32 at x=23
(not occupied) vs 31 at x=24 (occupied). Show distance of bot_y_raw to int 32.

Hook _render_wall_columns by re-deriving bot_y_raw from the captured seg/meta;
we monkeypatch _append_plane_marks to grab the live seg context via a closure
recorded in _render_wall_columns. Simpler: monkeypatch math.floor inside the
module is fragile, so instead re-run the exact loop but key off the captured
(seg_idx) that marked the plane.
"""

from __future__ import annotations

import math
from pathlib import Path

CONFIG = "configs/e1m1.yaml"
X, Y, ANGLE, VIEWZ = 1056.0, -3616.0, 64, 41.0
TP, TVP = 1, 1


def main() -> int:
    from torchwright_doom.render.config import apply_screen_env, load_render_config
    from torchwright_doom.render.wad_scene import (
        load_render_scene, pose_from_world, sandbox_scene_for,
    )

    cfg = load_render_config(CONFIG)
    apply_screen_env(cfg)
    scene = load_render_scene(cfg, base_dir=str(Path(CONFIG).parent))
    pose = pose_from_world(scene, x=X, y=Y, angle=ANGLE, viewz=VIEWZ)
    sb_scene = sandbox_scene_for(scene, pose)
    state = sb_scene.test_poses[0]

    import doom_sandbox.implementation.reference as ref

    # Capture the seg_idx + x-range that marked floor plane (1,1).
    seg_hits = set()
    orig_marks = ref._append_plane_marks
    cur = {"seg_idx": None}

    # We need the seg_idx; _append_plane_marks doesn't get it. Patch
    # _render_wall_columns to stash the current record before calling marks.
    orig_render = ref._render_wall_columns

    def patched_render(*args, **kw):
        cur["seg_idx"] = kw["record"].seg_idx
        cur["record"] = kw["record"]
        cur["seg"] = kw["seg"]
        cur["meta"] = kw["meta"]
        return orig_render(*args, **kw)

    def patched_marks(*, plane_marks, x, yl, yh, ceilingclip, floorclip,
                      markceiling, markfloor, ceiling_plane_id, ceiling_vp,
                      floor_plane_id, floor_vp, runtime_visplanes):
        if floor_plane_id == TP and floor_vp == TVP and markfloor:
            seg_hits.add(cur["seg_idx"])
        return orig_marks(
            plane_marks=plane_marks, x=x, yl=yl, yh=yh,
            ceilingclip=ceilingclip, floorclip=floorclip,
            markceiling=markceiling, markfloor=markfloor,
            ceiling_plane_id=ceiling_plane_id, ceiling_vp=ceiling_vp,
            floor_plane_id=floor_plane_id, floor_vp=floor_vp,
            runtime_visplanes=runtime_visplanes,
        )

    ref._render_wall_columns = patched_render
    ref._append_plane_marks = patched_marks
    try:
        ref.expected_wall_plane_mark_pass(sb_scene, state)
    finally:
        ref._render_wall_columns = orig_render
        ref._append_plane_marks = orig_marks

    print(f"floor plane ({TP},{TVP}) marked by seg_idx(s): {sorted(seg_hits)}")

    # Re-derive bot_y_raw for that seg across its x range.
    md = sb_scene.map_data
    segments = ref.bake_segments(md)
    viewz = ref._state_viewz(state)
    contexts = ref._horizontal_wall_range_contexts(sb_scene, state)
    by_seg = {c.record.seg_idx: c for c in contexts}

    for sidx in sorted(seg_hits):
        context = by_seg[sidx]
        record = context.record
        seg = segments[sidx]
        meta = ref._drawseg_meta(seg, md.segs[sidx], record, state, viewz)
        worldbottom = seg.front_floor - viewz
        worldtop = seg.front_ceiling - viewz
        print(f"\nseg_idx={sidx} x1={record.x1} x2={record.x2} wall_kind={meta.wall_kind}")
        print(f"  front_floor={seg.front_floor} front_ceiling={seg.front_ceiling} viewz={viewz}")
        print(f"  worldbottom={worldbottom!r} worldtop={worldtop!r}")
        print(f"  scale1={meta.scale1!r} scalestep={meta.scalestep!r}")
        print(f"  CENTER_Y={ref.CENTER_Y}")
        print("  x | scale_x | bot_y_raw | floor() | dist_to_32 | dist_to_int")
        for x in range(record.x1, record.x2 + 1):
            if not (20 <= x <= 31):
                continue
            scale_x = meta.scale1 + (x - record.x1) * meta.scalestep
            bot_y_raw = ref.CENTER_Y - worldbottom * scale_x
            fl = math.floor(bot_y_raw)
            nearest = round(bot_y_raw)
            d32 = bot_y_raw - 32.0
            dint = bot_y_raw - nearest
            flag = "  <== boundary col" if x in (23, 24) else ""
            print(f"  {x:3d} | {scale_x:.9f} | {bot_y_raw:.10f} | {fl:3d} | "
                  f"{d32:+.3e} | {dint:+.3e}{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
