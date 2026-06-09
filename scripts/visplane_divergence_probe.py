"""Pin the first STRUCTURAL visplane span divergence in the d8192 render.

Replays the d8192 emitted stream through the ARDrafter, finds the first long
mispredict run that lives inside a visplane's R_MakeSpans.col token run, and
reports:
  - which (plane_id, vp) visplane is being decomposed
  - the canonical R_MakeSpans.col / closeSlot / row token run for that visplane
  - the model's emitted token run for the same visplane
  - the canonical per-column (top[x], bottom[x]) coverage table for that plane
  - for the column whose membership flipped, the underlying top_y_raw/bot_y_raw
    that feeds yl/yh, and how close it sits to the integer ceil/floor boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG = "configs/e1m1_d8192_h16384.yaml"
DUMP = "out/render_d8192/token_dump.json"
X, Y, ANGLE, VIEWZ = 1056.0, -3616.0, 64, 41.0


def main() -> int:
    from torchwright_doom.render.config import apply_screen_env, load_render_config
    from torchwright_doom.render.wad_scene import (
        load_render_scene,
        pose_from_world,
        sandbox_scene_for,
    )

    cfg = load_render_config(CONFIG)
    apply_screen_env(cfg)
    scene = load_render_scene(cfg, base_dir=str(Path(CONFIG).parent))
    pose = pose_from_world(scene, x=X, y=Y, angle=ANGLE, viewz=VIEWZ)
    sb_scene = sandbox_scene_for(scene, pose)
    sb_pose = sb_scene.test_poses[0]

    from doom_sandbox.api.tokens import Token
    from doom_sandbox.implementation.reference_drafter import ARDrafter
    from torchwright_doom.render.tokens_bridge import _sandbox_types, sandbox_token_to_row

    sb_types = _sandbox_types()
    drafter = ARDrafter(sb_scene, sb_pose)

    stream = json.loads(Path(DUMP).read_text())["cases"][0]["predicted_next_tokens"]

    def to_tok(rec: dict) -> Token:
        return Token(sb_types[rec["type"]], dict(rec["values"]))

    # ---- replay, record per-position match flag + which actual token ----
    mispredict: list[bool] = []
    drafts: list[Token | None] = []
    for rec in stream:
        actual = to_tok(rec)
        draft = drafter.next_draft()
        drafts.append(draft)
        if draft is None:
            mispredict.append(True)
        else:
            try:
                matched = sandbox_token_to_row(draft) == sandbox_token_to_row(actual)
            except Exception:
                matched = False
            mispredict.append(not matched)
        drafter.consume(actual)

    # ---- find the FIRST long (>=20) mispredict run ----
    runs = []
    i = 0
    n = len(mispredict)
    while i < n:
        if mispredict[i]:
            j = i
            while j < n and mispredict[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    long_runs = [(a, b) for (a, b) in runs if b - a >= 20]
    print(f"total mispredict runs: {len(runs)}; long (>=20): {len(long_runs)}")
    if not long_runs:
        print("no long run found")
        return 1
    a, b = long_runs[0]
    print(f"FIRST long mispredict run: stream idx [{a}, {b}) length={b - a}")
    print(f"  model token at idx {a-1}: {stream[a-1]['type']} {stream[a-1].get('values')}")
    print(f"  model token at idx {a}:   {stream[a]['type']} {stream[a].get('values')}")

    # ---- find the visplaneBegin that this run lives inside (walk back) ----
    vp_begin_idx = None
    for k in range(a, -1, -1):
        if stream[k]["type"] == "visplaneBegin":
            vp_begin_idx = k
            break
    if vp_begin_idx is None:
        print("no visplaneBegin before the run")
    else:
        vpv = stream[vp_begin_idx]["values"]
        print(f"  enclosing visplaneBegin at idx {vp_begin_idx}: p={vpv.get('p')} vp={vpv.get('vp')}")

    # ---- find the next visplaneBegin / nextVp boundary after the run start ----
    vp_end_idx = None
    for k in range(a, n):
        if stream[k]["type"] in ("visplaneBegin", "R_DrawPlanes.nextVp", "R_DrawPlanes.nextPlane"):
            if k > (vp_begin_idx or -1):
                vp_end_idx = k
                break

    # ---- the MODEL's emitted col run for this visplane ----
    lo = (vp_begin_idx if vp_begin_idx is not None else a)
    hi = vp_end_idx if vp_end_idx is not None else min(a + (b - a) + 5, n)
    print("\n--- MODEL emitted tokens for this visplane (idx, type, values) ---")
    model_cols = []
    for k in range(lo, hi + 1):
        if k >= n:
            break
        t = stream[k]["type"]
        v = stream[k].get("values")
        if t in ("R_MakeSpans.col", "R_MakeSpans.closeSlot", "R_MapPlane.row",
                 "setCursorX", "setCursorY", "visplaneBegin", "R_DrawPlanes.nextVp"):
            print(f"   [{k}] {t} {v}")
            if t == "R_MakeSpans.col":
                model_cols.append(int(v["x"]))
    print(f"   MODEL R_MakeSpans.col x values: {model_cols}")

    # ---- canonical flat pass: compute the table + the canonical token run ----
    import doom_sandbox.implementation.reference as ref
    from doom_sandbox.implementation.reference_drafter import _flat_pass_tokens

    md = sb_scene.map_data
    state = sb_pose
    wall_pass = ref.expected_wall_plane_mark_pass(sb_scene, state)
    columns = ref._runtime_visplane_columns(wall_pass.plane_marks)

    canon = _flat_pass_tokens(sb_scene, state)
    # find canonical visplaneBegin for same (p, vp)
    pid = vpv.get("p")
    vpn = vpv.get("vp")
    print(f"\n--- CANONICAL flat tokens for visplane p={pid} vp={vpn} ---")
    canon_cols = []
    inside = False
    for tok in canon:
        ty = tok.type.name
        if ty == "visplaneBegin":
            inside = (int(tok.values["p"]) == int(pid) and int(tok.values["vp"]) == int(vpn))
        if inside and ty in ("visplaneBegin", "R_MakeSpans.col", "R_MakeSpans.closeSlot",
                              "R_MapPlane.row", "setCursorX", "setCursorY", "R_DrawPlanes.nextVp"):
            print(f"   {ty} {tok.values}")
            if ty == "R_MakeSpans.col":
                canon_cols.append(int(tok.values["x"]))
            if ty == "R_DrawPlanes.nextVp" and inside:
                inside = False
    print(f"   CANONICAL R_MakeSpans.col x values: {canon_cols}")

    # ---- the canonical (top, bottom) coverage table for this visplane ----
    table = columns.get((int(pid), int(vpn)))
    if table is None:
        print(f"\nNo canonical table for (p={pid}, vp={vpn})!  keys near: "
              f"{[k for k in columns if k[0] == int(pid)]}")
        return 0
    used = [(x, t, btm) for x, (t, btm) in enumerate(table) if t <= btm]
    print(f"\n--- CANONICAL coverage table (p={pid}, vp={vpn}); occupied columns (x: top..bottom) ---")
    for x, t, btm in used:
        print(f"   x={x:3d}  top={t:3d}  bottom={btm:3d}")

    # ---- diff the col sets ----
    model_set = set(model_cols)
    canon_set = set(canon_cols)
    extra = sorted(model_set - canon_set)
    missing = sorted(canon_set - model_set)
    print(f"\n--- COL-RUN DIFF (model vs canonical) ---")
    print(f"   extra in MODEL  (model emitted, canonical didn't): {extra}")
    print(f"   missing in MODEL (canonical had, model didn't):    {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
