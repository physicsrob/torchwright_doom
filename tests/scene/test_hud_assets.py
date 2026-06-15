"""Status-bar bake — real DOOM1.WAD anchors + faithful V_DrawPatch draw-list.

Pins the HUD patch bank against real game bytes (patch dimensions and the DOOM
picture offsets the blit relies on) and the E1M1 draw-list against the exact
``STlib_drawNum`` right-justification, so a regression in the bake or the
right-justify math is caught without a compiled render. The composite is the
pixel oracle's ground truth for the graph phase (Step 6).
"""

from __future__ import annotations

from torchwright_doom.hud_assets import (
    DrawCall,
    composite_bar,
    e1m1_draw_list,
    load_hud_patches,
)


def test_patch_bank_dimensions_and_offsets():
    bank = load_hud_patches()
    # (width, height, leftoffset, topoffset) straight from doom1.wad headers.
    assert (bank["STBAR"].width, bank["STBAR"].height) == (320, 32)
    assert (bank["STBAR"].leftoffset, bank["STBAR"].topoffset) == (0, 0)
    assert (bank["STTNUM0"].width, bank["STTNUM0"].height) == (14, 16)
    # "1" is narrower with a -1 leftoffset (centred in the 14-wide digit cell).
    assert (bank["STTNUM1"].width, bank["STTNUM1"].leftoffset) == (11, -1)
    assert (bank["STARMS"].width, bank["STARMS"].height) == (40, 32)
    assert (bank["STYSNUM2"].width, bank["STYSNUM2"].height) == (4, 6)
    # The face carries negative offsets, so V_DrawPatch shifts it DOWN+RIGHT
    # into the bar (148, 170) rather than above HUD_TOP.
    assert (bank["STFST01"].leftoffset, bank["STFST01"].topoffset) == (-5, -2)


def test_draw_list_right_justifies_numbers_faithfully():
    bank = load_hud_patches()
    calls = e1m1_draw_list(bank)

    # The plate is drawn first (painter order), at ST_X/ST_Y.
    assert calls[0] == DrawCall("STBAR", 0, 168)

    by_patch: dict[str, list[DrawCall]] = {}
    for c in calls:
        by_patch.setdefault(c.patch, []).append(c)

    # Ammo "50": digits emitted right-to-left from ST_AMMOX=44, advance 14
    # (STTNUM0 width). '0' at 44-14=30, '5' at 30-14=16.
    assert DrawCall("STTNUM0", 30, 171) in calls
    assert DrawCall("STTNUM5", 16, 171) in calls

    # Health "100%": percent at ST_HEALTHX=90, then 0@76, 0@62, 1@48.
    assert DrawCall("STTPRCNT", 90, 171) in calls
    assert DrawCall("STTNUM1", 48, 171) in calls

    # Armor "0%": the drawNum zero special-case draws a single 0 at 221-14=207.
    assert DrawCall("STTPRCNT", 221, 171) in calls
    assert DrawCall("STTNUM0", 207, 171) in calls


def test_draw_list_arms_face_and_ammo_counts():
    bank = load_hud_patches()
    calls = e1m1_draw_list(bank)

    # ARMS panel background then the six weapon numbers: "2" lit (pistol owned),
    # "3".."7" gray, in a 3x2 grid from (111,172) spaced (12,10).
    assert DrawCall("STARMS", 104, 168) in calls
    assert DrawCall("STYSNUM2", 111, 172) in calls  # lit pistol
    assert DrawCall("STGNUM3", 123, 172) in calls
    assert DrawCall("STGNUM5", 111, 182) in calls  # second row
    assert DrawCall("STGNUM7", 135, 182) in calls

    # Neutral face.
    assert DrawCall("STFST01", 143, 168) in calls

    # Right-panel counts (shortnum, advance 4): bullets 50/200 at x 288/314,
    # row y=173. '50' -> 0@284, 5@280; '200' -> 0@310, 0@306, 2@302.
    assert DrawCall("STYSNUM0", 284, 173) in calls
    assert DrawCall("STYSNUM5", 280, 173) in calls
    assert DrawCall("STYSNUM0", 310, 173) in calls
    assert DrawCall("STYSNUM2", 302, 173) in calls
    # CELL max is 300 on the bottom row (y=191, the am_cell remap slot).
    assert DrawCall("STYSNUM3", 302, 191) in calls


def test_composite_full_resolution_is_opaque_and_plate_backed():
    bar = composite_bar(1)
    assert bar.shape == (32, 320)
    # The plate covers the whole bar, so every cell is written (no -1).
    assert (bar < 0).sum() == 0
    # Top-left of the bar is the plate's top-left pixel (STBAR at 0,168).
    bank = load_hud_patches()
    assert bar[0, 0] == bank["STBAR"].pixels[0][0]
    # The ammo digits overwrite the plate: the ammo region differs from a bare
    # plate row beneath the digits (sanity that compositing actually happened).
    plate_only = bank["STBAR"].pixels
    digit_changed = any(
        bar[171 - 168, x] != plate_only[x][171 - 168] for x in range(16, 44)
    )
    assert digit_changed


def test_composite_half_resolution_decimates():
    bar = composite_bar(2)
    assert bar.shape == (16, 160)
    assert (bar < 0).sum() == 0
