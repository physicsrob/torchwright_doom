"""Fetch the reference render and report image-level agreement; write PNGs.

Success is image-level, not token-exact — the generated stream can diverge
benignly (off-by-one edges change pixel counts), so this module **reports
rich stats and renders PNGs**; it fixes no threshold. It is the standing
report-only correctness gate (``make run COMPARE=1``); the production
render's measured scores live in ``FACTS.md``.

The in-tree ``pydoom`` renderer is imported lazily inside the reference helpers,
so the stats/PNG helpers that take buffers load without touching it.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..model.constants import PIXEL_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH

Rgb = tuple[int, int, int]
PixelBuf = dict[tuple[int, int], Rgb]


# --- reference render (ground truth) --------------------------------------


def _weapon_reference_cells() -> list[tuple[int, int, Rgb]]:
    """Opaque player-weapon cells as ``(x, y, rgb)``, width-expanded to screen
    cells; empty when the HUD is off.

    Baked from ``configured_weapon_bake`` — the SAME bake the graph banks and the
    pydoom token reference use — so the weapon is consistent across all three.
    The baked value is a *lit* palette index; PLAYPAL maps it to RGB exactly as
    the host does for an emitted PIXEL token."""
    from ..model.assets.asset_banks import PLAYPAL, configured_weapon_bake
    from ..model.assets.weapon_assets import WEAPON_TRANSPARENT

    bake = configured_weapon_bake()
    if bake is None:
        return []
    cells: list[tuple[int, int, Rgb]] = []
    for col in range(bake.min_col, bake.max_col + 1):
        for row in range(bake.top, bake.bottom + 1):
            value = float(bake.table[row - bake.top, col - bake.min_col])
            if value >= WEAPON_TRANSPARENT - 0.5:
                continue
            r, g, b = PLAYPAL[int(value)]
            rgb = (int(r), int(g), int(b))
            for k in range(PIXEL_WIDTH):
                cells.append((col * PIXEL_WIDTH + k, row, rgb))
    return cells


def _hud_reference_cells() -> list[tuple[int, int, Rgb]]:
    """Opaque status-bar cells as ``(x, y, rgb)`` in painter (draw-list) order;
    empty when the HUD is off.

    Composited from ``configured_hud_draw_list`` + ``configured_hud_bake`` — the
    SAME tables the compiled graph's HUD pass and the pydoom token reference
    walk — so the bar is consistent across all three. The bar paints at native screen resolution
    (one cell per column, ``w = 1``, unlike the doubled weapon/3D view). Cells are
    emitted in draw-list order, so the consumer's last-write-wins puts each widget
    over the plate beneath it (DOOM's painter order)."""
    from ..model.assets.asset_banks import (
        PLAYPAL,
        configured_hud_bake,
        configured_hud_draw_list,
    )
    from ..model.assets.hud_assets import HUD_TRANSPARENT

    draw_list = configured_hud_draw_list()
    bake = configured_hud_bake()
    if draw_list is None or bake is None:
        return []
    cells: list[tuple[int, int, Rgb]] = []
    for i in range(draw_list.n_items):
        patch_id = draw_list.patch_id[i]
        ox, oy = draw_list.origin_x[i], draw_list.origin_y[i]
        base = bake.base_rows[patch_id]
        for col in range(draw_list.width[i]):
            for v in range(draw_list.height[i]):
                value = float(bake.table[base + v, col])
                if value >= HUD_TRANSPARENT - 0.5:
                    continue
                r, g, b = PLAYPAL[int(value)]
                cells.append((ox + col, oy + v, (int(r), int(g), int(b))))
    return cells


def reference_pixels(scene, pose) -> PixelBuf:
    """Reference frame as ``{(x, y): rgb}`` (first emission at each pixel wins).

    Low-detail: ``expected_pixel_pass`` emits one sample per rendered column at
    its left-edge screen x (``col * PIXEL_WIDTH``). Mirror the host blit
    (:func:`..decode.decode_rows_to_pixels`, which paints ``w`` cells in +X) by
    filling all ``PIXEL_WIDTH`` cells, so the reference is the true 320-wide
    frame, not a 1-wide comb. No-op at high-detail (``PIXEL_WIDTH == 1``).

    The player weapon is drawn last (DOOM's painter order), so its opaque cells
    OVERWRITE the 3D scene; transparent gaps leave the 3D pixel showing through.
    HUD off: no weapon, frame unchanged."""
    from ..pydoom import expected_pixel_pass

    out: PixelBuf = {}
    for p in expected_pixel_pass(scene, pose):
        rgb = (int(p.color[0]), int(p.color[1]), int(p.color[2]))
        for k in range(PIXEL_WIDTH):
            out.setdefault((int(p.x) + k, int(p.y)), rgb)
    for x, y, rgb in _weapon_reference_cells():
        out[(x, y)] = rgb  # last-write-wins: weapon over the 3D scene
    for x, y, rgb in _hud_reference_cells():
        out[(x, y)] = rgb  # the status bar paints its own (bottom) rows
    return out


def reference_options(scene, pose) -> dict[tuple[int, int], set[Rgb]]:
    """Per-pixel accepted-color option sets (texture-neighborhood + lighting tol).

    Width-expanded ``PIXEL_WIDTH`` cells wide to match the host blit (see
    :func:`reference_pixels`); no-op at high-detail. At weapon cells the model
    paints the baked weapon color exactly, so the option set there is that single
    color (replacing any 3D option underneath)."""
    from ..pydoom import expected_pixel_color_options

    opts: dict[tuple[int, int], set[Rgb]] = {}
    for o in expected_pixel_color_options(scene, pose):
        colors = {(int(r), int(g), int(b)) for r, g, b in o.colors}
        for k in range(PIXEL_WIDTH):
            opts.setdefault((int(o.x) + k, int(o.y)), set()).update(colors)
    for x, y, rgb in _weapon_reference_cells():
        opts[(x, y)] = {rgb}
    for x, y, rgb in _hud_reference_cells():
        opts[(x, y)] = {rgb}  # bar pixels: the single baked color, no tolerance
    return opts


# --- image-level comparison (report-first, no fixed bar) ------------------


@dataclass
class CompareReport:
    screen: tuple[int, int]
    n_gen_pixels: int
    n_ref_pixels: int
    only_in_gen: int
    only_in_ref: int
    in_both: int
    exact_color_matches: int  # of in_both, color identical
    gen_with_option: int  # gen (x,y) that have a reference option set
    in_option_matches: int  # of gen_with_option, color within the option set

    @property
    def exact_color_rate(self) -> float:
        return self.exact_color_matches / self.in_both if self.in_both else 0.0

    @property
    def in_option_rate(self) -> float:
        return (
            self.in_option_matches / self.gen_with_option
            if self.gen_with_option
            else 0.0
        )

    @property
    def coverage_rate(self) -> float:
        """Fraction of reference pixels the generated frame also drew."""
        return self.in_both / self.n_ref_pixels if self.n_ref_pixels else 0.0

    def format_short(self) -> str:
        w, h = self.screen
        return (
            f"image compare ({w}x{h}):\n"
            f"  generated pixels : {self.n_gen_pixels}\n"
            f"  reference pixels : {self.n_ref_pixels}\n"
            f"  (x,y) in both    : {self.in_both}  "
            f"(coverage {self.coverage_rate:.1%} of reference)\n"
            f"  only in generated: {self.only_in_gen}\n"
            f"  only in reference: {self.only_in_ref}\n"
            f"  exact color match: {self.exact_color_matches}/{self.in_both} "
            f"({self.exact_color_rate:.1%} of overlap)\n"
            f"  within option set: {self.in_option_matches}/{self.gen_with_option} "
            f"({self.in_option_rate:.1%} of generated-with-option)"
        )


def compare(
    gen: PixelBuf, ref: PixelBuf, options: dict[tuple[int, int], set[Rgb]]
) -> CompareReport:
    gen_keys = set(gen)
    ref_keys = set(ref)
    both = gen_keys & ref_keys
    exact = sum(1 for k in both if gen[k] == ref[k])
    gen_with_option = [k for k in gen_keys if k in options]
    in_option = sum(1 for k in gen_with_option if gen[k] in options[k])
    return CompareReport(
        screen=(SCREEN_WIDTH, SCREEN_HEIGHT),
        n_gen_pixels=len(gen),
        n_ref_pixels=len(ref),
        only_in_gen=len(gen_keys - ref_keys),
        only_in_ref=len(ref_keys - gen_keys),
        in_both=len(both),
        exact_color_matches=exact,
        gen_with_option=len(gen_with_option),
        in_option_matches=in_option,
    )


# --- PNG rendering (dependency-free) --------------------------------------


def _canvas(fill: Rgb = (0, 0, 0)) -> np.ndarray:
    a = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    a[:] = fill
    return a


def _blit(canvas: np.ndarray, buf: PixelBuf) -> None:
    for (x, y), rgb in buf.items():
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            canvas[y, x] = rgb


def _diff_canvas(
    gen: PixelBuf, ref: PixelBuf, options: dict[tuple[int, int], set[Rgb]]
) -> np.ndarray:
    """Color-coded diff: green=in-option or exact-match (exact-only where no
    option set exists), red=color-mismatch, blue=only-ref, yellow=only-gen,
    black=neither."""
    canvas = _canvas((0, 0, 0))
    gen_keys, ref_keys = set(gen), set(ref)
    for x in range(SCREEN_WIDTH):
        for y in range(SCREEN_HEIGHT):
            k = (x, y)
            in_gen, in_ref = k in gen_keys, k in ref_keys
            if in_gen and in_ref:
                ok = gen[k] in options.get(k, {ref[k]}) or gen[k] == ref[k]
                canvas[y, x] = (0, 200, 0) if ok else (220, 0, 0)
            elif in_ref:
                canvas[y, x] = (0, 0, 220)
            elif in_gen:
                canvas[y, x] = (220, 220, 0)
    return canvas


def _write_png(path, rgb: np.ndarray) -> None:
    """Write an ``(H, W, 3)`` uint8 array as a PNG using only stdlib zlib."""
    h, w, _ = rgb.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0 (None) per scanline
        raw.extend(rgb[y].tobytes())

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, truecolor RGB
    idat = zlib.compress(bytes(raw), 9)
    Path(path).write_bytes(
        sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    )


def _upscale(rgb: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def write_pngs(
    gen: PixelBuf,
    ref: PixelBuf,
    out_dir,
    *,
    options: dict[tuple[int, int], set[Rgb]] | None = None,
    scale: int = 8,
) -> list[Path]:
    """Write ``generated.png``, ``reference.png``, ``diff.png`` (nearest-upscaled)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = options or {}

    gen_c, ref_c = _canvas(), _canvas()
    _blit(gen_c, gen)
    _blit(ref_c, ref)
    diff_c = _diff_canvas(gen, ref, options)

    paths = []
    for name, canvas in (("generated", gen_c), ("reference", ref_c), ("diff", diff_c)):
        p = out_dir / f"{name}.png"
        _write_png(p, _upscale(canvas, scale))
        paths.append(p)
    return paths
