"""Fetch the reference render and report image-level agreement; write PNGs.

Plan K §2: success is image-level, not token-exact — the generated stream will
diverge (off-by-one edges change pixel counts), so we **report rich stats and
render PNGs first**, and pick the acceptance bar once the real divergence is
visible. This module fixes no threshold; it measures.

The in-tree ``pydoom`` renderer is imported lazily inside the reference helpers,
so the stats/PNG helpers that take buffers load without touching it.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import PIXEL_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH

Rgb = tuple[int, int, int]
PixelBuf = dict[tuple[int, int], Rgb]


# --- reference render (ground truth) --------------------------------------


def reference_pixels(scene, pose) -> PixelBuf:
    """Reference frame as ``{(x, y): rgb}`` (first emission at each pixel wins).

    Low-detail: ``expected_pixel_pass`` emits one sample per rendered column at
    its left-edge screen x (``col * PIXEL_WIDTH``). Mirror the host blit
    (:func:`..decode.decode_rows_to_pixels`, which paints ``w`` cells in +X) by
    filling all ``PIXEL_WIDTH`` cells, so the reference is the true 320-wide
    frame, not a 1-wide comb. No-op at high-detail (``PIXEL_WIDTH == 1``)."""
    from ..pydoom import expected_pixel_pass

    out: PixelBuf = {}
    for p in expected_pixel_pass(scene, pose):
        rgb = (int(p.color[0]), int(p.color[1]), int(p.color[2]))
        for k in range(PIXEL_WIDTH):
            out.setdefault((int(p.x) + k, int(p.y)), rgb)
    return out


def reference_options(scene, pose) -> dict[tuple[int, int], set[Rgb]]:
    """Per-pixel accepted-color option sets (texture-neighborhood + lighting tol).

    Width-expanded ``PIXEL_WIDTH`` cells wide to match the host blit (see
    :func:`reference_pixels`); no-op at high-detail."""
    from ..pydoom import expected_pixel_color_options

    opts: dict[tuple[int, int], set[Rgb]] = {}
    for o in expected_pixel_color_options(scene, pose):
        colors = {(int(r), int(g), int(b)) for r, g, b in o.colors}
        for k in range(PIXEL_WIDTH):
            opts.setdefault((int(o.x) + k, int(o.y)), set()).update(colors)
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
    """Color-coded diff: green=in-option, red=color-mismatch, blue=only-ref,
    yellow=only-gen, black=neither."""
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
