"""Load DOOM WAD assets the texture read-surface compiles into weights.

Data-source **B**: a self-contained real-side WAD loader rather than reusing
compiled banks. Narrowed to the asset payloads the forward path
needs: PLAYPAL, COLORMAP, TEXTURE1/TEXTURE2 wall textures, patch pictures, and
flats. The composite must byte-match the reference renderer (pydoom)'s
``ASSET_BOOK``
(column-major ``pixels[u][v]``, missing-texture handling, 64x64 flats).

This is the WAD-loading machinery the prefill-side ``asset_config`` deliberately
omits (it snapshots PLAYPAL as a literal). ``prompt/wad.py`` parses map
*geometry* from the same WAD and states textures are out of scope; this module
is the texture/flat counterpart, kept separate exactly as the original split
geometry (``prompt/wad.py``) from assets (``wad_assets.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable

from ..asset_config import FLAT_NAMES, WALL_TEXTURE_NAMES

# The DOOM1 IWAD lives at the submodule root (same file the reference
# renderer (pydoom) loads): parents[3] climbs assets/ -> model/ -> the
# package -> the submodule root.
DOOM1_WAD_PATH = Path(__file__).resolve().parents[3] / "doom1.wad"

# DOOM asset dimensions.
PALETTE_SIZE = 256  # colors in a PLAYPAL palette
FLAT_SIZE = 64  # a flat (floor/ceiling texture) is FLAT_SIZE x FLAT_SIZE pixels


@dataclass(frozen=True)
class TextureImage:
    """A native-size texture/flat: column-major ``pixels[u][v]`` palette indices.

    Real-side stand-in for the original ``TextureImage`` type —
    only the fields the loader and bank builder touch (``name``/``width``/
    ``height``/``pixels``), as a plain frozen dataclass (no pydantic dep).
    """

    name: str
    width: int
    height: int
    pixels: list[list[int]]  # pixels[column][row]


def _decode_name(raw: bytes) -> str:
    return raw.rstrip(b"\x00").decode("ascii", errors="replace").upper()


@dataclass(frozen=True)
class LumpInfo:
    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class TexturePatchRef:
    origin_x: int
    origin_y: int
    patch_index: int


@dataclass(frozen=True)
class TextureDef:
    name: str
    width: int
    height: int
    patches: tuple[TexturePatchRef, ...]


@dataclass(frozen=True)
class PatchImage:
    width: int
    height: int
    # Column-major. Transparent pixels are None.
    pixels: list[list[int | None]]
    # DOOM picture offsets (``leftoffset``/``topoffset`` in the lump header).
    # ``V_DrawPatch`` draws so patch-local (0, 0) lands at screen
    # ``(x - leftoffset, y - topoffset)`` — the HUD blit needs these to place
    # widgets faithfully (e.g. the face STFST01 carries ``(-5, -2)``).
    leftoffset: int = 0
    topoffset: int = 0


@dataclass(frozen=True)
class AssetBook:
    wall_textures: tuple[TextureImage, ...]
    flat_textures: tuple[TextureImage, ...]
    palette: tuple[tuple[int, int, int], ...]
    colormap: tuple[tuple[int, ...], ...]


class WADReader:
    """Small dependency-free WAD reader for texture/flat assets."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            self._data = f.read()
        magic = self._data[:4]
        if magic not in {b"IWAD", b"PWAD"}:
            raise ValueError(f"{self.path} is not an IWAD/PWAD file")
        numlumps = struct.unpack_from("<I", self._data, 4)[0]
        dir_offset = struct.unpack_from("<I", self._data, 8)[0]
        self._lump_order: list[LumpInfo] = []
        self._lumps_by_name: dict[str, list[LumpInfo]] = {}
        for i in range(numlumps):
            base = dir_offset + i * 16
            offset = struct.unpack_from("<I", self._data, base)[0]
            size = struct.unpack_from("<I", self._data, base + 4)[0]
            name = _decode_name(self._data[base + 8 : base + 16])
            lump = LumpInfo(name=name, offset=offset, size=size)
            self._lump_order.append(lump)
            self._lumps_by_name.setdefault(name, []).append(lump)

    def lump(self, name: str) -> bytes:
        lump = self._find_lump(name)
        return self._slice(lump)

    def palette(self) -> tuple[tuple[int, int, int], ...]:
        """Return the first PLAYPAL palette as 256 RGB triples."""

        playpal = self.lump("PLAYPAL")
        if len(playpal) < PALETTE_SIZE * 3:
            raise ValueError("PLAYPAL is too short for one 256-color palette")
        return tuple(
            (
                int(playpal[i * 3]),
                int(playpal[i * 3 + 1]),
                int(playpal[i * 3 + 2]),
            )
            for i in range(PALETTE_SIZE)
        )

    def colormap(self) -> tuple[tuple[int, ...], ...]:
        """Return the 33 COLORMAP rows."""

        colormap = self.lump("COLORMAP")
        if len(colormap) < 33 * PALETTE_SIZE:
            raise ValueError("COLORMAP is too short for 33 rows")
        return tuple(
            tuple(
                int(v) for v in colormap[row * PALETTE_SIZE : (row + 1) * PALETTE_SIZE]
            )
            for row in range(33)
        )

    def texture(self, name: str) -> TextureImage:
        texture_defs = self._texture_defs()
        key = name.upper()
        if key not in texture_defs:
            raise KeyError(f"wall texture {key!r} not found in TEXTURE1/2")
        texture = texture_defs[key]
        pnames = self._pnames()
        pixels = [[0 for _ in range(texture.height)] for _ in range(texture.width)]
        for patch_ref in texture.patches:
            try:
                patch_name = pnames[patch_ref.patch_index]
            except IndexError as exc:
                raise ValueError(
                    f"texture {texture.name!r} references missing patch "
                    f"index {patch_ref.patch_index}"
                ) from exc
            patch = self._patch_image(patch_name)
            for px in range(patch.width):
                tx = patch_ref.origin_x + px
                if tx < 0 or tx >= texture.width:
                    continue
                column = patch.pixels[px]
                for py, color in enumerate(column):
                    if color is None:
                        continue
                    ty = patch_ref.origin_y + py
                    if 0 <= ty < texture.height:
                        pixels[tx][ty] = int(color)
        return TextureImage(
            name=texture.name,
            width=texture.width,
            height=texture.height,
            pixels=pixels,
        )

    def flat(self, name: str) -> TextureImage:
        key = name.upper()
        flat = self.lump(key)
        if len(flat) != FLAT_SIZE * FLAT_SIZE:
            raise ValueError(
                f"flat {key!r} has size {len(flat)}, expected {FLAT_SIZE * FLAT_SIZE}"
            )
        pixels = [
            [int(flat[y * FLAT_SIZE + x]) for y in range(FLAT_SIZE)]
            for x in range(FLAT_SIZE)
        ]
        return TextureImage(name=key, width=FLAT_SIZE, height=FLAT_SIZE, pixels=pixels)

    def _find_lump(self, name: str) -> LumpInfo:
        key = name.upper()
        matches = self._lumps_by_name.get(key)
        if not matches:
            raise KeyError(f"lump {key!r} not found in {self.path}")
        return matches[0]

    def _slice(self, lump: LumpInfo) -> bytes:
        return self._data[lump.offset : lump.offset + lump.size]

    def _pnames(self) -> tuple[str, ...]:
        buf = self.lump("PNAMES")
        count = struct.unpack_from("<I", buf, 0)[0]
        expected = 4 + count * 8
        if len(buf) < expected:
            raise ValueError(
                f"PNAMES has size {len(buf)}, expected at least {expected}"
            )
        return tuple(_decode_name(buf[4 + i * 8 : 12 + i * 8]) for i in range(count))

    def _texture_defs(self) -> dict[str, TextureDef]:
        out: dict[str, TextureDef] = {}
        for lump_name in ("TEXTURE1", "TEXTURE2"):
            if lump_name not in self._lumps_by_name:
                continue
            out.update(_parse_texture_defs(self.lump(lump_name)))
        return out

    def patch(self, name: str) -> PatchImage:
        """Public by-name patch loader (the thin wrapper the HUD bake needs).

        Returns the decoded masked picture (``pixels[col][row]``, ``None`` for
        transparent) with its DOOM offsets. Same decode the texture compositor
        uses internally.
        """
        return self._patch_image(name)

    def _patch_image(self, name: str) -> PatchImage:
        buf = self.lump(name)
        if len(buf) < 8:
            raise ValueError(f"patch {name!r} is too short")
        width, height, leftoffset, topoffset = struct.unpack_from("<hhhh", buf, 0)
        if width < 0 or height < 0:
            raise ValueError(f"patch {name!r} has invalid size {width}x{height}")
        column_table_size = 8 + width * 4
        if len(buf) < column_table_size:
            raise ValueError(f"patch {name!r} is too short for its column table")
        pixels: list[list[int | None]] = [
            [None for _ in range(height)] for _ in range(width)
        ]
        for x in range(width):
            col_offset = struct.unpack_from("<I", buf, 8 + x * 4)[0]
            pos = col_offset
            while True:
                if pos >= len(buf):
                    raise ValueError(f"patch {name!r} column {x} is unterminated")
                topdelta = buf[pos]
                pos += 1
                if topdelta == 0xFF:
                    break
                if pos + 2 > len(buf):
                    raise ValueError(f"patch {name!r} column {x} has truncated post")
                length = buf[pos]
                pos += 1
                pos += 1  # unused padding byte
                if pos + length + 1 > len(buf):
                    raise ValueError(f"patch {name!r} column {x} has truncated pixels")
                for dy in range(length):
                    y = topdelta + dy
                    if 0 <= y < height:
                        pixels[x][y] = int(buf[pos + dy])
                pos += length
                pos += 1  # unused padding byte
        return PatchImage(
            width=width,
            height=height,
            pixels=pixels,
            leftoffset=leftoffset,
            topoffset=topoffset,
        )


def _parse_texture_defs(buf: bytes) -> dict[str, TextureDef]:
    count = struct.unpack_from("<I", buf, 0)[0]
    offsets = [struct.unpack_from("<I", buf, 4 + i * 4)[0] for i in range(count)]
    out: dict[str, TextureDef] = {}
    for offset in offsets:
        name, _masked, width, height, _col_dir, patch_count = struct.unpack_from(
            "<8sIhhIh", buf, offset
        )
        patches: list[TexturePatchRef] = []
        patch_base = offset + 22
        for p in range(patch_count):
            origin_x, origin_y, patch_index, _stepdir, _colormap = struct.unpack_from(
                "<hhhhh", buf, patch_base + p * 10
            )
            patches.append(
                TexturePatchRef(
                    origin_x=origin_x,
                    origin_y=origin_y,
                    patch_index=patch_index,
                )
            )
        texture = TextureDef(
            name=_decode_name(name),
            width=width,
            height=height,
            patches=tuple(patches),
        )
        out[texture.name] = texture
    return out


def load_asset_book(
    wad_path: str | Path = DOOM1_WAD_PATH,
    *,
    wall_texture_names: Iterable[str] = WALL_TEXTURE_NAMES,
    flat_names: Iterable[str] = FLAT_NAMES,
) -> AssetBook:
    wad = WADReader(wad_path)
    return AssetBook(
        wall_textures=tuple(wad.texture(name) for name in wall_texture_names),
        flat_textures=tuple(wad.flat(name) for name in flat_names),
        palette=wad.palette(),
        colormap=wad.colormap(),
    )
