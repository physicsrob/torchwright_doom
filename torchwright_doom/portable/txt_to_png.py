"""Pure-stdlib canonical Doom text to PNG decoder shipped with bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_bundle(bundle: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads((bundle / "doom_bundle_manifest.json").read_text())
    if not manifest.get("validation", {}).get("complete"):
        raise ValueError("Doom bundle manifest is not complete")
    vocab = json.loads((bundle / "doom_vocab.json").read_text())
    palette = json.loads((bundle / "doom_palette.json").read_text())
    files = manifest.get("files", {})
    for name in ("doom_vocab.json", "doom_palette.json"):
        expected = files.get(name, {}).get("sha256")
        if expected and _sha_bytes((bundle / name).read_bytes()) != expected:
            raise ValueError(f"Doom bundle hash mismatch for {name}")
    return manifest, vocab, palette


def _rows(raw: str, vocab: dict) -> list[int]:
    mapping = {word: row for row, word in enumerate(vocab["words"])}
    out = []
    for word in raw.split():
        try:
            out.append(mapping[word])
        except KeyError:
            raise ValueError(f"unknown canonical Doom word: {word!r}") from None
    return out


def _pixels(rows: list[int], vocab: dict, palette: list[list[int]]):
    records = vocab["rows"]
    x = y = None
    dx, dy = 0, 1
    pixels: dict[tuple[int, int], tuple[int, int, int]] = {}
    for row in rows:
        record = records[row]
        kind = record["type"]
        values = record["values"]
        if kind == "setCursorDirectionX":
            dx, dy = 1, 0
        elif kind == "setCursorDirectionY":
            dx, dy = 0, 1
        elif kind == "setCursorX":
            x = int(values["x"])
        elif kind == "setCursorY":
            y = int(values["y"])
        elif kind == "pixel" and x is not None and y is not None:
            width = int(values["w"])
            channels = palette[int(values["color"])]
            if len(channels) != 3:
                raise ValueError("Doom palette entry is not RGB")
            rgb = (int(channels[0]), int(channels[1]), int(channels[2]))
            for offset in range(width):
                pixels[(x + offset, y)] = rgb
            if dx:
                x += width
            else:
                y += dy
    return pixels


def _write_png(path: Path, width: int, height: int, pixels) -> None:
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixels.get((x, y), (0, 0, 0)))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        signature
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode canonical Doom text to PNG")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args(argv)
    bundle = args.bundle or Path(__file__).resolve().parent.parent
    manifest, vocab, palette_blob = _load_bundle(bundle)
    raw_bytes = args.input.read_bytes()
    rows = _rows(raw_bytes.decode("utf-8"), vocab)
    screen = manifest["screen"]
    pixels = _pixels(rows, vocab, palette_blob["colors"])
    _write_png(args.output, int(screen["width"]), int(screen["height"]), pixels)
    if args.provenance:
        ids_bytes = json.dumps(rows, separators=(",", ":")).encode()
        args.provenance.write_text(
            json.dumps(
                {
                    "raw_text_sha256": _sha_bytes(raw_bytes),
                    "row_ids_sha256": _sha_bytes(ids_bytes),
                    "row_vocab_fingerprint": manifest["row_vocab_fingerprint"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
