"""Pure-stdlib, bundle-driven Doom text prettifier.

This file is also copied into published bundles as ``tools/pretty_text.py``.
It therefore must not import TorchWright, torch, Transformers, or
``torchwright_doom``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan(text: str) -> list[tuple[str, list[str] | None]]:
    body = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    out: list[tuple[str, list[str] | None]] = []
    i, n = 0, len(body)
    while i < n:
        if body[i].isspace():
            i += 1
            continue
        start = i
        while i < n and not body[i].isspace() and body[i] != "(":
            i += 1
        name = body[start:i]
        args = None
        if i < n and body[i] == "(":
            close = body.find(")", i)
            if close < 0:
                raise ValueError(f"unclosed token arguments after {name!r}")
            inner = body[i + 1 : close].strip()
            args = [part.strip() for part in inner.split(",")] if inner else []
            i = close + 1
        if name:
            out.append((name, args))
    return out


def _label(name: str, args: list[str] | None, *, compact: bool = False) -> str:
    if not args:
        return name
    separator = "," if compact else ", "
    return f"{name}({separator.join(args)})"


def _fmt_decimal(value: float, places: int) -> str:
    if places <= 0:
        return str(int(round(value)))
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _decode_float(lo: float, hi: float, encoded: float) -> float:
    return lo + (float(encoded) + 1.0) * 0.5 * (hi - lo)


def _encode_float(lo: float, hi: float, value: float) -> float:
    return (2.0 / (hi - lo)) * float(value) - (hi + lo) / (hi - lo)


def _level(value: float, steps: int) -> int:
    return round((float(value) + 1.0) * 0.5 * steps)


class DoomTextFormatter:
    def __init__(self, vocab: dict, tables: dict):
        self.vocab_blob = vocab
        self.tables = tables
        self.words = list(vocab["words"])
        self.labels = list(vocab["labels"])
        if len(self.words) != int(vocab["n_rows"]) or len(self.labels) != len(
            self.words
        ):
            raise ValueError("frozen Doom vocabulary arrays have inconsistent widths")
        self.word_to_id = {word: row for row, word in enumerate(self.words)}
        self.label_to_id = {label: row for row, label in enumerate(self.labels)}
        if len(self.word_to_id) != len(self.words) or len(self.label_to_id) != len(
            self.labels
        ):
            raise ValueError("frozen Doom vocabulary is not injective")

        carrier = tables["carrier"]
        self.value_start = int(carrier["value"]["start"])
        self.value_size = int(carrier["value"]["size"])
        self.angle_start = int(carrier["angle"]["start"])
        self.angle_size = int(carrier["angle"]["size"])
        self.angle_lo = int(carrier["angle"]["lo"])
        self.value_steps = int(tables["value_steps"])
        self.angle_bam = int(tables["angle_bam"])
        self.sentinel_value = float(tables["back_height_sentinel"])
        self.marker_range = {
            key: (float(value[0]), float(value[1]))
            for key, value in tables["marker_range"].items()
        }
        self.angle_markers = set(tables["angle_markers"])
        self.sentinel_markers = set(tables["sentinel_markers"])
        self.x_markers = set(tables["x_coord_markers"])
        self.y_markers = set(tables["y_coord_markers"])
        origin = tables.get("origin", [0.0, 0.0])
        self.origin = (float(origin[0]), float(origin[1]))
        self.header_levels = {
            str(key): int(value)
            for key, value in tables.get("header_levels", {}).items()
        }
        layout = tables.get("layout", {})
        self.indent_unit = int(layout.get("indent_unit", 2))
        self.field_indent = int(layout.get("field_indent", 4))

    @classmethod
    def from_bundle(
        cls, bundle_dir: str | Path, *, allow_incomplete: bool = False
    ) -> "DoomTextFormatter":
        directory = Path(bundle_dir)
        manifest_path = directory / "doom_bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not allow_incomplete and not manifest.get("validation", {}).get("complete"):
            raise ValueError("Doom bundle manifest is not complete")
        files = manifest.get("files", {})
        for name in ("doom_vocab.json", "doom_tables.json"):
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(f"Doom bundle is missing {name}")
            expected = files.get(name, {}).get("sha256")
            if expected and _sha256(path) != expected:
                raise ValueError(f"Doom bundle hash mismatch for {name}")
        vocab = json.loads((directory / "doom_vocab.json").read_text(encoding="utf-8"))
        tables = json.loads(
            (directory / "doom_tables.json").read_text(encoding="utf-8")
        )
        if int(vocab["n_rows"]) != int(manifest["vocab_size"]):
            raise ValueError("Doom formatter vocabulary width disagrees with manifest")
        if vocab.get("fingerprint") != manifest.get("row_vocab_fingerprint"):
            raise ValueError("Doom formatter row-vocabulary fingerprint mismatch")
        screen = vocab.get("screen", {})
        manifest_screen = manifest.get("screen", {})
        if (screen.get("width"), screen.get("height")) != (
            manifest_screen.get("width"),
            manifest_screen.get("height"),
        ):
            raise ValueError("Doom formatter screen identity mismatch")
        words_digest = hashlib.sha256(
            json.dumps(
                vocab["words"], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if words_digest != manifest.get("tokenizer_vocab_sha256"):
            raise ValueError("Doom formatter tokenizer-word identity mismatch")
        return cls(vocab, tables)

    def rows_from_raw_text(self, raw_text: str) -> list[int]:
        rows = []
        for word in raw_text.split():
            try:
                rows.append(self.word_to_id[word])
            except KeyError:
                raise ValueError(f"unknown canonical Doom word: {word!r}") from None
        return rows

    def raw_text_from_rows(self, rows: list[int]) -> str:
        try:
            return " ".join(self.words[int(row)] for row in rows)
        except (IndexError, TypeError):
            raise ValueError("Doom row outside frozen vocabulary") from None

    def _carrier_kind(self, row: int) -> str | None:
        if self.value_start <= row < self.value_start + self.value_size:
            return "value"
        if self.angle_start <= row < self.angle_start + self.angle_size:
            return "angle"
        return None

    def _origin_shift(self, marker: str) -> float:
        if marker in self.x_markers:
            return self.origin[0]
        if marker in self.y_markers:
            return self.origin[1]
        return 0.0

    def _shortest_value(
        self, lo: float, hi: float, carrier: float, shift: float
    ) -> str:
        target = _level(carrier, self.value_steps)
        physical = _decode_float(lo, hi, carrier) + shift
        for places in range(10):
            candidate = round(physical, places)
            if (
                _level(_encode_float(lo, hi, candidate - shift), self.value_steps)
                == target
            ):
                return _fmt_decimal(candidate, places)
        return repr(physical)

    def _render_carrier(self, marker: str, row: int) -> str:
        if self._carrier_kind(row) == "value":
            if marker not in self.marker_range:
                raise ValueError(f"value follows non-marker {marker!r}")
            lo, hi = self.marker_range[marker]
            carrier = -1.0 + (row - self.value_start) / self.value_steps * 2.0
            shift = self._origin_shift(marker)
            physical = _decode_float(lo, hi, carrier) + shift
            if (
                marker in self.sentinel_markers
                and abs(physical - self.sentinel_value) < 0.5
            ):
                return "none"
            return self._shortest_value(lo, hi, carrier, shift)
        if marker not in self.angle_markers:
            raise ValueError(f"angle carrier follows non-angle marker {marker!r}")
        bam = row - self.angle_start + self.angle_lo
        physical = bam * 360.0 / self.angle_bam
        for places in range(10):
            candidate = round(physical, places)
            if round(candidate * self.angle_bam / 360.0) == bam:
                return _fmt_decimal(candidate, places)
        return repr(physical)

    def _encode_carrier(self, marker: str, value: str) -> int:
        if marker in self.marker_range:
            lo, hi = self.marker_range[marker]
            if marker in self.sentinel_markers and value == "none":
                carrier = _encode_float(lo, hi, self.sentinel_value)
            else:
                carrier = _encode_float(
                    lo, hi, float(value) - self._origin_shift(marker)
                )
            return self.value_start + _level(carrier, self.value_steps)
        if marker in self.angle_markers:
            bam = round(float(value) * self.angle_bam / 360.0)
            return self.angle_start + bam - self.angle_lo
        raise ValueError(f"token {marker!r} cannot carry value {value!r}")

    def _pretty_flat(self, rows: list[int]) -> str:
        units = []
        i = 0
        while i < len(rows):
            row = rows[i]
            if self._carrier_kind(row):
                raise ValueError(f"carrier at row-stream position {i} has no marker")
            label = self.labels[row]
            if i + 1 < len(rows) and self._carrier_kind(rows[i + 1]):
                name, args = _scan(label)[0]
                args = list(args or ())
                args.append(self._render_carrier(name, rows[i + 1]))
                label = _label(name, args)
                i += 1
            units.append(label)
            i += 1
        return " ".join(units)

    def _layout(self, flat: str) -> str:
        lines: list[str] = []
        group: list[str] = []
        level = 0

        def flush() -> None:
            if not group:
                return
            if group[0].split("(", 1)[0] in self.header_levels:
                lines.append(" " * (level * self.indent_unit) + group[0])
                if len(group) > 1:
                    lines.append(
                        " " * (level * self.indent_unit + self.field_indent)
                        + " ".join(group[1:])
                    )
            else:
                lines.append(" ".join(group))
            group.clear()

        for name, args in _scan(flat):
            if name in self.header_levels:
                flush()
                level = self.header_levels[name]
            group.append(_label(name, args))
        flush()
        return "\n".join(lines)

    def format_text(self, raw_tokenizer_text: str) -> str:
        return self._layout(
            self._pretty_flat(self.rows_from_raw_text(raw_tokenizer_text))
        )

    def parse_pretty_text(self, pretty_text: str) -> str:
        rows: list[int] = []
        for name, args in _scan(pretty_text):
            pretty = _label(name, args)
            row = self.label_to_id.get(pretty)
            if row is not None:
                rows.append(row)
                continue
            if not args:
                raise ValueError(f"unknown pretty Doom token: {pretty!r}")
            base = _label(name, args[:-1])
            try:
                rows.append(self.label_to_id[base])
            except KeyError:
                raise ValueError(f"unknown pretty Doom token: {pretty!r}") from None
            rows.append(self._encode_carrier(name, args[-1]))
        return self.raw_text_from_rows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Format canonical Doom tokenizer text")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    bundle = args.bundle or Path(__file__).resolve().parent.parent
    formatter = DoomTextFormatter.from_bundle(bundle)
    raw = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
    rendered = formatter.format_text(raw) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
