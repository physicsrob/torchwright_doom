"""YAML render-job config and cache-key helpers."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..asset_config import (
    FLAT_NAMES,
    N_FLATS,
    N_WALL_TEXTURES,
    WALL_TEXTURE_NAMES,
    AssetConfig,
)


@dataclass(frozen=True)
class ModelConfig:
    d: int = 4096
    d_head: int = 32
    scale: int = 4
    d_hidden: int | None = None
    max_layers: int = 200
    trim_heads: bool = True
    assume_zero_init: bool = True
    max_seq_len: int = 65536
    optimize: int = 0
    # Static KV-cache slot count S baked into the compiled ONNX (the
    # arange_S mask constant + the past_K_i shapes).  A compiled model
    # hard-caps at prefill + decode <= S.  12288 covers the e1m1 frame
    # (~3613 prefill + 8000 decode + headroom); the L4 gate config uses a
    # smaller stride.  Must be <= max_seq_len.  NOTE: adding this field
    # busts every compile-cache key once (it enters the payload via
    # asdict(config.model)) — intended.
    cache_stride: int = 12288
    # Windowed-cache protocol (attention sink + sliding window — the ring
    # plan, ring_idea.md): the committed KV cache becomes a fixed
    # cache_window-slot host-managed window.  The runtime pins the
    # prefill in the sink slots [0, prefill_len) and wraps rollout rows
    # through the remaining ring slots, so committed pixel rows
    # evaporate by overwrite and the attention width stays CONSTANT
    # (cache_window + pass width) for the whole frame.  None = the
    # unbounded static cache above.  When set, cache_stride is IGNORED
    # (the window IS the slot count; the exporter rejects both).
    # Positions stay absolute and run to max_seq_len regardless.
    # Output equals the unbounded cache ONLY IF every attention read's
    # span fits what the window keeps resident (the span condition —
    # ring_idea.md); size it from the span census, with margin.  Enters
    # the compile-cache key via asdict like every model field (and like
    # cache_stride, adding it busts every key once — intended).
    cache_window: int | None = None


@dataclass(frozen=True)
class RegionConfig:
    x1: float = 627.2
    y1: float = -3760.0
    x2: float = 1395.2
    y2: float = -2800.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class TextureConfig:
    wall: tuple[str, ...] = WALL_TEXTURE_NAMES
    flat: tuple[str, ...] = FLAT_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "wall", tuple(name.upper() for name in self.wall))
        object.__setattr__(self, "flat", tuple(name.upper() for name in self.flat))

    def asset_config(self) -> AssetConfig:
        return AssetConfig(wall_names=self.wall, flat_names=self.flat)


@dataclass(frozen=True)
class RenderConfig:
    wad: str = "doom1.wad"
    map: str = "E1M1"
    model: ModelConfig = ModelConfig()
    region: RegionConfig = RegionConfig()
    textures: TextureConfig = TextureConfig()

    @property
    def screen(self) -> tuple[int, int]:
        return screen_dims_for_scale(self.model.scale)

    def asset_config(self) -> AssetConfig:
        return self.textures.asset_config()


def screen_dims_for_scale(scale: int) -> tuple[int, int]:
    if scale not in (2, 4):
        raise ValueError("scale must be 4 (80x50) or 2 (160x100); scale=1 is deferred")
    return 320 // scale, 200 // scale


def apply_screen_env(config: RenderConfig) -> None:
    """Set renderer + sandbox screen env vars before graph modules import."""
    width, height = config.screen
    os.environ["TORCHWRIGHT_DOOM_RENDER_SCALE"] = str(config.model.scale)
    os.environ["TORCHWRIGHT_DOOM_SCREEN_WIDTH"] = str(width)
    os.environ["TORCHWRIGHT_DOOM_SCREEN_HEIGHT"] = str(height)
    os.environ["DOOM_SANDBOX_SCREEN_WIDTH"] = str(width)
    os.environ["DOOM_SANDBOX_SCREEN_HEIGHT"] = str(height)


def load_render_config(path: str | Path) -> RenderConfig:
    path = Path(path)
    data = _load_yaml_subset(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    model = _mapping(data.get("model") or {}, "model")
    region = _mapping(data.get("region") or {}, "region")
    textures = _mapping(data.get("textures") or {}, "textures")
    cfg = RenderConfig(
        wad=str(data.get("wad", "doom1.wad")),
        map=str(data.get("map", "E1M1")),
        model=ModelConfig(
            d=int(model.get("d", 4096)),
            d_head=int(model.get("d_head", 32)),
            scale=int(model.get("scale", 4)),
            d_hidden=_optional_int(model.get("d_hidden")),
            max_layers=int(model.get("max_layers", 200)),
            trim_heads=bool(model.get("trim_heads", True)),
            assume_zero_init=bool(model.get("assume_zero_init", True)),
            max_seq_len=int(model.get("max_seq_len", 65536)),
            optimize=int(model.get("optimize", 0)),
            cache_stride=int(model.get("cache_stride", 12288)),
            cache_window=_optional_int(model.get("cache_window")),
        ),
        region=RegionConfig(
            x1=float(region.get("x1", 627.2)),
            y1=float(region.get("y1", -3760.0)),
            x2=float(region.get("x2", 1395.2)),
            y2=float(region.get("y2", -2800.0)),
        ),
        textures=TextureConfig(
            wall=tuple(str(v) for v in textures.get("wall", WALL_TEXTURE_NAMES)),
            flat=tuple(str(v) for v in textures.get("flat", FLAT_NAMES)),
        ),
    )
    _validate_config(cfg)
    return cfg


def resolve_wad_path(
    config: RenderConfig, *, base_dir: str | Path | None = None
) -> Path:
    wad = Path(config.wad)
    candidates = []
    if wad.is_absolute():
        candidates.append(wad)
    else:
        if base_dir is not None:
            candidates.append(Path(base_dir) / wad)
        candidates.append(Path.cwd() / wad)
        candidates.append(Path(__file__).resolve().parents[2] / wad)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"WAD {config.wad!r} not found. Checked: {[str(c) for c in candidates]}"
    )


def compile_cache_dir(config: RenderConfig, wad_path: str | Path) -> Path:
    return (
        Path.home()
        / ".cache"
        / "torchwright_doom"
        / "compiled"
        / cache_key(config, wad_path)
    )


def cache_key(config: RenderConfig, wad_path: str | Path) -> str:
    return cache_key_from_payload(canonical_compile_payload(config, wad_path))


def cache_key_from_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_compile_payload(
    config: RenderConfig, wad_path: str | Path
) -> dict[str, Any]:
    wad_path = Path(wad_path)
    return {
        "wad": str(wad_path.resolve()),
        "wad_sha256": _file_sha256(wad_path),
        "map": config.map,
        "region": asdict(config.region),
        "wall_names": list(config.textures.wall),
        "flat_names": list(config.textures.flat),
        "model": asdict(config.model),
        "screen": {"width": config.screen[0], "height": config.screen[1]},
        "git": {
            "torchwright_doom": _git_sha(Path(__file__).resolve().parents[2]),
            "torchwright": _git_sha(
                Path(__file__).resolve().parents[3] / "torchwright"
            ),
        },
    }


def _validate_config(config: RenderConfig) -> None:
    screen_dims_for_scale(config.model.scale)
    if not (1 <= config.model.cache_stride <= config.model.max_seq_len):
        raise ValueError(
            f"model.cache_stride {config.model.cache_stride} must be in "
            f"[1, max_seq_len={config.model.max_seq_len}]"
        )
    if config.model.cache_window is not None and not (
        1 <= config.model.cache_window <= config.model.max_seq_len
    ):
        raise ValueError(
            f"model.cache_window {config.model.cache_window} must be in "
            f"[1, max_seq_len={config.model.max_seq_len}] (a window wider "
            f"than the position space can never fill)"
        )
    if len(config.textures.wall) != N_WALL_TEXTURES:
        raise ValueError(
            "this graph still requires exactly "
            f"{N_WALL_TEXTURES} wall textures; got {len(config.textures.wall)}"
        )
    if len(config.textures.flat) != N_FLATS:
        raise ValueError(
            f"this graph still requires exactly {N_FLATS} flats; "
            f"got {len(config.textures.flat)}"
        )


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_yaml_subset(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = _strip_comment(raw_line).rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"{path}:{lineno}: indentation must use multiples of two")
        text = line.strip()
        if ":" not in text:
            raise ValueError(f"{path}:{lineno}: expected key: value")
        key, value_text = text.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value_text:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value_text)
    return root


def _strip_comment(line: str) -> str:
    in_quote: str | None = None
    for i, ch in enumerate(line):
        if ch in ("'", '"'):
            in_quote = None if in_quote == ch else ch if in_quote is None else in_quote
        elif ch == "#" and in_quote is None:
            return line[:i]
    return line


def _parse_scalar(text: str) -> Any:
    if text.startswith("[") or text.startswith("{"):
        return ast.literal_eval(_quote_bare_words(text))
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text.strip("'\"")


def _quote_bare_words(text: str) -> str:
    out: list[str] = []
    token: list[str] = []
    in_quote: str | None = None

    def flush() -> None:
        if not token:
            return
        value = "".join(token).strip()
        token.clear()
        if not value:
            return
        lower = value.lower()
        if lower in ("true", "false", "none", "null"):
            out.append(
                {"true": "True", "false": "False", "none": "None", "null": "None"}[
                    lower
                ]
            )
            return
        try:
            float(value)
        except ValueError:
            out.append(repr(value))
        else:
            out.append(value)

    for ch in text:
        if in_quote is not None:
            out.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            flush()
            in_quote = ch
            out.append(ch)
        elif ch in "[]{}:,":
            flush()
            out.append(ch)
        elif ch.isspace():
            flush()
            out.append(ch)
        else:
            token.append(ch)
    flush()
    return "".join(out)


def _git_sha(repo: Path) -> str:
    """Committed HEAD sha, extended with a working-tree digest when dirty.

    The compile cache key embeds this. Without the dirty suffix, an
    uncommitted edit ships to the compile (``add_local_python_source`` sends
    the working tree) while the key stays pinned at HEAD — every edit-run
    cycle silently HITs the artifact compiled from the *first* iteration and
    the gate "validates" code that was never compiled.  sha256 of
    ``git diff HEAD`` covers tracked edits; ``git status --porcelain`` adds
    untracked (non-ignored) file *names* so new files also move the key.
    """
    git = ["git", "-C", str(repo)]
    try:
        head = subprocess.check_output(
            [*git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
    try:
        diff = subprocess.check_output(
            [*git, "diff", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        status = subprocess.check_output(
            [*git, "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return f"{head}-dirty.unknown"
    if not diff and not status:
        return head
    digest = hashlib.sha256((diff + "\0" + status).encode("utf-8")).hexdigest()[:12]
    return f"{head}-dirty.{digest}"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
