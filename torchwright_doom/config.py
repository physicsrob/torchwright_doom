"""The render-job spec: YAML config loading, validation, run-knob resolution.

Shared root authority: ``bundle/`` and the run
path both consume it, so it is owned by neither. Cache identity (compile
payloads, cache keys, git/WAD hashing) lives in the sibling root
``identity.py``. Note ``torchwright_doom/identity.py`` (cache identity)
coexists with ``torchwright_doom/tokenizer/identity.py`` (vocab fingerprint)
— different concerns, same basename; grep with paths.

``apply_screen_env`` is the import-time screen-environment mechanism: the
graph modules size themselves from env vars AT IMPORT, so every command must
call it before importing anything that reaches ``constants`` / ``embedding``.
Removing that mechanism is a separate graph-wide project.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model.asset_config import (
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
    # Partial-rotary width (vanilla HF partial_rotary_factor): the first d_rot
    # dims of every head rotate, the last d_head - d_rot are the unrotated
    # "NoPE tail" — dims with no positional encoding, whose values read back
    # identically from any query position.  None = full rotary
    # (d_rot == d_head).  The two committed configs set 64 (with d_head 128)
    # so the wide per-column clip values (the floor/ceiling clip arrays that
    # attention must recover position-free) fit the tail, while the
    # global-position recency tiebreak (see GLOSSARY.md: recency) rides a
    # rotated plane.  Rides the compile-cache key via asdict(config.model).
    d_rot: int | None = None
    scale: int = 4
    # Pixel paint detail: "high" = 1 screen column per rendered column,
    # "low" = 2 (DOOM low-detail; the host blits W=2). The 3D view renders
    # SCREEN_WIDTH // pixel_width columns either way. Both committed configs
    # set "low"; the "high" default exists so a config that omits the field
    # keeps every column (a "low" default would silently halve its column
    # count). Rides the compile-cache key via asdict(config.model).
    detail: str = "high"
    # Status bar / viewport split. When True, the 3D view shrinks to the top
    # VIEW_HEIGHT rows (the projection horizon moves to the view centre) and the
    # bottom BAR_HEIGHT rows are reserved for DOOM's status bar — DOOM's default
    # screenblocks-10 layout. False (the default) renders a bar-less
    # full-screen view (VIEW_HEIGHT collapses to SCREEN_HEIGHT); both
    # committed configs set true. Changes the graph geometry, so it rides
    # the compile-cache key via asdict(config.model).
    hud: bool = False
    d_hidden: int | None = None
    # Per-layer attention-head capacity cap (stock num_attention_heads is the
    # max ACTUAL head count over layers, so this bounds the published global
    # head width). None = d // d_head. Production sets 32 with d_head 128;
    # the consumer config sets 16. This is what makes their respective KV
    # caches fit one render GPU. Rides the compile-cache key via
    # asdict(config.model).
    n_heads: int | None = None
    max_layers: int = 200
    trim_heads: bool = True
    max_seq_len: int = 65536
    optimize: int = 0
    # Production is one fixed physical contract: stock Phi-3, SwiGLU,
    # biasless, RMSNorm-certified. Bias and ONNX static-cache settings are
    # deliberately not configuration surfaces.


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
class PoseConfig:
    """Default render pose, in world coordinates — map data, the same kind
    of fact as ``region:`` (the E1M1 start room for THE config)."""

    x: float = 1056.0
    y: float = -3616.0
    angle: int = 64
    viewz: float = 41.0


@dataclass(frozen=True)
class RunConfig:
    """Render-job runtime defaults (the optional ``run:`` YAML section).

    ``pose`` and ``max_new_tokens`` define the executable prompt and default
    generation bound shipped inside a complete bundle, so both content-bearing
    values are included in bundle identity. CLI flags remain render-only
    overrides and never mutate the published bundle.
    """

    # 61440 covers a full 320x200 frame with ample headroom; the cap is the
    # pos-encoding table (max_seq_len 65536): infer.py rejects a prompt +
    # generation demand beyond max_position_embeddings before generating.
    max_new_tokens: int = 61440
    pose: PoseConfig = PoseConfig()


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
    # Render-job runtime defaults (optional ``run:`` section) — a RUNTIME knob
    # set, never part of the compile payload.
    run: RunConfig = RunConfig()

    @property
    def screen(self) -> tuple[int, int]:
        return screen_dims_for_scale(self.model.scale)

    def asset_config(self) -> AssetConfig:
        return self.textures.asset_config()


@dataclass(frozen=True)
class ResolvedRunArgs:
    x: float
    y: float
    angle: int
    viewz: float
    max_new_tokens: int


def resolve_run_args(
    config: RenderConfig,
    *,
    x: float | None = None,
    y: float | None = None,
    angle: int | None = None,
    viewz: float | None = None,
    max_new_tokens: int | None = None,
) -> ResolvedRunArgs:
    """Resolve render-job knobs: explicit caller value > config ``run:``.

    ``None`` means "the caller didn't say" — every entry point (Makefile,
    modal_render, argparse) passes ``None`` unless the user set the flag,
    so the config stays the single default source.
    """
    run = config.run
    return ResolvedRunArgs(
        x=run.pose.x if x is None else float(x),
        y=run.pose.y if y is None else float(y),
        angle=run.pose.angle if angle is None else int(angle),
        viewz=run.pose.viewz if viewz is None else float(viewz),
        max_new_tokens=(
            run.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        ),
    )


def screen_dims_for_scale(scale: int) -> tuple[int, int]:
    if scale not in (1, 2, 4):
        raise ValueError("scale must be 1 (320x200), 2 (160x100), or 4 (80x50)")
    return 320 // scale, 200 // scale


def apply_screen_env(config: RenderConfig) -> None:
    """Set the renderer screen env vars before graph modules import."""
    width, height = config.screen
    os.environ["TORCHWRIGHT_DOOM_RENDER_SCALE"] = str(config.model.scale)
    os.environ["TORCHWRIGHT_DOOM_SCREEN_WIDTH"] = str(width)
    os.environ["TORCHWRIGHT_DOOM_SCREEN_HEIGHT"] = str(height)
    os.environ["TORCHWRIGHT_DOOM_DETAIL"] = config.model.detail
    os.environ["TORCHWRIGHT_DOOM_HUD"] = "1" if config.model.hud else "0"


def load_render_config(path: str | Path) -> RenderConfig:
    path = Path(path)
    data = _load_yaml_subset(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    model = _mapping(data.get("model") or {}, "model")
    region = _mapping(data.get("region") or {}, "region")
    textures = _mapping(data.get("textures") or {}, "textures")
    run = _mapping(data.get("run") or {}, "run")
    pose = _mapping(run.get("pose") or {}, "run.pose")
    _known_keys(data, {"wad", "map", "model", "region", "textures", "run"}, str(path))
    _known_keys(
        model,
        {
            "d",
            "d_head",
            "d_rot",
            "scale",
            "detail",
            "hud",
            "d_hidden",
            "n_heads",
            "max_layers",
            "trim_heads",
            "max_seq_len",
            "optimize",
        },
        "model",
    )
    _known_keys(region, {"x1", "y1", "x2", "y2"}, "region")
    _known_keys(textures, {"wall", "flat"}, "textures")
    _known_keys(run, {"max_new_tokens", "pose"}, "run")
    _known_keys(pose, {"x", "y", "angle", "viewz"}, "run.pose")
    run_defaults = RunConfig()
    pose_defaults = PoseConfig()
    cfg = RenderConfig(
        wad=str(data.get("wad", "doom1.wad")),
        map=str(data.get("map", "E1M1")),
        model=ModelConfig(
            d=int(model.get("d", 4096)),
            d_head=int(model.get("d_head", 32)),
            d_rot=_optional_int(model.get("d_rot")),
            scale=int(model.get("scale", 4)),
            detail=str(model.get("detail", "high")),
            hud=bool(model.get("hud", False)),
            d_hidden=_optional_int(model.get("d_hidden")),
            n_heads=_optional_int(model.get("n_heads")),
            max_layers=int(model.get("max_layers", 200)),
            trim_heads=bool(model.get("trim_heads", True)),
            max_seq_len=int(model.get("max_seq_len", 65536)),
            optimize=int(model.get("optimize", 0)),
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
        run=RunConfig(
            max_new_tokens=int(run.get("max_new_tokens", run_defaults.max_new_tokens)),
            pose=PoseConfig(
                x=float(pose.get("x", pose_defaults.x)),
                y=float(pose.get("y", pose_defaults.y)),
                angle=int(pose.get("angle", pose_defaults.angle)),
                viewz=float(pose.get("viewz", pose_defaults.viewz)),
            ),
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
        # Repository-root fallback (this file sits at the package root, one
        # directory below the repo).
        candidates.append(Path(__file__).resolve().parents[1] / wad)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"WAD {config.wad!r} not found. Checked: {[str(c) for c in candidates]}"
    )


def _validate_config(config: RenderConfig) -> None:
    screen_dims_for_scale(config.model.scale)
    if config.model.detail not in ("low", "high"):
        raise ValueError(
            f"model.detail {config.model.detail!r} must be 'low' or 'high'"
        )
    if config.run.max_new_tokens < 1:
        raise ValueError(f"run.max_new_tokens {config.run.max_new_tokens} must be >= 1")
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


def _known_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name}: unknown key(s): {', '.join(unknown)}")


def _load_yaml_subset(path: Path) -> dict[str, Any]:
    # Hand-rolled ON PURPOSE: no workspace package declares a YAML
    # dependency (pyyaml reaches the lockfile only as another tool's
    # transitive dependency) and one render config does not justify adding
    # a runtime import surface.  The accepted
    # grammar is the subset the committed config uses — nested mappings by
    # two-space indentation, inline [list] / {mapping} literals, scalars
    # (int / float / bool / null / quoted or bare strings), '#' comments.
    # Block lists ("- item") are NOT accepted and fail loud below.
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
