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
    # Partial-rotary width (vanilla HF partial_rotary_factor): the first d_rot
    # dims of every head rotate, the last d_head - d_rot are the unrotated NoPE
    # tail that content rides (exactly position-free).  None = full rotary
    # (d_rot == d_head).  The two committed configs set 64 (with d_head 128) so
    # the wide clip content fits the tail and the global position tiebreak rides
    # a rotated plane.  Rides the compile-cache key via asdict(config.model).
    d_rot: int | None = None
    scale: int = 4
    # Pixel paint detail: "high" = 1 screen column per rendered column (today),
    # "low" = 2 (DOOM low-detail; the host blits W=2). The 3D view renders
    # SCREEN_WIDTH // pixel_width columns either way. Default "high" preserves
    # today's render for any config that doesn't opt in (a "low" default would
    # silently halve the column count of the scale-2 config). Rides the
    # compile-cache key via asdict(config.model) (busts every key once).
    detail: str = "high"
    # Status bar / viewport split. When True, the 3D view shrinks to the top
    # VIEW_HEIGHT rows (the projection horizon moves to the view centre) and the
    # bottom BAR_HEIGHT rows are reserved for DOOM's status bar — DOOM's default
    # screenblocks-10 layout. False (default) keeps today's full-screen view,
    # bit-identical (VIEW_HEIGHT collapses to SCREEN_HEIGHT). Changes the graph
    # geometry, so it rides the compile-cache key via asdict(config.model).
    hud: bool = False
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

    These are RUNTIME knobs — deliberately NOT part of
    ``canonical_compile_payload``, so changing them never recompiles
    (pinned by ``tests/inference/test_config.py``).  CLI flags override
    field-by-field; the dataclass defaults apply when a config omits the
    section.  Single-sourcing these here is what removed the Makefile /
    cli.py / modal_render.py three-way default drift (a bare direct
    invocation used to truncate the frame).
    """

    # 61440 covers a full 320x200 frame with ample headroom; the cap is the
    # pos-encoding table (max_seq_len 65536), and an oversized demand makes
    # empty_past() reject before prefill.
    max_positions: int = 61440
    # Prefill chunk size: a memory knob that bounds the per-pass transient
    # activations.  Semantically identical to a single pass.
    prefill_chunk_size: int = 128
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
    max_positions: int
    prefill_chunk_size: int


def resolve_run_args(
    config: RenderConfig,
    *,
    x: float | None = None,
    y: float | None = None,
    angle: int | None = None,
    viewz: float | None = None,
    max_positions: int | None = None,
    prefill_chunk_size: int | None = None,
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
        max_positions=(
            run.max_positions if max_positions is None else int(max_positions)
        ),
        prefill_chunk_size=(
            run.prefill_chunk_size
            if prefill_chunk_size is None
            else int(prefill_chunk_size)
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
            max_layers=int(model.get("max_layers", 200)),
            trim_heads=bool(model.get("trim_heads", True)),
            assume_zero_init=bool(model.get("assume_zero_init", True)),
            max_seq_len=int(model.get("max_seq_len", 65536)),
            optimize=int(model.get("optimize", 0)),
            cache_stride=int(model.get("cache_stride", 12288)),
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
            max_positions=int(run.get("max_positions", run_defaults.max_positions)),
            prefill_chunk_size=int(
                run.get("prefill_chunk_size", run_defaults.prefill_chunk_size)
            ),
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
    git_shas = {
        "torchwright_doom": _git_sha(Path(__file__).resolve().parents[2]),
        "torchwright": _git_sha(Path(__file__).resolve().parents[3] / "torchwright"),
    }
    # Enforce "the container must never derive its own key" (see
    # compile_cached): a git-less caller would mint a fixed
    # "unknown"-keyed payload that never changes on a code edit, so the
    # cache would silently serve artifacts compiled from other code
    # states.  endswith also catches "<head>-dirty.unknown" (rev-parse
    # worked but diff/status failed) — that key would be blind to
    # working-tree edits, the same stale-artifact hazard.  Compute the
    # payload where .git is available and hand it over explicitly
    # (compile_payload=...).
    unresolved = {k: v for k, v in git_shas.items() if v.endswith("unknown")}
    if unresolved:
        raise RuntimeError(
            f"cannot derive a compile-cache key here: git sha unresolvable "
            f"for {sorted(unresolved)} — compute canonical_compile_payload "
            f"where .git is available and pass it through compile_payload=."
        )
    return {
        "wad": str(wad_path.resolve()),
        "wad_sha256": _file_sha256(wad_path),
        "map": config.map,
        "region": asdict(config.region),
        "wall_names": list(config.textures.wall),
        "flat_names": list(config.textures.flat),
        "model": asdict(config.model),
        "screen": {"width": config.screen[0], "height": config.screen[1]},
        "git": git_shas,
    }


def _validate_config(config: RenderConfig) -> None:
    screen_dims_for_scale(config.model.scale)
    if config.model.detail not in ("low", "high"):
        raise ValueError(
            f"model.detail {config.model.detail!r} must be 'low' or 'high'"
        )
    if config.run.max_positions < 1:
        raise ValueError(f"run.max_positions {config.run.max_positions} must be >= 1")
    if config.run.prefill_chunk_size < 1:
        raise ValueError(
            f"run.prefill_chunk_size {config.run.prefill_chunk_size} must be >= 1"
        )
    if not (1 <= config.model.cache_stride <= config.model.max_seq_len):
        raise ValueError(
            f"model.cache_stride {config.model.cache_stride} must be in "
            f"[1, max_seq_len={config.model.max_seq_len}]"
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
    # Hand-rolled ON PURPOSE: pyyaml is not in the workspace lockfile and
    # one render config does not justify the dependency.  The accepted
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
