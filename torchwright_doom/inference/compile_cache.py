"""Compile-artifact cache for YAML render jobs (the KV cache lives in kv_cache.py)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..asset_config import MISSING_TEXTURE_ID
from ..vocab import DONE, PIXEL
from .compiled_model import compile_to_onnx_path
from .config import (
    RenderConfig,
    canonical_compile_payload,
    compile_cache_dir,
    resolve_wad_path,
)
from .onnx_runtime import OnnxTokenRuntime
from .tokens_bridge import row_index

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torchwright.debug.onnx_debug import OnnxDebugSession


def compile_cached(
    config: RenderConfig,
    *,
    base_dir: str | Path | None = None,
    verbose: bool = False,
    cache_dir: str | Path | None = None,
    compile_payload: dict[str, Any] | None = None,
) -> Path:
    """Compile ``config`` into the ONNX cache (skipping on a cache hit).

    ``cache_dir`` / ``compile_payload`` exist for the Modal compile path:
    the cache key embeds git SHAs that are unresolvable inside a Modal
    container (no ``.git``), so the local entrypoint computes both and
    hands them over — the container must never derive its own key.
    """
    wad_path = resolve_wad_path(config, base_dir=base_dir)
    if cache_dir is None:
        cache_dir = compile_cache_dir(config, wad_path)
    else:
        cache_dir = Path(cache_dir)
    onnx_path = cache_dir / "model.onnx"
    meta_path = cache_dir / "model.meta.json"
    if onnx_path.exists() and meta_path.exists():
        print(f"[compile] cache hit {cache_dir}", flush=True)
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compile] cache miss {cache_dir}", flush=True)
    build_info = compile_to_onnx_path(
        onnx_path,
        d=config.model.d,
        d_head=config.model.d_head,
        d_hidden=config.model.d_hidden,
        max_layers=config.model.max_layers,
        max_seq_len=config.model.max_seq_len,
        cache_stride=config.model.cache_stride,
        cache_window=config.model.cache_window,
        trim_heads=config.model.trim_heads,
        optimize=config.model.optimize,
        assume_zero_init=config.model.assume_zero_init,
        verbose=verbose,
        asset_config=config.asset_config(),
        wad_path=wad_path,
        # Render geometry travels into the debug sidecar's free-form
        # "extra" (torchwright owns the sidecar schema but not these
        # doom-domain keys); the floor-plan viz reads them from there.
        extra_metadata={
            "screen": {"width": config.screen[0], "height": config.screen[1]},
            "scale": config.model.scale,
        },
    )
    _write_render_meta(
        meta_path,
        config=config,
        wad_path=wad_path,
        build_info=build_info,
        compile_payload=compile_payload,
    )
    # Always surface the compiled depth (the head-to-head metric across
    # configs) without needing a render or VERBOSE_COMPILE — read back
    # from the meta just written so the number is the artifact's, not a
    # build-info guess.
    meta = json.loads(meta_path.read_text())
    print(
        f"[compile] {meta.get('n_layers')} layers "
        f"(d={config.model.d}, d_hidden={config.model.d_hidden or config.model.d}, "
        f"optimize={config.model.optimize}, scale={config.model.scale}, "
        f"d_embed={meta.get('d_embed')}, vocab_rows={meta.get('n_vocab_rows')}) "
        f"-> {cache_dir}",
        flush=True,
    )
    return cache_dir


def load_debug_session(
    cache_dir: str | Path | None,
    config: RenderConfig,
    *,
    base_dir: str | Path | None = None,
    providers=None,
) -> "OnnxDebugSession":
    """Open an :class:`OnnxDebugSession` over a cached compiled artifact.

    Rebuilds the forward graph deterministically (``build_graph``) and pairs
    it with ``<cache_dir>/model.onnx`` — the session's fingerprint check
    fails loud if the rebuilt graph differs from the one the artifact was
    compiled from.  ``cache_dir=None`` resolves the config's own cache entry.

    This NEVER compiles: a missing artifact (or its debug sidecar) raises
    with the recompile recipe instead — production compiles are
    Modal-scale jobs, not something to trigger from a debug session.
    """
    from .compiled_model import build_graph

    wad_path = resolve_wad_path(config, base_dir=base_dir)
    if cache_dir is None:
        cache_dir = compile_cache_dir(config, wad_path)
    else:
        cache_dir = Path(cache_dir)
    onnx_path = cache_dir / "model.onnx"
    sidecar_path = cache_dir / "model.debug.json"
    recipe = (
        f"recompile via `python -m torchwright_doom.inference compile "
        f"--config <yaml>` (delete {cache_dir} first if it exists; "
        f"compile_to_onnx writes the debug sidecar by default)"
    )
    if not onnx_path.exists():
        raise FileNotFoundError(f"no cached artifact at {onnx_path} — {recipe}")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"cached artifact {onnx_path} has no debug sidecar "
            f"{sidecar_path.name} (pre-sidecar cache entry?) — {recipe}"
        )
    # The graph rebuild must see the same screen dims the artifact was
    # compiled at; constants.py reads them from env at import, so a caller
    # that skipped apply_screen_env() would otherwise surface as an opaque
    # fingerprint mismatch.
    from ..constants import SCREEN_HEIGHT, SCREEN_WIDTH

    if (SCREEN_WIDTH, SCREEN_HEIGHT) != config.screen:
        raise RuntimeError(
            f"screen dims {SCREEN_WIDTH}x{SCREEN_HEIGHT} were imported before "
            f"this config's {config.screen[0]}x{config.screen[1]} was applied "
            f"— call apply_screen_env(config) before importing graph modules"
        )
    from torchwright.debug.onnx_debug import OnnxDebugSession

    next_token, pos, _emb, _banks = build_graph(
        asset_config=config.asset_config(), wad_path=wad_path
    )
    return OnnxDebugSession(str(onnx_path), next_token, pos, providers=providers)


def load_cached_runtime(
    cache_dir: str | Path,
    *,
    enable_profiling: bool = False,
    profile_dir: str | Path | None = None,
    attention_buckets: list[int] | None = None,
    expiring_types: tuple[str, ...] | None = None,
) -> OnnxTokenRuntime:
    import os

    from .onnx_runtime import env_flag

    providers = ["CPUExecutionProvider"] if env_flag("TWDOOM_FORCE_CPU") else None
    # Windowed-cache expiry policy (runtime knob, no recompile): token
    # type names whose rows may be recycled once the window fills.
    # Resolution order: TWDOOM_EXPIRING_TYPES (comma-separated, ad-hoc
    # override) > the ``expiring_types`` argument (the config field,
    # RenderConfig.expiring_types) > pixel-only.  Default: pixel rows
    # only — they publish no channels and are read only at offset <= 3,
    # so evicting old ones is safe by construction; everything else
    # stays resident.
    env_types = os.environ.get("TWDOOM_EXPIRING_TYPES")
    if env_types is not None:
        expiring_types = tuple(t.strip() for t in env_types.split(",") if t.strip())
    elif expiring_types is None:
        expiring_types = ("pixel",)
    return OnnxTokenRuntime(
        Path(cache_dir) / "model.onnx",
        providers=providers,
        enable_profiling=enable_profiling,
        profile_dir=profile_dir,
        attention_buckets=attention_buckets,
        expiring_types=expiring_types,
    )


def _write_render_meta(
    path: Path,
    *,
    config: RenderConfig,
    wad_path: Path,
    build_info: dict[str, Any],
    compile_payload: dict[str, Any] | None = None,
) -> None:
    from ..embedding import TOKEN_VOCAB

    asset_config = config.asset_config()
    existing = json.loads(path.read_text()) if path.exists() else {}
    pixel_start, _ = TOKEN_VOCAB.type_to_row_range[PIXEL]
    terminal_row = row_index(DONE, {})
    rows = [
        {"type": t.name, "values": dict(values)}
        for t, values in TOKEN_VOCAB.row_to_token
    ]
    payload = {
        **existing,
        "render_meta_format": "torchwright_doom.inference.v1",
        # Prefer the handed-over payload: computed remotely it would carry
        # "unknown" git SHAs (see compile_cached docstring).
        "compile_payload": (
            compile_payload
            if compile_payload is not None
            else canonical_compile_payload(config, wad_path)
        ),
        "model": asdict(config.model),
        "screen": {"width": config.screen[0], "height": config.screen[1]},
        "n_layers": _read_n_layers_from_onnx_inputs(path.parent / "model.onnx"),
        "d_embed": build_info.get("d_embed"),
        "n_vocab_rows": build_info.get("n_rows"),
        "wall_name_to_id": {
            **{name: idx for name, idx in asset_config.wall_id_by_name.items()},
            "-": MISSING_TEXTURE_ID,
        },
        "wall_id_to_name": {
            str(idx): name for idx, name in asset_config.wall_name_by_id().items()
        },
        "flat_name_to_id": asset_config.flat_id_by_name,
        "flat_id_to_name": {
            str(idx): name for idx, name in asset_config.flat_name_by_id().items()
        },
        "palette": [list(rgb) for rgb in build_info["asset_banks"].playpal],
        "pixel_row_start": pixel_start,
        "terminal_row": terminal_row,
        "row_to_token": rows,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_n_layers_from_onnx_inputs(onnx_path: Path) -> int | None:
    try:
        import onnx

        model = onnx.load(str(onnx_path), load_external_data=False)
        return sum(1 for inp in model.graph.input if inp.name.startswith("past_K_"))
    except Exception as exc:
        # Degraded, not fatal — but say why, or meta records n_layers: null
        # and the "[compile] N layers" headline (the head-to-head depth
        # metric) silently prints None.
        print(f"[compile] n_layers probe failed ({exc!r})", flush=True)
        return None
