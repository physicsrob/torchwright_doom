"""Compile cache for YAML render jobs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..asset_config import MISSING_TEXTURE_ID
from ..vocab import DONE, PIXEL
from .compiled_model import compile_to_onnx_path
from .config import (
    RenderConfig,
    canonical_compile_payload,
    compile_cache_dir,
    resolve_wad_path,
)
from .inference import OnnxTokenRuntime
from .tokens_bridge import row_index


def compile_cached(
    config: RenderConfig,
    *,
    base_dir: str | Path | None = None,
    verbose: bool = False,
) -> Path:
    wad_path = resolve_wad_path(config, base_dir=base_dir)
    cache_dir = compile_cache_dir(config, wad_path)
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
        trim_heads=config.model.trim_heads,
        optimize=config.model.optimize,
        assume_zero_init=config.model.assume_zero_init,
        verbose=verbose,
        asset_config=config.asset_config(),
        wad_path=wad_path,
    )
    _write_render_meta(
        meta_path,
        config=config,
        wad_path=wad_path,
        build_info=build_info,
    )
    return cache_dir


def load_cached_runtime(cache_dir: str | Path) -> OnnxTokenRuntime:
    return OnnxTokenRuntime(Path(cache_dir) / "model.onnx")


def _write_render_meta(
    path: Path,
    *,
    config: RenderConfig,
    wad_path: Path,
    build_info: dict[str, Any],
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
        "render_meta_format": "torchwright_doom.render.v1",
        "compile_payload": canonical_compile_payload(config, wad_path),
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
    except Exception:
        return None
