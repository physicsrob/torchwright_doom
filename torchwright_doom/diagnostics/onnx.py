"""Explicit ONNX diagnostic backend; never a production render input.

The single retained diagnostic command is

    python -m torchwright_doom compile-onnx-debug --config configs/e1m1.yaml

It emits an explicitly diagnostic artifact and never populates or satisfies
the HF bundle cache. The module top level is graph-free (only the root job
spec and identity); every graph-reaching import happens inside a business
function after ``apply_screen_env`` has run.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..asset_config import MISSING_TEXTURE_ID, AssetConfig
from ..config import (
    RenderConfig,
    apply_screen_env,
    load_render_config,
    resolve_wad_path,
)
from ..identity import cache_key_from_payload, canonical_compile_payload

_ONNX_DEBUG_CACHE_STRIDE = 12288


def canonical_onnx_debug_payload(
    config: RenderConfig, wad_path: str | Path, *, cache_stride: int = 12288
) -> dict[str, Any]:
    if not (1 <= cache_stride <= config.model.max_seq_len):
        raise ValueError(
            f"ONNX diagnostic cache_stride {cache_stride} must be in "
            f"[1, max_seq_len={config.model.max_seq_len}]"
        )
    payload = canonical_compile_payload(config, wad_path)
    payload["artifact"] = {
        "kind": "onnx_debug",
        "format": 1,
        "architecture": "phi3",
        "cache_stride": int(cache_stride),
    }
    return payload


def onnx_debug_cache_dir(config: RenderConfig, wad_path: str | Path) -> Path:
    return (
        Path.home()
        / ".cache"
        / "torchwright_doom"
        / "onnx_debug"
        / cache_key_from_payload(canonical_onnx_debug_payload(config, wad_path))
    )


def compile_onnx_debug_config(
    *, config_path: str | Path, verbose_compile: bool = False
) -> dict[str, Any]:
    """Explicit diagnostic backend dispatch; output is never accepted by
    rendering."""
    config_path = Path(config_path)
    config = load_render_config(config_path)
    apply_screen_env(config)
    cache_dir = compile_onnx_debug_cached(
        config, base_dir=config_path.parent, verbose=verbose_compile
    )
    return {"cache_dir": str(cache_dir), "artifact_kind": "onnx_debug"}


def compile_onnx_debug_cached(
    config: RenderConfig,
    *,
    base_dir: str | Path | None = None,
    verbose: bool = False,
    cache_dir: str | Path | None = None,
    compile_payload: dict[str, Any] | None = None,
) -> Path:
    """Compile ``config`` into the ONNX diagnostic cache on a miss.

    ``cache_dir`` / ``compile_payload`` exist for the Modal compile path:
    the cache key embeds git SHAs that are unresolvable inside a Modal
    container (no ``.git``), so the local entrypoint computes both and
    hands them over — the container must never derive its own key.
    """
    wad_path = resolve_wad_path(config, base_dir=base_dir)
    from torchwright.compiler import CompileProfile

    if cache_dir is None:
        cache_dir = onnx_debug_cache_dir(config, wad_path)
    else:
        cache_dir = Path(cache_dir)
    onnx_path = cache_dir / "model.onnx"
    meta_path = cache_dir / "model.meta.json"
    if onnx_path.exists() and meta_path.exists():
        print(f"[compile] cache hit {cache_dir}", flush=True)
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compile] cache miss {cache_dir}", flush=True)
    build_info = compile_onnx_debug_path(
        onnx_path,
        d=config.model.d,
        d_head=config.model.d_head,
        d_rot=config.model.d_rot,
        d_hidden=config.model.d_hidden,
        max_layers=config.model.max_layers,
        max_seq_len=config.model.max_seq_len,
        cache_stride=_ONNX_DEBUG_CACHE_STRIDE,
        trim_heads=config.model.trim_heads,
        optimize=config.model.optimize,
        profile=CompileProfile.PHI3,
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
    # Lead with the schedule actually emitted. Solver-attempt status is a
    # separate diagnostic and must not be mistaken for artifact identity.
    schedule = meta.get("schedule") or {}
    selected_origin = schedule.get("selected_origin") or "heuristic"
    delivery = schedule.get("delivery") or "fresh"
    selected_objective = schedule.get("selected_objective")
    print(
        f"[compile] {meta.get('n_layers')} layers "
        f"(d={config.model.d}, d_hidden={config.model.d_hidden or config.model.d}, "
        f"optimize={config.model.optimize}, selected={selected_origin}, "
        f"delivery={delivery}, objective={selected_objective}, "
        f"scale={config.model.scale}, "
        f"d_embed={meta.get('d_embed')}, vocab_rows={meta.get('n_vocab_rows')}) "
        f"-> {cache_dir}",
        flush=True,
    )
    return cache_dir


def compile_onnx_debug_path(
    output_path: str | Path,
    *,
    d: int = 4096,
    d_head: int = 32,
    d_rot: int | None = None,
    max_layers: int = 200,
    max_seq_len: int = 65536,
    cache_stride: int = 12288,
    verbose: bool = False,
    trim_heads: bool = True,
    optimize: int = 0,
    d_hidden: int | None = None,
    rms_norm_const_exp: int = 63,
    asset_config: AssetConfig | None = None,
    wad_path: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
    profile=None,
) -> dict[str, Any]:
    """Compile the token-id forward to a diagnostic ONNX artifact.

    ``cache_stride`` sets the diagnostic ONNX static KV-cache slot count S. It
    is intentionally absent from production ``ModelConfig`` because direct HF
    uses a growing ``DynamicCache``.

    ``rms_norm_const_exp`` (``q``) is the pinned-constant exponent for the
    identity RMSNorm (on by default at the production power-of-two ``d``).  The
    doom forward carries fixed-point coordinates, so its residual energy bound
    is ~2^99.6 — far above the calculator's ~2^44 default.  ``q=63`` is the
    largest the fp32 pinned energy allows (2^127 at the production odd
    ``b=log2(8192)=13``); its budget ``2^(2q-24)=2^102`` clears the doom energy
    with ~5x margin.  A future residual-energy increase past 2^102 would make
    the identity infeasible (q can't go higher) and re-break the compile with a
    clear "rms_norm identity not certified" error.
    """
    from torchwright.compiler.export import compile_to_onnx

    from ..embedding import TOKEN_VOCAB
    from ..model_graph import build_graph

    next_token, rope, emb, asset_banks = build_graph(
        d_head=d_head,
        max_positions=max_seq_len,
        d_rot=d_rot,
        asset_config=asset_config,
        wad_path=wad_path,
    )
    # Linear-layer fusion is owned by the compiler: ``lower()`` runs
    # ``fuse_consecutive_linears`` on its compiler-private copy of the
    # graph (torchwright eb4a0f8).  The explicit pre-pass that used to
    # live here was redundant with that AND mutated doom's source graph
    # in place — removed.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "d": d,
        "d_head": d_head,
        "max_seq_len": max_seq_len,
        "max_layers": max_layers,
        "verbose": verbose,
        "trim_heads": trim_heads,
        "optimize": optimize,
        "cache_stride": cache_stride,
        "rms_norm_const_exp": rms_norm_const_exp,
        "bias": False,
    }
    if profile is not None:
        kwargs["profile"] = profile
    if d_hidden is not None:
        kwargs["d_hidden"] = d_hidden
    if extra_metadata is not None:
        kwargs["extra_metadata"] = extra_metadata
    compile_to_onnx(
        next_token,
        embedding=emb,
        output_path=str(output_path),
        **kwargs,
    )
    return {
        "n_rows": TOKEN_VOCAB.n_rows,
        "d_embed": TOKEN_VOCAB.layout.d_embed,
        "asset_banks": asset_banks,
    }


def _write_render_meta(
    path: Path,
    *,
    config: RenderConfig,
    wad_path: Path,
    build_info: dict[str, Any],
    compile_payload: dict[str, Any] | None = None,
) -> None:
    from ..embedding import TOKEN_VOCAB
    from ..tokenizer.rows import row_index
    from ..vocab import DONE, PIXEL

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
        # "unknown" git SHAs (see compile_onnx_debug_cached docstring).
        "compile_payload": (
            compile_payload
            if compile_payload is not None
            else canonical_onnx_debug_payload(
                config, wad_path, cache_stride=_ONNX_DEBUG_CACHE_STRIDE
            )
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
