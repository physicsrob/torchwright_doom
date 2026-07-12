"""Compile the token-id ``forward`` graph to the diagnostic ONNX artifact.

The graph construction itself is the root ``model_graph.build_graph`` (the
shared source-graph authority); this module only owns the explicit ONNX
diagnostic compile. ``OnnxDebugSession`` over the cached artifact (see
``compile_cache.load_onnx_debug_session``) reuses the same construction,
which is what its graph-fingerprint check requires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..asset_config import AssetConfig
from ..model_graph import build_graph


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
