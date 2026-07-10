"""Build the token-id ``forward`` graph and compile it to the K ONNX artifact.

``build_graph`` is the one place the doom forward graph is constructed for
compilation and for artifact debugging (asset banks -> ``AssetIndex`` ->
``build_doom_embedding("token_ids")`` -> ``forward``).  ``compile_to_onnx_path``
compiles that graph to the production ONNX artifact; ``OnnxDebugSession`` over
the cached artifact (see ``compile_cache.load_debug_session``) reuses the same
construction, which is what its graph-fingerprint check requires.  Graph nodes
are built **inside** ``build_graph`` — never at import (the
import-time-node-free rule, twdoom CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..asset_banks import build_asset_banks
from ..asset_config import DEFAULT_ASSET_CONFIG, AssetConfig
from ..assets import AssetIndex

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torchwright.graph.node import Node
    from torchwright.graph.rope import RopeConfig


def build_graph(
    *,
    d_head: int,
    max_positions: int,
    d_rot: int | None = None,
    asset_config: AssetConfig | None = None,
    wad_path: str | Path | None = None,
) -> tuple["Node", "RopeConfig", "Node", Any]:
    """Construct the token-id forward graph.

    Returns ``(next_token, rope, emb, asset_banks)`` — the output node, the RoPE
    config the graph was built against, the embedding input node, and the asset
    banks.  ``d_head`` MUST equal the ``d_head`` the compile entry point is
    called with (the compiler asserts it); ``max_positions`` sizes the
    graph-derived absolute-position scalar (``global_position_from_bos``) and
    must cover the longest rollout (pass ``max_seq_len``).  ``d_rot`` is the
    partial-rotary width (``None`` = full rotary); the production configs pass
    ``64`` so wide content rides the NoPE tail and the position tiebreak rides a
    rotated plane.
    """
    from torchwright.ops.inout_nodes import create_rope_config

    from ..embedding import build_doom_embedding
    from ..past import GraphPast
    from ..render_main import forward

    asset_config = asset_config or DEFAULT_ASSET_CONFIG
    asset_banks = build_asset_banks(
        wad_path=wad_path or None,
        wall_names=asset_config.wall_names,
        flat_names=asset_config.flat_names,
    )
    asset_index = AssetIndex(asset_banks)
    emb = build_doom_embedding("token_ids")
    rope = create_rope_config(d_head=d_head, max_positions=max_positions, d_rot=d_rot)
    next_token = forward(
        emb,
        GraphPast(input_vec=emb, rope=rope),
        asset_index=asset_index,
    )
    return next_token, rope, emb, asset_banks


def compile_to_onnx_path(
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
    bias: bool = True,
    asset_config: AssetConfig | None = None,
    wad_path: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the token-id forward to ONNX and return basic build metadata.

    ``cache_stride`` sets the static KV-cache slot count S baked into the
    exported graph (see ModelConfig.cache_stride); the cache is unbounded.

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
        "bias": bias,
    }
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
