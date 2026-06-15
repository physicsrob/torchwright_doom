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
    from torchwright.graph.pos_encoding import PosEncoding


def build_graph(
    *,
    asset_config: AssetConfig | None = None,
    wad_path: str | Path | None = None,
) -> tuple["Node", "PosEncoding", "Node", Any]:
    """Construct the token-id forward graph.

    Returns ``(next_token, pos, emb, asset_banks)`` — the output node, the
    positional encoding, the embedding input node, and the asset banks the
    graph was built against.
    """
    from torchwright.ops.inout_nodes import create_pos_encoding

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
    pos = create_pos_encoding()
    next_token = forward(
        emb,
        GraphPast(input_vec=emb, pos_encoding=pos),
        pos,
        asset_index=asset_index,
    )
    return next_token, pos, emb, asset_banks


def compile_to_onnx_path(
    output_path: str | Path,
    *,
    d: int = 4096,
    d_head: int = 32,
    max_layers: int = 200,
    max_seq_len: int = 65536,
    cache_stride: int = 12288,
    cache_window: int | None = None,
    verbose: bool = False,
    trim_heads: bool = True,
    optimize: int = 0,
    assume_zero_init: bool = True,
    d_hidden: int | None = None,
    asset_config: AssetConfig | None = None,
    wad_path: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the token-id forward to ONNX and return basic build metadata.

    ``cache_window`` selects the windowed-cache export (the host-managed
    permanent/expiring slot policy with an in-graph writtenness mask; see
    ModelConfig.cache_window).  The exporter rejects cache_stride +
    cache_window together, so the window replaces the stride in the
    kwargs rather than riding alongside it.
    """
    from torchwright.compiler.export import compile_to_onnx

    from ..embedding import TOKEN_VOCAB

    next_token, pos, emb, asset_banks = build_graph(
        asset_config=asset_config, wad_path=wad_path
    )
    # Linear-layer fusion before compile (always on): the width-safe gate skips
    # the fusions that would eject a downstream ReLU into the residual stream,
    # which drops the production compile ~57 -> 48 layers without busting the
    # width budget. See torchwright.graph.optimize.fuse_consecutive_linears.
    from torchwright.graph.optimize import fuse_consecutive_linears

    n_fused = fuse_consecutive_linears(
        {next_token}, verbose=False, skip_relu_ejecting=True
    )
    print(f"[compile] linear fusion (width-safe): fused {n_fused} pairs", flush=True)
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
        "assume_zero_init": assume_zero_init,
    }
    if cache_window is not None:
        kwargs["cache_window"] = cache_window
    else:
        kwargs["cache_stride"] = cache_stride
    if d_hidden is not None:
        kwargs["d_hidden"] = d_hidden
    if extra_metadata is not None:
        kwargs["extra_metadata"] = extra_metadata
    compile_to_onnx(
        next_token,
        pos,
        embedding=emb,
        output_path=str(output_path),
        **kwargs,
    )
    return {
        "n_rows": TOKEN_VOCAB.n_rows,
        "d_embed": TOKEN_VOCAB.layout.d_embed,
        "asset_banks": asset_banks,
    }
