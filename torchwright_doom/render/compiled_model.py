"""Build + compile the token-id ``forward`` (the K artifact) and decode outputs.

This is the one place a *compiled* doom forward is constructed for inference. The
build mirrors ``tests/scene/test_forward_ar_rollout.py::_compiled_rollout`` exactly
(``build_doom_embedding("token_ids")`` -> ``forward`` -> ``compile_headless``), but
generalized to return the output ``Node`` too (the diagnostic's ``probe_compiled``
targets it). Graph nodes are built **inside** ``build_compiled`` — never at import
(the import-time-node-free rule, twdoom CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..asset_banks import build_asset_banks
from ..asset_config import DEFAULT_ASSET_CONFIG, AssetConfig
from ..assets import AssetIndex

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torchwright.compiler.export import CompiledHeadless
    from torchwright.graph.node import Node

# Default compiled config — H's d=4096 / d_head=32 working point (the radixed-key
# target the J forward compiles at; see test_forward_compiles / test_forward_ar_rollout).
DEFAULT_D = 4096
DEFAULT_D_HEAD = 32


def build_compiled(
    device,
    *,
    d: int = DEFAULT_D,
    d_head: int = DEFAULT_D_HEAD,
    max_layers: int = 200,
    verbose: bool = False,
    asset_config: AssetConfig | None = None,
    wad_path: str | Path | None = None,
) -> tuple["CompiledHeadless", "Node"]:
    """Compile the token-id forward. Returns ``(compiled, output_node)``."""
    from torchwright.compiler.export import compile_headless
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
    compiled = compile_headless(
        next_token,
        pos,
        d=d,
        d_head=d_head,
        max_layers=max_layers,
        verbose=verbose,
        device=str(device),
    )
    return compiled, next_token


def compile_to_onnx_path(
    output_path: str | Path,
    *,
    d: int = DEFAULT_D,
    d_head: int = DEFAULT_D_HEAD,
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
) -> dict[str, Any]:
    """Compile the token-id forward to ONNX and return basic build metadata.

    ``cache_window`` selects the windowed-cache export (attention sink +
    sliding window; see ModelConfig.cache_window).  The exporter rejects
    cache_stride + cache_window together, so the window replaces the
    stride in the kwargs rather than riding alongside it.
    """
    from torchwright.compiler.export import compile_to_onnx
    from torchwright.ops.inout_nodes import create_pos_encoding

    from ..embedding import TOKEN_VOCAB, build_doom_embedding
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


from .inference import OnnxTokenRuntime, argmax_rows  # noqa: E402,F401
