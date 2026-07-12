"""``build_graph``: the one place the Doom forward graph is constructed.

Shared source-graph authority at the package root (like ``config.py`` /
``identity.py``): publication (``bundle/``) and the ONNX diagnostics both
compile from this exact construction (asset banks -> ``AssetIndex`` ->
``build_doom_embedding("token_ids")`` -> ``forward``). Graph nodes are built
**inside** ``build_graph`` — never at import (the import-time-node-free rule,
twdoom CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .model.assets.asset_banks import build_asset_banks
from .model.asset_config import DEFAULT_ASSET_CONFIG, AssetConfig
from .model.assets.assets import AssetIndex

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

    from .model.embedding import build_doom_embedding
    from .model.past import GraphPast
    from .model.render_main import forward

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
