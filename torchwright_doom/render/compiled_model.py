"""Build + compile the token-id ``forward`` (the K artifact) and decode outputs.

This is the one place a *compiled* doom forward is constructed for inference. The
build mirrors ``tests/scene/test_forward_ar_rollout.py::_compiled_rollout`` exactly
(``build_doom_embedding("token_ids")`` -> ``forward`` -> ``compile_headless``), but
generalized to return the output ``Node`` too (the diagnostic's ``probe_compiled``
targets it). Graph nodes are built **inside** ``build_compiled`` — never at import
(the import-time-node-free rule, twdoom CLAUDE.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..embedding import W_EMBED

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torchwright.compiler.export import CompiledHeadless
    from torchwright.graph.node import Node

# Default compiled config — H's d=4096 / d_head=32 working point (the radixed-key
# target the J forward compiles at; see test_forward_compiles / test_forward_ar_rollout).
DEFAULT_D = 4096
DEFAULT_D_HEAD = 32

# W_EMBED.t() cached per device (a tensor transpose, not a graph node). Decode runs
# on the outputs' own device so inference keeps it on-GPU and only the argmax ids
# cross back to host.
_W_EMBED_T_BY_DEVICE: dict[str, torch.Tensor] = {}


def _w_embed_t(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _W_EMBED_T_BY_DEVICE.get(key)
    if t is None:
        t = W_EMBED.t().contiguous().to(device)
        _W_EMBED_T_BY_DEVICE[key] = t
    return t


def build_compiled(
    device,
    *,
    d: int = DEFAULT_D,
    d_head: int = DEFAULT_D_HEAD,
    max_layers: int = 200,
    verbose: bool = False,
) -> tuple["CompiledHeadless", "Node"]:
    """Compile the token-id forward. Returns ``(compiled, output_node)``."""
    from torchwright.compiler.export import compile_headless
    from torchwright.ops.inout_nodes import create_pos_encoding

    from ..embedding import build_doom_embedding
    from ..past import GraphPast
    from ..render_main import forward

    emb = build_doom_embedding("token_ids")
    pos = create_pos_encoding()
    next_token = forward(emb, GraphPast(input_vec=emb, pos_encoding=pos), pos)
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


def argmax_rows(outputs: torch.Tensor) -> list[int]:
    """Argmax-decode ``compiled.step`` outputs ``(n, d_embed)`` to ``W_EMBED`` row ids.

    The standard LLM decode: nearest ``W_EMBED`` row by dot product. Runs on the
    outputs' device; only the resulting ids cross to host.
    """
    o = outputs.detach()
    wt = _w_embed_t(o.device).to(o.dtype)
    return (o @ wt).argmax(dim=-1).cpu().tolist()
