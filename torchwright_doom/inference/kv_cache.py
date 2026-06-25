"""The unbounded host-owned KV cache type the generation loops thread.

The vanilla-transformer analog is HF's ``DynamicCache``: slot == position, the
committed prefix is ``[:length]``, committing just raises ``length``. The
production runtime
(:class:`~torchwright_doom.inference.hf_runtime.HfTokenRuntime`) drives its own
``transformers.DynamicCache`` behind a duck-typed
:class:`~torchwright_doom.inference.hf_runtime.HfCache`; this :class:`KVCache`
remains the type the generation-loop signatures name and the type the
in-process test runtimes hand back from ``empty_past``.

The windowed/expiring slot-recycling protocol (and its ONNX runtime) was
retired with the move to the native HF model — the unbounded cache is the only
one now.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    """Runtime-owned KV cache: one buffer per layer per side, slot == position.

    ``length`` counts the committed positions; the committed prefix is
    ``k[i][:length]``. ``max_len`` is the run's demand cap. ``windowed`` is a
    vestigial always-``None`` field that lets the shared generation loops read
    ``cache.windowed`` uniformly across this type and the HF runtime's
    duck-typed cache (both report ``None`` — there is no slot-recycling policy
    anymore).
    """

    k: list[torch.Tensor]  # each (n_slots, n_heads, d_head)
    v: list[torch.Tensor]
    length: int
    max_len: int
    windowed: None = None


def commit(cache, target: int):
    """Set the committed length to ``target`` (in place, no copy) and return it.

    Works on any cache exposing a writable ``length`` — the production
    :class:`~torchwright_doom.inference.hf_runtime.HfCache` and this
    :class:`KVCache` alike.
    """
    cache.length = target
    return cache
