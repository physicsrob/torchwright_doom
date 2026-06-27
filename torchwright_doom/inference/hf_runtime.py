"""The production runtime: a native HuggingFace ``TorchwrightForCausalLM``.

This is the sole DOOM render runtime. The compiled artifact still ships as
ONNX (the compiler's export plus the ONNX->HF convert step are unchanged);
this module loads that artifact as a standard ``transformers`` causal LM and
drives it through the validated generation loops in :mod:`generation`.

The :class:`~torchwright_doom.inference.generation.TokenRuntime` seam is in
integer row-id space — ``run_prefill`` / ``pure_ar_rollout`` build a
``(n_new, 1)`` float token-id tensor with ``rows_to_input`` and decode logits
with ``argmax_rows`` — and the HF model takes integer ``input_ids`` and returns
logits, so the seam matches with no protocol shim. :class:`HfTokenRuntime`
reshapes the seam's token-id tensor to ``(1, n_new)`` long ``input_ids``, builds
a ``cache_position``, and steps a stock ``transformers.DynamicCache``.

The cache is unbounded (no windowing, no slot recycling): a full frame fits a
big GPU at fp32 (~28 GB weights + tens of GB KV). The trade-off vs. the retired
ONNX runtime — greedy-only, ~30 min/frame, no bounded cache — is accepted in
exchange for being, with no asterisk, a standard transformer.

**Screen-config caveat** (shared with ``hf_export``): the DOOM token vocab is
built at import time from the screen-size env vars, so :func:`apply_screen_env`
MUST run before any vocab/embedding import. Every DOOM import here is deferred
into the functions, which the caller invokes only after applying the screen env.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .generation import TokenRuntime


@dataclass
class HfCache:
    """Duck-typed stand-in for :class:`~.kv_cache.KVCache` over a HF cache.

    The generation loops only read ``.length`` (the committed position count);
    the actual key/value tensors live in the wrapped
    ``transformers.DynamicCache``. ``commit`` just raises ``length``.
    """

    dynamic: Any  # transformers.DynamicCache
    length: int = 0


class HfTokenRuntime(TokenRuntime[HfCache]):
    """Drive a native ``TorchwrightForCausalLM`` over the row-id seam.

    Owns the loaded model + device; the generation loops own the cache (one
    :class:`HfCache` per run via :meth:`empty_past`).
    """

    def __init__(self, model, device: str | torch.device) -> None:
        self.model = model
        self.device = device

    def empty_past(self, max_len: int) -> HfCache:
        """A fresh empty cache for the run (``max_len`` is advisory — the HF
        ``DynamicCache`` grows as needed, unbounded)."""
        from transformers import DynamicCache

        return HfCache(dynamic=DynamicCache(), length=0)

    def step(
        self,
        inputs: torch.Tensor,
        cache: HfCache,
        past_len: int | None = None,
    ) -> tuple[torch.Tensor, HfCache]:
        """One forward pass: ``(n_new, 1)`` float token-id tensor in, the
        ``(n_new, vocab)`` logits for this pass + the advanced cache out."""
        base = cache.length if past_len is None else int(past_len)
        n_new = inputs.shape[0]
        input_ids = inputs.reshape(1, n_new).to(dtype=torch.long, device=self.device)
        cache_position = torch.arange(base, base + n_new, device=self.device)
        with torch.no_grad():
            res = self.model(
                input_ids=input_ids,
                past_key_values=cache.dynamic,
                cache_position=cache_position,
                use_cache=True,
            )
        cache.dynamic = res.past_key_values
        cache.length = base + n_new
        return res.logits[0], cache

    def max_safe_prefill_chunk(self, planned_rows: int | None = None) -> int:
        """No int32 transient-indexability clamp on the torch path — the only
        prefill-chunk bound is GPU memory, handled by ``prefill_chunk_size``."""
        return 2**31 - 1


def load_hf_runtime(
    cache_dir: str | Path,
    *,
    device: str = "cpu",
) -> HfTokenRuntime:
    """Load a compiled DOOM artifact as a native HF runtime (parallel to the
    retired ``load_cached_runtime``).

    Reads the artifact meta, resolves the BEGIN/DONE bos/eos strings, converts
    the ONNX to ``TorchwrightForCausalLM`` (fp32, eval), moves it to ``device``,
    and wraps it in :class:`HfTokenRuntime`.
    """
    from torchwright.compiler.hf.convert import convert_onnx_to_hf

    from .hf_export import _doom_bos_eos_strings, _meta_path

    cache_dir = Path(cache_dir)
    onnx_path = str(cache_dir / "model.onnx")
    meta = json.loads(_meta_path(onnx_path).read_text())
    bos_str, eos_str = _doom_bos_eos_strings(meta)

    model = convert_onnx_to_hf(onnx_path, bos_token=bos_str, eos_token=eos_str)
    model = model.to(torch.float32).eval()
    if device != "cpu":
        model = model.to(device)
    return HfTokenRuntime(model, device)
