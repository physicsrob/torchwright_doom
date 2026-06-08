"""Compatibility wrapper for prefill helpers.

Implementation lives in ``render.inference``.
"""

from __future__ import annotations

from .inference import DEFAULT_PREFILL_CHUNK_SIZE, run_prefill

__all__ = ["DEFAULT_PREFILL_CHUNK_SIZE", "run_prefill"]
