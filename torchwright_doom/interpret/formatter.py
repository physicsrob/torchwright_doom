"""Public text-only presentation surface for stock-tokenizer Doom text."""

from __future__ import annotations

from ..portable.pretty_text import DoomTextFormatter


class DoomFormatter(DoomTextFormatter):
    """Display-only wrapper over the bundle-shipped prettifier
    (``portable/pretty_text.py``): turns the tokenizer's raw output text into
    the readable surface, driven by the bundle's frozen vocab/tables. Pretty
    output is never parsed back or used for validation."""
