"""Public text-only presentation surface for stock-tokenizer Doom text."""

from __future__ import annotations

from ..portable.pretty_text import DoomTextFormatter


class DoomFormatter(DoomTextFormatter):
    """Bundle-driven canonical-text ↔ contextual pretty-text formatter."""
