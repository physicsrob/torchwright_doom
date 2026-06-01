"""Shared type-match helper for narrow input-type slices (Plan D / D2).

``SceneTokenView`` and ``ProtocolTokenView`` both need to compare a previous
position's ``GraphPast.input_type()`` slice (the compact 8-wide E8 code)
against a ``TokenType``. The implementation is identical, so it lives here
rather than being duplicated in each view (where it previously had to be kept
byte-for-byte in sync).

Shared infrastructure: ported once here, imported — not re-created — by both
views; kept at a neutral top-level path rather than scene-private because the
protocol view reaches across to it too.

The sandbox built a one-hot ``linear`` over ``VOCAB.types``; the real side
repoints the body at Plan A's :func:`torchwright_doom.extract.type_matches`,
which does the E8 dot-product plus ``compare`` directly on the 8-wide code.
"""

from __future__ import annotations

from torchwright.graph import Node

from .extract import type_matches
from .tokens import TokenType


def input_type_matches(type_vec: Node, token_type: TokenType) -> Node:
    """Return a ±1 type-match bit from a narrow ``input_type()`` E8 slice.

    ``TokenType.check()`` intentionally requires a full token embedding; this
    helper is for the 8-wide type slices recovered from previous positions.
    Token types compare by name, matching the vocab's ``TokenType`` equality.
    """
    return type_matches(type_vec, token_type)
