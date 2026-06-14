"""Structural header context for scene indexing.

The prefill stream groups facts by the most recent structural header. A `NODE`
header provides the context for following node fields, an `SS` header provides
the context for following subsector facts, and a `SEG` header provides the
context for following seg facts.

The same context pattern is instantiated three times (and a fourth for plane
defs, from ``scene_index``):

- `node`: current id is the most recent `NODE(j=...)`.
- `subsector`: current id is the most recent `SS(s=...)`.
- `seg`: current id is the most recent `SEG(i=...)`.

Changes from the original: the import block (``Vec`` -> ``Node``; ``Past`` ->
``GraphPast``; std/constants from the real-side shim) and ``Vec.shape`` ->
``len(node)`` in the ``split`` sizes. ``HeaderContext.publish`` packs
``concat(id_value, key_value)`` into the residual; ``pick_most_recent`` recovers
that packed state, which ``split`` recovers as ``(current_id, current_key)`` —
so the context exposes both the scalar id and the lifted-equality key.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node

from .past import GraphPast, PastHandle
from .render_constants import MATCH_GAIN_LONG
from .std import concat, constant, split


@dataclass(frozen=True)
class HeaderContext:
    """Current scene-entity context for one kind of structural header.

    For the node context, `header_active` marks positions where `NODE` appears
    and `header_id` stores that header's `j` value. `current_id` is recovered
    by attending to the most recent active header, and `current_key` is the
    lifted equality key used to key facts in that context.
    """

    header_active: PastHandle
    header_id: PastHandle
    current_id: Node
    current_key: Node

    @classmethod
    def publish(
        cls,
        past: GraphPast,
        *,
        active_name: str,
        id_name: str,
        key_name: str,
        is_header: Node,
        id_value: Node,
        key_value: Node,
    ) -> "HeaderContext":
        """Publish one header kind and derive the current context key."""
        # The active channel says "a header of this kind occurred here"; the
        # id/key channels carry that header's id in scalar and lifted forms.
        # Attending to the most recent active header recovers the scene entity
        # currently contextualizing this input.
        header_active = past.publish(active_name, is_header)
        header_id = past.publish(id_name, id_value)
        header_state = past.publish(key_name, concat(id_value, key_value))
        current_id, current_key = split(
            past.pick_most_recent(
                constant(1.0),
                header_active,
                header_state,
                match_gain=MATCH_GAIN_LONG,
            ),
            [len(id_value), len(key_value)],
        )
        return cls(
            header_active=header_active,
            header_id=header_id,
            current_id=current_id,
            current_key=current_key,
        )
