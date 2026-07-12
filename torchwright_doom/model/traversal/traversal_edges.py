"""Dynamic traversal edges for BSP return handling.

The AR protocol walks the BSP tree depth-first. `TRAVERSE_ENTER` and
`TRAVERSE_BETWEEN` choose a child and descend one depth level; later
`TRAVERSE_RETURN(entity_u, depth)` has to recover the active parent frame.

That return cannot safely rely on a static child-to-parent lookup. Static
reverse lookup answers "which node owns this child in the scene graph"; a
runtime return is a stack pop and must answer "which parent frame did this
rollout descend from". This implementation records the edge actually taken,
keyed by `(child entity, child depth)`, and stores the parent plus whether that
child was visited as the first or second child.

The only changes from the original are the import block (``Vec`` -> ``Node``;
``Past`` -> ``GraphPast``; std helpers from the real-side shim; ``make_token``
-> ``emit_token``; ``add_const``/``DEPTH_NONZERO`` from ``render_ops``).
"""

from __future__ import annotations

from dataclasses import dataclass

from torchwright.graph import Node
from torchwright.graph import annotated

from ..past import PastHandle, PastHandleScope
from ..protocol.protocol_tokens import ProtocolTokenView
from ..render_ops import DEPTH_NONZERO, add_const
from ..attention_handles import PastLike, RecentKeyHandle, lifted_id_query
from ..std import bool_or, concat, make_token_head, one_hot, select, split
from ..vocab import (
    DRAW_PLANES_BEGIN,
    N_DEPTH_MAX,
    TRAVERSE_BETWEEN,
    TRAVERSE_RETURN,
)


@dataclass(frozen=True)
class TraversalEdges:
    """Published dynamic edges used to resolve `TRAVERSE_RETURN`.

    `edge_lookup` stores `(child entity, child depth)` at edge-taking positions.
    `parent` stores the node that chose the child. `was_first` is true for
    `TRAVERSE_ENTER` edges and false for `TRAVERSE_BETWEEN` edges.
    """

    edge_lookup: RecentKeyHandle
    edge_state: PastHandle

    @classmethod
    @annotated("bsp")
    def publish(
        cls,
        past: PastHandleScope,
        inp: ProtocolTokenView,
        enter_child: Node,
        between_child: Node,
        enter_child_lifted: Node,
        between_child_lifted: Node,
    ) -> "TraversalEdges":
        """Publish dynamic edges taken by ENTER/BETWEEN positions.

        ``enter_child_lifted`` / ``between_child_lifted`` are the width-3
        ``[child, -child^2, 1]`` producer keys for the same child ids carried
        by the scalar ``enter_child`` / ``between_child``. The published edge
        KEY uses the lifted child form so the return-frame query can match it
        with a width-3 lifted-equality query instead of a width-128 one-hot.
        """
        enter_child_tree_depth = add_const(inp.enter_depth, 1.0)
        between_child_tree_depth = add_const(inp.between_depth, 1.0)
        edge_active = bool_or(inp.is_enter, inp.is_between)
        edge_child_lifted = _edge_publish_value(
            inp, enter_child_lifted, between_child_lifted
        )
        edge_tree_depth = _edge_publish_value(
            inp,
            enter_child_tree_depth,
            between_child_tree_depth,
        )
        edge_parent = _edge_publish_value(inp, inp.enter_node, inp.between_node)

        return cls(
            edge_lookup=RecentKeyHandle.publish(
                past,
                "traversal_edge",
                edge_active,
                _edge_producer_key(edge_child_lifted, edge_tree_depth),
            ),
            edge_state=past.publish(
                "traversal_edge_state",
                concat(edge_parent, inp.is_enter),
            ),
        )

    @annotated("bsp")
    def after_return(self, past: PastLike, entity_u: Node, tree_depth: Node) -> Node:
        """Emit the next token after `TRAVERSE_RETURN(entity_u, depth)`.

        If the returning child was first, resume the parent as
        `TRAVERSE_BETWEEN`. If it was second, both children are done, so keep
        returning upward. At BSP tree depth zero, the traversal is complete.
        """
        edge_query = _edge_query_key(entity_u, tree_depth)
        parent, child_was_first = split(
            self.edge_lookup.pick(past, edge_query, self.edge_state),
            [1, 1],
        )
        parent_tree_depth = add_const(tree_depth, -1.0)
        pop_parent = select(
            child_was_first,
            make_token_head(TRAVERSE_BETWEEN, node=parent, depth=parent_tree_depth),
            make_token_head(TRAVERSE_RETURN, entity_u=parent, depth=parent_tree_depth),
        )
        return select(
            DEPTH_NONZERO(tree_depth),
            pop_parent,
            make_token_head(DRAW_PLANES_BEGIN),
        )


def _edge_producer_key(child_lifted: Node, tree_depth: Node) -> Node:
    """Producer KEY for a published edge row.

    The entity field is the width-3 lifted child key ``[child, -child^2, 1]``;
    the ``-child^2`` term must live on this producer side so the consumer's
    ``[2q, 1, 1]`` query scores an exact match as ``1 + q^2 - (child-q)^2`` —
    a query-side square would make the score linear in the child id and pick
    the largest id instead of the equal one.
    """
    return concat(child_lifted, one_hot(tree_depth, N_DEPTH_MAX))


def _edge_query_key(entity_u: Node, tree_depth: Node) -> Node:
    """Consumer QUERY for a TRAVERSE_RETURN row.

    The entity field is the lifted-equality query ``[2*entity, 1, 1]``; paired
    with the producer key above it resolves to the unique row whose published
    child equals ``entity_u``. Width is ``3 + N_DEPTH_MAX`` to match the key.
    """
    return concat(lifted_id_query(entity_u), one_hot(tree_depth, N_DEPTH_MAX))


def _edge_publish_value(
    inp: ProtocolTokenView,
    enter_value: Node,
    between_value: Node,
) -> Node:
    """Choose the edge field for ENTER/BETWEEN publishing.

    The returned value is meaningful only on `TRAVERSE_ENTER` and
    `TRAVERSE_BETWEEN` rows. `TraversalEdges.publish()` masks all other rows
    with `edge_active` before publishing the key, so inactive rows may harmlessly
    carry the BETWEEN-shaped fallback. This is intentionally not `type_switch`:
    most rows are neither ENTER nor BETWEEN, while `type_switch`'s contract says
    exactly one branch should be active.
    """
    return select(inp.is_enter, enter_value, between_value)
