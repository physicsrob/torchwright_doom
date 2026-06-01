"""Small wrappers around ``GraphPast`` attention storage patterns.

This module is deliberately generic: it knows how to publish residual-stream
channels and recover values through ``GraphPast``, but it does not know why a
value is useful. Algorithm modules should still wrap these helpers in domain
names such as ``SideTable``, ``ProcessSegCycle``, or ``TraversalEdges``.

The helpers cover three recurring shapes:

- keyed lookup: publish ``(key, value)`` rows, then ``pick_argmax`` by id;
- optional/presence lookup: keyed lookup plus a marker for absent records;
- recency lookup: publish an active marker or dynamic key, then recover the
  most recent matching row.

Ported from ``doom_sandbox/implementation/forward/attention_handles.py``
(Plan D / D1). The only changes from the sandbox source are the import
block (``Vec`` -> ``Node``; ``Past`` -> ``GraphPast``; the std helpers and
constants now come from the real-side shim) — the dataclasses, the lifted-id
key scheme, and the publish/pick structure are a line-for-line port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from torchwright.graph import Node

from .past import GraphPast, PastHandle, PastHandleScope
from .render_constants import MATCH_GAIN_LONG
from .render_ops import MARKER_PRESENT
from .std import concat, constant, gate, linear, one_hot

PastLike: TypeAlias = GraphPast | PastHandleScope


_LIFTED_ID_QUERY = [[2.0, 0.0, 0.0], [0.0, 1.0, 1.0]]


def lifted_id_query(query_id: Node) -> Node:
    """Return the query vector for exact lifted scalar-id equality.

    Producer keys have shape ``[id, -id^2, 1]``. The query
    ``[2q, 1, 1]`` scores a producer as ``1 + q^2 - (id-q)^2``, so
    an exact match beats every other active id by at least one logit unit
    before MATCH_GAIN scaling. The positive constant also makes id=0
    exact matches beat masked zero keys.
    """
    return linear(concat(query_id, constant(1.0)), _LIFTED_ID_QUERY)


@dataclass(frozen=True)
class ValidValueHandle:
    """Validity/value channel pair consumed by ``mean_where``."""

    validity: PastHandle
    value: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastLike,
        name: str,
        valid: Node,
        value: Node,
    ) -> "ValidValueHandle":
        return cls(
            validity=past.publish(f"{name}_valid", valid),
            value=past.publish(name, value),
        )

    def mean(self, past: PastLike) -> Node:
        """Average this value over positions marked valid."""
        return past.mean_where(self.validity, self.value)


@dataclass(frozen=True)
class KeyValueHandle:
    """Key/value channel pair gated by the key channel."""

    key: PastHandle
    value: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastLike,
        name: str,
        active: Node,
        key: Node,
        value: Node,
    ) -> "KeyValueHandle":
        """Publish a gated key and an always-present value channel.

        ``active`` controls only the key: inactive rows publish a semantic zero
        key through ``gate(active, key)``, so they cannot win a keyed lookup. The
        value channel is still published on every row because ``GraphPast`` handles
        are per-channel; consumers must treat it as meaningful only at rows
        whose key was active.
        """
        return cls(
            key=past.publish(f"{name}_key", gate(active, key)),
            value=past.publish(f"{name}_value", value),
        )

    def pick(self, past: PastLike, query_id: Node, width: int) -> Node:
        """Read the value whose published key matches ``query_id``."""
        return past.pick_argmax(one_hot(query_id, width), self.key, self.value)


@dataclass(frozen=True)
class LiftedKeyValueHandle:
    """Key/value channel pair using lifted scalar-id equality keys."""

    key: PastHandle
    value: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastLike,
        name: str,
        active: Node,
        key: Node,
        value: Node,
    ) -> "LiftedKeyValueHandle":
        return cls(
            key=past.publish(
                f"{name}_key",
                gate(active, key),
            ),
            value=past.publish(f"{name}_value", value),
        )

    def pick(self, past: PastLike, query_id: Node) -> Node:
        return past.pick_argmax(lifted_id_query(query_id), self.key, self.value)


@dataclass(frozen=True)
class OptionalKeyValueHandle:
    """Keyed value plus a marker for detecting absent records."""

    key_value: KeyValueHandle
    marker: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastLike,
        name: str,
        active: Node,
        key: Node,
        value: Node,
    ) -> "OptionalKeyValueHandle":
        """Publish a keyed value and a same-row presence marker.

        The marker follows the key, not the value. A lookup first chooses the
        best key match, then reads this marker from that same row so callers can
        distinguish "real producer matched" from "all rows had zero keys".
        """
        return cls(
            key_value=KeyValueHandle.publish(
                past,
                name,
                active,
                key,
                value,
            ),
            marker=past.publish(f"{name}_marker", active),
        )

    def pick(self, past: PastLike, query_id: Node, width: int) -> Node:
        """Read the optional value whose published key matches ``query_id``."""
        return self.key_value.pick(past, query_id, width)

    def pick_marker(self, past: PastLike, query_id: Node, width: int) -> Node:
        """Read the presence marker at the key matching ``query_id``."""
        return past.pick_argmax(
            one_hot(query_id, width),
            self.key_value.key,
            self.marker,
        )


@dataclass(frozen=True)
class KeyMarkerHandle:
    """Key/marker channel pair for keyed presence tests."""

    key: PastHandle
    marker: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastLike,
        name: str,
        active: Node,
        key: Node,
    ) -> "KeyMarkerHandle":
        return cls(
            key=past.publish(f"{name}_key", gate(active, key)),
            marker=past.publish(f"{name}_marker", active),
        )

    def pick(self, past: PastLike, query_id: Node, width: int) -> Node:
        """Read the marker whose published key matches ``query_id``."""
        return past.pick_argmax(one_hot(query_id, width), self.key, self.marker)


@dataclass(frozen=True)
class RecentMarkerHandle:
    """Marker for recovering values from the most recent active row."""

    marker: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastLike,
        name: str,
        active: Node,
    ) -> "RecentMarkerHandle":
        """Publish a recency marker.

        Values recovered through this marker are normally published in sibling
        channels on every row. The marker is what says which rows are valid
        members of the logical record stream.
        """
        return cls(marker=past.publish(f"{name}_marker", active))

    def pick(
        self,
        past: PastLike,
        value: PastHandle,
        *,
        match_gain: float = MATCH_GAIN_LONG,
    ) -> Node:
        """Read ``value`` from the most recent row where this marker was active."""
        return past.pick_most_recent(
            constant(1.0),
            self.marker,
            value,
            match_gain=match_gain,
        )


@dataclass(frozen=True)
class RecentKeyHandle:
    """Dynamic key for recovering values from the most recent matching row."""

    key: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastLike,
        name: str,
        active: Node,
        key: Node,
    ) -> "RecentKeyHandle":
        return cls(key=past.publish(f"{name}_key", gate(active, key)))

    def pick(
        self,
        past: PastLike,
        query: Node,
        value: PastHandle,
        *,
        match_gain: float = MATCH_GAIN_LONG,
    ) -> Node:
        """Read ``value`` from the most recent row whose key matches ``query``."""
        return past.pick_most_recent(
            query,
            self.key,
            value,
            match_gain=match_gain,
        )


@dataclass(frozen=True)
class KeyValueLookup:
    """Callable keyed-value lookup backed by a published key/value handle."""

    past: PastLike
    handle: KeyValueHandle
    width: int

    def __call__(self, query_id: Node) -> Node:
        """Read the value whose published key matches ``query_id``."""
        return self.handle.pick(self.past, query_id, self.width)


@dataclass(frozen=True)
class LiftedKeyValueLookup:
    """Callable lifted-id lookup backed by a published key/value handle."""

    past: PastLike
    handle: LiftedKeyValueHandle

    def __call__(self, query_id: Node) -> Node:
        return self.handle.pick(self.past, query_id)


@dataclass(frozen=True)
class OptionalKeyValueLookup:
    """Callable optional keyed-value lookup with a presence test."""

    past: PastLike
    handle: OptionalKeyValueHandle
    width: int

    def __call__(self, query_id: Node) -> Node:
        """Read the value whose published key matches ``query_id``."""
        return self.handle.pick(self.past, query_id, self.width)

    def present(self, query_id: Node) -> Node:
        """Read whether an optional value was published at ``query_id``."""
        marker = self.handle.pick_marker(self.past, query_id, self.width)
        return MARKER_PRESENT(marker)


@dataclass(frozen=True)
class KeyPresenceLookup:
    """Callable keyed-marker lookup returning a boolean presence value."""

    past: PastLike
    handle: KeyMarkerHandle
    width: int

    def __call__(self, query_id: Node) -> Node:
        """Read whether a marker was published at ``query_id``."""
        marker = self.handle.pick(self.past, query_id, self.width)
        return MARKER_PRESENT(marker)
