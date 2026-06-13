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

The keyed lookups use *lifted* id keys: an integer id is encoded as
``[id, -id^2, 1]`` so one attention dot-product peaks at exact id equality (see
:func:`lifted_id_query` and ``GLOSSARY.md``), instead of a width-N one-hot.

Ported from ``doom_sandbox/implementation/forward/attention_handles.py``.
The only changes from the sandbox source are the import
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
from .render_ops import MARKER_PRESENT, same_int
from .std import concat, constant, gate, linear, one_hot, select

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
        """Read ``value`` from the most recent row where this marker was active.

        This is a long-range recency read (it defaults to ``MATCH_GAIN_LONG``).
        Per the windowed KV cache, the marker channel it lands on must be a
        PERMANENT (non-expiring) published channel: matching an expiring-type
        row beyond the resident window reads a recycled slot and returns a
        wrong value SILENTLY. See CLAUDE.md "Windowed KV cache — the protocol
        invariant".
        """
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
        """Read ``value`` from the most recent row whose key matches ``query``.

        This is a long-range recency read (it defaults to ``MATCH_GAIN_LONG``).
        Per the windowed KV cache, the key channel it lands on must be a
        PERMANENT (non-expiring) published channel: matching an expiring-type
        row beyond the resident window reads a recycled slot and returns a
        wrong value SILENTLY. See CLAUDE.md "Windowed KV cache — the protocol
        invariant".
        """
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


# Presence query offset. A one-hot presence key detects absence cleanly (an
# absent query scores 0 against every key). A lifted-equality key has no
# "no-match" state -- it always returns the NEAREST id, scoring positive -- so
# absence is detected by recovering the matched row's id and testing equality
# to the query. For that recovery to be robust the winning row must always be a
# real (active) producer, never a gated-zero inactive row: an inactive row
# scores 0, while an active row scores ``K + q^2 - (id-q)^2``. The offset ``K``
# must exceed the largest ``(id-q)^2`` gap (max id ~128 -> 128^2 = 16384) so the
# nearest present id always outscores inactive rows even at q=0 (where the bare
# 1 + q^2 ~ 1 would tie them and blend the recovered id). K is common-mode
# across rows, so it leaves the match-vs-nearest margin at the >=1 logit unit of
# the (id-q)^2 term.
_PRESENCE_OFFSET = 20000.0
_LIFTED_PRESENCE_QUERY = [[2.0, 0.0, 0.0], [0.0, 1.0, _PRESENCE_OFFSET]]

# Sentinel id published on INACTIVE rows of a lifted presence lookup. Any value
# that can never equal a valid (non-negative) query id works. It keeps the
# degenerate empty-publisher case correct: if no active producer exists, every
# key scores 0, the softmax blends the inactive rows, and the recovered id must
# read ABSENT rather than blending to 0 and falsely matching a q=0 probe.
_ABSENT_ID_SENTINEL = -1.0


def lifted_presence_query(query_id: Node) -> Node:
    """Query ``[2q, 1, K]`` for lifted-equality presence detection.

    Paired with a producer key ``[id, -id^2, 1]`` (the shared ``id_lifted_key``
    derived) it scores ``K + q^2 - (id-q)^2``; see ``_PRESENCE_OFFSET``.
    """
    return linear(concat(query_id, constant(1.0)), _LIFTED_PRESENCE_QUERY)


@dataclass(frozen=True)
class LiftedKeyPresenceHandle:
    """Lifted-equality presence test: recover the matched id, compare to query.

    Replaces a width-``N`` ``one_hot`` presence key with the width-3 lifted key
    ``[id, -id^2, 1]``. The recoverable ``id_value`` is the id itself; a lookup
    recovers the nearest published id and the caller tests equality to the
    query, so a missing id (whose nearest neighbour is returned) reads absent.
    """

    key: PastHandle
    id_value: PastHandle

    @classmethod
    def publish(
        cls,
        past: PastLike,
        name: str,
        active: Node,
        lifted_key: Node,
        id_scalar: Node,
    ) -> "LiftedKeyPresenceHandle":
        return cls(
            key=past.publish(f"{name}_key", gate(active, lifted_key)),
            # Active rows publish the real id; inactive rows publish a sentinel
            # (not 0) so a degenerate query against an empty publisher set
            # recovers the sentinel and reads ABSENT -- never a false PRESENT at
            # q == 0. (q=0 is unreachable from current call sites, but this makes
            # presence correct by construction, not by call-site luck.)
            id_value=past.publish(
                f"{name}_id",
                select(active, id_scalar, constant(_ABSENT_ID_SENTINEL)),
            ),
        )

    def present(self, past: PastLike, query_id: Node) -> Node:
        """±1: was a producer with id == ``query_id`` published?"""
        recovered = past.pick_argmax(
            lifted_presence_query(query_id), self.key, self.id_value
        )
        return same_int(recovered, query_id)


@dataclass(frozen=True)
class LiftedKeyPresenceLookup:
    """Callable lifted presence test returning a boolean (±1) presence value."""

    past: PastLike
    handle: LiftedKeyPresenceHandle

    def __call__(self, query_id: Node) -> Node:
        return self.handle.present(self.past, query_id)
