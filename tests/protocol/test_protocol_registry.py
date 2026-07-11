"""Protocol-registry coverage and classification."""

from __future__ import annotations

from torchwright_doom import protocol_registry as reg
from torchwright_doom.protocol_tokens import ProtocolTokenView
from torchwright_doom.vocab import SEG_LIGHT_STATIC, VOCAB_TYPES


def test_registry_covers_vocab_exactly_once() -> None:
    reg_tokens = list(reg.PROTOCOL_BY_TOKEN)
    assert len(reg_tokens) == len(reg.PROTOCOL_ENTRIES)  # no duplicate tokens
    assert set(reg_tokens) == set(VOCAB_TYPES)
    assert len(reg.PROTOCOL_ENTRIES) == len(VOCAB_TYPES)


def test_seg_light_static_is_inert() -> None:
    assert SEG_LIGHT_STATIC in set(reg.INERT_NON_PAYLOAD_TYPES)


def test_grouped_predicate_excluded_from_token_checks() -> None:
    names = {p.predicate for p in reg.TOKEN_CHECK_PREDICATES}
    # is_inert_non_payload covers many tokens -> grouped, hand-written.
    assert "is_inert_non_payload" not in names
    # carrier one-token predicates are present.
    assert "is_value" in names
    assert "is_angle_value" in names


def test_every_token_check_predicate_exists_on_view() -> None:
    for spec in reg.TOKEN_CHECK_PREDICATES:
        assert hasattr(ProtocolTokenView, spec.predicate), spec.predicate


def test_every_dispatch_predicate_resolvable() -> None:
    """Each dispatch transition predicate is either installed from the registry
    or a hand-written ProtocolTokenView attribute."""
    for transition in reg.DISPATCH_TRANSITIONS:
        assert hasattr(ProtocolTokenView, transition.predicate), transition.predicate


def test_prefill_replay_predicates_are_derived() -> None:
    assert reg.PREFILL_REPLAY_PREDICATES == (
        "is_inert_non_payload",
        "is_scene_value_payload",
        "is_scene_angle_payload",
    )


def test_protocol_table_is_whitespace_clean() -> None:
    table = reg.render_protocol_table()
    for line in table.splitlines():
        assert line == line.rstrip(), f"trailing whitespace: {line!r}"
