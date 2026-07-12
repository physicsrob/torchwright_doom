"""``TokenType`` equality is name-based, not identity-based."""

from __future__ import annotations

from torchwright_doom.model.tokens import FloatSlot, IntSlot, TokenType


def test_same_name_compares_equal_and_hashes_equal() -> None:
    a = TokenType("value", slots={"v": FloatSlot(-1.0, 1.0)})
    b = TokenType("value", slots={"v": FloatSlot(-1.0, 1.0)})
    assert a is not b  # genuinely distinct instances
    assert a == b  # ...but equal by name
    assert hash(a) == hash(b)
    # Usable interchangeably as dict / set keys by name.
    assert {a: 1}[b] == 1
    assert len({a, b}) == 1


def test_slots_do_not_affect_identity() -> None:
    # Same name, different slot definitions -> still equal (name is the key).
    a = TokenType("seg", slots={"i": IntSlot(0, 10)})
    b = TokenType("seg", slots={"i": IntSlot(0, 999)})
    assert a == b
    assert hash(a) == hash(b)


def test_different_name_not_equal() -> None:
    assert TokenType("value") != TokenType("angleValue")


def test_not_equal_to_non_tokentype() -> None:
    assert TokenType("value") != "value"
    # != None (not `is not None`) on purpose: this routes through __eq__/
    # __ne__ with a None operand, which `is not` would bypass.
    assert TokenType("value") != None  # noqa: E711
