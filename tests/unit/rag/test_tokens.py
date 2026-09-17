"""Tests for the deterministic approximate token counter."""

import pytest

from repomind.rag.tokens import estimate_tokens


def test_empty_text_is_zero_tokens() -> None:
    assert estimate_tokens("") == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a", 1),
        ("abcd", 1),
        ("abcde", 2),
        ("x" * 100, 25),
        ("x" * 101, 26),
    ],
)
def test_estimate_matches_documented_ceil_div_four_rule(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


def test_estimate_is_deterministic() -> None:
    text = "def value():\n    return 1\n"
    assert estimate_tokens(text) == estimate_tokens(text)


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        estimate_tokens(123)  # type: ignore[arg-type]
