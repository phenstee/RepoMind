"""Tests for deterministic persistence helpers."""

from repomind.db import content_sha256


def test_content_sha256_is_deterministic_and_has_expected_shape() -> None:
    first = content_sha256("def café():\r\n    return '雪'\r\n")
    second = content_sha256("def café():\r\n    return '雪'\r\n")

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_content_sha256_distinguishes_newline_encoding() -> None:
    assert content_sha256("line\n") != content_sha256("line\r\n")
