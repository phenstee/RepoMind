"""Tests for deterministic identifier-shaped candidate extraction."""

import pytest

from repomind.retrieval.identifiers import (
    IdentifierConfidence,
    extract_identifier_candidates,
)


def _by_text(query: str) -> dict[str, tuple[bool, IdentifierConfidence]]:
    return {
        candidate.text: (candidate.qualified, candidate.confidence)
        for candidate in extract_identifier_candidates(query)
    }


def test_qualified_dotted_name_is_strong_and_whole() -> None:
    candidates = _by_text("UserService.login")

    assert candidates["UserService.login"] == (True, IdentifierConfidence.STRONG)
    assert "UserService" not in candidates
    assert "login" not in candidates


def test_snake_case_identifier_is_strong() -> None:
    candidates = _by_text("load_neighbor_chunks")

    assert candidates["load_neighbor_chunks"] == (False, IdentifierConfidence.STRONG)


def test_pascal_case_identifier_is_strong() -> None:
    candidates = _by_text("ContextAssembler")

    assert candidates["ContextAssembler"] == (False, IdentifierConfidence.STRONG)


def test_camel_case_identifier_is_strong() -> None:
    candidates = _by_text("camelCaseName")

    assert candidates["camelCaseName"] == (False, IdentifierConfidence.STRONG)


def test_plain_lowercase_word_is_weak() -> None:
    for word in ("run", "get", "set", "test", "save", "load", "login"):
        candidates = _by_text(word)
        assert candidates[word] == (False, IdentifierConfidence.WEAK), word


def test_natural_language_question_extracts_qualified_identifier() -> None:
    candidates = _by_text("Where is UserService.login implemented?")

    assert candidates["UserService.login"] == (True, IdentifierConfidence.STRONG)
    # Ordinary prose words are still emitted, but only as weak candidates.
    assert candidates["Where"][1] is IdentifierConfidence.WEAK
    assert candidates["implemented"][1] is IdentifierConfidence.WEAK


def test_natural_language_question_extracts_snake_case_identifier() -> None:
    candidates = _by_text("How does load_neighbor_chunks avoid N+1 queries?")

    assert candidates["load_neighbor_chunks"] == (False, IdentifierConfidence.STRONG)


def test_common_word_query_has_no_qualified_or_strong_candidates() -> None:
    candidates = extract_identifier_candidates("run validation before saving")

    assert all(not c.qualified for c in candidates)
    assert all(c.confidence is IdentifierConfidence.WEAK for c in candidates)


def test_original_query_text_is_never_modified() -> None:
    query = "How does UserService.login validate credentials?"

    extract_identifier_candidates(query)

    assert query == "How does UserService.login validate credentials?"


def test_duplicate_words_are_deduplicated_preserving_first_seen_order() -> None:
    candidates = extract_identifier_candidates("login attempt then login again")

    texts = [c.text for c in candidates]
    assert texts.count("login") == 1
    assert texts.index("login") < texts.index("attempt")


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        extract_identifier_candidates(123)  # type: ignore[arg-type]


def test_short_and_numeric_tokens_are_ignored() -> None:
    candidates = extract_identifier_candidates("a 42 is ok")

    texts = {c.text for c in candidates}
    assert "a" not in texts
    assert "42" not in texts
    assert "is" in texts
    assert "ok" in texts
