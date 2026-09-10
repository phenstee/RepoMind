"""Tests for RAG configuration and grounded-answer models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.rag import RAGConfig, RepositoryAnswer, SourceCitation


def test_rag_config_defaults_are_bounded() -> None:
    config = RAGConfig()

    assert config.top_k == 5
    assert config.max_context_chars == 20_000


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": 0},
        {"top_k": -1},
        {"top_k": True},
        {"max_context_chars": 0},
        {"max_context_chars": -1},
        {"max_context_chars": True},
    ],
)
def test_rag_config_rejects_invalid_limits(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RAGConfig(**overrides)  # type: ignore[arg-type]


def test_source_citation_accepts_safe_repository_location() -> None:
    citation = SourceCitation(
        relative_path="src/repomind/llm/client.py",
        start_line=10,
        end_line=30,
    )

    assert citation.relative_path == Path("src/repomind/llm/client.py")
    assert citation.start_line == 10
    assert citation.end_line == 30


@pytest.mark.parametrize(
    "relative_path",
    ["../secret.py", "/etc/passwd", "C:\\secret.py"],
)
def test_source_citation_reuses_safe_relative_path_validation(relative_path: str) -> None:
    with pytest.raises(ValidationError, match="safe repository-relative"):
        SourceCitation(relative_path=relative_path, start_line=1, end_line=1)


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    [(0, 1), (2, 1)],
)
def test_source_citation_rejects_invalid_line_range(
    start_line: int, end_line: int
) -> None:
    with pytest.raises(ValidationError):
        SourceCitation(
            relative_path="src/example.py",
            start_line=start_line,
            end_line=end_line,
        )


def test_repository_answer_accepts_grounded_citation() -> None:
    citation = SourceCitation(relative_path="src/example.py", start_line=1, end_line=2)
    answer = RepositoryAnswer(
        answer="The example is implemented in the cited source.",
        citations=[citation],
    )

    assert answer.citations == [citation]
    assert answer.insufficient_evidence is False


def test_repository_answer_accepts_insufficient_evidence_without_citations() -> None:
    answer = RepositoryAnswer(
        answer="The supplied context is insufficient.",
        insufficient_evidence=True,
    )

    assert answer.citations == []
    assert answer.insufficient_evidence is True


def test_repository_answer_rejects_sufficient_answer_without_citation() -> None:
    with pytest.raises(ValidationError, match="must include a citation"):
        RepositoryAnswer(answer="An unsupported factual answer.")


def test_repository_answer_rejects_empty_answer_text() -> None:
    with pytest.raises(ValidationError):
        RepositoryAnswer(answer="", insufficient_evidence=True)
