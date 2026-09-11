"""Tests for deterministic code-aware lexical tokenization."""

import pytest

from repomind.retrieval import tokenize_code


def test_plain_english_is_case_normalized_and_frequency_is_preserved() -> None:
    assert tokenize_code("Token token AUTHENTICATION") == (
        "token",
        "token",
        "authentication",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "persist_embedded_chunks",
            ("persist_embedded_chunks", "persist", "embedded", "chunks"),
        ),
        (
            "OPENAI_EMBEDDING_MODEL",
            ("openai_embedding_model", "openai", "embedding", "model"),
        ),
        (
            "OpenAILLMClient",
            ("openaillmclient", "open", "ai", "llm", "client"),
        ),
        (
            "RepositoryIngestionError",
            ("repositoryingestionerror", "repository", "ingestion", "error"),
        ),
        (
            "src/repomind/retrieval",
            ("src/repomind/retrieval", "src", "repomind", "retrieval"),
        ),
        ("repository.py", ("repository.py", "repository", "py")),
        ("llm.client", ("llm.client", "llm", "client")),
        ("code-aware", ("code-aware", "code", "aware")),
        ("caféVariable", ("cafévariable", "café", "variable")),
    ],
)
def test_code_forms_keep_compound_and_split_components(
    text: str,
    expected: tuple[str, ...],
) -> None:
    assert tokenize_code(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "...", "---///___"])
def test_empty_or_punctuation_only_text_has_no_tokens(text: str) -> None:
    assert tokenize_code(text) == ()


def test_tokenizer_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        tokenize_code(123)  # type: ignore[arg-type]
