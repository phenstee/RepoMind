"""Tests for deterministic in-memory semantic search."""

import pytest
from pydantic import ValidationError

from repomind.ingestion import CodeChunk
from repomind.retrieval import (
    EmbeddedChunk,
    EmbeddingVector,
    SemanticSearchError,
    SemanticSearchResult,
    SimilarityError,
    rank_by_similarity,
    semantic_search,
)


def _embedded_chunk(
    path: str,
    values: tuple[float, ...],
    *,
    model: str = "test-model",
) -> EmbeddedChunk:
    chunk = CodeChunk(
        relative_path=path,
        language="python",
        start_line=1,
        end_line=1,
        content=f"content from {path}\n",
        chunk_index=0,
    )
    return EmbeddedChunk(
        chunk=chunk,
        embedding=EmbeddingVector(model=model, values=values),
    )


def _query(
    values: tuple[float, ...] = (1.0, 0.0), *, model: str = "test-model"
) -> EmbeddingVector:
    return EmbeddingVector(model=model, values=values)


def test_rank_by_similarity_orders_known_vectors_and_assigns_ranks() -> None:
    chunks = [
        _embedded_chunk("auth.py", (0.95, 0.05)),
        _embedded_chunk("database.py", (0.7, 0.3)),
        _embedded_chunk("styles.css", (0.0, 1.0)),
        _embedded_chunk("opposite.py", (-1.0, 0.0)),
    ]

    results = rank_by_similarity(_query(), chunks, top_k=4)

    assert [str(result.chunk.relative_path) for result in results] == [
        "auth.py",
        "database.py",
        "styles.css",
        "opposite.py",
    ]
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )


@pytest.mark.parametrize(("top_k", "expected_count"), [(1, 1), (2, 2), (10, 3)])
def test_rank_by_similarity_honors_top_k(top_k: int, expected_count: int) -> None:
    chunks = [
        _embedded_chunk("first.py", (1.0, 0.0)),
        _embedded_chunk("second.py", (0.5, 0.5)),
        _embedded_chunk("third.py", (0.0, 1.0)),
    ]

    assert len(rank_by_similarity(_query(), chunks, top_k=top_k)) == expected_count


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_rank_by_similarity_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(SemanticSearchError, match="positive integer"):
        rank_by_similarity(_query(), [], top_k=top_k)  # type: ignore[arg-type]


def test_rank_by_similarity_returns_empty_result_for_empty_corpus() -> None:
    assert rank_by_similarity(_query(), [], top_k=5) == []


def test_rank_by_similarity_preserves_input_order_for_ties() -> None:
    chunks = [
        _embedded_chunk("first.py", (1.0, 1.0)),
        _embedded_chunk("second.py", (1.0, 1.0)),
        _embedded_chunk("third.py", (1.0, 1.0)),
    ]

    first_run = rank_by_similarity(_query((1.0, 1.0)), chunks, top_k=3)
    second_run = rank_by_similarity(_query((1.0, 1.0)), chunks, top_k=3)

    expected_paths = ["first.py", "second.py", "third.py"]
    assert [str(result.chunk.relative_path) for result in first_run] == expected_paths
    assert [str(result.chunk.relative_path) for result in second_run] == expected_paths


def test_rank_by_similarity_rejects_embedding_model_mismatch() -> None:
    chunks = [_embedded_chunk("chunk.py", (1.0, 0.0), model="other-model")]

    with pytest.raises(SemanticSearchError, match="model"):
        rank_by_similarity(_query(model="query-model"), chunks, top_k=1)


def test_rank_by_similarity_rejects_mixed_corpus_models() -> None:
    chunks = [
        _embedded_chunk("valid.py", (1.0, 0.0)),
        _embedded_chunk("invalid.py", (1.0, 0.0), model="other-model"),
    ]

    with pytest.raises(SemanticSearchError, match="invalid.py"):
        rank_by_similarity(_query(), chunks, top_k=2)


def test_rank_by_similarity_rejects_dimension_mismatch() -> None:
    chunks = [_embedded_chunk("chunk.py", (1.0, 0.0, 0.0))]

    with pytest.raises(SimilarityError, match="same dimensions"):
        rank_by_similarity(_query(), chunks, top_k=1)


@pytest.mark.parametrize(
    ("score", "rank"),
    [
        (float("nan"), 1),
        (float("inf"), 1),
        (1.01, 1),
        (-1.01, 1),
        (0.5, 0),
    ],
)
def test_semantic_search_result_rejects_invalid_values(score: float, rank: int) -> None:
    with pytest.raises(ValidationError):
        SemanticSearchResult(
            chunk=_embedded_chunk("chunk.py", (1.0, 0.0)).chunk,
            score=score,
            rank=rank,
        )


class _FakeEmbeddingProvider:
    def __init__(self, embedding: EmbeddingVector) -> None:
        self.embedding = embedding
        self.calls: list[str] = []

    def embed_text(self, text: str) -> EmbeddingVector:
        self.calls.append(text)
        return self.embedding


def test_semantic_search_embeds_query_once_and_preserves_exact_text() -> None:
    provider = _FakeEmbeddingProvider(_query())
    chunks = [
        _embedded_chunk("auth.py", (1.0, 0.0)),
        _embedded_chunk("styles.css", (0.0, 1.0)),
    ]
    query = "  authentication middleware?  "

    results = semantic_search(query, chunks, provider, top_k=1)

    assert provider.calls == [query]
    assert [str(result.chunk.relative_path) for result in results] == ["auth.py"]


@pytest.mark.parametrize("query", ["", " ", "\t\r\n"])
def test_semantic_search_rejects_blank_query_without_embedding(query: str) -> None:
    provider = _FakeEmbeddingProvider(_query())

    with pytest.raises(SemanticSearchError, match="must not be empty"):
        semantic_search(query, [_embedded_chunk("chunk.py", (1.0, 0.0))], provider)

    assert provider.calls == []


def test_semantic_search_skips_embedding_for_empty_corpus() -> None:
    provider = _FakeEmbeddingProvider(_query())

    assert semantic_search("valid query", [], provider) == []
    assert provider.calls == []
