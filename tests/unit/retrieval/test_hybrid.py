"""Tests for pure RRF and end-to-end offline hybrid retrieval."""

import pytest
from pydantic import ValidationError

from repomind.ingestion import CodeChunk
from repomind.rag import build_repository_context
from repomind.retrieval import (
    BM25Index,
    BM25SearchResult,
    EmbeddedChunk,
    EmbeddingVector,
    HybridSearchError,
    HybridSearchResult,
    SemanticSearchResult,
    chunk_identity,
    hybrid_search,
    reciprocal_rank_fusion,
)


def _chunk(path: str, content: str, index: int = 0) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=index + 1,
        end_line=index + max(1, len(content.splitlines())),
        content=content,
        chunk_index=index,
    )


def _semantic(chunk: CodeChunk, rank: int) -> SemanticSearchResult:
    return SemanticSearchResult(chunk=chunk, score=1.0 - rank / 10, rank=rank)


def _lexical(chunk: CodeChunk, rank: int) -> BM25SearchResult:
    return BM25SearchResult(chunk=chunk, score=10.0 / rank, rank=rank)


def _embedded(chunk: CodeChunk, values: tuple[float, ...]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=chunk,
        embedding=EmbeddingVector(values=values, model="test-model"),
    )


class _FakeEmbeddingProvider:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.embedding = EmbeddingVector(values=values, model="test-model")
        self.calls: list[str] = []

    def embed_text(self, text: str) -> EmbeddingVector:
        self.calls.append(text)
        return self.embedding


def test_rrf_rewards_chunks_present_in_both_rankings_with_exact_scores() -> None:
    a = _chunk("a.py", "A")
    b = _chunk("b.py", "B")
    c = _chunk("c.py", "C")
    d = _chunk("d.py", "D")

    results = reciprocal_rank_fusion(
        [_semantic(a, 1), _semantic(b, 2), _semantic(c, 3)],
        [_lexical(c, 1), _lexical(a, 2), _lexical(d, 3)],
        top_k=4,
        rrf_k=60,
    )

    assert [result.chunk.relative_path.as_posix() for result in results] == [
        "a.py",
        "c.py",
        "b.py",
        "d.py",
    ]
    assert results[0].fusion_score == pytest.approx(1 / 61 + 1 / 62)
    assert results[1].fusion_score == pytest.approx(1 / 63 + 1 / 61)
    assert results[2].fusion_score == pytest.approx(1 / 62)
    assert results[3].fusion_score == pytest.approx(1 / 63)
    assert (results[0].semantic_rank, results[0].lexical_rank) == (1, 2)


def test_chunk_identity_matches_equal_metadata_not_object_identity() -> None:
    first = _chunk("same.py", "first object")
    second = first.model_copy()

    assert first is not second
    assert chunk_identity(first) == chunk_identity(second)
    fused = reciprocal_rank_fusion(
        [_semantic(first, 1)],
        [_lexical(second, 1)],
        top_k=1,
    )
    assert len(fused) == 1
    assert fused[0].semantic_rank == fused[0].lexical_rank == 1


def test_rrf_rejects_conflicting_chunks_with_the_same_source_identity() -> None:
    first = _chunk("same.py", "first content")
    conflicting = first.model_copy(update={"content": "different content"})

    with pytest.raises(HybridSearchError, match="Conflicting chunks"):
        reciprocal_rank_fusion(
            [_semantic(first, 1)],
            [_lexical(conflicting, 1)],
            top_k=1,
        )


def test_rrf_exact_ties_use_best_rank_then_chunk_identity() -> None:
    z = _chunk("z.py", "Z")
    a = _chunk("a.py", "A")

    first = reciprocal_rank_fusion(
        [_semantic(z, 1), _semantic(a, 2)],
        [_lexical(a, 1), _lexical(z, 2)],
        top_k=2,
    )
    second = reciprocal_rank_fusion(
        [_semantic(z, 1), _semantic(a, 2)],
        [_lexical(a, 1), _lexical(z, 2)],
        top_k=2,
    )

    assert [result.chunk.relative_path.as_posix() for result in first] == [
        "a.py",
        "z.py",
    ]
    assert first == second


def test_rrf_rejects_duplicate_identity_within_one_ranking() -> None:
    chunk = _chunk("same.py", "same")
    with pytest.raises(HybridSearchError, match="Duplicate semantic"):
        reciprocal_rank_fusion(
            [_semantic(chunk, 1), _semantic(chunk, 2)],
            [],
            top_k=2,
        )


def test_hybrid_result_requires_a_contributing_rank() -> None:
    with pytest.raises(ValidationError, match="contributing rank"):
        HybridSearchResult(
            chunk=_chunk("none.py", "none"),
            rank=1,
            fusion_score=0.1,
        )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_rrf_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(HybridSearchError, match="positive integer"):
        reciprocal_rank_fusion([], [], top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("rrf_k", [0, -1, True, float("inf"), float("nan")])
def test_rrf_rejects_invalid_constant(rrf_k: object) -> None:
    with pytest.raises(HybridSearchError, match="positive finite"):
        reciprocal_rank_fusion([], [], top_k=1, rrf_k=rrf_k)  # type: ignore[arg-type]


def test_hybrid_search_embeds_exact_query_once_and_uses_default_candidate_depth() -> None:
    chunks = [_chunk(f"{index}.py", f"term {index}", index) for index in range(6)]
    embedded = [_embedded(chunk, (1.0, float(index))) for index, chunk in enumerate(chunks)]
    provider = _FakeEmbeddingProvider((1.0, 0.0))
    query = "  term  "

    results = hybrid_search(
        query,
        embedded,
        BM25Index.from_chunks(chunks),
        provider,
        top_k=1,
    )

    assert provider.calls == [query]
    assert len(results) == 1


def test_exact_identifier_lexical_rank_improves_hybrid_position() -> None:
    conceptual = _chunk("concept.py", "provider settings and configuration")
    unrelated = _chunk("search.py", "vector similarity implementation")
    exact = _chunk("settings.py", 'OPENAI_EMBEDDING_MODEL = "model"')
    embedded = [
        _embedded(conceptual, (1.0, 0.0)),
        _embedded(unrelated, (0.8, 0.2)),
        _embedded(exact, (0.6, 0.4)),
    ]
    provider = _FakeEmbeddingProvider((1.0, 0.0))

    results = hybrid_search(
        "OPENAI_EMBEDDING_MODEL",
        embedded,
        BM25Index.from_chunks([conceptual, unrelated, exact]),
        provider,
        top_k=3,
    )

    assert results[0].chunk == exact
    assert results[0].semantic_rank == 3
    assert results[0].lexical_rank == 1


def test_semantic_signal_keeps_natural_language_paraphrase_highly_ranked() -> None:
    relevant = _chunk("retry.py", "bounded exponential backoff retries model requests")
    lexical = _chunk("transport.py", "application model requests requests requests")
    unrelated = _chunk("styles.py", "page colors and spacing")
    embedded = [
        _embedded(relevant, (1.0, 0.0)),
        _embedded(lexical, (0.2, 0.8)),
        _embedded(unrelated, (0.0, 1.0)),
    ]
    provider = _FakeEmbeddingProvider((1.0, 0.0))

    results = hybrid_search(
        "Where does the application retry failed model requests?",
        embedded,
        BM25Index.from_chunks([relevant, lexical, unrelated]),
        provider,
        top_k=3,
    )

    assert results[0].chunk == relevant
    assert results[0].semantic_rank == 1
    assert results[0].lexical_rank is not None


def test_hybrid_results_feed_existing_rag_context_builder() -> None:
    chunk = _chunk("auth.py", "def authenticate(): return True")
    provider = _FakeEmbeddingProvider((1.0, 0.0))
    results = hybrid_search(
        "authenticate",
        [_embedded(chunk, (1.0, 0.0))],
        BM25Index.from_chunks([chunk]),
        provider,
        top_k=1,
    )

    context = build_repository_context(results)

    assert context.sources[0].chunk == chunk
    assert "auth.py" in context.text


@pytest.mark.parametrize("query", ["", " ", "\t\r\n"])
def test_hybrid_search_rejects_blank_query_without_embedding(query: str) -> None:
    chunk = _chunk("one.py", "content")
    provider = _FakeEmbeddingProvider((1.0, 0.0))

    with pytest.raises(HybridSearchError, match="must not be empty"):
        hybrid_search(
            query,
            [_embedded(chunk, (1.0, 0.0))],
            BM25Index.from_chunks([chunk]),
            provider,
        )

    assert provider.calls == []


@pytest.mark.parametrize("candidate_name", ["semantic_candidates", "lexical_candidates"])
def test_hybrid_search_rejects_invalid_candidate_depth(candidate_name: str) -> None:
    provider = _FakeEmbeddingProvider((1.0, 0.0))
    arguments = {candidate_name: 0}

    with pytest.raises(HybridSearchError, match=candidate_name):
        hybrid_search(
            "query",
            [],
            BM25Index.from_chunks([]),
            provider,
            **arguments,  # type: ignore[arg-type]
        )

    assert provider.calls == []
