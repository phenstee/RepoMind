"""Offline tests for versioned strategy-independent retrieval evaluation."""

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from repomind.evaluation import (
    EvaluationMode,
    RetrievalBenchmarkCase,
    RetrievalBenchmarkSuite,
    evaluate_retrieval,
)
from repomind.ingestion import CodeChunk
from repomind.retrieval import (
    BM25Index,
    EmbeddedChunk,
    EmbeddingVector,
    RankedChunk,
    RerankedSearchResult,
    SemanticSearchResult,
    chunk_identity,
    hybrid_search,
    semantic_search,
)

EXACT_QUERY = "OPENAI_EMBEDDING_MODEL"
NATURAL_QUERY = "Where is retry behavior for failed model requests implemented?"


def _chunk(path: str, content: str, index: int) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=1,
        end_line=max(1, len(content.splitlines())),
        content=content,
        chunk_index=index,
    )


def _corpus() -> list[EmbeddedChunk]:
    chunks_and_vectors = [
        (
            _chunk(
                "src/llm/client.py",
                "async def retry_failed_request():\n    await asyncio.sleep(delay)\n",
                0,
            ),
            (0.0, 1.0, 0.0),
        ),
        (
            _chunk("tests/test_llm.py", "def test_failed_request_retry(): pass\n", 0),
            (0.0, 0.8, 0.6),
        ),
        (
            _chunk(
                "src/config.py",
                'OPENAI_EMBEDDING_MODEL = "text-embedding-test"\n',
                0,
            ),
            (0.55, 0.1, 0.83),
        ),
        (
            _chunk("src/embeddings.py", "def embed_chunks(texts): pass\n", 0),
            (0.95, 0.0, 0.31),
        ),
        (
            _chunk("src/db.py", "def persist_chunks_to_postgresql(): pass\n", 0),
            (0.0, 0.2, 0.98),
        ),
        (
            _chunk("src/hybrid.py", "def reciprocal_rank_fusion(): pass\n", 0),
            (0.3, 0.3, 0.9),
        ),
        (
            _chunk("web/styles.css", ".button { color: blue; }\n", 0),
            (0.1, 0.0, 0.99),
        ),
    ]
    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=EmbeddingVector(values=vector, model="fixture-model"),
        )
        for chunk, vector in chunks_and_vectors
    ]


class _QueryEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_text(self, text: str) -> EmbeddingVector:
        self.calls.append(text)
        vector = (1.0, 0.0, 0.0) if text == EXACT_QUERY else (0.0, 1.0, 0.0)
        return EmbeddingVector(values=vector, model="fixture-model")


class _DeterministicReranker:
    def rerank(
        self,
        query: str,
        candidates: Sequence[RankedChunk],
        *,
        top_k: int,
    ) -> list[RerankedSearchResult]:
        preferred = "src/config.py" if query == EXACT_QUERY else "src/llm/client.py"
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.chunk.relative_path.as_posix() != preferred,
                candidate.rank,
            ),
        )[:top_k]
        return [
            RerankedSearchResult(
                chunk=candidate.chunk,
                rank=rank,
                original_rank=candidate.rank,
            )
            for rank, candidate in enumerate(ordered, start=1)
        ]


def _suite(corpus: Sequence[EmbeddedChunk]) -> RetrievalBenchmarkSuite:
    by_path = {item.chunk.relative_path.as_posix(): item.chunk for item in corpus}
    return RetrievalBenchmarkSuite(
        cases=(
            RetrievalBenchmarkCase(
                id="exact-identifier",
                query=EXACT_QUERY,
                relevant_chunks=(chunk_identity(by_path["src/config.py"]),),
            ),
            RetrievalBenchmarkCase(
                id="natural-language-retry",
                query=NATURAL_QUERY,
                relevant_chunks=(chunk_identity(by_path["src/llm/client.py"]),),
            ),
        )
    )


def test_same_suite_runs_semantic_bm25_hybrid_and_reranked_strategies() -> None:
    corpus = _corpus()
    suite = _suite(corpus)
    chunks = [item.chunk for item in corpus]
    bm25 = BM25Index.from_chunks(chunks)
    semantic_provider = _QueryEmbeddingProvider()
    hybrid_provider = _QueryEmbeddingProvider()

    def semantic(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return semantic_search(query, corpus, semantic_provider, top_k=top_k)

    def lexical(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return bm25.search(query, top_k=top_k)

    def hybrid(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return hybrid_search(query, corpus, bm25, hybrid_provider, top_k=top_k)

    reranker = _DeterministicReranker()

    def reranked(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        candidates = hybrid_search(
            query,
            corpus,
            bm25,
            hybrid_provider,
            top_k=len(corpus),
        )
        return reranker.rerank(query, candidates, top_k=top_k)

    reports = [
        evaluate_retrieval(suite, "semantic", semantic, k=3),
        evaluate_retrieval(suite, "bm25", lexical, k=3),
        evaluate_retrieval(suite, "hybrid", hybrid, k=3),
        evaluate_retrieval(suite, "hybrid+rerank", reranked, k=3),
    ]

    assert [report.strategy for report in reports] == [
        "semantic",
        "bm25",
        "hybrid",
        "hybrid+rerank",
    ]
    assert all(report.mode is EvaluationMode.OFFLINE_FIXTURE for report in reports)
    assert all(report.case_count == 2 for report in reports)
    assert reports[0].case_results[0].first_relevant_rank > 1
    assert reports[1].case_results[0].first_relevant_rank == 1
    assert reports[2].case_results[0].first_relevant_rank == 1
    assert reports[3].case_results[0].first_relevant_rank == 1
    assert semantic_provider.calls == [EXACT_QUERY, NATURAL_QUERY]
    assert hybrid_provider.calls == [
        EXACT_QUERY,
        NATURAL_QUERY,
        EXACT_QUERY,
        NATURAL_QUERY,
    ]
    assert reports[0].model_dump(mode="json")["mode"] == "offline_fixture"


def test_relevance_labels_are_used_only_after_retrieval() -> None:
    chunk_a = _chunk("a.py", "A\n", 0)
    chunk_b = _chunk("b.py", "B\n", 0)
    fixed = [SemanticSearchResult(chunk=chunk_a, score=1.0, rank=1)]
    calls: list[tuple[str, int]] = []

    def retriever(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        calls.append((query, top_k))
        return fixed

    suite_a = RetrievalBenchmarkSuite(
        cases=(
            RetrievalBenchmarkCase(
                id="case-a",
                query="same query",
                relevant_chunks=(chunk_identity(chunk_a),),
            ),
        )
    )
    suite_b = RetrievalBenchmarkSuite(
        cases=(
            RetrievalBenchmarkCase(
                id="case-b",
                query="same query",
                relevant_chunks=(chunk_identity(chunk_b),),
            ),
        )
    )

    result_a = evaluate_retrieval(suite_a, "fixed", retriever, k=1)
    result_b = evaluate_retrieval(suite_b, "fixed", retriever, k=1)

    assert calls == [("same query", 1), ("same query", 1)]
    assert result_a.case_results[0].retrieved_chunk_ids == (
        chunk_identity(chunk_a),
    )
    assert result_b.case_results[0].retrieved_chunk_ids == (
        chunk_identity(chunk_a),
    )
    assert result_a.mean_recall_at_k == 1.0
    assert result_b.mean_recall_at_k == 0.0


def test_benchmark_models_reject_empty_gold_and_duplicate_case_ids() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        RetrievalBenchmarkCase(id="empty", query="query", relevant_chunks=())

    chunk = _chunk("a.py", "A\n", 0)
    case = RetrievalBenchmarkCase(
        id="duplicate",
        query="query",
        relevant_chunks=(chunk_identity(chunk),),
    )
    with pytest.raises(ValidationError, match="case IDs must be unique"):
        RetrievalBenchmarkSuite(cases=(case, case))
