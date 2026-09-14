"""Offline integration tests for configurable high-level repository RAG."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from repomind.ingestion import CodeChunk
from repomind.rag import (
    RAGConfig,
    RAGError,
    answer_repository_question,
    answer_repository_question_with_retriever,
)
from repomind.retrieval import (
    BM25Index,
    EmbeddedChunk,
    EmbeddingVector,
    HybridSearchResult,
    LLMReranker,
    RankedChunk,
    RerankingError,
    hybrid_search,
)


def _chunk(
    path: str,
    content: str,
    *,
    chunk_index: int,
    start_line: int = 1,
) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=start_line,
        end_line=start_line + max(1, len(content.splitlines())) - 1,
        content=content,
        chunk_index=chunk_index,
    )


def _embedded(chunk: CodeChunk, vector: tuple[float, ...]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=chunk,
        embedding=EmbeddingVector(values=vector, model="test-model"),
    )


class _FakeEmbeddingProvider:
    def __init__(self, vector: tuple[float, ...] = (1.0, 0.0)) -> None:
        self.vector = EmbeddingVector(values=vector, model="test-model")
        self.calls: list[str] = []

    def embed_text(self, text: str) -> EmbeddingVector:
        self.calls.append(text)
        return self.vector


class _FakeAnswerLLM:
    def __init__(self, source_ids: list[str] | None = None) -> None:
        self.source_ids = ["S1"] if source_ids is None else source_ids
        self.calls: list[dict[str, Any]] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
            }
        )
        return response_model(
            answer="Grounded answer.",
            source_ids=self.source_ids,
            insufficient_evidence=False,
        )


class _FakeRerankLLM:
    def __init__(self, candidate_ids: list[str]) -> None:
        self.candidate_ids = candidate_ids
        self.calls = 0

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        self.calls += 1
        return response_model(ranked_candidate_ids=self.candidate_ids)


class _HybridRetriever:
    def __init__(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
        embedding_provider: _FakeEmbeddingProvider,
    ) -> None:
        self.embedded_chunks = embedded_chunks
        self.embedding_provider = embedding_provider
        self.bm25_index = BM25Index.from_chunks(
            [embedded.chunk for embedded in embedded_chunks]
        )
        self.results: list[HybridSearchResult] = []

    def __call__(self, query: str, *, top_k: int) -> Sequence[RankedChunk]:
        self.results = hybrid_search(
            query,
            self.embedded_chunks,
            self.bm25_index,
            self.embedding_provider,
            top_k=top_k,
        )
        return self.results


class _RerankedRetriever:
    def __init__(
        self,
        candidates: Sequence[HybridSearchResult],
        reranker: LLMReranker,
    ) -> None:
        self.candidates = candidates
        self.reranker = reranker

    def __call__(self, query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return self.reranker.rerank(query, self.candidates, top_k=top_k)


def test_semantic_only_public_rag_remains_the_default_baseline() -> None:
    semantic_first = _chunk(
        "src/semantic.py",
        "def explain_retry_policy(): pass\n",
        chunk_index=0,
    )
    corpus = [
        _embedded(semantic_first, (1.0, 0.0)),
        _embedded(
            _chunk("src/other.py", "OTHER = True\n", chunk_index=0),
            (0.0, 1.0),
        ),
    ]
    embedding_provider = _FakeEmbeddingProvider()

    answer = answer_repository_question(
        "Where is retry policy explained?",
        corpus,
        embedding_provider,
        _FakeAnswerLLM(),
        config=RAGConfig(top_k=1),
    )

    assert embedding_provider.calls == ["Where is retry policy explained?"]
    assert answer.citations[0].relative_path == Path("src/semantic.py")


def test_hybrid_public_rag_promotes_an_exact_identifier_over_semantic_first() -> None:
    semantic_first = _chunk(
        "src/retry.py",
        "DEFAULT_TIMEOUT = 30\n",
        chunk_index=0,
    )
    identifier = _chunk(
        "src/config.py",
        'OPENAI_EMBEDDING_MODEL = "text-embedding-test"\n',
        chunk_index=0,
        start_line=12,
    )
    corpus = [
        _embedded(semantic_first, (1.0, 0.0)),
        _embedded(identifier, (0.8, 0.6)),
        _embedded(
            _chunk("src/colors.py", "BACKGROUND = 'blue'\n", chunk_index=0),
            (0.0, 1.0),
        ),
    ]

    semantic_answer = answer_repository_question(
        "OPENAI_EMBEDDING_MODEL",
        corpus,
        _FakeEmbeddingProvider(),
        _FakeAnswerLLM(),
        config=RAGConfig(top_k=1),
    )
    hybrid_embedding_provider = _FakeEmbeddingProvider()
    retriever = _HybridRetriever(corpus, hybrid_embedding_provider)
    final_llm = _FakeAnswerLLM()
    hybrid_answer = answer_repository_question_with_retriever(
        "OPENAI_EMBEDDING_MODEL",
        retriever,
        final_llm,
        config=RAGConfig(top_k=2),
    )

    assert semantic_answer.citations[0].relative_path == Path("src/retry.py")
    assert retriever.results[0].chunk == identifier
    assert retriever.results[0].semantic_rank == 2
    assert retriever.results[0].lexical_rank == 1
    assert hybrid_embedding_provider.calls == ["OPENAI_EMBEDDING_MODEL"]
    assert "<path>src/config.py</path>" in final_llm.calls[0]["prompt"]
    assert hybrid_answer.citations[0].relative_path == Path("src/config.py")
    assert hybrid_answer.citations[0].start_line == 12


def test_hybrid_public_rag_preserves_semantically_relevant_implementation() -> None:
    question = "Where is retry behavior for failed model requests implemented?"
    implementation = _chunk(
        "src/llm/client.py",
        "async def retry_failed_request():\n    await asyncio.sleep(delay)\n",
        chunk_index=0,
        start_line=40,
    )
    documentation = _chunk(
        "README.md",
        "Retry behavior for failed model requests is implemented in the client.\n",
        chunk_index=0,
    )
    corpus = [
        _embedded(implementation, (1.0, 0.0)),
        _embedded(
            _chunk("src/logging.py", "def log_request(): pass\n", chunk_index=0),
            (0.7, 0.7),
        ),
        _embedded(documentation, (0.0, 1.0)),
    ]
    retriever = _HybridRetriever(corpus, _FakeEmbeddingProvider())
    final_llm = _FakeAnswerLLM()

    answer = answer_repository_question_with_retriever(
        question,
        retriever,
        final_llm,
        config=RAGConfig(top_k=2),
    )

    implementation_result = next(
        result for result in retriever.results if result.chunk == implementation
    )
    assert implementation_result.semantic_rank == 1
    assert implementation_result.lexical_rank is not None
    assert "<path>src/llm/client.py</path>" in final_llm.calls[0]["prompt"]
    assert answer.citations[0].relative_path == Path("src/llm/client.py")


def _hybrid_candidates() -> list[HybridSearchResult]:
    return [
        HybridSearchResult(
            chunk=_chunk("src/one.py", "one\n", chunk_index=0, start_line=10),
            rank=1,
            fusion_score=0.03,
            semantic_rank=1,
        ),
        HybridSearchResult(
            chunk=_chunk("src/two.py", "two\n", chunk_index=0, start_line=20),
            rank=2,
            fusion_score=0.02,
            lexical_rank=1,
        ),
        HybridSearchResult(
            chunk=_chunk(
                "src/three.py",
                "def three():\n    return 3\n",
                chunk_index=0,
                start_line=30,
            ),
            rank=3,
            fusion_score=0.01,
            semantic_rank=2,
            lexical_rank=2,
        ),
    ]


def test_reranked_hybrid_order_becomes_context_and_real_citation_order() -> None:
    rerank_llm = _FakeRerankLLM(["C3", "C1", "C2"])
    retriever = _RerankedRetriever(
        _hybrid_candidates(),
        LLMReranker(rerank_llm),
    )
    final_llm = _FakeAnswerLLM(["S1"])

    answer = answer_repository_question_with_retriever(
        "Which candidate is most relevant?",
        retriever,
        final_llm,
        config=RAGConfig(top_k=3),
    )

    prompt = final_llm.calls[0]["prompt"]
    assert prompt.index("<path>src/three.py</path>") < prompt.index(
        "<path>src/one.py</path>"
    )
    assert rerank_llm.calls == 1
    assert answer.citations[0].relative_path == Path("src/three.py")
    assert answer.citations[0].start_line == 30
    assert answer.citations[0].end_line == 31


def test_invalid_reranker_candidate_id_stops_before_final_generation() -> None:
    retriever = _RerankedRetriever(
        _hybrid_candidates(),
        LLMReranker(_FakeRerankLLM(["C3", "C1", "C99"])),
    )
    final_llm = _FakeAnswerLLM()

    with pytest.raises(RerankingError, match="unknown candidate IDs"):
        answer_repository_question_with_retriever(
            "Which candidate is most relevant?",
            retriever,
            final_llm,
            config=RAGConfig(top_k=3),
        )

    assert final_llm.calls == []


def test_reranked_context_still_rejects_forged_final_source_id() -> None:
    retriever = _RerankedRetriever(
        _hybrid_candidates(),
        LLMReranker(_FakeRerankLLM(["C3", "C1", "C2"])),
    )

    with pytest.raises(RAGError, match="unknown source ID.*S99"):
        answer_repository_question_with_retriever(
            "Which candidate is most relevant?",
            retriever,
            _FakeAnswerLLM(["S99"]),
            config=RAGConfig(top_k=3),
        )
