"""Tests wiring the context assembler into the RAG pipeline entry points."""

from pathlib import Path

import pytest

from repomind.ingestion import ChunkingConfig, ChunkingStrategy, SourceFile, chunk_source_file
from repomind.observability import InMemoryTraceRecorder, RunType, TraceContext
from repomind.rag import (
    ContextStrategy,
    RAGConfig,
    answer_repository_question,
    answer_repository_question_with_retriever,
)
from repomind.rag.assembly import InMemoryNeighborLoader
from repomind.retrieval import EmbeddedChunk, EmbeddingVector, SemanticSearchResult

_SOURCE = (
    "def verify_token(token):\n"
    "    if not token:\n"
    "        raise ValueError('token required')\n"
    "    return _lookup(token)\n"
)


class _FixedEmbeddingProvider:
    def __init__(self, vector: tuple[float, ...] = (1.0, 0.0)) -> None:
        self.embedding = EmbeddingVector(values=vector, model="test-model")

    def embed_text(self, text: str) -> EmbeddingVector:
        return self.embedding


class _RecordingLLM:
    def __init__(self, source_ids: list[str] | None = None) -> None:
        self.source_ids = source_ids or ["S1"]
        self.calls: list[dict] = []

    def generate_structured(self, prompt, response_model, *, system_prompt=None, temperature=None):
        self.calls.append({"prompt": prompt})
        return response_model(
            answer="Tokens are verified by checking presence then looking them up.",
            source_ids=self.source_ids,
            insufficient_evidence=False,
        )


def _split_chunks():
    source_file = SourceFile(
        relative_path=Path("src/token.py"),
        language="python",
        content=_SOURCE,
        size_bytes=len(_SOURCE.encode()),
        line_count=len(_SOURCE.splitlines()),
    )
    chunks = chunk_source_file(
        source_file,
        ChunkingConfig(strategy=ChunkingStrategy.LINE, max_lines_per_chunk=2, overlap_lines=0),
    )
    assert len(chunks) >= 2
    return chunks


def _embedded(chunks):
    provider = _FixedEmbeddingProvider()
    return [EmbeddedChunk(chunk=chunk, embedding=provider.embedding) for chunk in chunks]


def test_seeds_only_does_not_include_neighbor_implementation_lines() -> None:
    chunks = _split_chunks()
    llm = _RecordingLLM()

    answer_repository_question(
        "How are tokens verified?",
        _embedded(chunks),
        _FixedEmbeddingProvider(),
        llm,
        config=RAGConfig(top_k=1, context_strategy=ContextStrategy.SEEDS_ONLY),
    )

    prompt = llm.calls[0]["prompt"]
    assert "_lookup(token)" not in prompt


def test_expanded_strategy_pulls_in_neighbor_implementation_lines() -> None:
    chunks = _split_chunks()
    llm = _RecordingLLM()

    answer_repository_question(
        "How are tokens verified?",
        _embedded(chunks),
        _FixedEmbeddingProvider(),
        llm,
        config=RAGConfig(top_k=1, context_strategy=ContextStrategy.EXPANDED),
    )

    prompt = llm.calls[0]["prompt"]
    assert "def verify_token" in prompt
    assert "_lookup(token)" in prompt


def test_expanded_strategy_emits_context_assembled_trace_event() -> None:
    chunks = _split_chunks()
    llm = _RecordingLLM()
    trace = TraceContext(InMemoryTraceRecorder(), RunType.RAG)

    answer_repository_question(
        "How are tokens verified?",
        _embedded(chunks),
        _FixedEmbeddingProvider(),
        llm,
        config=RAGConfig(top_k=1, context_strategy=ContextStrategy.EXPANDED),
        trace=trace,
    )

    events = [event for event in trace.events if event.event_type == "context.assembled"]
    assert len(events) == 1
    metadata = events[0].metadata
    assert metadata["strategy"] == "expanded"
    assert metadata["seed_count"] == 1
    assert metadata["packed_count"] >= metadata["seed_count"]
    assert "estimated_tokens" in metadata
    assert "budget_tokens" in metadata


def test_seeds_only_never_emits_context_assembled_event() -> None:
    chunks = _split_chunks()
    llm = _RecordingLLM()
    trace = TraceContext(InMemoryTraceRecorder(), RunType.RAG)

    answer_repository_question(
        "How are tokens verified?",
        _embedded(chunks),
        _FixedEmbeddingProvider(),
        llm,
        config=RAGConfig(top_k=1, context_strategy=ContextStrategy.SEEDS_ONLY),
        trace=trace,
    )

    assert not any(event.event_type == "context.assembled" for event in trace.events)


def test_expanded_citation_for_neighbor_is_source_correct() -> None:
    chunks = _split_chunks()
    seed_chunk = chunks[0]
    neighbor_chunk = chunks[1]

    def retriever(query, *, top_k):
        return [SemanticSearchResult(chunk=seed_chunk, score=0.9, rank=1)]

    llm = _RecordingLLM(source_ids=["S2"])
    neighbor_loader = InMemoryNeighborLoader(chunks)

    answer = answer_repository_question_with_retriever(
        "How are tokens verified?",
        retriever,
        llm,
        config=RAGConfig(top_k=1, context_strategy=ContextStrategy.EXPANDED),
        neighbor_loader=neighbor_loader,
    )

    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.relative_path == neighbor_chunk.relative_path
    assert citation.start_line == neighbor_chunk.start_line
    assert citation.end_line == neighbor_chunk.end_line


def test_context_assembled_trace_event_never_carries_source_content_or_secrets() -> None:
    secret_source = (
        "SECRET_SOURCE_CONTENT = 'sk-test-secret'\n"
        "DB_URL = 'postgresql://user:password@host/db'\n"
        "def verify_token(token):\n"
        "    return token\n"
    )
    source_file = SourceFile(
        relative_path=Path("src/secret.py"),
        language="python",
        content=secret_source,
        size_bytes=len(secret_source.encode()),
        line_count=len(secret_source.splitlines()),
    )
    chunks = chunk_source_file(
        source_file,
        ChunkingConfig(strategy=ChunkingStrategy.LINE, max_lines_per_chunk=2, overlap_lines=0),
    )
    llm = _RecordingLLM()
    trace = TraceContext(InMemoryTraceRecorder(), RunType.RAG)

    answer_repository_question(
        "What is in this file?",
        _embedded(chunks),
        _FixedEmbeddingProvider(),
        llm,
        config=RAGConfig(top_k=1, context_strategy=ContextStrategy.EXPANDED),
        trace=trace,
    )

    event = next(event for event in trace.events if event.event_type == "context.assembled")
    serialized = str(event.metadata)
    assert "SECRET_SOURCE_CONTENT" not in serialized
    assert "sk-test-secret" not in serialized
    assert "password" not in serialized
    assert "src/secret.py" not in serialized
    allowed_keys = {
        "strategy",
        "seed_count",
        "expanded_count",
        "deduplicated_count",
        "dropped_count",
        "packed_count",
        "estimated_tokens",
        "budget_tokens",
    }
    assert set(event.metadata) <= allowed_keys


def test_retriever_based_expanded_strategy_requires_neighbor_loader() -> None:
    chunks = _split_chunks()

    def retriever(query, *, top_k):
        return [SemanticSearchResult(chunk=chunks[0], score=0.9, rank=1)]

    with pytest.raises(ValueError, match="neighbor_loader"):
        answer_repository_question_with_retriever(
            "How are tokens verified?",
            retriever,
            _RecordingLLM(),
            config=RAGConfig(top_k=1, context_strategy=ContextStrategy.EXPANDED),
        )
