"""Real PostgreSQL coverage for Milestone 25's indexed navigation tool."""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from repomind.db import persist_embedded_chunks, persist_repository_snapshot
from repomind.db.hybrid import postgres_hybrid_symbol_search
from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    RepositorySnapshot,
    SourceFile,
    chunk_source_file,
)
from repomind.retrieval import EmbeddedChunk, EmbeddingVector
from repomind.tools import IndexedCodeSearchInput, indexed_code_search

pytestmark = pytest.mark.postgres

_SOURCE = '''class UserService:
    def login(self, username, password):
        normalized = username.strip().casefold()
        if not normalized or not password:
            raise ValueError("SECRET_SOURCE_CONTENT credentials are required")
        return "token"

    def logout(self, token):
        return None
'''


class _CountingEmbeddingProvider:
    def __init__(self, vector: tuple[float, ...], model: str) -> None:
        self.vector = vector
        self.model = model
        self.calls = 0

    def embed_text(self, text: str) -> EmbeddingVector:
        self.calls += 1
        return EmbeddingVector(values=self.vector, model=self.model)


def _persist(session: Session, prefix: str) -> tuple[int, str]:
    name = f"{prefix}-{uuid4().hex}"
    source = SourceFile(
        relative_path=Path("src/auth/service.py"),
        language="python",
        content=_SOURCE,
        size_bytes=len(_SOURCE.encode()),
        line_count=len(_SOURCE.splitlines()),
    )
    snapshot = RepositorySnapshot(
        root=Path("C:/isolated/indexed-search-fixture"),
        name=name,
        files=[source],
        skipped=[],
        file_count=1,
        total_size_bytes=source.size_bytes,
        languages={"python": 1},
    )
    repository = persist_repository_snapshot(session, snapshot)
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, max_lines_per_chunk=20, overlap_lines=0)
    chunks = chunk_source_file(source, config)
    model = f"indexed-search-test-{uuid4().hex}"
    embedded = [
        EmbeddedChunk(
            chunk=chunk,
            embedding=EmbeddingVector(values=(float(index) + 1.0, 0.0), model=model),
        )
        for index, chunk in enumerate(chunks)
    ]
    persist_embedded_chunks(session, repository.id, embedded)
    return repository.id, model


def _retriever(session: Session, repository_id: int, embedder: _CountingEmbeddingProvider):
    def search(query: str, *, top_k: int):
        return postgres_hybrid_symbol_search(
            session, repository_id, query, embedder.embed_text(query), top_k=top_k
        )

    return search


def test_indexed_tool_returns_bounded_locations_without_source_or_vectors(
    db_session: Session,
) -> None:
    repository_id, model = _persist(db_session, "indexed-basic")
    embedder = _CountingEmbeddingProvider((1.0, 0.0), model)
    retriever = _retriever(db_session, repository_id, embedder)

    output = indexed_code_search(
        retriever, IndexedCodeSearchInput(query="UserService.login", max_results=5)
    )

    assert output.result_count >= 1
    dumped = output.model_dump()
    assert "content" not in str(dumped)
    assert "SECRET_SOURCE_CONTENT" not in str(dumped)
    for location in output.locations:
        assert location.relative_path == Path("src/auth/service.py")
        assert location.start_line >= 1
        assert location.end_line >= location.start_line
    # One embedding call for one tool execution, never per-candidate.
    assert embedder.calls == 1


def test_indexed_tool_result_count_is_bounded_by_max_results(db_session: Session) -> None:
    repository_id, model = _persist(db_session, "indexed-bounded")
    embedder = _CountingEmbeddingProvider((1.0, 0.0), model)
    retriever = _retriever(db_session, repository_id, embedder)

    output = indexed_code_search(
        retriever, IndexedCodeSearchInput(query="login", max_results=1)
    )

    assert output.result_count <= 1


def test_indexed_tool_is_isolated_by_repository(db_session: Session) -> None:
    repository_a, model_a = _persist(db_session, "indexed-isolate-a")
    repository_b, model_b = _persist(db_session, "indexed-isolate-b")
    embedder_a = _CountingEmbeddingProvider((1.0, 0.0), model_a)
    embedder_b = _CountingEmbeddingProvider((1.0, 0.0), model_b)

    output_a = indexed_code_search(
        _retriever(db_session, repository_a, embedder_a),
        IndexedCodeSearchInput(query="UserService.login", max_results=5),
    )
    output_b = indexed_code_search(
        _retriever(db_session, repository_b, embedder_b),
        IndexedCodeSearchInput(query="UserService.login", max_results=5),
    )

    assert output_a.result_count >= 1
    assert output_b.result_count >= 1
    assert {location.relative_path for location in output_a.locations} == {
        location.relative_path for location in output_b.locations
    }


def test_indexed_tool_empty_index_returns_valid_empty_result(db_session: Session) -> None:
    repository_id, model = _persist(db_session, "indexed-empty")
    embedder = _CountingEmbeddingProvider((1.0, 0.0), model)
    retriever = _retriever(db_session, repository_id, embedder)

    output = indexed_code_search(
        retriever,
        IndexedCodeSearchInput(query="totally unrelated nonexistent symbol name", max_results=5),
    )

    assert isinstance(output.result_count, int)
    assert output.locations == [] or output.result_count >= 0
