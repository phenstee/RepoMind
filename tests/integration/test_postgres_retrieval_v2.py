"""Real PostgreSQL coverage for Retrieval V2's HNSW path."""

import math
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import ClauseElement, Executable, Select, text
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from repomind.db import (
    persist_embedded_chunks,
    persist_repository_snapshot,
    pgvector_semantic_search,
)
from repomind.db.models import HNSW_INDEX_NAME
from repomind.db.repositories import _semantic_search_statement
from repomind.ingestion import (
    ChunkingStrategy,
    ChunkKind,
    CodeChunk,
    RepositorySnapshot,
    SourceFile,
)
from repomind.retrieval import EmbeddedChunk, EmbeddingVector, SemanticSearchMode

pytestmark = pytest.mark.postgres

_DIMENSIONS = 1536
_MODEL = "offline-hnsw-1536"


def _vector(angle: float) -> tuple[float, ...]:
    return (math.cos(angle), math.sin(angle), *([0.0] * (_DIMENSIONS - 2)))


def _repository(session: Session, prefix: str, count: int) -> int:
    name = f"{prefix}-{uuid4().hex}"
    content = "".join(f"value_{index} = {index}\n" for index in range(count))
    source = SourceFile(
        relative_path=Path("src/vectors.py"),
        language="python",
        content=content,
        size_bytes=len(content.encode()),
        line_count=count,
    )
    snapshot = RepositorySnapshot(
        root=Path("C:/isolated/retrieval-v2-fixture"),
        name=name,
        files=[source],
        skipped=[],
        file_count=1,
        total_size_bytes=source.size_bytes,
        languages={"python": 1},
    )
    return persist_repository_snapshot(session, snapshot).id


def _embedded(index: int, *, model: str = _MODEL) -> EmbeddedChunk:
    chunk = CodeChunk(
        relative_path="src/vectors.py",
        language="python",
        start_line=index + 1,
        end_line=index + 1,
        content=f"value_{index} = {index}\n",
        chunk_index=index,
        chunking_strategy=ChunkingStrategy.STRUCTURAL,
        chunk_kind=ChunkKind.MODULE,
    )
    return EmbeddedChunk(
        chunk=chunk,
        embedding=EmbeddingVector(values=_vector(index * 0.01), model=model),
    )


def test_hnsw_index_exists_is_valid_and_uses_cosine_ops(db_session: Session) -> None:
    row = db_session.execute(
        text(
            "SELECT i.indisvalid, pg_get_indexdef(i.indexrelid) "
            "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname = :name"
        ),
        {"name": HNSW_INDEX_NAME},
    ).one()

    assert row[0] is True
    definition = row[1].lower()
    assert "using hnsw" in definition
    assert "vector_cosine_ops" in definition
    assert "vector(1536)" in definition


def test_ann_recall_filters_and_roundtrip(db_session: Session) -> None:
    count = 128
    repository_id = _repository(db_session, "ann-primary", count + 1)
    other_repository_id = _repository(db_session, "ann-other", 1)
    corpus = [_embedded(index) for index in range(count)]
    corpus.append(_embedded(count, model="other-model"))
    persist_embedded_chunks(db_session, repository_id, corpus)
    persist_embedded_chunks(db_session, other_repository_id, [_embedded(999)])
    query = EmbeddingVector(values=_vector(0.403), model=_MODEL)

    exact = pgvector_semantic_search(
        db_session,
        repository_id,
        query,
        top_k=10,
        mode=SemanticSearchMode.EXACT,
    )
    # This test proves ANN quality and eligibility, not the tiny-fixture cost
    # decision. Production retrieval never changes enable_seqscan.
    db_session.execute(text("SET LOCAL enable_seqscan = off"))
    approximate = pgvector_semantic_search(
        db_session,
        repository_id,
        query,
        top_k=10,
        mode=SemanticSearchMode.ANN,
    )

    exact_ids = {result.chunk.chunk_index for result in exact}
    ann_ids = {result.chunk.chunk_index for result in approximate}
    # HNSW is approximate: 0.8 catches severe candidate loss without demanding
    # exact equality from a deterministic but implementation-dependent graph.
    assert len(exact_ids & ann_ids) / len(exact_ids) >= 0.8
    assert all(result.chunk.chunk_index < count for result in approximate)
    assert all(
        result.chunk.chunking_strategy is ChunkingStrategy.STRUCTURAL
        for result in approximate
    )


class _Explain(Executable, ClauseElement):
    """``EXPLAIN`` the exact production statement, binds and all."""

    inherit_cache = False

    def __init__(self, statement: Select) -> None:
        self.statement = statement


@compiles(_Explain, "postgresql")
def _compile_explain(element: _Explain, compiler, **kwargs) -> str:
    return "EXPLAIN (COSTS OFF) " + compiler.process(element.statement, **kwargs)


def _plan(session: Session, statement: Select) -> str:
    return "\n".join(session.execute(_Explain(statement)).scalars())


def test_production_ann_statement_walks_the_hnsw_index(db_session: Session) -> None:
    count = 128
    repository_id = _repository(db_session, "ann-plan", count)
    persist_embedded_chunks(
        db_session, repository_id, [_embedded(index) for index in range(count)]
    )
    query = EmbeddingVector(values=_vector(0.403), model=_MODEL)

    # The tiny fixture would otherwise make a sequential scan cheapest; this
    # proves index eligibility, not the planner's cost choice. Production
    # retrieval never changes enable_seqscan, and SET LOCAL ends with the test
    # transaction.
    db_session.execute(text("SET LOCAL enable_seqscan = off"))
    ann_plan = _plan(
        db_session,
        _semantic_search_statement(
            repository_id, query, top_k=10, mode=SemanticSearchMode.ANN
        ),
    )
    exact_plan = _plan(
        db_session,
        _semantic_search_statement(
            repository_id, query, top_k=10, mode=SemanticSearchMode.EXACT
        ),
    )

    # The nearest-neighbor LIMIT must be fed by the HNSW index in distance
    # order, never by sorting every repository match; the deterministic
    # tie-break sort may only run above it, over the top_k rows.
    nearest_step = ann_plan[ann_plan.index("Limit") :]
    assert f"Index Scan using {HNSW_INDEX_NAME}" in nearest_step
    assert "Sort" not in nearest_step
    # Exact mode stays an exact baseline that cannot use the ANN index.
    assert HNSW_INDEX_NAME not in exact_plan
