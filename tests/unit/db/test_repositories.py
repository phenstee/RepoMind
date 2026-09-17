"""Database-independent validation tests for persistence operations."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from repomind.db import (
    PersistenceError,
    RepositoryNotFoundError,
    persist_embedded_chunks,
    pgvector_semantic_search,
)
from repomind.db.models import RepositoryRecord
from repomind.ingestion import ChunkingStrategy, ChunkKind, CodeChunk
from repomind.retrieval import (
    EmbeddedChunk,
    EmbeddingVector,
    SemanticSearchError,
    SemanticSearchMode,
)


def _embedded(values: tuple[float, ...], *, chunk_index: int = 0) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=CodeChunk(
            relative_path="src/example.py",
            language="python",
            start_line=chunk_index + 1,
            end_line=chunk_index + 1,
            content=f"chunk {chunk_index}\n",
            chunk_index=chunk_index,
        ),
        embedding=EmbeddingVector(values=values, model="model-a"),
    )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_pgvector_search_rejects_invalid_top_k_without_database(top_k: object) -> None:
    with pytest.raises(SemanticSearchError, match="positive integer"):
        pgvector_semantic_search(
            MagicMock(),
            1,
            EmbeddingVector(values=(1.0, 0.0), model="model-a"),
            top_k=top_k,  # type: ignore[arg-type]
        )


def test_pgvector_search_rejects_unknown_repository() -> None:
    session = MagicMock()
    session.get.return_value = None

    with pytest.raises(RepositoryNotFoundError, match="does not exist"):
        pgvector_semantic_search(
            session,
            404,
            EmbeddingVector(values=(1.0, 0.0), model="model-a"),
        )


def test_pgvector_search_returns_empty_for_missing_embedding_model() -> None:
    session = MagicMock()
    session.get.return_value = RepositoryRecord(id=1, name="repository")
    session.scalars.return_value.all.return_value = []

    results = pgvector_semantic_search(
        session,
        1,
        EmbeddingVector(values=(1.0, 0.0), model="unknown-model"),
    )

    assert results == []
    session.execute.assert_not_called()


def test_pgvector_search_rejects_query_dimension_mismatch_before_ranking() -> None:
    session = MagicMock()
    session.get.return_value = RepositoryRecord(id=1, name="repository")
    session.scalars.return_value.all.return_value = [3]

    with pytest.raises(PersistenceError, match="stored.*uses 3"):
        pgvector_semantic_search(
            session,
            1,
            EmbeddingVector(values=(1.0, 0.0), model="model-a"),
        )

    session.execute.assert_not_called()


def test_pgvector_search_rejects_zero_query_vector() -> None:
    session = MagicMock()
    session.get.return_value = RepositoryRecord(id=1, name="repository")

    with pytest.raises(PersistenceError, match="unusable"):
        pgvector_semantic_search(
            session,
            1,
            EmbeddingVector(values=(0.0, 0.0), model="model-a"),
        )

    session.scalars.assert_not_called()


def test_ann_search_rejects_dimensions_without_an_hnsw_expression_index() -> None:
    session = MagicMock()
    session.get.return_value = RepositoryRecord(id=1, name="repository")
    session.scalars.return_value.all.return_value = [2]

    with pytest.raises(PersistenceError, match="ANN search requires 1536-dimensional"):
        pgvector_semantic_search(
            session,
            1,
            EmbeddingVector(values=(1.0, 0.0), model="model-a"),
            mode=SemanticSearchMode.ANN,
        )

    session.execute.assert_not_called()


def test_pgvector_search_rejects_unknown_mode() -> None:
    with pytest.raises(SemanticSearchError, match="exact.*ann"):
        pgvector_semantic_search(
            MagicMock(),
            1,
            EmbeddingVector(values=(1.0, 0.0), model="model-a"),
            mode="other",  # type: ignore[arg-type]
        )


def test_ann_search_uses_transaction_local_tuning_and_index_expression() -> None:
    session = MagicMock()
    session.get.return_value = RepositoryRecord(id=1, name="repository")
    session.scalars.return_value.all.return_value = [1536]
    query = EmbeddingVector(values=(1.0, *([0.0] * 1535)), model="model-a")

    results = pgvector_semantic_search(
        session,
        1,
        query,
        mode=SemanticSearchMode.ANN,
    )

    assert results == []
    assert str(session.execute.call_args_list[0].args[0]) == (
        "SET LOCAL hnsw.iterative_scan = 'strict_order'"
    )
    statement = session.execute.call_args_list[1].args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "CAST(code_chunks.embedding AS VECTOR(1536)) <=>" in sql
    assert "code_chunks.embedding_model" in sql
    assert "repository_files.repository_id" in sql


def test_persist_embedded_chunks_rejects_mixed_dimensions_for_one_model() -> None:
    session = MagicMock()

    with pytest.raises(PersistenceError, match="inconsistent dimensions"):
        persist_embedded_chunks(
            session,
            1,
            [_embedded((1.0, 0.0)), _embedded((1.0, 0.0, 0.0), chunk_index=1)],
        )

    session.get.assert_not_called()


def test_persist_embedded_chunks_rejects_zero_vector_before_database_work() -> None:
    session = MagicMock()

    with pytest.raises(PersistenceError, match="unusable embedding"):
        persist_embedded_chunks(session, 1, [_embedded((0.0, 0.0))])

    session.get.assert_not_called()


def test_structural_metadata_is_copied_to_persistence_records() -> None:
    session = MagicMock()
    file_record = MagicMock(relative_path="src/example.py")
    file_record.chunks = []
    session.get.return_value = RepositoryRecord(id=1, name="repository")
    session.scalars.return_value.all.return_value = [file_record]
    embedded = _embedded((1.0, 0.0))
    embedded = embedded.model_copy(
        update={
            "chunk": embedded.chunk.model_copy(
                update={
                    "chunking_strategy": ChunkingStrategy.STRUCTURAL,
                    "chunk_kind": ChunkKind.METHOD,
                    "symbol_name": "run",
                    "qualified_symbol_name": "Worker.run",
                    "parent_symbol": "Worker",
                }
            )
        }
    )

    records = persist_embedded_chunks(session, 1, [embedded])

    assert records[0].chunking_strategy == "python_ast_v1"
    assert records[0].chunk_kind == "method"
    assert records[0].qualified_symbol_name == "Worker.run"
