"""Database-independent validation tests for persistence operations."""

from unittest.mock import MagicMock

import pytest

from repomind.db import (
    PersistenceError,
    RepositoryNotFoundError,
    persist_embedded_chunks,
    pgvector_semantic_search,
)
from repomind.db.models import RepositoryRecord
from repomind.ingestion import CodeChunk
from repomind.retrieval import (
    EmbeddedChunk,
    EmbeddingVector,
    SemanticSearchError,
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
