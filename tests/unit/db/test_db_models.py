"""Tests for ORM metadata and domain reconstruction without a database."""

import pytest

from repomind.db import PersistenceError
from repomind.db.models import (
    HNSW_EMBEDDING_DIMENSIONS,
    HNSW_INDEX_NAME,
    CodeChunkRecord,
    RepositoryFileRecord,
)
from repomind.db.repositories import _embedded_chunk_from_record


def test_vector_column_is_dimension_flexible_but_metadata_checked() -> None:
    embedding_type = CodeChunkRecord.__table__.c.embedding.type
    constraint_names = {
        constraint.name for constraint in CodeChunkRecord.__table__.constraints
    }

    assert embedding_type.dim is None
    assert "ck_code_chunks_embedding_metadata" in constraint_names
    assert HNSW_EMBEDDING_DIMENSIONS == 1536
    assert HNSW_INDEX_NAME == "ix_code_chunks_embedding_hnsw_1536_cosine"


def test_foreign_keys_cascade_on_repository_deletion() -> None:
    file_foreign_key = next(
        iter(RepositoryFileRecord.__table__.c.repository_id.foreign_keys)
    )
    chunk_foreign_key = next(
        iter(CodeChunkRecord.__table__.c.repository_file_id.foreign_keys)
    )

    assert file_foreign_key.ondelete == "CASCADE"
    assert chunk_foreign_key.ondelete == "CASCADE"


def test_embedded_chunk_record_reconstructs_domain_models() -> None:
    record = CodeChunkRecord(
        id=7,
        repository_file_id=3,
        chunk_index=2,
        start_line=10,
        end_line=11,
        content="def café():\r\n    return '雪'\r\n",
        content_hash="a" * 64,
        embedding=[1.0, 0.5, 0.0],
        embedding_model="model-a",
        embedding_dimensions=3,
    )

    embedded = _embedded_chunk_from_record(record, "src/example.py", "python")

    assert embedded.chunk.relative_path.as_posix() == "src/example.py"
    assert embedded.chunk.start_line == 10
    assert embedded.chunk.end_line == 11
    assert embedded.chunk.content == record.content
    assert embedded.chunk.chunk_index == 2
    assert embedded.embedding.values == (1.0, 0.5, 0.0)
    assert embedded.embedding.model == "model-a"


def test_record_reconstruction_rejects_dimension_mismatch() -> None:
    record = CodeChunkRecord(
        id=7,
        repository_file_id=3,
        chunk_index=0,
        start_line=1,
        end_line=1,
        content="content\n",
        content_hash="a" * 64,
        embedding=[1.0, 0.0],
        embedding_model="model-a",
        embedding_dimensions=3,
    )

    with pytest.raises(PersistenceError, match="inconsistent embedding dimensions"):
        _embedded_chunk_from_record(record, "src/example.py", "python")
