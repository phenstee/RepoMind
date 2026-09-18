"""SQLAlchemy models for persistent repository indexes."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from repomind.db.base import Base

HNSW_EMBEDDING_DIMENSIONS = 1536
HNSW_INDEX_NAME = "ix_code_chunks_embedding_hnsw_1536_cosine"


class RepositoryRecord(Base):
    """A named repository index without a machine-specific absolute root."""

    __tablename__ = "repositories"
    __table_args__ = (
        UniqueConstraint("name", name="uq_repositories_name"),
        CheckConstraint("char_length(name) > 0", name="ck_repositories_name_nonempty"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    workspace_relative_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    files: Mapped[list[RepositoryFileRecord]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class RepositoryFileRecord(Base):
    """Source content and metadata captured at repository indexing time."""

    __tablename__ = "repository_files"
    __table_args__ = (
        UniqueConstraint(
            "repository_id",
            "relative_path",
            name="uq_repository_files_repository_path",
        ),
        CheckConstraint("char_length(relative_path) > 0", name="ck_repository_files_path"),
        CheckConstraint("size_bytes >= 0", name="ck_repository_files_size"),
        CheckConstraint("line_count >= 0", name="ck_repository_files_lines"),
        CheckConstraint(
            "char_length(content_hash) = 64",
            name="ck_repository_files_hash_length",
        ),
        Index("ix_repository_files_repository_id", "repository_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    line_count: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    repository: Mapped[RepositoryRecord] = relationship(back_populates="files")
    chunks: Mapped[list[CodeChunkRecord]] = relationship(
        back_populates="repository_file",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CodeChunkRecord(Base):
    """A persisted source chunk with an optional pgvector embedding."""

    __tablename__ = "code_chunks"
    __table_args__ = (
        UniqueConstraint(
            "repository_file_id",
            "chunk_index",
            name="uq_code_chunks_file_index",
        ),
        CheckConstraint("chunk_index >= 0", name="ck_code_chunks_index"),
        CheckConstraint("start_line >= 1", name="ck_code_chunks_start_line"),
        CheckConstraint("end_line >= start_line", name="ck_code_chunks_line_range"),
        CheckConstraint(
            "chunking_strategy IN ('line_v1', 'python_ast_v1')",
            name="ck_code_chunks_chunking_strategy",
        ),
        CheckConstraint(
            "chunk_kind IN ('line', 'line_fallback', 'module', 'function', "
            "'class', 'method', 'structural_fragment')",
            name="ck_code_chunks_kind",
        ),
        CheckConstraint(
            "(chunk_kind <> 'structural_fragment' AND fragment_index IS NULL "
            "AND fragment_count IS NULL) OR "
            "(chunk_kind = 'structural_fragment' AND fragment_index >= 1 "
            "AND fragment_count >= fragment_index)",
            name="ck_code_chunks_fragment",
        ),
        CheckConstraint(
            "char_length(content_hash) = 64",
            name="ck_code_chunks_hash_length",
        ),
        CheckConstraint(
            "(embedding IS NULL AND embedding_model IS NULL "
            "AND embedding_dimensions IS NULL) OR "
            "(embedding IS NOT NULL AND embedding_model IS NOT NULL "
            "AND char_length(embedding_model) > 0 "
            "AND embedding_dimensions > 0 "
            "AND vector_dims(embedding) = embedding_dimensions)",
            name="ck_code_chunks_embedding_metadata",
        ),
        Index("ix_code_chunks_repository_file_id", "repository_file_id"),
        Index("ix_code_chunks_embedding_model", "embedding_model"),
        Index("ix_code_chunks_symbol_name", "symbol_name"),
        Index("ix_code_chunks_qualified_symbol_name", "qualified_symbol_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repository_file_id: Mapped[int] = mapped_column(
        ForeignKey("repository_files.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunking_strategy: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="line_v1"
    )
    chunk_kind: Mapped[str] = mapped_column(String(32), nullable=False, server_default="line")
    symbol_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    qualified_symbol_name: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    parent_symbol: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    fragment_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fragment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    repository_file: Mapped[RepositoryFileRecord] = relationship(back_populates="chunks")


class TraceRunRecord(Base):
    """Run summaries, separate from repository indexing transactions."""

    __tablename__ = "trace_runs"
    __table_args__ = (
        Index("ix_trace_runs_started_at", "started_at"),
        Index("ix_trace_runs_type_status", "run_type", "status"),
    )
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    run_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(256))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float] = mapped_column(Float)
    llm_calls: Mapped[int] = mapped_column(Integer)
    tool_calls: Mapped[int] = mapped_column(Integer)
    successful_mutations: Mapped[int] = mapped_column(Integer)
    error_count: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    details_json: Mapped[dict] = mapped_column(JSONB)


class TraceEventRecord(Base):
    """Ordered events; a composite primary key also indexes timeline lookup."""

    __tablename__ = "trace_events"
    __table_args__ = (CheckConstraint("sequence >= 1", name="ck_trace_events_sequence"),)
    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("trace_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    metadata_json: Mapped[dict] = mapped_column(JSONB)


class JobRecord(Base):
    """Durable worker request state; payload/result are bounded application contracts."""

    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint("job_type IN ('index', 'rag', 'agent', 'coding')", name="ck_jobs_type"),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_jobs_status",
        ),
        CheckConstraint("payload_version = 1", name="ck_jobs_payload_version"),
        CheckConstraint("attempt_count >= 0", name="ck_jobs_attempt_count"),
        Index("ix_jobs_claim", "status", "created_at"),
        Index("ix_jobs_repository", "repository_id", "created_at"),
        Index("ix_jobs_lease", "status", "lease_expires_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    job_type: Mapped[str] = mapped_column(String(16), nullable=False)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    payload_version: Mapped[int] = mapped_column(Integer, nullable=False)
    request_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    result_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    trace_run_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    side_effect_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
