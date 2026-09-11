"""SQLAlchemy models for persistent repository indexes."""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from repomind.db.base import Base


class RepositoryRecord(Base):
    """A named repository index without a machine-specific absolute root."""

    __tablename__ = "repositories"
    __table_args__ = (
        UniqueConstraint("name", name="uq_repositories_name"),
        CheckConstraint("char_length(name) > 0", name="ck_repositories_name_nonempty"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
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
    embedding: Mapped[list[float] | None] = mapped_column(Vector(), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    repository_file: Mapped[RepositoryFileRecord] = relationship(back_populates="chunks")
