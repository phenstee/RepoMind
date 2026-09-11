"""Create persistent repository index and pgvector storage.

Revision ID: 20260910_01
Revises:
Create Date: 2026-09-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "20260910_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the extension and normalized persistent repository schema."""

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "repositories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "char_length(name) > 0",
            name="ck_repositories_name_nonempty",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_repositories_name"),
    )
    op.create_table(
        "repository_files",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("repository_id", sa.Integer(), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("line_count", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.CheckConstraint(
            "char_length(content_hash) = 64",
            name="ck_repository_files_hash_length",
        ),
        sa.CheckConstraint(
            "line_count >= 0",
            name="ck_repository_files_lines",
        ),
        sa.CheckConstraint(
            "char_length(relative_path) > 0",
            name="ck_repository_files_path",
        ),
        sa.CheckConstraint(
            "size_bytes >= 0",
            name="ck_repository_files_size",
        ),
        sa.ForeignKeyConstraint(
            ["repository_id"],
            ["repositories.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "repository_id",
            "relative_path",
            name="uq_repository_files_repository_path",
        ),
    )
    op.create_index(
        "ix_repository_files_repository_id",
        "repository_files",
        ["repository_id"],
    )
    op.create_table(
        "code_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("repository_file_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "char_length(content_hash) = 64",
            name="ck_code_chunks_hash_length",
        ),
        sa.CheckConstraint("chunk_index >= 0", name="ck_code_chunks_index"),
        sa.CheckConstraint(
            "(embedding IS NULL AND embedding_model IS NULL "
            "AND embedding_dimensions IS NULL) OR "
            "(embedding IS NOT NULL AND embedding_model IS NOT NULL "
            "AND char_length(embedding_model) > 0 "
            "AND embedding_dimensions > 0 "
            "AND vector_dims(embedding) = embedding_dimensions)",
            name="ck_code_chunks_embedding_metadata",
        ),
        sa.CheckConstraint(
            "end_line >= start_line",
            name="ck_code_chunks_line_range",
        ),
        sa.CheckConstraint(
            "start_line >= 1",
            name="ck_code_chunks_start_line",
        ),
        sa.ForeignKeyConstraint(
            ["repository_file_id"],
            ["repository_files.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "repository_file_id",
            "chunk_index",
            name="uq_code_chunks_file_index",
        ),
    )
    op.create_index(
        "ix_code_chunks_embedding_model",
        "code_chunks",
        ["embedding_model"],
    )
    op.create_index(
        "ix_code_chunks_repository_file_id",
        "code_chunks",
        ["repository_file_id"],
    )


def downgrade() -> None:
    """Remove RepoMind tables while leaving the shared vector extension installed."""

    op.drop_index("ix_code_chunks_repository_file_id", table_name="code_chunks")
    op.drop_index("ix_code_chunks_embedding_model", table_name="code_chunks")
    op.drop_table("code_chunks")
    op.drop_index(
        "ix_repository_files_repository_id",
        table_name="repository_files",
    )
    op.drop_table("repository_files")
    op.drop_table("repositories")
