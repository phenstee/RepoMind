"""Add structural chunk metadata and a cosine HNSW index.

Revision ID: 20260917_01
Revises: 20260916_02
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260917_01"
down_revision: str | None = "20260916_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

HNSW_INDEX_NAME = "ix_code_chunks_embedding_hnsw_1536_cosine"


def upgrade() -> None:
    """Persist chunk provenance and index the default 1,536-dimensional vectors."""

    op.add_column(
        "code_chunks",
        sa.Column(
            "chunking_strategy",
            sa.String(length=32),
            server_default="line_v1",
            nullable=False,
        ),
    )
    op.add_column(
        "code_chunks",
        sa.Column(
            "chunk_kind",
            sa.String(length=32),
            server_default="line",
            nullable=False,
        ),
    )
    op.add_column("code_chunks", sa.Column("symbol_name", sa.String(length=512)))
    op.add_column(
        "code_chunks", sa.Column("qualified_symbol_name", sa.String(length=2048))
    )
    op.add_column("code_chunks", sa.Column("parent_symbol", sa.String(length=2048)))
    op.add_column("code_chunks", sa.Column("fragment_index", sa.Integer()))
    op.add_column("code_chunks", sa.Column("fragment_count", sa.Integer()))
    op.create_check_constraint(
        "ck_code_chunks_chunking_strategy",
        "code_chunks",
        "chunking_strategy IN ('line_v1', 'python_ast_v1')",
    )
    op.create_check_constraint(
        "ck_code_chunks_kind",
        "code_chunks",
        "chunk_kind IN ('line', 'line_fallback', 'module', 'function', "
        "'class', 'method', 'structural_fragment')",
    )
    op.create_check_constraint(
        "ck_code_chunks_fragment",
        "code_chunks",
        "(chunk_kind <> 'structural_fragment' AND fragment_index IS NULL "
        "AND fragment_count IS NULL) OR "
        "(chunk_kind = 'structural_fragment' AND fragment_index >= 1 "
        "AND fragment_count >= fragment_index)",
    )
    # The column remains dimension-flexible for exact baselines and test models.
    # pgvector requires a fixed-dimensional expression for an HNSW index.
    op.execute(
        f"CREATE INDEX {HNSW_INDEX_NAME} ON code_chunks "
        "USING hnsw ((embedding::vector(1536)) vector_cosine_ops) "
        "WHERE embedding IS NOT NULL AND embedding_dimensions = 1536"
    )


def downgrade() -> None:
    """Remove Retrieval V2 metadata and ANN acceleration."""

    op.drop_index(HNSW_INDEX_NAME, table_name="code_chunks")
    op.drop_constraint("ck_code_chunks_fragment", "code_chunks", type_="check")
    op.drop_constraint("ck_code_chunks_kind", "code_chunks", type_="check")
    op.drop_constraint("ck_code_chunks_chunking_strategy", "code_chunks", type_="check")
    op.drop_column("code_chunks", "fragment_count")
    op.drop_column("code_chunks", "fragment_index")
    op.drop_column("code_chunks", "parent_symbol")
    op.drop_column("code_chunks", "qualified_symbol_name")
    op.drop_column("code_chunks", "symbol_name")
    op.drop_column("code_chunks", "chunk_kind")
    op.drop_column("code_chunks", "chunking_strategy")
