"""Add B-tree indexes for bounded persisted-symbol-metadata lookup.

Revision ID: 20260918_01
Revises: 20260917_01
Create Date: 2026-09-18
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260918_01"
down_revision: str | None = "20260917_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SYMBOL_NAME_INDEX = "ix_code_chunks_symbol_name"
_QUALIFIED_SYMBOL_NAME_INDEX = "ix_code_chunks_qualified_symbol_name"


def upgrade() -> None:
    """Index the two exact-match columns Milestone 24 symbol lookup filters on.

    Plain B-tree indexes only; repository scoping still comes from the
    existing join to ``repository_files``, and pgvector/FTS/trigram
    infrastructure is intentionally not introduced for string identifiers.
    """

    op.create_index(_SYMBOL_NAME_INDEX, "code_chunks", ["symbol_name"])
    op.create_index(_QUALIFIED_SYMBOL_NAME_INDEX, "code_chunks", ["qualified_symbol_name"])


def downgrade() -> None:
    op.drop_index(_QUALIFIED_SYMBOL_NAME_INDEX, table_name="code_chunks")
    op.drop_index(_SYMBOL_NAME_INDEX, table_name="code_chunks")
