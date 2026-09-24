"""Persist the indexing-configuration fingerprint for incremental indexing.

Revision ID: 20260923_01
Revises: 20260918_01
Create Date: 2026-09-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260923_01"
down_revision: str | None = "20260918_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FINGERPRINT_CONSTRAINT = "ck_repositories_index_fingerprint_length"


def upgrade() -> None:
    """Add a nullable fingerprint describing how the stored index was built.

    The column is deliberately nullable with no backfill: a repository indexed
    before this migration has no recorded provenance, so it must be treated as
    incompatible and rebuilt once rather than assumed reusable.
    """

    op.add_column(
        "repositories",
        sa.Column("index_fingerprint", sa.String(length=64), nullable=True),
    )
    op.create_check_constraint(
        _FINGERPRINT_CONSTRAINT,
        "repositories",
        "index_fingerprint IS NULL OR char_length(index_fingerprint) = 64",
    )


def downgrade() -> None:
    op.drop_constraint(_FINGERPRINT_CONSTRAINT, "repositories", type_="check")
    op.drop_column("repositories", "index_fingerprint")
