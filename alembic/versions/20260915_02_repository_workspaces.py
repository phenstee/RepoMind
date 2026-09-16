"""Bind registered repositories to workspace-relative locations.

Revision ID: 20260915_02
Revises: 20260915_01
"""

import sqlalchemy as sa

from alembic import op

revision = "20260915_02"
down_revision = "20260915_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "repositories", sa.Column("workspace_relative_path", sa.String(1024), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("repositories", "workspace_relative_path")
