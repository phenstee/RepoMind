"""Add durable cooperative job cancellation state.

Revision ID: 20260916_02
Revises: 20260916_01
"""

import sqlalchemy as sa

from alembic import op

revision = "20260916_02"
down_revision = "20260916_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_jobs_status", "jobs", type_="check")
    op.create_check_constraint(
        "ck_jobs_status",
        "jobs",
        "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
    )
    op.add_column("jobs", sa.Column("cancel_requested_at", sa.DateTime(timezone=True)))
    op.add_column("jobs", sa.Column("cancelled_at", sa.DateTime(timezone=True)))
    op.add_column("jobs", sa.Column("side_effect_started_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.execute("UPDATE jobs SET status = 'failed', error_code = 'job_cancelled' WHERE status = 'cancelled'")
    op.drop_column("jobs", "side_effect_started_at")
    op.drop_column("jobs", "cancelled_at")
    op.drop_column("jobs", "cancel_requested_at")
    op.drop_constraint("ck_jobs_status", "jobs", type_="check")
    op.create_check_constraint(
        "ck_jobs_status",
        "jobs",
        "status IN ('queued', 'running', 'succeeded', 'failed')",
    )
