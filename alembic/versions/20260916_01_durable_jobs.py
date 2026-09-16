"""Add durable PostgreSQL job records.

Revision ID: 20260916_01
Revises: 20260915_02
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260916_01"
down_revision = "20260915_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_type", sa.String(length=16), nullable=False),
        sa.Column("repository_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("payload_version", sa.Integer(), nullable=False),
        sa.Column("request_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trace_run_id", sa.Uuid(), nullable=True),
        sa.CheckConstraint("job_type IN ('index', 'rag', 'agent', 'coding')", name="ck_jobs_type"),
        sa.CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed')", name="ck_jobs_status"),
        sa.CheckConstraint("payload_version = 1", name="ck_jobs_payload_version"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_jobs_attempt_count"),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_claim", "jobs", ["status", "created_at"])
    op.create_index("ix_jobs_repository", "jobs", ["repository_id", "created_at"])
    op.create_index("ix_jobs_lease", "jobs", ["status", "lease_expires_at"])


def downgrade() -> None:
    op.drop_table("jobs")
