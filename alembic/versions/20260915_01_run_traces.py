"""Add optional run summaries and ordered trace events.

Revision ID: 20260915_01
Revises: 20260910_01
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260915_01"
down_revision = "20260910_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trace_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("model", sa.String(256), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("llm_calls", sa.Integer(), nullable=False),
        sa.Column("tool_calls", sa.Integer(), nullable=False),
        sa.Column("successful_mutations", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("details_json", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_trace_runs_started_at", "trace_runs", ["started_at"])
    op.create_index("ix_trace_runs_type_status", "trace_runs", ["run_type", "status"])
    op.create_table(
        "trace_events",
        sa.Column(
            "run_id",
            sa.Uuid(),
            sa.ForeignKey("trace_runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("sequence", sa.Integer(), primary_key=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("sequence >= 1", name="ck_trace_events_sequence"),
    )


def downgrade() -> None:
    op.drop_table("trace_events")
    op.drop_index("ix_trace_runs_type_status", table_name="trace_runs")
    op.drop_index("ix_trace_runs_started_at", table_name="trace_runs")
    op.drop_table("trace_runs")
