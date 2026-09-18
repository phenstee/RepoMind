"""Serializable trace contracts, independent of database sessions."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from repomind.llm.models import TokenUsage
from repomind.observability.sanitization import redact, sanitize_metadata


class RunType(StrEnum):
    INDEX = "index"
    RAG = "rag"
    READ_ONLY_AGENT = "read_only_agent"
    EDITING_AGENT = "editing_agent"
    CODING_TASK = "coding_task"
    EVALUATION = "evaluation"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


EventType = Literal[
    "run.started",
    "run.completed",
    "run.failed",
    "job.cancel_requested",
    "job.cancelled",
    "job.cancellation_deferred",
    "index.started",
    "ingestion.completed",
    "chunking.completed",
    "embedding.started",
    "embedding.completed",
    "persistence.completed",
    "model.started",
    "model.attempt",
    "model.completed",
    "model.failed",
    "model.usage",
    "tool.started",
    "tool.completed",
    "tool.failed",
    "tool.blocked",
    "agent.decision",
    "agent.stopped",
    "file.mutated",
    "preflight.started",
    "preflight.passed",
    "preflight.failed",
    "planning.started",
    "planning.completed",
    "planning.failed",
    "verification.started",
    "verification.completed",
    "verification.failed",
    "completion.requested",
    "completion.blocked",
    "completion.completed",
    "final_review.started",
    "final_review.completed",
    "review.started",
    "review.completed",
    "review.blocked",
    "review.failed",
    "retrieval.started",
    "retrieval.completed",
    "retrieval.failed",
    "rag.context",
    "rag.answer",
    "context.assembled",
    "symbol.matched",
    "evaluation.case.started",
    "evaluation.case.completed",
    "evaluation.case.failed",
]


class TraceEvent(BaseModel):
    """Sequences start at one; metadata is allowlisted on construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    event_type: EventType
    sequence: int = Field(ge=1, strict=True)
    timestamp: AwareDatetime
    duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @field_validator("metadata", mode="before")
    @classmethod
    def _sanitize(cls, value: Any) -> dict[str, Any]:
        return sanitize_metadata(value)


class RunTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: UUID
    run_type: RunType
    status: RunStatus = RunStatus.RUNNING
    domain_status: str | None = None
    model: str | None = None
    started_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    llm_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    successful_mutations: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    events: tuple[TraceEvent, ...] = ()
    token_usage: TokenUsage | None = None
    usage_reported_calls: int = Field(default=0, ge=0)
    estimated_cost_usd: None = None
    evaluation_summary: dict[str, Any] | None = None

    @field_validator("started_at", "ended_at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None

    @field_validator("token_usage")
    @classmethod
    def _usage(cls, value: TokenUsage | None) -> TokenUsage | None:
        if (
            value is not None
            and min(value.prompt_tokens, value.completion_tokens, value.total_tokens) < 0
        ):
            raise ValueError("trace token counts cannot be negative")
        return value

    @field_validator("domain_status", "model")
    @classmethod
    def _safe_label(cls, value: str | None) -> str | None:
        return redact(value) if value is not None else None

    @field_validator("evaluation_summary", mode="before")
    @classmethod
    def _summary(cls, value: Any) -> Any:
        return sanitize_metadata(value) if value is not None else None

    @model_validator(mode="after")
    def _consistent(self) -> "RunTrace":
        if [e.sequence for e in self.events] != list(range(1, len(self.events) + 1)):
            raise ValueError("event sequences must be contiguous, starting at one")
        terminal = self.status != RunStatus.RUNNING
        if not terminal and (self.ended_at is not None or self.duration_ms is not None):
            raise ValueError("running traces cannot have an end time or duration")
        if terminal != (self.ended_at is not None and self.duration_ms is not None):
            raise ValueError("terminal runs require end time and duration")
        return self
