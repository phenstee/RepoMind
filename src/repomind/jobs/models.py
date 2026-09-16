"""Small, durable execution contracts; job IDs are not trace run IDs."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class JobType(StrEnum):
    INDEX = "index"
    RAG = "rag"
    AGENT = "agent"
    CODING = "coding"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CancellationState(StrEnum):
    AVAILABLE = "available"
    REQUESTED = "requested"
    DEFERRED = "deferred"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """Application-owned persisted state, including bounded execution inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    job_type: JobType
    repository_id: int = Field(ge=1)
    status: JobStatus
    payload_version: int = Field(default=1, ge=1, le=1)
    request_payload: dict[str, Any]
    result_payload: dict[str, Any] | None = None
    error_code: str | None = None
    attempt_count: int = Field(ge=0)
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    trace_run_id: UUID | None = None
    cancel_requested_at: datetime | None = None
    cancelled_at: datetime | None = None
    side_effect_started_at: datetime | None = None

    @property
    def terminal(self) -> bool:
        return self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_requested_at is not None

    @property
    def cancellation_state(self) -> CancellationState:
        if self.status == JobStatus.CANCELLED:
            return CancellationState.CANCELLED
        unsafe_coding = self.job_type == JobType.CODING and self.side_effect_started_at is not None
        if unsafe_coding and self.cancel_requested:
            return CancellationState.DEFERRED
        if self.terminal or unsafe_coding:
            return CancellationState.UNAVAILABLE
        if self.cancel_requested:
            return CancellationState.REQUESTED
        return CancellationState.AVAILABLE
