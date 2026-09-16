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

    @property
    def terminal(self) -> bool:
        return self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}
