"""Bounded HTTP contracts; private domain payloads are deliberately excluded."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from repomind.agent import AgentRunStatus
from repomind.coding import CodingPlan, CodingReview, CodingTaskStatus
from repomind.ingestion import validate_repository_relative_path
from repomind.jobs import CancellationState, JobStatus, JobType
from repomind.llm.models import TokenUsage
from repomind.observability import RunStatus, RunType, TraceEvent
from repomind.tools import RunRuffInput, RunTestsInput

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10_000)]
RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=1024)]
Strategy = Literal["semantic", "hybrid", "hybrid_rerank"]
PublicRelativePath = Annotated[
    str, AfterValidator(lambda value: validate_repository_relative_path(Path(value)).as_posix())
]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisterRepositoryRequest(RequestModel):
    name: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=255,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._-]*$",
        ),
    ]
    path: RelativePath


class RepositoryResponse(BaseModel):
    id: int
    name: str
    created_at: datetime


class RepositoryListItemResponse(RepositoryResponse):
    """Compact, safe binding metadata for the repository selector."""

    workspace_relative_path: PublicRelativePath | None


class RepositoryListResponse(BaseModel):
    repositories: list[RepositoryListItemResponse]


class RepositoryFileResponse(BaseModel):
    relative_path: PublicRelativePath
    language: str | None
    size_bytes: int
    line_count: int


class RepositoryFilesResponse(BaseModel):
    repository_id: int
    files: list[RepositoryFileResponse]
    limit: int
    offset: int


class IndexResponse(BaseModel):
    repository_id: int
    files_indexed: int
    chunks_indexed: int
    embedding_model: str | None


class RAGRequest(RequestModel):
    question: NonBlank
    strategy: Strategy = "semantic"
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    trace: bool = False


class CitationResponse(BaseModel):
    relative_path: str
    start_line: int
    end_line: int

    @field_validator("relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return validate_repository_relative_path(Path(value)).as_posix()


class RAGResponse(BaseModel):
    answer: str
    insufficient_evidence: bool
    citations: list[CitationResponse]
    trace_run_id: UUID | None = None


class AgentRequest(RequestModel):
    query: NonBlank
    max_iterations: int = Field(default=8, ge=1, le=20, strict=True)
    trace: bool = False


class AgentResponse(BaseModel):
    status: AgentRunStatus
    final_answer: str | None
    iterations: int
    llm_calls: int
    tool_execution_attempts: int
    trace_run_id: UUID | None = None


class VerificationRequest(RequestModel):
    """Clients may choose verifier scopes, but may not disable completion gates."""

    test_paths: list[RelativePath] = Field(
        default_factory=lambda: ["tests"], min_length=1, max_length=32
    )
    ruff_paths: list[RelativePath] = Field(
        default_factory=lambda: ["."], min_length=1, max_length=32
    )

    @field_validator("test_paths")
    @classmethod
    def _tests(cls, values: list[str]) -> list[str]:
        RunTestsInput(paths=values)
        return values

    @field_validator("ruff_paths")
    @classmethod
    def _ruff(cls, values: list[str]) -> list[str]:
        RunRuffInput(paths=values)
        return values


class CodingRequest(RequestModel):
    objective: NonBlank
    acceptance_criteria: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    ] = Field(default_factory=list, max_length=50)
    verification: VerificationRequest = Field(default_factory=VerificationRequest)
    max_iterations: int = Field(default=8, ge=1, le=20, strict=True)
    trace: bool = False


class VerificationSummary(BaseModel):
    passed: bool | None
    workspace_revision: int | None
    exit_code: int | None
    timed_out: bool | None
    execution_failed: bool


class CodingResponse(BaseModel):
    status: CodingTaskStatus
    final_answer: str | None
    tests: VerificationSummary
    ruff: VerificationSummary
    changed_files: list[PublicRelativePath]
    completion_attempts: int
    workspace_revision: int
    plan: CodingPlan | None = None
    review: CodingReview | None = None
    review_attempts: int = Field(default=0, ge=0)
    review_blocks: int = Field(default=0, ge=0)
    trace_run_id: UUID | None = None


class RunSummaryResponse(BaseModel):
    run_id: UUID
    run_type: RunType
    status: RunStatus
    domain_status: str | None
    model: str | None
    started_at: datetime
    ended_at: datetime | None
    duration_ms: float | None
    llm_calls: int
    tool_calls: int
    successful_mutations: int
    errors: int
    token_usage: TokenUsage | None
    usage_reported_calls: int
    estimated_cost_usd: None = None


class RunDetailResponse(RunSummaryResponse):
    events: tuple[TraceEvent, ...]


class RunListResponse(BaseModel):
    runs: list[RunSummaryResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["repomind"] = "repomind"


class ErrorDetail(BaseModel):
    code: str
    message: str
    trace_run_id: UUID | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class JobQueuedResponse(BaseModel):
    job_id: UUID
    job_type: JobType
    status: JobStatus


class JobSummaryResponse(JobQueuedResponse):
    repository_id: int
    attempt_count: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    trace_run_id: UUID | None
    cancel_requested: bool
    cancel_requested_at: datetime | None
    cancelled_at: datetime | None
    cancellation_control: CancellationState


class JobDetailResponse(JobSummaryResponse):
    result: IndexResponse | RAGResponse | AgentResponse | CodingResponse | None = None
    error: ErrorDetail | None = None


class JobListResponse(BaseModel):
    jobs: list[JobSummaryResponse]


class JobCancelResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    cancel_requested: bool
    cancel_requested_at: datetime | None
    cancelled_at: datetime | None
    cancellation_control: CancellationState
