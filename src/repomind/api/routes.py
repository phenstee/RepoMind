"""Synchronous HTTP translation only; FastAPI runs these handlers in its thread pool."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query, Request
from starlette.responses import StreamingResponse

from repomind.api.dependencies import Services, get_services
from repomind.api.models import (
    AgentRequest,
    AgentResponse,
    CodingRequest,
    CodingResponse,
    ErrorResponse,
    HealthResponse,
    IndexResponse,
    JobCancelResponse,
    JobDetailResponse,
    JobListResponse,
    JobQueuedResponse,
    RAGRequest,
    RAGResponse,
    RegisterRepositoryRequest,
    RepositoryFilesResponse,
    RepositoryListResponse,
    RepositoryResponse,
    RunDetailResponse,
    RunListResponse,
)
from repomind.jobs import JobStatus, JobType
from repomind.observability import RunStatus, RunType

router = APIRouter(
    prefix="/api/v1",
    responses={code: {"model": ErrorResponse} for code in (400, 404, 409, 413, 422, 500, 503)},
)
ServiceDep = Annotated[Services, Depends(get_services)]
RepositoryID = Annotated[int, Path(ge=1)]
SSE_RESPONSE = {
    200: {
        "description": "Semantic progress frames followed by one terminal result or error frame.",
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }
}


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.post("/repositories", response_model=RepositoryResponse, status_code=201)
def register_repository(body: RegisterRepositoryRequest, services: ServiceDep):
    return services.repositories.register(body)


@router.get("/repositories", response_model=RepositoryListResponse)
def list_repositories(
    services: ServiceDep, limit: Annotated[int, Query(ge=1, le=100)] = 50
):
    return services.repositories.list(limit)


@router.get("/repositories/{repository_id}", response_model=RepositoryResponse)
def get_repository(repository_id: RepositoryID, services: ServiceDep):
    return services.repositories.get(repository_id)


@router.post("/repositories/{repository_id}/index", response_model=IndexResponse)
def index_repository(repository_id: RepositoryID, services: ServiceDep):
    return services.execution.index(repository_id)


def _job_response(job, *, detail: bool = False):
    base = {
        "job_id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "repository_id": job.repository_id,
        "attempt_count": job.attempt_count,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "trace_run_id": job.trace_run_id,
        "cancel_requested": job.cancel_requested,
        "cancel_requested_at": job.cancel_requested_at,
        "cancelled_at": job.cancelled_at,
        "cancellation_control": job.cancellation_state,
    }
    if not detail:
        return base
    result = None
    if job.result_payload is not None:
        models = {JobType.INDEX: IndexResponse, JobType.RAG: RAGResponse, JobType.AGENT: AgentResponse, JobType.CODING: CodingResponse}
        result = models[job.job_type].model_validate(job.result_payload)
    base["result"] = result
    base["error"] = None if job.error_code is None else {"code": job.error_code, "message": "The operation failed."}
    return base


@router.post("/repositories/{repository_id}/jobs/index", response_model=JobQueuedResponse, status_code=202)
def queue_index(repository_id: RepositoryID, services: ServiceDep):
    job = services.jobs.enqueue(JobType.INDEX, repository_id)
    return {"job_id": job.id, "job_type": job.job_type, "status": job.status}


@router.post("/repositories/{repository_id}/jobs/rag", response_model=JobQueuedResponse, status_code=202)
def queue_rag(repository_id: RepositoryID, body: RAGRequest, services: ServiceDep):
    job = services.jobs.enqueue(JobType.RAG, repository_id, body)
    return {"job_id": job.id, "job_type": job.job_type, "status": job.status}


@router.post("/repositories/{repository_id}/jobs/agent", response_model=JobQueuedResponse, status_code=202)
def queue_agent(repository_id: RepositoryID, body: AgentRequest, services: ServiceDep):
    job = services.jobs.enqueue(JobType.AGENT, repository_id, body)
    return {"job_id": job.id, "job_type": job.job_type, "status": job.status}


@router.post("/repositories/{repository_id}/jobs/coding", response_model=JobQueuedResponse, status_code=202)
def queue_coding(repository_id: RepositoryID, body: CodingRequest, services: ServiceDep):
    job = services.jobs.enqueue(JobType.CODING, repository_id, body)
    return {"job_id": job.id, "job_type": job.job_type, "status": job.status}


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    services: ServiceDep,
    status: JobStatus | None = None,
    job_type: JobType | None = None,
    repository_id: Annotated[int | None, Query(ge=1)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    return {"jobs": [_job_response(job) for job in services.jobs.list(status=status, job_type=job_type, repository_id=repository_id, limit=limit)]}


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
def get_job(job_id: UUID, services: ServiceDep):
    return _job_response(services.jobs.get(job_id), detail=True)


@router.post("/jobs/{job_id}/cancel", response_model=JobCancelResponse)
def cancel_job(job_id: UUID, services: ServiceDep):
    job = services.jobs.cancel(job_id)
    return {
        "job_id": job.id,
        "status": job.status,
        "cancel_requested": job.cancel_requested,
        "cancel_requested_at": job.cancel_requested_at,
        "cancelled_at": job.cancelled_at,
        "cancellation_control": job.cancellation_state,
    }


@router.get("/jobs/{job_id}/events", response_class=StreamingResponse, responses=SSE_RESPONSE)
def stream_job(request: Request, job_id: UUID, services: ServiceDep):
    return services.jobs.stream(request, job_id)


@router.post(
    "/repositories/{repository_id}/index/stream",
    response_class=StreamingResponse,
    responses=SSE_RESPONSE,
)
def stream_index(request: Request, repository_id: RepositoryID, services: ServiceDep):
    return services.streaming.index(request, repository_id)


@router.get("/repositories/{repository_id}/files", response_model=RepositoryFilesResponse)
def list_files(
    repository_id: RepositoryID,
    services: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
):
    return services.repositories.files(repository_id, limit, offset)


@router.post("/repositories/{repository_id}/rag", response_model=RAGResponse)
def rag(repository_id: RepositoryID, body: RAGRequest, services: ServiceDep):
    return services.execution.rag(repository_id, body)


@router.post(
    "/repositories/{repository_id}/rag/stream",
    response_class=StreamingResponse,
    responses=SSE_RESPONSE,
)
def stream_rag(
    request: Request, repository_id: RepositoryID, body: RAGRequest, services: ServiceDep
):
    return services.streaming.rag(request, repository_id, body)


@router.post("/repositories/{repository_id}/agent/runs", response_model=AgentResponse)
def agent(repository_id: RepositoryID, body: AgentRequest, services: ServiceDep):
    return services.execution.agent(repository_id, body)


@router.post(
    "/repositories/{repository_id}/agent/runs/stream",
    response_class=StreamingResponse,
    responses=SSE_RESPONSE,
)
def stream_agent(
    request: Request, repository_id: RepositoryID, body: AgentRequest, services: ServiceDep
):
    return services.streaming.agent(request, repository_id, body)


@router.post("/repositories/{repository_id}/coding/runs", response_model=CodingResponse)
def coding(repository_id: RepositoryID, body: CodingRequest, services: ServiceDep):
    return services.execution.coding(repository_id, body)


@router.post(
    "/repositories/{repository_id}/coding/runs/stream",
    response_class=StreamingResponse,
    responses=SSE_RESPONSE,
)
def stream_coding(
    request: Request, repository_id: RepositoryID, body: CodingRequest, services: ServiceDep
):
    return services.streaming.coding(request, repository_id, body)


@router.get("/runs", response_model=RunListResponse)
def list_runs(
    services: ServiceDep,
    run_type: RunType | None = None,
    status: RunStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    return services.runs.list(run_type, status, limit)


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run(run_id: UUID, services: ServiceDep):
    return services.runs.get(run_id)
