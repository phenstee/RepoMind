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
    RAGRequest,
    RAGResponse,
    RegisterRepositoryRequest,
    RepositoryFilesResponse,
    RepositoryResponse,
    RunDetailResponse,
    RunListResponse,
)
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


@router.get("/repositories/{repository_id}", response_model=RepositoryResponse)
def get_repository(repository_id: RepositoryID, services: ServiceDep):
    return services.repositories.get(repository_id)


@router.post("/repositories/{repository_id}/index", response_model=IndexResponse)
def index_repository(repository_id: RepositoryID, services: ServiceDep):
    return services.execution.index(repository_id)


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
