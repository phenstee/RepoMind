"""Small sanitized projections of Milestone 15 persisted traces."""

from typing import Protocol
from uuid import UUID

from repomind.api.errors import APIError
from repomind.api.models import RunDetailResponse, RunListResponse, RunSummaryResponse
from repomind.api.privacy import public_metadata
from repomind.observability import RunStatus, RunTrace, RunType


class TraceStore(Protocol):
    def persist_run_trace(self, trace: RunTrace) -> None: ...
    def get_run_trace(self, run_id: UUID) -> RunTrace | None: ...
    def list_run_traces(
        self, *, run_type: RunType | None = None, status: RunStatus | None = None, limit: int = 20
    ) -> tuple[RunTrace, ...]: ...


def safe_trace(trace: RunTrace) -> dict:
    # Revalidation is intentional: model_copy/legacy JSON may bypass original validators.
    validated = RunTrace.model_validate(trace.model_dump(mode="json"))
    return public_metadata(validated.model_dump(mode="json"))


class RunService:
    def __init__(self, store: TraceStore):
        self.store = store

    def list(
        self, run_type: RunType | None, status: RunStatus | None, limit: int
    ) -> RunListResponse:
        return RunListResponse(
            runs=[
                RunSummaryResponse.model_validate(safe_trace(trace))
                for trace in self.store.list_run_traces(
                    run_type=run_type, status=status, limit=limit
                )
            ]
        )

    def get(self, run_id: UUID) -> RunDetailResponse:
        trace = self.store.get_run_trace(run_id)
        if trace is None:
            raise APIError(404, "run_not_found", "Run not found.")
        return RunDetailResponse.model_validate(safe_trace(trace))
