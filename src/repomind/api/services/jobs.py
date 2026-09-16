"""Typed durable-job submission, public projection, and detached SSE observation."""

from uuid import UUID

import anyio
from fastapi import Request
from starlette.responses import StreamingResponse

from repomind.api.errors import APIError
from repomind.api.services.repositories import RepositoryService
from repomind.api.streaming import encode_sse_event
from repomind.jobs.broker import JobBroker
from repomind.jobs.models import Job, JobStatus, JobType
from repomind.jobs.store import JobNotFoundError, PostgresJobStore


class JobService:
    def __init__(self, store: PostgresJobStore, broker: JobBroker, repositories: RepositoryService) -> None:
        self.store = store
        self.broker = broker
        self.repositories = repositories

    def enqueue(self, job_type: JobType, repository_id: int, request: object | None = None) -> Job:
        self.repositories.locate(repository_id)
        payload = {} if request is None else request.model_dump(mode="json")
        job = self.store.create(job_type, repository_id, payload)
        self.broker.notify_job(job.id)
        return job

    def get(self, job_id: UUID) -> Job:
        try:
            return self.store.get(job_id)
        except JobNotFoundError as exc:
            raise APIError(404, "job_not_found", "Job not found.") from exc

    def list(self, **filters) -> tuple[Job, ...]:
        return tuple(self.store.list(**filters))

    def stream(self, request: Request, job_id: UUID) -> StreamingResponse:
        self.get(job_id)

        async def events():
            subscription = self.broker.subscribe(job_id)
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    job = await anyio.to_thread.run_sync(self.get, job_id)
                    if job.status == JobStatus.SUCCEEDED:
                        yield encode_sse_event(
                            "result", {"job_id": str(job.id), "result": job.result_payload, "trace_run_id": str(job.trace_run_id) if job.trace_run_id else None}
                        )
                        return
                    if job.status == JobStatus.FAILED:
                        yield encode_sse_event("error", {"job_id": str(job.id), "error": {"code": job.error_code or "operation_failed", "message": "The operation failed."}})
                        return
                    progress = await anyio.to_thread.run_sync(subscription.next, 1.0)
                    if progress is not None:
                        yield encode_sse_event(progress.event, progress, event_id=progress.sequence)
            finally:
                subscription.close()

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
