"""One explicit durable worker loop that reuses existing execution services."""

import logging
from threading import Event, Thread
from uuid import uuid4

from repomind.api.errors import APIError
from repomind.api.models import AgentRequest, CodingRequest, RAGRequest
from repomind.api.services.execution import ExecutionService
from repomind.api.streaming import safe_progress_event
from repomind.jobs.broker import JobBroker
from repomind.jobs.models import Job, JobType
from repomind.jobs.store import PostgresJobStore
from repomind.observability import InMemoryTraceRecorder, RunType, TraceContext

logger = logging.getLogger(__name__)
_RUN_TYPES = {
    JobType.INDEX: RunType.INDEX,
    JobType.RAG: RunType.RAG,
    JobType.AGENT: RunType.READ_ONLY_AGENT,
    JobType.CODING: RunType.CODING_TASK,
}


class JobWorker:
    def __init__(
        self,
        store: PostgresJobStore,
        broker: JobBroker,
        execution: ExecutionService,
        *,
        worker_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> None:
        self.store = store
        self.broker = broker
        self.execution = execution
        self.worker_id = worker_id or f"worker-{uuid4()}"
        self.lease_seconds = lease_seconds

    def run_once(self) -> bool:
        self.store.recover_expired()
        job = self.store.claim_next(self.worker_id, self.lease_seconds)
        if job is None:
            return False
        self._execute(job)
        return True

    def _execute(self, job: Job) -> None:
        def forward(run_id, event) -> None:
            self.broker.publish_progress(job.id, safe_progress_event(run_id, event))

        recorder = InMemoryTraceRecorder(sink=self.execution.trace_store.persist_run_trace, listener=forward)
        trace = TraceContext(recorder, _RUN_TYPES[job.job_type])
        assert trace.run_id is not None
        self.store.link_trace(job.id, self.worker_id, trace.run_id)
        done = Event()
        heartbeat = Thread(target=self._heartbeat, args=(job.id, done), daemon=True)
        heartbeat.start()
        try:
            result = self._dispatch(job, trace)
            self.store.mark_succeeded(job.id, self.worker_id, result.model_dump(mode="json"))
        except APIError as exc:
            self.store.mark_failed(job.id, self.worker_id, exc.code)
        except Exception:
            logger.exception("Durable job failed")
            self.store.mark_failed(job.id, self.worker_id, "operation_failed")
        finally:
            done.set()
            heartbeat.join(timeout=1)

    def _heartbeat(self, job_id, done: Event) -> None:
        while not done.wait(self.lease_seconds / 3):
            if not self.store.renew_lease(job_id, self.worker_id, self.lease_seconds):
                return

    def _dispatch(self, job: Job, trace: TraceContext):
        if job.job_type == JobType.INDEX:
            return self.execution.index(job.repository_id, trace=trace)
        if job.job_type == JobType.RAG:
            return self.execution.rag(job.repository_id, RAGRequest.model_validate(job.request_payload), trace=trace)
        if job.job_type == JobType.AGENT:
            return self.execution.agent(job.repository_id, AgentRequest.model_validate(job.request_payload), trace=trace)
        if job.job_type == JobType.CODING:
            return self.execution.coding(job.repository_id, CodingRequest.model_validate(job.request_payload), trace=trace)
        raise ValueError("Unsupported durable job type")
