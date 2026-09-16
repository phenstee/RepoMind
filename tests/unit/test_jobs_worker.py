from types import SimpleNamespace
from uuid import uuid4

from repomind.api.models import RAGResponse
from repomind.jobs.models import Job, JobStatus, JobType
from repomind.jobs.worker import JobWorker


class FakeStore:
    def __init__(self, job):
        self.job = job
        self.trace_run_id = None
        self.result = None
        self.error = None

    def recover_expired(self):
        return ()

    def claim_next(self, worker_id, lease_seconds):
        self.job = self.job.model_copy(update={"status": JobStatus.RUNNING, "lease_owner": worker_id})
        return self.job

    def link_trace(self, job_id, worker_id, trace_run_id):
        self.trace_run_id = trace_run_id

    def renew_lease(self, job_id, worker_id, lease_seconds):
        return True

    def mark_succeeded(self, job_id, worker_id, result):
        self.result = result

    def mark_failed(self, job_id, worker_id, error_code):
        self.error = error_code


class FakeBroker:
    def __init__(self):
        self.events = []

    def publish_progress(self, job_id, event):
        self.events.append(event)


class FakeExecution:
    def __init__(self, *, fail=False):
        self.trace_store = SimpleNamespace(persist_run_trace=lambda trace: None)
        self.fail = fail

    def rag(self, repository_id, request, *, trace):
        trace.emit("retrieval.started", strategy=request.strategy, reranking_enabled=False)
        if self.fail:
            raise RuntimeError("sk-secret postgresql://private:password@host/db")
        trace.finish("completed")
        return RAGResponse(
            answer="safe answer",
            insufficient_evidence=False,
            citations=[],
            trace_run_id=trace.run_id,
        )


def _job():
    from datetime import UTC, datetime

    return Job(
        id=uuid4(),
        job_type=JobType.RAG,
        repository_id=1,
        status=JobStatus.QUEUED,
        request_payload={"question": "Where is the entry point?"},
        attempt_count=0,
        created_at=datetime.now(UTC),
    )


def test_worker_reuses_rag_service_persists_safe_result_and_forwards_progress():
    store, broker = FakeStore(_job()), FakeBroker()
    assert JobWorker(store, broker, FakeExecution(), worker_id="test", lease_seconds=30).run_once()
    assert store.trace_run_id is not None
    assert store.result["answer"] == "safe answer"
    assert store.error is None
    assert [event.event for event in broker.events] == [
        "run.started",
        "retrieval.started",
        "run.completed",
    ]


def test_worker_failure_never_persists_raw_exception_text():
    store, broker = FakeStore(_job()), FakeBroker()
    assert JobWorker(store, broker, FakeExecution(fail=True), worker_id="test", lease_seconds=30).run_once()
    assert store.error == "operation_failed"
    assert store.result is None
    assert "secret" not in str(store.error)
