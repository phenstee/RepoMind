from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4

from repomind.api.models import CodingResponse, RAGResponse, VerificationSummary
from repomind.coding import CodingPlan, CodingReview, CodingTaskStatus
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

    def get(self, job_id):
        assert job_id == self.job.id
        return self.job

    def request_cancel(self):
        self.job = self.job.model_copy(update={"cancel_requested_at": self.job.created_at})

    def mark_side_effect_started(self, job_id, worker_id):
        self.job = self.job.model_copy(update={"side_effect_started_at": self.job.created_at})
        return self.job

    def mark_cancelled(self, job_id, worker_id):
        self.job = self.job.model_copy(update={"status": JobStatus.CANCELLED})

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

    def rag(self, repository_id, request, *, trace, cancellation):
        trace.emit("retrieval.started", strategy=request.strategy, reranking_enabled=False)
        if self.fail:
            raise RuntimeError("sk-secret postgresql://private:password@host/db")
        cancellation.checkpoint()
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


def _coding_job():
    from datetime import UTC, datetime

    return Job(
        id=uuid4(),
        job_type=JobType.CODING,
        repository_id=1,
        status=JobStatus.QUEUED,
        request_payload={"objective": "Make one bounded change."},
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


def test_durable_coding_keeps_plan_executor_review_and_result_under_one_job_trace():
    class CodingExecution(FakeExecution):
        def coding(self, repository_id, request, *, trace, cancellation):
            del repository_id, request
            cancellation.checkpoint()
            trace.emit("planning.started", criteria_count=0)
            trace.emit(
                "planning.completed",
                criteria_count=0,
                step_count=1,
                criteria_covered=0,
                uncertainty_count=0,
            )
            trace.emit("agent.decision", iteration=1, action="final")
            trace.emit("review.started", workspace_revision=0)
            trace.emit(
                "review.completed",
                workspace_revision=0,
                verdict="approve",
                criteria_satisfied=0,
                criteria_unsatisfied=0,
                finding_count=0,
            )
            trace.finish("completed")
            empty_verification = VerificationSummary(
                passed=None,
                workspace_revision=None,
                exit_code=None,
                timed_out=None,
                execution_failed=False,
            )
            return CodingResponse(
                status=CodingTaskStatus.COMPLETED,
                final_answer="Completed.",
                tests=empty_verification,
                ruff=empty_verification,
                changed_files=[],
                completion_attempts=1,
                workspace_revision=0,
                plan=CodingPlan(
                    task_summary="Make one bounded change.",
                    steps=[{"step_id": 1, "action": "Inspect and complete the task."}],
                ),
                review=CodingReview(verdict="approve", workspace_revision=0),
                review_attempts=1,
                review_blocks=0,
                trace_run_id=trace.run_id,
            )

    store, broker = FakeStore(_coding_job()), FakeBroker()
    worker = JobWorker(store, broker, CodingExecution(), worker_id="test", lease_seconds=30)

    assert worker.run_once()
    assert store.error is None
    assert store.result["plan"]["steps"][0]["step_id"] == 1
    assert store.result["review"]["verdict"] == "approve"
    assert store.result["trace_run_id"] == str(store.trace_run_id)
    assert [event.event for event in broker.events] == [
        "run.started",
        "planning.started",
        "planning.completed",
        "agent.decision",
        "review.started",
        "review.completed",
        "run.completed",
    ]


def test_worker_failure_never_persists_raw_exception_text():
    store, broker = FakeStore(_job()), FakeBroker()
    assert JobWorker(store, broker, FakeExecution(fail=True), worker_id="test", lease_seconds=30).run_once()
    assert store.error == "operation_failed"
    assert store.result is None
    assert "secret" not in str(store.error)


def test_worker_cooperatively_cancels_at_checkpoint_without_marking_failure():
    store, broker = FakeStore(_job()), FakeBroker()

    class CancellingExecution(FakeExecution):
        def rag(self, repository_id, request, *, trace, cancellation):
            del repository_id, request
            store.request_cancel()
            cancellation.checkpoint()

    assert JobWorker(
        store, broker, CancellingExecution(), worker_id="test", lease_seconds=30
    ).run_once()
    assert store.job.status == JobStatus.CANCELLED
    assert store.result is None
    assert store.error is None
    assert [event.event for event in broker.events] == [
        "run.started",
        "job.cancel_requested",
        "job.cancelled",
    ]


def test_running_blocking_operation_finishes_before_cancellation_is_acknowledged():
    store, broker = FakeStore(_job()), FakeBroker()
    started = Event()
    release = Event()

    class BlockingExecution(FakeExecution):
        def rag(self, repository_id, request, *, trace, cancellation):
            del repository_id, request
            started.set()
            assert release.wait(timeout=5)
            cancellation.checkpoint()

    worker = JobWorker(
        store, broker, BlockingExecution(), worker_id="test", lease_seconds=30
    )
    thread = Thread(target=worker.run_once)
    thread.start()
    assert started.wait(timeout=5)

    store.request_cancel()
    assert thread.is_alive()
    assert store.job.status == JobStatus.RUNNING

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert store.job.status == JobStatus.CANCELLED
