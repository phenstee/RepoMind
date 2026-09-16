"""Real PostgreSQL coverage for durable-job claiming, recovery, and repository locks."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import Barrier
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, delete, select, text
from sqlalchemy.orm import Session, sessionmaker

from repomind.api.errors import APIError
from repomind.api.models import RAGResponse
from repomind.api.streaming import ProgressEvent
from repomind.db.models import JobRecord, RepositoryRecord, TraceEventRecord, TraceRunRecord
from repomind.jobs.locks import PostgresRepositoryExecutionLock
from repomind.jobs.models import JobStatus, JobType
from repomind.jobs.store import PostgresJobStore
from repomind.jobs.worker import JobWorker
from repomind.observability import PostgresTraceStore

pytestmark = pytest.mark.postgres


@dataclass
class JobDatabase:
    factory: sessionmaker[Session]
    repository_ids: list[int] = field(default_factory=list)

    def create_repository(self) -> int:
        with self.factory.begin() as session:
            record = RepositoryRecord(
                name=f"jobs-integration-{uuid4().hex}",
                workspace_relative_path=f"jobs/{uuid4().hex}",
            )
            session.add(record)
            session.flush()
            self.repository_ids.append(record.id)
            return record.id

    def cleanup(self) -> None:
        if not self.repository_ids:
            return
        with self.factory.begin() as session:
            trace_ids = tuple(
                trace_id
                for trace_id in session.scalars(
                    select(JobRecord.trace_run_id).where(JobRecord.repository_id.in_(self.repository_ids))
                )
                if trace_id is not None
            )
            session.execute(delete(JobRecord).where(JobRecord.repository_id.in_(self.repository_ids)))
            if trace_ids:
                session.execute(delete(TraceEventRecord).where(TraceEventRecord.run_id.in_(trace_ids)))
                session.execute(delete(TraceRunRecord).where(TraceRunRecord.id.in_(trace_ids)))
            session.execute(delete(RepositoryRecord).where(RepositoryRecord.id.in_(self.repository_ids)))


@pytest.fixture
def job_database(postgres_engine: Engine):
    database = JobDatabase(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    try:
        yield database
    finally:
        database.cleanup()


def _clock(initial: datetime) -> tuple[Callable[[], datetime], list[datetime]]:
    values = [initial]
    return lambda: values[0], values


def test_atomic_claim_assigns_one_worker_and_one_attempt(job_database: JobDatabase):
    repository_id = job_database.create_repository()
    store = PostgresJobStore(job_database.factory)
    job = store.create(JobType.RAG, repository_id, {"question": "Where is the entry point?"})
    barrier = Barrier(2)

    def claim(worker_id: str):
        barrier.wait()
        return PostgresJobStore(job_database.factory).claim_next(worker_id, lease_seconds=60)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(claim, ("worker-one", "worker-two")))

    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1
    assert claimed[0].id == job.id
    assert claimed[0].status == JobStatus.RUNNING
    assert claimed[0].lease_owner in {"worker-one", "worker-two"}
    assert claimed[0].attempt_count == 1
    assert store.get(job.id) == claimed[0]


def test_two_workers_claim_distinct_queued_jobs(job_database: JobDatabase):
    repository_id = job_database.create_repository()
    clock, values = _clock(datetime(2026, 9, 16, tzinfo=UTC))
    store = PostgresJobStore(job_database.factory, clock=clock)
    first = store.create(JobType.RAG, repository_id, {"question": "first"})
    values[0] += timedelta(microseconds=1)
    second = store.create(JobType.RAG, repository_id, {"question": "second"})
    barrier = Barrier(2)

    def claim(worker_id: str):
        barrier.wait()
        return PostgresJobStore(job_database.factory, clock=clock).claim_next(
            worker_id, lease_seconds=60
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = tuple(pool.map(claim, ("worker-one", "worker-two")))

    assert {job.id for job in claimed if job is not None} == {first.id, second.id}
    assert {job.lease_owner for job in claimed if job is not None} == {"worker-one", "worker-two"}
    assert all(job is not None and job.attempt_count == 1 for job in claimed)


def test_queued_cancellation_is_terminal_idempotent_and_unclaimable(
    job_database: JobDatabase,
):
    clock, _ = _clock(datetime(2026, 9, 16, tzinfo=UTC))
    store = PostgresJobStore(job_database.factory, clock=clock)
    job = store.create(JobType.RAG, job_database.create_repository(), {"question": "cancel"})

    cancelled = store.request_cancel(job.id)
    repeated = store.request_cancel(job.id)

    assert cancelled == repeated
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.cancel_requested_at == cancelled.cancelled_at == cancelled.finished_at
    assert cancelled.result_payload is None
    assert cancelled.error_code is None
    assert store.claim_next("worker", lease_seconds=60) is None


def test_claim_and_cancel_are_serialized_without_losing_the_request(
    job_database: JobDatabase,
):
    repository_id = job_database.create_repository()
    store = PostgresJobStore(job_database.factory)
    job = store.create(JobType.RAG, repository_id, {"question": "race"})
    barrier = Barrier(2)

    def claim():
        barrier.wait()
        return PostgresJobStore(job_database.factory).claim_next("worker", lease_seconds=60)

    def cancel():
        barrier.wait()
        return PostgresJobStore(job_database.factory).request_cancel(job.id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed_future = pool.submit(claim)
        cancelled_future = pool.submit(cancel)
        claimed = claimed_future.result()
        cancelled_future.result()

    final = store.get(job.id)
    assert final.cancel_requested_at is not None
    if claimed is None:
        assert final.status == JobStatus.CANCELLED
    else:
        assert final.status == JobStatus.RUNNING
        assert final.lease_owner == "worker"


def test_running_cancellation_is_completed_by_lease_owner(job_database: JobDatabase):
    store = PostgresJobStore(job_database.factory)
    job = store.create(
        JobType.AGENT, job_database.create_repository(), {"query": "investigate"}
    )
    assert store.claim_next("worker", lease_seconds=60).id == job.id

    requested = store.request_cancel(job.id)
    assert requested.status == JobStatus.RUNNING
    assert requested.cancel_requested_at is not None
    assert store.claim_next("worker-two", lease_seconds=60) is None
    store.mark_cancelled(job.id, "worker")

    cancelled = store.get(job.id)
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.cancelled_at is not None
    assert cancelled.finished_at == cancelled.cancelled_at
    assert cancelled.lease_owner is None
    assert cancelled.lease_expires_at is None


def test_coding_cancellation_is_deferred_after_first_side_effect(
    job_database: JobDatabase,
):
    store = PostgresJobStore(job_database.factory)
    job = store.create(
        JobType.CODING, job_database.create_repository(), {"objective": "edit"}
    )
    assert store.claim_next("worker", lease_seconds=60).id == job.id
    store.mark_side_effect_started(job.id, "worker")

    requested = store.request_cancel(job.id)

    assert requested.cancellation_state.value == "deferred"
    with pytest.raises(ValueError, match="cannot be cancelled"):
        store.mark_cancelled(job.id, "worker")
    store.mark_succeeded(job.id, "worker", {"status": "completed"})
    assert store.get(job.id).status == JobStatus.SUCCEEDED


@pytest.mark.parametrize("terminal", [JobStatus.SUCCEEDED, JobStatus.FAILED])
def test_cancel_request_does_not_rewrite_completed_job(
    job_database: JobDatabase, terminal: JobStatus
):
    store = PostgresJobStore(job_database.factory)
    job = store.create(JobType.RAG, job_database.create_repository(), {"question": "done"})
    store.claim_next("worker", lease_seconds=60)
    if terminal == JobStatus.SUCCEEDED:
        store.mark_succeeded(job.id, "worker", {"answer": "safe"})
    else:
        store.mark_failed(job.id, "worker", "operation_failed")
    before = store.get(job.id)

    after = store.request_cancel(job.id)

    assert after == before
    assert after.cancel_requested_at is None


def test_skip_locked_claims_another_queued_job_without_waiting(job_database: JobDatabase):
    repository_id = job_database.create_repository()
    clock, values = _clock(datetime(2026, 9, 16, tzinfo=UTC))
    store = PostgresJobStore(job_database.factory, clock=clock)
    first = store.create(JobType.RAG, repository_id, {"question": "first"})
    values[0] += timedelta(microseconds=1)
    second = store.create(JobType.RAG, repository_id, {"question": "second"})

    with job_database.factory() as locked_session:
        transaction = locked_session.begin()
        locked_session.execute(text("SET LOCAL lock_timeout = '500ms'"))
        locked_session.scalar(
            select(JobRecord).where(JobRecord.id == first.id).with_for_update()
        )
        claimed = PostgresJobStore(job_database.factory, clock=clock).claim_next(
            "worker-two", lease_seconds=60
        )
        transaction.rollback()

    assert claimed is not None
    assert claimed.id == second.id
    assert claimed.lease_owner == "worker-two"
    assert PostgresJobStore(job_database.factory, clock=clock).claim_next(
        "worker-one", lease_seconds=60
    ).id == first.id


def test_expired_non_coding_job_is_requeued_with_lease_cleared(job_database: JobDatabase):
    clock, values = _clock(datetime(2026, 9, 16, tzinfo=UTC))
    store = PostgresJobStore(job_database.factory, clock=clock)
    job = store.create(JobType.RAG, job_database.create_repository(), {"question": "recover me"})
    assert store.claim_next("worker-one", lease_seconds=30).id == job.id

    values[0] += timedelta(seconds=31)
    assert store.recover_expired() == (job.id,)
    recovered = store.get(job.id)
    assert recovered.status == JobStatus.QUEUED
    assert recovered.attempt_count == 1
    assert recovered.started_at is None
    assert recovered.finished_at is None
    assert recovered.lease_owner is None
    assert recovered.lease_expires_at is None
    assert store.claim_next("worker-two", lease_seconds=30).attempt_count == 2


def test_expired_cancel_requested_job_becomes_cancelled(job_database: JobDatabase):
    clock, values = _clock(datetime(2026, 9, 16, tzinfo=UTC))
    store = PostgresJobStore(job_database.factory, clock=clock)
    job = store.create(JobType.RAG, job_database.create_repository(), {"question": "stop"})
    assert store.claim_next("worker-one", lease_seconds=30).id == job.id
    store.request_cancel(job.id)

    values[0] += timedelta(seconds=31)
    assert store.recover_expired() == (job.id,)

    recovered = store.get(job.id)
    assert recovered.status == JobStatus.CANCELLED
    assert recovered.cancelled_at == values[0]
    assert recovered.lease_owner is None
    assert store.claim_next("worker-two", lease_seconds=30) is None


def test_expired_coding_job_fails_without_replay(job_database: JobDatabase):
    clock, values = _clock(datetime(2026, 9, 16, tzinfo=UTC))
    store = PostgresJobStore(job_database.factory, clock=clock)
    job = store.create(JobType.CODING, job_database.create_repository(), {"objective": "change one line"})
    assert store.claim_next("worker-one", lease_seconds=30).id == job.id

    values[0] += timedelta(seconds=31)
    assert store.recover_expired() == (job.id,)
    interrupted = store.get(job.id)
    assert interrupted.status == JobStatus.FAILED
    assert interrupted.error_code == "job_interrupted"
    assert interrupted.finished_at == values[0]
    assert interrupted.lease_owner is None
    assert interrupted.lease_expires_at is None
    assert store.claim_next("worker-two", lease_seconds=30) is None


def test_expired_cancel_requested_coding_job_still_fails_without_replay(
    job_database: JobDatabase,
):
    clock, values = _clock(datetime(2026, 9, 16, tzinfo=UTC))
    store = PostgresJobStore(job_database.factory, clock=clock)
    job = store.create(
        JobType.CODING, job_database.create_repository(), {"objective": "change one line"}
    )
    assert store.claim_next("worker-one", lease_seconds=30).id == job.id
    store.request_cancel(job.id)

    values[0] += timedelta(seconds=31)
    store.recover_expired()

    interrupted = store.get(job.id)
    assert interrupted.status == JobStatus.FAILED
    assert interrupted.error_code == "job_interrupted"
    assert interrupted.cancel_requested_at is not None


def test_index_and_coding_share_a_real_repository_advisory_lock(job_database: JobDatabase):
    same_repository = job_database.create_repository()
    other_repository = job_database.create_repository()
    index_lock = PostgresRepositoryExecutionLock(job_database.factory)
    coding_lock = PostgresRepositoryExecutionLock(job_database.factory)

    with index_lock.hold(same_repository):
        with pytest.raises(APIError) as busy, coding_lock.hold(same_repository):
            pytest.fail("the competing coding operation must not acquire the lock")
        assert busy.value.status == 409
        assert busy.value.code == "repository_busy"
        with coding_lock.hold(other_repository):
            pass

    with coding_lock.hold(same_repository):
        pass


@dataclass
class _Broker:
    events: list[ProgressEvent] = field(default_factory=list)

    def publish_progress(self, job_id: UUID, event: ProgressEvent) -> None:
        del job_id
        self.events.append(event)


class _Execution:
    def __init__(self, trace_store: PostgresTraceStore) -> None:
        self.trace_store = trace_store

    def rag(self, repository_id: int, request, *, trace, cancellation):
        del repository_id, request
        cancellation.checkpoint()
        trace.emit("retrieval.started", strategy="semantic", reranking_enabled=False)
        trace.finish()
        return RAGResponse(
            answer="safe answer",
            insufficient_evidence=False,
            citations=[],
            trace_run_id=trace.run_id,
        )


def test_worker_persists_trace_and_links_it_to_durable_job(job_database: JobDatabase):
    repository_id = job_database.create_repository()
    store = PostgresJobStore(job_database.factory)
    job = store.create(JobType.RAG, repository_id, {"question": "Where is the entry point?"})
    trace_store = PostgresTraceStore(job_database.factory)

    assert JobWorker(store, _Broker(), _Execution(trace_store), worker_id="integration").run_once()

    completed = store.get(job.id)
    assert completed.status == JobStatus.SUCCEEDED
    assert completed.trace_run_id is not None
    persisted = trace_store.get_run_trace(completed.trace_run_id)
    assert persisted is not None
    assert persisted.run_id == completed.trace_run_id
    assert [event.event_type for event in persisted.events] == [
        "run.started",
        "retrieval.started",
        "run.completed",
    ]


def test_worker_persists_normal_cancellation_as_cancelled_not_failed(
    job_database: JobDatabase,
):
    repository_id = job_database.create_repository()
    store = PostgresJobStore(job_database.factory)
    job = store.create(JobType.RAG, repository_id, {"question": "cancel safely"})
    trace_store = PostgresTraceStore(job_database.factory)

    class CancellingExecution:
        def __init__(self) -> None:
            self.trace_store = trace_store

        def rag(self, repository_id, request, *, trace, cancellation):
            del repository_id, request
            store.request_cancel(job.id)
            cancellation.checkpoint()

    assert JobWorker(
        store, _Broker(), CancellingExecution(), worker_id="integration-cancel"
    ).run_once()

    cancelled = store.get(job.id)
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.error_code is None
    persisted = trace_store.get_run_trace(cancelled.trace_run_id)
    assert persisted is not None
    assert persisted.status.value == "cancelled"
    assert [event.event_type for event in persisted.events] == [
        "run.started",
        "job.cancel_requested",
        "job.cancelled",
    ]
