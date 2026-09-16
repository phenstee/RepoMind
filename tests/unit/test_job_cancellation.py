from datetime import UTC, datetime
from uuid import uuid4

import pytest

from repomind.jobs import (
    CancellationState,
    DurableCancellationToken,
    Job,
    JobCancellationRequested,
    JobStatus,
    JobType,
)
from repomind.jobs.store import InMemoryJobStore


class _Trace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type, **metadata) -> None:
        self.events.append((event_type, metadata))


def _running_coding_job() -> Job:
    return Job(
        id=uuid4(),
        job_type=JobType.CODING,
        repository_id=1,
        status=JobStatus.RUNNING,
        request_payload={"objective": "edit"},
        attempt_count=1,
        created_at=datetime.now(UTC),
        lease_owner="worker",
    )


def test_coding_cancellation_before_mutation_raises_at_checkpoint() -> None:
    store = InMemoryJobStore()
    job = _running_coding_job()
    store.jobs[job.id] = job
    store.request_cancel(job.id)
    trace = _Trace()
    token = DurableCancellationToken(store, job.id, "worker", JobType.CODING, trace)

    with pytest.raises(JobCancellationRequested):
        token.checkpoint()

    assert trace.events == [
        (
            "job.cancel_requested",
            {"job_type": "coding", "control": CancellationState.REQUESTED.value},
        )
    ]


def test_coding_cancellation_after_mutation_is_deferred_and_non_interrupting() -> None:
    store = InMemoryJobStore()
    job = _running_coding_job()
    store.jobs[job.id] = job
    trace = _Trace()
    token = DurableCancellationToken(store, job.id, "worker", JobType.CODING, trace)
    token.side_effect_started()
    store.request_cancel(job.id)

    token.checkpoint()
    token.checkpoint()

    assert store.get(job.id).cancellation_state == CancellationState.DEFERRED
    assert trace.events == [
        (
            "job.cancellation_deferred",
            {"job_type": "coding", "control": CancellationState.DEFERRED.value},
        )
    ]
