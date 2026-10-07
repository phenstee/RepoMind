"""The durable worker must outlive database/Redis blips, lost leases and shutdown signals."""

import os
import signal
import time
from datetime import UTC, datetime
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.exc import OperationalError

from repomind import worker as worker_main
from repomind.api.models import IndexResponse
from repomind.jobs.models import Job, JobStatus, JobType
from repomind.jobs.store import JobNotFoundError
from repomind.jobs.worker import JobWorker


def _db_down(*args, **kwargs):
    raise OperationalError("SELECT 1", {}, Exception("connection refused"))


class RecordingStop(Event):
    """A stop event whose backoff waits return immediately but are recorded."""

    def __init__(self):
        super().__init__()
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.is_set()


class ScriptedWorker:
    def __init__(self, stop, steps):
        self.stop = stop
        self.steps = list(steps)

    def run_once(self):
        step = self.steps.pop(0)
        if not self.steps:
            self.stop.set()
        if isinstance(step, Exception):
            raise step
        return step


def test_worker_loop_survives_failures_with_capped_exponential_backoff():
    stop = RecordingStop()
    failure = OperationalError("SELECT 1", {}, Exception("down"))
    steps = [failure] * 7 + [True, failure, True]
    broker = SimpleNamespace(wait_for_work=lambda timeout: False)

    worker_main.run(ScriptedWorker(stop, steps), broker, stop, poll_seconds=1.0)

    # Doubling backoff is capped at 30s and resets after a successful iteration.
    assert stop.waits == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 1.0]


def test_idle_wait_returns_on_wakeup_and_never_blocks_shutdown_for_long():
    timeouts = []
    results = iter([False, False, True])

    def wait_for_work(timeout):
        timeouts.append(timeout)
        return next(results)

    broker = SimpleNamespace(wait_for_work=wait_for_work)
    worker_main.wait_for_work(broker, 60.0, Event())

    assert len(timeouts) == 3
    assert all(timeout <= 1.0 for timeout in timeouts)

    stop = Event()

    def stop_requested(timeout):
        stop.set()
        return False

    started = time.monotonic()
    worker_main.wait_for_work(SimpleNamespace(wait_for_work=stop_requested), 60.0, stop)
    assert time.monotonic() - started < 1.0


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="POSIX signals")
def test_sigterm_requests_a_graceful_stop():
    previous = signal.getsignal(signal.SIGTERM)
    stop = Event()
    try:
        worker_main.install_shutdown_handler(stop)
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.wait(timeout=2)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _job(job_type=JobType.RAG, payload=None):
    return Job(
        id=uuid4(),
        job_type=job_type,
        repository_id=1,
        status=JobStatus.QUEUED,
        request_payload={"question": "Where?"} if payload is None else payload,
        attempt_count=0,
        created_at=datetime.now(UTC),
    )


class FakeStore:
    def __init__(self, job):
        self.job = job
        self.renewals = 0
        self.outcomes = []

    def recover_expired(self):
        return ()

    def claim_next(self, worker_id, lease_seconds):
        self.job = self.job.model_copy(
            update={"status": JobStatus.RUNNING, "lease_owner": worker_id}
        )
        return self.job

    def link_trace(self, job_id, worker_id, trace_run_id):
        return None

    def renew_lease(self, job_id, worker_id, lease_seconds):
        self.renewals += 1
        return True

    def get(self, job_id):
        return self.job

    def request_cancel(self):
        self.job = self.job.model_copy(update={"cancel_requested_at": datetime.now(UTC)})

    def mark_succeeded(self, job_id, worker_id, result):
        self.outcomes.append(("succeeded", result))

    def mark_failed(self, job_id, worker_id, error_code):
        self.outcomes.append(("failed", error_code))

    def mark_cancelled(self, job_id, worker_id):
        self.outcomes.append(("cancelled", None))


class FakeExecution:
    def __init__(self, operation):
        self.trace_store = SimpleNamespace(persist_run_trace=lambda trace: None)
        self.operation = operation

    def rag(self, repository_id, request, *, trace, cancellation):
        return self.operation(trace, cancellation)

    def index(self, repository_id, *, trace, cancellation):
        return self.operation(trace, cancellation)


def _broker():
    return SimpleNamespace(publish_progress=lambda job_id, event: None)


def _index_response(trace):
    trace.finish("completed")
    return IndexResponse(
        repository_id=1, files_indexed=1, chunks_indexed=1, embedding_model="offline-model"
    )


def test_cancel_request_after_dispatch_returns_still_records_success():
    store = FakeStore(_job(JobType.INDEX, {}))

    def index_then_cancel(trace, cancellation):
        cancellation.checkpoint()
        response = _index_response(trace)
        store.request_cancel()  # arrives after the index has been committed
        return response

    assert JobWorker(store, _broker(), FakeExecution(index_then_cancel), worker_id="w").run_once()

    assert [outcome for outcome, _ in store.outcomes] == ["succeeded"]
    assert store.outcomes[0][1]["files_indexed"] == 1


@pytest.mark.parametrize("error", [JobNotFoundError("Job lease is no longer active"), None])
@pytest.mark.parametrize("operation_fails", [False, True])
def test_lost_lease_or_database_blip_while_recording_outcome_does_not_crash(
    monkeypatch, error, operation_fails
):
    store = FakeStore(_job())

    def operation(trace, cancellation):
        if operation_fails:
            raise RuntimeError("provider exploded")
        return _index_response(trace)

    def unavailable(*args):
        if error is not None:
            raise error
        _db_down()

    monkeypatch.setattr(store, "mark_succeeded", unavailable)
    monkeypatch.setattr(store, "mark_failed", unavailable)

    assert JobWorker(store, _broker(), FakeExecution(operation), worker_id="w").run_once()


def test_lost_lease_before_start_skips_the_job_without_running_it():
    store = FakeStore(_job())
    ran = []

    def lease_lost(*args):
        raise JobNotFoundError("Job lease is no longer active")

    store.link_trace = lease_lost
    execution = FakeExecution(lambda trace, cancellation: ran.append(True))

    assert JobWorker(store, _broker(), execution, worker_id="w").run_once()
    assert not ran and not store.outcomes


def test_heartbeat_keeps_renewing_after_a_transient_database_error():
    store = FakeStore(_job())
    calls = []

    def flaky_renew(job_id, worker_id, lease_seconds):
        calls.append(time.monotonic())
        if len(calls) == 1:
            _db_down()
        return True

    store.renew_lease = flaky_renew
    worker = JobWorker(store, _broker(), FakeExecution(None), worker_id="w", lease_seconds=0.15)
    done = Event()
    heartbeat = Thread(target=worker._heartbeat, args=(store.job.id, done), daemon=True)
    heartbeat.start()
    try:
        deadline = time.monotonic() + 5
        while len(calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        done.set()
        heartbeat.join(timeout=2)

    assert len(calls) >= 3
    assert not heartbeat.is_alive()


def test_heartbeat_stops_once_the_lease_is_lost():
    store = FakeStore(_job())
    store.renew_lease = lambda job_id, worker_id, lease_seconds: False
    worker = JobWorker(store, _broker(), FakeExecution(None), worker_id="w", lease_seconds=0.03)
    heartbeat = Thread(target=worker._heartbeat, args=(store.job.id, Event()), daemon=True)
    heartbeat.start()
    heartbeat.join(timeout=2)
    assert not heartbeat.is_alive()
