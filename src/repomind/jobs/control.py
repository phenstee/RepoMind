"""Transport-neutral cooperative cancellation checked only at safe boundaries."""

from typing import Protocol
from uuid import UUID

from repomind.jobs.models import CancellationState, Job, JobType


class JobCancellationRequested(RuntimeError):
    """Raised only at a safe cooperative checkpoint."""


class CancellationStore(Protocol):
    def get(self, job_id: UUID) -> Job: ...
    def mark_side_effect_started(self, job_id: UUID, worker_id: str) -> Job: ...


class CancellationTrace(Protocol):
    def emit(self, event_type, **metadata) -> None: ...


class CooperativeCancellation(Protocol):
    def checkpoint(self) -> None: ...
    def side_effect_started(self) -> None: ...


class NoCancellation:
    def checkpoint(self) -> None:
        return None

    def side_effect_started(self) -> None:
        return None


class DurableCancellationToken:
    def __init__(
        self,
        store: CancellationStore,
        job_id: UUID,
        worker_id: str,
        job_type: JobType,
        trace: CancellationTrace,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.worker_id = worker_id
        self.job_type = job_type
        self.trace = trace
        self._request_emitted = False
        self._deferred_emitted = False

    def checkpoint(self) -> None:
        job = self.store.get(self.job_id)
        if not job.cancel_requested:
            return
        if job.cancellation_state == CancellationState.DEFERRED:
            if not self._deferred_emitted:
                self.trace.emit(
                    "job.cancellation_deferred",
                    job_type=self.job_type.value,
                    control=CancellationState.DEFERRED.value,
                )
                self._deferred_emitted = True
            return
        if not self._request_emitted:
            self.trace.emit(
                "job.cancel_requested",
                job_type=self.job_type.value,
                control=CancellationState.REQUESTED.value,
            )
            self._request_emitted = True
        raise JobCancellationRequested("job cancellation requested")

    def side_effect_started(self) -> None:
        if self.job_type == JobType.CODING:
            self.store.mark_side_effect_started(self.job_id, self.worker_id)
