"""PostgreSQL-backed durable job state with atomic PostgreSQL claims."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from repomind.db.models import JobRecord
from repomind.db.session import session_scope
from repomind.jobs.models import Job, JobStatus, JobType


class JobNotFoundError(ValueError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


def _job(record: JobRecord) -> Job:
    return Job(
        id=record.id,
        job_type=record.job_type,
        repository_id=record.repository_id,
        status=record.status,
        payload_version=record.payload_version,
        request_payload=record.request_json,
        result_payload=record.result_json,
        error_code=record.error_code,
        attempt_count=record.attempt_count,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        lease_owner=record.lease_owner,
        lease_expires_at=record.lease_expires_at,
        trace_run_id=record.trace_run_id,
    )


class PostgresJobStore:
    def __init__(self, factory: sessionmaker[Session], *, clock=utc_now) -> None:
        self.factory = factory
        self.clock = clock

    def create(self, job_type: JobType, repository_id: int, payload: dict) -> Job:
        now = self.clock()
        with session_scope(self.factory) as session:
            record = JobRecord(
                id=uuid4(),
                job_type=job_type.value,
                repository_id=repository_id,
                status=JobStatus.QUEUED.value,
                payload_version=1,
                request_json=payload,
                attempt_count=0,
                created_at=now,
            )
            session.add(record)
            session.flush()
            return _job(record)

    def get(self, job_id: UUID) -> Job:
        with self.factory() as session:
            record = session.get(JobRecord, job_id)
            if record is None:
                raise JobNotFoundError("Job not found")
            return _job(record)

    def list(
        self,
        *,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
        repository_id: int | None = None,
        limit: int = 20,
    ) -> Sequence[Job]:
        query = select(JobRecord)
        if status is not None:
            query = query.where(JobRecord.status == status.value)
        if job_type is not None:
            query = query.where(JobRecord.job_type == job_type.value)
        if repository_id is not None:
            query = query.where(JobRecord.repository_id == repository_id)
        query = query.order_by(JobRecord.created_at.desc(), JobRecord.id).limit(limit)
        with self.factory() as session:
            return tuple(_job(record) for record in session.scalars(query))

    def claim_next(self, worker_id: str, lease_seconds: float) -> Job | None:
        now = self.clock()
        with session_scope(self.factory) as session:
            record = session.scalar(
                select(JobRecord)
                .where(JobRecord.status == JobStatus.QUEUED.value)
                .order_by(JobRecord.created_at, JobRecord.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if record is None:
                return None
            record.status = JobStatus.RUNNING.value
            record.attempt_count += 1
            record.started_at = now
            record.lease_owner = worker_id
            record.lease_expires_at = now + timedelta(seconds=lease_seconds)
            session.flush()
            return _job(record)

    def renew_lease(self, job_id: UUID, worker_id: str, lease_seconds: float) -> bool:
        now = self.clock()
        with session_scope(self.factory) as session:
            record = session.get(JobRecord, job_id)
            if record is None or record.status != JobStatus.RUNNING.value or record.lease_owner != worker_id:
                return False
            record.lease_expires_at = now + timedelta(seconds=lease_seconds)
            return True

    def link_trace(self, job_id: UUID, worker_id: str, trace_run_id: UUID) -> None:
        with session_scope(self.factory) as session:
            record = session.get(JobRecord, job_id)
            if record is None or record.status != JobStatus.RUNNING.value or record.lease_owner != worker_id:
                raise JobNotFoundError("Job lease is no longer active")
            record.trace_run_id = trace_run_id

    def mark_succeeded(self, job_id: UUID, worker_id: str, result: dict) -> None:
        self._terminal(job_id, worker_id, JobStatus.SUCCEEDED, result=result)

    def mark_failed(self, job_id: UUID, worker_id: str, error_code: str) -> None:
        self._terminal(job_id, worker_id, JobStatus.FAILED, error_code=error_code)

    def _terminal(
        self,
        job_id: UUID,
        worker_id: str,
        status: JobStatus,
        *,
        result: dict | None = None,
        error_code: str | None = None,
    ) -> None:
        with session_scope(self.factory) as session:
            record = session.get(JobRecord, job_id)
            if record is None or record.status != JobStatus.RUNNING.value or record.lease_owner != worker_id:
                raise JobNotFoundError("Job lease is no longer active")
            record.status = status.value
            record.result_json = result
            record.error_code = error_code
            record.finished_at = self.clock()
            record.lease_owner = None
            record.lease_expires_at = None

    def recover_expired(self) -> tuple[UUID, ...]:
        now = self.clock()
        recovered: list[UUID] = []
        with session_scope(self.factory) as session:
            records = session.scalars(
                select(JobRecord)
                .where(
                    JobRecord.status == JobStatus.RUNNING.value,
                    JobRecord.lease_expires_at.is_not(None),
                    JobRecord.lease_expires_at <= now,
                )
                .with_for_update(skip_locked=True)
            ).all()
            for record in records:
                recovered.append(record.id)
                if record.job_type == JobType.CODING.value:
                    record.status = JobStatus.FAILED.value
                    record.error_code = "job_interrupted"
                    record.finished_at = now
                else:
                    record.status = JobStatus.QUEUED.value
                    record.started_at = None
                record.lease_owner = None
                record.lease_expires_at = None
        return tuple(recovered)


class InMemoryJobStore:
    """Injection-only store for offline API tests; production always uses PostgreSQL."""

    def __init__(self) -> None:
        self.jobs: dict[UUID, Job] = {}

    def create(self, job_type: JobType, repository_id: int, payload: dict) -> Job:
        job = Job(id=uuid4(), job_type=job_type, repository_id=repository_id, status=JobStatus.QUEUED, request_payload=payload, attempt_count=0, created_at=utc_now())
        self.jobs[job.id] = job
        return job

    def get(self, job_id: UUID) -> Job:
        if job_id not in self.jobs:
            raise JobNotFoundError("Job not found")
        return self.jobs[job_id]

    def list(self, **filters) -> Sequence[Job]:
        jobs = tuple(self.jobs.values())
        return jobs[: filters.get("limit", 20)]
