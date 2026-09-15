"""Optional PostgreSQL store with one independent transaction per completed trace."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from repomind.db.models import TraceEventRecord, TraceRunRecord
from repomind.db.session import session_scope
from repomind.observability.models import RunStatus, RunTrace, RunType, TraceEvent


class PostgresTraceStore:
    """Inject a factory bound to an Engine, never a primary operation's Session.

    Direct calls raise normal database exceptions. Use ``store.persist_run_trace``
    as an InMemoryTraceRecorder sink for best-effort persistence and diagnostics.
    No connection or schema changes occur during construction.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def persist_run_trace(self, trace: RunTrace) -> None:
        trace = RunTrace.model_validate(trace.model_dump(mode="json"))
        if trace.status == RunStatus.RUNNING:
            raise ValueError("only completed/failed runs can be persisted")
        usage = trace.token_usage
        with session_scope(self.session_factory) as session:
            session.add(
                TraceRunRecord(
                    id=trace.run_id,
                    run_type=trace.run_type.value,
                    status=trace.status.value,
                    model=trace.model,
                    started_at=trace.started_at,
                    ended_at=trace.ended_at,
                    duration_ms=trace.duration_ms,
                    llm_calls=trace.llm_calls,
                    tool_calls=trace.tool_calls,
                    successful_mutations=trace.successful_mutations,
                    error_count=trace.errors,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                    total_tokens=usage.total_tokens if usage else None,
                    details_json={
                        "domain_status": trace.domain_status,
                        "usage_reported_calls": trace.usage_reported_calls,
                        "evaluation_summary": trace.evaluation_summary,
                    },
                )
            )
            session.flush()
            session.add_all(
                [
                    TraceEventRecord(
                        run_id=trace.run_id,
                        sequence=e.sequence,
                        event_type=e.event_type,
                        timestamp=e.timestamp,
                        duration_ms=e.duration_ms,
                        metadata_json=e.metadata,
                    )
                    for e in trace.events
                ]
            )

    def get_run_trace(self, run_id: UUID) -> RunTrace | None:
        with self.session_factory() as session:
            row = session.get(TraceRunRecord, run_id)
            return self._load(session, row) if row is not None else None

    def list_run_traces(
        self, *, run_type: RunType | None = None, status: RunStatus | None = None, limit: int = 20
    ) -> tuple[RunTrace, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        query = select(TraceRunRecord)
        if run_type is not None:
            query = query.where(TraceRunRecord.run_type == RunType(run_type).value)
        if status is not None:
            query = query.where(TraceRunRecord.status == RunStatus(status).value)
        query = query.order_by(TraceRunRecord.started_at.desc(), TraceRunRecord.id).limit(limit)
        with self.session_factory() as session:
            return tuple(self._load(session, row) for row in session.scalars(query))

    @staticmethod
    def _load(session: Session, row: TraceRunRecord) -> RunTrace:
        events = session.scalars(
            select(TraceEventRecord)
            .where(TraceEventRecord.run_id == row.id)
            .order_by(TraceEventRecord.sequence)
        ).all()
        usage = (
            None
            if row.total_tokens is None
            else {
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "total_tokens": row.total_tokens,
            }
        )
        return RunTrace(
            run_id=row.id,
            run_type=row.run_type,
            status=row.status,
            model=row.model,
            started_at=row.started_at,
            ended_at=row.ended_at,
            duration_ms=row.duration_ms,
            llm_calls=row.llm_calls,
            tool_calls=row.tool_calls,
            successful_mutations=row.successful_mutations,
            errors=row.error_count,
            token_usage=usage,
            **row.details_json,
            events=tuple(
                TraceEvent(
                    event_type=e.event_type,
                    sequence=e.sequence,
                    timestamp=e.timestamp,
                    duration_ms=e.duration_ms,
                    metadata=e.metadata_json,
                )
                for e in events
            ),
        )
