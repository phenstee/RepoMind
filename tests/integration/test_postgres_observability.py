"""Real PostgreSQL trace roundtrip; migrations must be applied as for indexing tests."""

from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from repomind.db import create_session_factory
from repomind.db.models import TraceEventRecord, TraceRunRecord
from repomind.observability import InMemoryTraceRecorder, TraceContext
from repomind.observability.persistence import PostgresTraceStore

pytestmark = pytest.mark.postgres


def test_trace_migration_roundtrip_listing_and_cascade(postgres_engine):
    with postgres_engine.connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == (
            ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
        )
    factory = create_session_factory(postgres_engine)
    store = PostgresTraceStore(factory)
    recorder = InMemoryTraceRecorder(sink=store.persist_run_trace)
    context = TraceContext(recorder, "coding_task")
    context.emit("completion.blocked", blocker_codes=["tests_failed"], workspace_revision=1)
    context.finish("verification_failed")
    trace = recorder.traces[context.run_id]
    try:
        assert recorder.diagnostics == []
        reconstructed = store.get_run_trace(context.run_id)
        assert reconstructed == trace
        assert reconstructed.started_at.utcoffset().total_seconds() == 0
        assert [e.sequence for e in reconstructed.events] == [1, 2, 3]
        listed = store.list_run_traces(run_type="coding_task", status="failed", limit=100)
        assert context.run_id in {item.run_id for item in listed}
        assert context.run_id not in {
            item.run_id for item in store.list_run_traces(status="completed", limit=100)
        }
        assert store.get_run_trace(uuid4()) is None
        with pytest.raises(IntegrityError):
            store.persist_run_trace(trace)
        assert store.get_run_trace(context.run_id) == trace
    finally:
        with factory.begin() as session:
            session.execute(delete(TraceRunRecord).where(TraceRunRecord.id == context.run_id))
    with factory() as session:
        assert (
            session.scalars(
                select(TraceEventRecord).where(TraceEventRecord.run_id == context.run_id)
            ).all()
            == []
        )
