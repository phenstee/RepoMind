from io import StringIO
from unittest.mock import MagicMock

import pytest
from alembic.config import Config
from sqlalchemy.dialects.postgresql import JSONB

from alembic import command
from repomind.db.models import TraceEventRecord, TraceRunRecord
from repomind.observability import InMemoryTraceRecorder, TraceContext
from repomind.observability.persistence import PostgresTraceStore


def test_orm_metadata_and_offline_migration_sql(monkeypatch):
    monkeypatch.setattr("logging.config.fileConfig", lambda *a, **k: None)
    assert isinstance(TraceEventRecord.__table__.c.metadata_json.type, JSONB)
    assert TraceRunRecord.__table__.c.started_at.type.timezone
    assert next(iter(TraceEventRecord.__table__.c.run_id.foreign_keys)).ondelete == "CASCADE"
    output = StringIO()
    config = Config("alembic.ini", output_buffer=output)
    command.upgrade(config, "20260910_01:head", sql=True)
    sql = output.getvalue()
    assert "CREATE TABLE trace_runs" in sql
    assert "CREATE TABLE trace_events" in sql
    assert "ON DELETE CASCADE" in sql
    assert "TIMESTAMP WITH TIME ZONE" in sql


def test_store_revalidates_metadata_and_uses_own_transaction():
    recorder = InMemoryTraceRecorder()
    context = TraceContext(recorder, "rag")
    context.emit("tool.completed", tool="read_file", path="app.py")
    context.finish()
    trace = recorder.traces[context.run_id]
    trace.events[1].metadata["content"] = "PRIVATE SOURCE"
    factory = MagicMock()
    session = factory.return_value
    store = PostgresTraceStore(factory)
    store.persist_run_trace(trace)
    session.commit.assert_called_once()
    session.close.assert_called_once()
    events = session.add_all.call_args.args[0]
    assert events[1].metadata_json == {"tool": "read_file", "path": "app.py"}
    session.reset_mock()
    session.flush.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError):
        store.persist_run_trace(trace)
    session.rollback.assert_called_once()
    session.close.assert_called_once()
    session.commit.assert_not_called()


def test_store_rejects_running_trace_and_unbounded_listing():
    recorder = InMemoryTraceRecorder()
    context = TraceContext(recorder, "rag")
    factory = MagicMock()
    store = PostgresTraceStore(factory)
    with pytest.raises(ValueError, match="completed/failed"):
        store.persist_run_trace(recorder.traces[context.run_id])
    for limit in (0, 101, True):
        with pytest.raises(ValueError, match="limit"):
            store.list_run_traces(limit=limit)
    factory.assert_not_called()
