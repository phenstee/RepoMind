from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest

from repomind.observability import InMemoryTraceRecorder, NoOpTraceRecorder, TraceContext
from repomind.observability.instrumentation import record_model_usage


def test_sequence_fake_clocks_and_token_aggregation() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    elapsed = [10.0]
    recorder = InMemoryTraceRecorder(
        id_provider=lambda: UUID(int=1), utc_clock=lambda: now, monotonic_clock=lambda: elapsed[0]
    )
    trace = TraceContext(recorder, "rag")
    for prompt, completion in ((100, 20), (50, 10)):
        with trace.operation("model", model="fake", operation="text"):
            record_model_usage(
                trace,
                SimpleNamespace(
                    usage=SimpleNamespace(
                        prompt_tokens=prompt,
                        completion_tokens=completion,
                        total_tokens=prompt + completion,
                    )
                ),
                1,
            )
    elapsed[0] = 10.25
    now += timedelta(seconds=1)
    trace.finish()
    result = recorder.traces[UUID(int=1)]
    assert result.duration_ms == 250
    assert result.started_at.tzinfo == UTC
    assert result.ended_at == now
    assert [e.sequence for e in result.events] == list(range(1, 9))
    assert result.llm_calls == 2
    assert result.token_usage.model_dump() == {
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "total_tokens": 180,
    }
    assert result.usage_reported_calls == 2
    assert result.estimated_cost_usd is None
    before = result.model_dump(mode="json")
    trace.finish()
    trace.emit("tool.started", tool="late")
    assert recorder.traces[UUID(int=1)].model_dump(mode="json") == before


def test_absent_usage_stays_unknown() -> None:
    recorder = InMemoryTraceRecorder()
    trace = TraceContext(recorder, "rag")
    with trace.operation("model", operation="text"):
        record_model_usage(trace, SimpleNamespace(usage=None), 1)
    trace.finish()
    result = recorder.traces[trace.run_id]
    assert result.token_usage is None
    assert result.usage_reported_calls == 0


def test_noop_keeps_no_events() -> None:
    trace = TraceContext(NoOpTraceRecorder(), "rag")
    trace.emit("tool.started", tool="read_file")
    trace.finish()
    assert trace.run_id is None
    assert trace.events == []
    def invalid_projection():
        raise AssertionError("no-op must not extract payload metadata")
    assert trace.project(invalid_projection) == {}


def test_sink_failure_retains_trace_and_safe_diagnostic(caplog) -> None:
    def fail(trace):
        raise RuntimeError("postgresql://user:password@host/db")

    recorder = InMemoryTraceRecorder(sink=fail)
    trace = TraceContext(recorder, "editing_agent")
    trace.emit("file.mutated", path="app.py")
    trace.finish()
    assert recorder.traces[trace.run_id].successful_mutations == 1
    assert len(recorder.diagnostics) == 1
    assert "password" not in str(recorder.diagnostics) + caplog.text
    assert "persistence failed" in caplog.text


@pytest.mark.parametrize("method", ["start_run", "record_event", "finish_run"])
def test_custom_recorder_failure_is_best_effort(method, monkeypatch) -> None:
    recorder = InMemoryTraceRecorder()

    def fail(*args):
        raise RuntimeError("sk-never-store-this")

    monkeypatch.setattr(recorder, method, fail)
    trace = TraceContext(recorder, "rag")
    trace.emit("model.started", model="fake")
    trace.finish()
    assert trace.diagnostics
    assert "sk-never-store-this" not in str(trace.diagnostics)


def test_separate_contexts_have_independent_ids_and_sequences() -> None:
    recorder = InMemoryTraceRecorder()
    first, second = TraceContext(recorder, "rag"), TraceContext(recorder, "rag")
    first.emit("model.started")
    second.finish()
    first.finish()
    assert first.run_id != second.run_id
    assert len(recorder.traces[first.run_id].events) == 3
    assert len(recorder.traces[second.run_id].events) == 2
