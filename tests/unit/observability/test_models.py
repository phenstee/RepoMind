from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from repomind.observability import RunTrace, TraceEvent


@pytest.mark.parametrize(
    "update",
    [
        {"sequence": 0},
        {"sequence": True},
        {"event_type": "made.up"},
        {"timestamp": datetime(2026, 1, 1)},  # noqa: DTZ001 - invalid input under test
        {"duration_ms": -1},
        {"duration_ms": float("nan")},
    ],
)
def test_invalid_events_rejected(update) -> None:
    data = {
        "event_type": "run.started",
        "sequence": 1,
        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
    } | update
    with pytest.raises(ValidationError):
        TraceEvent(**data)


def test_trace_rejects_noncontiguous_events_and_unfinished_terminal_state() -> None:
    data = {
        "run_id": UUID(int=1),
        "run_type": "rag",
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    with pytest.raises(ValidationError, match="contiguous"):
        RunTrace(
            **data,
            events=[TraceEvent(event_type="run.started", sequence=2, timestamp=data["started_at"])],
        )
    with pytest.raises(ValidationError, match="end time"):
        RunTrace(**data, status="completed")


def test_domain_json_roundtrip_omits_unapproved_metadata() -> None:
    event = TraceEvent(
        event_type="tool.completed",
        sequence=1,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"tool": "read_file", "content": "secret source", "prompt": "private"},
    )
    assert event.metadata == {"tool": "read_file"}
    assert TraceEvent.model_validate(event.model_dump(mode="json")) == event


def test_public_exports_keep_database_optional():
    from repomind.llm import LLMResponse, OpenAILLMClient, TokenUsage
    from repomind.observability import PostgresTraceStore, TraceRecorder

    assert LLMResponse and OpenAILLMClient and TokenUsage and PostgresTraceStore and TraceRecorder
