"""Persistence output is revalidated and reduced at the HTTP boundary."""

from datetime import UTC, datetime
from uuid import UUID

from repomind.api.privacy import public_text
from repomind.observability import RunStatus, RunTrace, RunType, TraceEvent


def test_run_list_filters_and_sanitized_detail(api):
    stamp = datetime(2026, 9, 15, tzinfo=UTC)
    event = TraceEvent(event_type="tool.completed", sequence=1, timestamp=stamp)
    # Bypass Pydantic validation to simulate suspicious metadata already in storage.
    event.metadata.update(
        {
            "prompt": "FULL PROMPT",
            "stdout": "SECRET OUTPUT",
            "content": "SOURCE",
            "old_text": "REPLACEMENT",
            "embedding": [1, 2],
            "path": "C:/private/secret.py",
            "message": "sk-abcdef",
            "model": "postgresql://private:password@host/database",
            "operation": "C:/private/workspace",
        }
    )
    trace = RunTrace(
        run_id=UUID(int=1),
        run_type=RunType.RAG,
        status=RunStatus.COMPLETED,
        started_at=stamp,
        ended_at=stamp,
        duration_ms=0,
    ).model_copy(update={"events": (event,)})
    api.traces.runs[trace.run_id] = trace
    response = api.client.get(f"/api/v1/runs/{trace.run_id}")
    assert response.status_code == 200, response.text
    metadata = response.json()["events"][0]["metadata"]
    for key in ("prompt", "stdout", "content", "old_text", "embedding"):
        assert key not in metadata
    for private in ("private", "password", "abcdef", "FULL PROMPT", "REPLACEMENT", "SOURCE"):
        assert private not in response.text
    assert response.json()["estimated_cost_usd"] is None
    listing = api.client.get("/api/v1/runs?run_type=rag&status=completed&limit=1").json()["runs"]
    assert len(listing) == 1 and "events" not in listing[0]
    assert api.client.get("/api/v1/runs?run_type=coding_task").json()["runs"] == []
    assert api.client.get("/api/v1/runs?status=failed").json()["runs"] == []


def test_model_authored_answers_redact_host_paths_and_recognizable_secrets(api):
    api.llm.responses.append(
        {
            "action": "final",
            "final_answer": r"See C:\Users\private\app.py or /home/private/app.py. sk-abcdef src/app.py stays relative.",
        }
    )
    response = api.client.post("/api/v1/repositories/1/agent/runs", json={"query": "inspect"})
    assert response.status_code == 200
    assert "private" not in response.text and "abcdef" not in response.text
    assert "src/app.py" in response.json()["final_answer"]


def test_redaction_preserves_urls_relative_paths_and_long_answers():
    answer = "Read https://example.invalid/docs and src/app.py.\n" * 30
    assert public_text(answer) == answer
