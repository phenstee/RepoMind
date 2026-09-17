"""Offline semantic SSE contracts driven by the existing trace timeline."""

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from uuid import UUID

import pytest

from repomind.api.services.streaming import _ProgressChannel
from repomind.api.streaming import encode_sse_event, safe_progress_event
from repomind.ingestion import CodeChunk
from repomind.observability import TraceEvent
from repomind.retrieval import SemanticSearchResult


def _events(response):
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = []
    for block in response.text.split("\n\n"):
        if not block:
            continue
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append(
            {
                "id": int(fields["id"]) if "id" in fields else None,
                "event": fields["event"],
                "data": json.loads(fields["data"]),
            }
        )
    return events


def _candidates():
    return [
        SemanticSearchResult(
            chunk=CodeChunk(
                relative_path="src/app.py",
                language="python",
                start_line=1,
                end_line=2,
                content="def value():\n    return 1\n",
                chunk_index=0,
            ),
            score=1,
            rank=1,
        )
    ]


def _stream(api, path, body=None):
    return _events(api.client.post("/api/v1" + path, json=body))


def _git(root, *arguments):
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


@pytest.fixture
def coding_project(api):
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    (api.repo / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n.ruff_cache/\n*.pyc\n", encoding="utf-8"
    )
    (api.repo / "tests").mkdir()
    (api.repo / "tests/test_app.py").write_text(
        "from app import value\n\n\ndef test_value():\n    assert value() == 2\n", encoding="utf-8"
    )
    _git(api.repo, "init", "-b", "main")
    _git(api.repo, "config", "user.email", "stream-tests@example.invalid")
    _git(api.repo, "config", "user.name", "Stream Tests")
    _git(api.repo, "add", ".")
    _git(api.repo, "commit", "-m", "temporary stream fixture baseline")
    return api


def test_sse_encoding_is_framed_json_and_newline_safe():
    encoded = encode_sse_event(
        "tool.completed",
        {"note": "line one\nevent: injected", "nested": {"text": "x\ny"}},
        event_id=4,
    )
    assert encoded.startswith("id: 4\nevent: tool.completed\ndata: ")
    assert encoded.endswith("\n\n")
    assert "\nevent: injected" not in encoded
    payload = json.loads(encoded.split("data: ", 1)[1])
    assert payload["note"] == "line one\nevent: injected"
    with pytest.raises(ValueError):
        encode_sse_event("event\ninjected", {})


def test_cancelled_durable_job_stream_has_safe_terminal_frame(api):
    queued = api.client.post(
        "/api/v1/repositories/1/jobs/rag", json={"question": "private question"}
    ).json()
    api.client.post(f"/api/v1/jobs/{queued['job_id']}/cancel")

    events = _events(api.client.get(f"/api/v1/jobs/{queued['job_id']}/events"))

    assert len(events) == 1
    assert events[0]["event"] == "cancelled"
    assert events[0]["data"]["status"] == "cancelled"
    assert "result" not in events[0]["data"]
    assert "error" not in events[0]["data"]
    assert "private question" not in api.client.get(f"/api/v1/jobs/{queued['job_id']}/events").text


def test_progress_projection_is_allowlisted_and_relative_path_only():
    event = TraceEvent(
        event_type="tool.completed",
        sequence=4,
        timestamp=datetime(2026, 9, 16, tzinfo=UTC),
        metadata={
            "tool": "read_file",
            "path": r"C:\\Users\\private\\secret.py",
            "content": "SECRET_SOURCE_CONTENT",
            "stdout": "NEW_SECRET_SOURCE",
            "message": "sk-test-secret Bearer supersecret postgresql://user:password@localhost/db",
            "passed": True,
        },
    )
    progress = safe_progress_event(UUID(int=4), event)
    assert progress.data == {"tool": "read_file", "passed": True}
    body = encode_sse_event(progress.event, progress, event_id=progress.sequence)
    for private in (
        "SECRET_SOURCE_CONTENT",
        "NEW_SECRET_SOURCE",
        "sk-test-secret",
        "supersecret",
        "password",
        "Users",
    ):
        assert private not in body


def test_review_progress_exposes_counts_and_verdict_without_review_text():
    event = TraceEvent(
        event_type="review.completed",
        sequence=7,
        timestamp=datetime(2026, 9, 16, tzinfo=UTC),
        metadata={
            "workspace_revision": 2,
            "verdict": "changes_required",
            "criteria_satisfied": 1,
            "criteria_unsatisfied": 1,
            "finding_count": 1,
            "findings": ["SECRET_SOURCE_CONTENT sk-test-secret"],
            "repository_diff": "private diff",
        },
    )

    progress = safe_progress_event(UUID(int=7), event)

    assert progress.data == {
        "workspace_revision": 2,
        "verdict": "changes_required",
        "criteria_satisfied": 1,
        "criteria_unsatisfied": 1,
        "finding_count": 1,
    }
    assert "SECRET_SOURCE_CONTENT" not in progress.model_dump_json()


def test_index_stream_has_coarse_stages_terminal_result_and_retained_trace(api):
    events = _stream(api, "/repositories/1/index/stream")
    names = [event["event"] for event in events]
    assert names == [
        "run.started",
        "index.started",
        "ingestion.completed",
        "chunking.completed",
        "embedding.started",
        "embedding.completed",
        "persistence.completed",
        "run.completed",
        "result",
    ]
    progress = events[:-1]
    assert [event["id"] for event in progress] == list(range(1, len(progress) + 1))
    result = events[-1]
    assert result["id"] is None
    assert result["data"]["result"] == {
        "repository_id": 1,
        "files_indexed": 1,
        "chunks_indexed": 1,
        "embedding_model": "offline-model",
    }
    run_id = UUID(result["data"]["run_id"])
    trace = api.traces.runs[run_id]
    assert [event.event_type for event in trace.events] == names[:-1]
    assert [event.sequence for event in trace.events] == [event["id"] for event in progress]
    assert str(api.root) not in json.dumps(events)
    assert "def value" not in json.dumps(events)


@pytest.mark.parametrize(
    "strategy,trace_strategy,reranking",
    [
        ("semantic", "semantic", False),
        ("hybrid", "hybrid", False),
        ("hybrid_rerank", "hybrid+rerank", True),
    ],
)
def test_rag_stream_reuses_service_result_and_strategy_projection(
    api, strategy, trace_strategy, reranking
):
    api.store.candidates = _candidates()
    if reranking:
        api.llm.responses.append({"ranked_candidate_ids": ["C1"]})
    api.llm.responses.append(
        {"answer": "value returns one", "source_ids": ["S1"], "insufficient_evidence": False}
    )
    events = _stream(
        api,
        "/repositories/1/rag/stream",
        {"question": "What does value do?", "strategy": strategy, "top_k": 3},
    )
    names = [event["event"] for event in events]
    assert names[0] == "run.started" and names[-2:] == ["run.completed", "result"]
    retrieval = next(event for event in events if event["event"] == "retrieval.completed")
    assert {
        key: value for key, value in retrieval["data"]["data"].items() if key != "duration_ms"
    } == {
        "strategy": trace_strategy,
        "reranking_enabled": reranking,
        "candidate_count": 1,
    }
    result = events[-1]["data"]["result"]
    assert result["answer"] == "value returns one"
    assert result["citations"] == [{"relative_path": "src/app.py", "start_line": 1, "end_line": 2}]
    assert "score" not in json.dumps(events)
    assert "def value" not in json.dumps(events)


def test_rag_stream_result_matches_non_streaming_contract(api):
    api.store.candidates = _candidates()
    api.llm.responses.append(
        {"answer": "value returns one", "source_ids": ["S1"], "insufficient_evidence": False}
    )
    normal = api.client.post(
        "/api/v1/repositories/1/rag",
        json={"question": "What does value do?", "strategy": "semantic", "trace": True},
    ).json()
    api.llm.responses.append(
        {"answer": "value returns one", "source_ids": ["S1"], "insufficient_evidence": False}
    )
    streamed = _stream(
        api,
        "/repositories/1/rag/stream",
        {"question": "What does value do?", "strategy": "semantic"},
    )[-1]["data"]["result"]
    assert {key: value for key, value in normal.items() if key != "trace_run_id"} == {
        key: value for key, value in streamed.items() if key != "trace_run_id"
    }
    assert streamed["trace_run_id"] == streamed["trace_run_id"]


def test_read_only_agent_streams_actions_without_editing_capability(api):
    api.llm.responses.extend(
        [
            {"action": "tool", "tool_name": "search_code", "tool_arguments": {"query": "value"}},
            {"action": "tool", "tool_name": "read_file", "tool_arguments": {"path": "app.py"}},
            {"action": "final", "final_answer": "Inspected app.py."},
        ]
    )
    events = _stream(api, "/repositories/1/agent/runs/stream", {"query": "inspect"})
    tools = [event["data"]["data"]["tool"] for event in events if event["event"] == "tool.started"]
    assert tools == ["search_code", "read_file"]
    assert events[-1]["data"]["result"]["status"] == "completed"
    body = json.dumps(events)
    for private in ("return 1", "create_file", "replace_text", "run_tests", "run_ruff"):
        assert private not in body


def test_coding_recovery_streams_failure_then_correction_and_completion(coding_project):
    api = coding_project
    before = (api.repo / "app.py").read_bytes()
    incorrect = before.replace(b"return 1", b"return 3")
    api.llm.responses.extend(
        [
            {
                "action": "tool",
                "tool_name": "replace_text",
                "tool_arguments": {
                    "path": "app.py",
                    "old_text": "return 1",
                    "new_text": "return 3",
                    "expected_sha256": hashlib.sha256(before).hexdigest(),
                },
            },
            {"action": "final", "final_answer": "This should pass."},
            {
                "action": "tool",
                "tool_name": "replace_text",
                "tool_arguments": {
                    "path": "app.py",
                    "old_text": "return 3",
                    "new_text": "return 2",
                    "expected_sha256": hashlib.sha256(incorrect).hexdigest(),
                },
            },
            {"action": "final", "final_answer": "Corrected."},
        ]
    )
    events = _stream(api, "/repositories/1/coding/runs/stream", {"objective": "Return two"})
    names = [event["event"] for event in events]
    assert names.index("planning.started") < names.index("planning.completed")
    assert names.count("file.mutated") == 2
    assert names.count("completion.requested") == 2
    assert names.count("verification.completed") >= 4
    assert "completion.blocked" in names
    assert names.index("review.started") < names.index("review.completed")
    assert names[-2:] == ["run.completed", "result"]
    blocked = next(event for event in events if event["event"] == "completion.blocked")
    assert "tests_failed" in blocked["data"]["data"]["blocker_codes"]
    result = events[-1]["data"]["result"]
    assert result["status"] == "completed"
    assert result["tests"]["passed"] and result["ruff"]["passed"]
    assert "new_text" not in json.dumps(events) and "return 3" not in json.dumps(events)


def test_runtime_failure_after_start_is_one_safe_terminal_error(api):
    api.llm.responses.append(
        RuntimeError(
            r"C:\\Users\\private\\secret sk-test-secret postgresql://user:password@host/db"
        )
    )
    events = _stream(api, "/repositories/1/agent/runs/stream", {"query": "inspect"})
    assert events[-1]["event"] == "error"
    assert sum(event["event"] == "error" for event in events) == 1
    assert "result" not in [event["event"] for event in events]
    assert events[-1]["data"]["error"] == {
        "code": "operation_failed",
        "message": "The operation failed.",
    }
    for private in ("private", "secret", "password", "sk-test-secret"):
        assert private not in json.dumps(events)


@pytest.mark.parametrize(
    "path,body,status",
    [
        ("/repositories/999/rag/stream", {"question": "q"}, 404),
        ("/repositories/1/rag/stream", {"question": " "}, 422),
        ("/repositories/1/agent/runs/stream", {"query": "q", "tools": ["create_file"]}, 422),
    ],
)
def test_pre_stream_failures_retain_normal_http_errors(api, path, body, status):
    response = api.client.post("/api/v1" + path, json=body)
    assert response.status_code == status
    assert not response.headers.get("content-type", "").startswith("text/event-stream")


def test_per_request_channels_cannot_mix_runs():
    stamp = datetime(2026, 9, 16, tzinfo=UTC)
    first, second = UUID(int=1), UUID(int=2)
    channel_a, channel_b = _ProgressChannel(first), _ProgressChannel(second)
    event = TraceEvent(event_type="run.started", sequence=1, timestamp=stamp)
    channel_a.receive(second, event)
    channel_b.receive(first, event)
    assert channel_a.next_event() is None and channel_b.next_event() is None
    channel_a.receive(first, event)
    channel_b.receive(second, event)
    assert channel_a.next_event().run_id == first
    assert channel_b.next_event().run_id == second
