"""Real strategy orchestration and distinct read-only/editing HTTP capabilities."""

import hashlib
import shutil
import subprocess

import pytest

from repomind.ingestion import CodeChunk
from repomind.retrieval import SemanticSearchResult


def candidates():
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


@pytest.mark.parametrize(
    "strategy,hybrid,depth,model_calls",
    [
        ("semantic", False, 3, 1),
        ("hybrid", True, 3, 1),
        ("hybrid_rerank", True, 20, 2),
    ],
)
def test_rag_real_strategy_routing_and_repository_owned_citations(
    api, strategy, hybrid, depth, model_calls
):
    api.store.candidates = candidates()
    if strategy == "hybrid_rerank":
        api.llm.responses.append({"ranked_candidate_ids": ["C1"]})
    api.llm.responses.append(
        {"answer": "value returns one", "source_ids": ["S1"], "insufficient_evidence": False}
    )
    response = api.client.post(
        "/api/v1/repositories/1/rag",
        json={"question": "What does value do?", "strategy": strategy, "top_k": 3, "trace": True},
    )
    assert response.status_code == 200, response.text
    assert response.json()["citations"] == [
        {"relative_path": "src/app.py", "start_line": 1, "end_line": 2}
    ]
    assert api.store.search_calls == [(1, "What does value do?", hybrid, depth)]
    assert api.embeddings.calls == ["What does value do?"]
    assert len(api.llm.calls) == model_calls
    run_id = response.json()["trace_run_id"]
    trace = api.client.get(f"/api/v1/runs/{run_id}").json()
    assert trace["run_type"] == "rag" and trace["status"] == "completed"
    assert trace["llm_calls"] == model_calls
    assert "def value" not in str(trace)


@pytest.mark.parametrize("strategy", ["semantic", "hybrid", "hybrid_rerank"])
def test_no_evidence_skips_answer_and_rerank_calls(api, strategy):
    response = api.client.post(
        "/api/v1/repositories/1/rag", json={"question": "q", "strategy": strategy}
    )
    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True
    assert response.json()["citations"] == []
    assert response.json()["trace_run_id"] is None
    assert not api.llm.calls
    assert not api.traces.runs


def test_trace_sink_failure_does_not_fail_answer(api):
    api.traces.fail = True
    response = api.client.post("/api/v1/repositories/1/rag", json={"question": "q", "trace": True})
    assert response.status_code == 200
    assert response.json()["trace_run_id"]
    assert "password" not in response.text


def test_read_only_agent_cannot_execute_mutation_or_verifiers(api):
    api.llm.responses.extend(
        [
            {"action": "tool", "tool_name": tool, "tool_arguments": args}
            for tool, args in [
                ("create_file", {"path": "new.py", "content": "bad"}),
                ("replace_text", {"path": "app.py"}),
                ("run_tests", {}),
                ("run_ruff", {}),
            ]
        ]
    )
    api.llm.responses.append({"action": "final", "final_answer": "Inspection done."})
    before = (api.repo / "app.py").read_bytes()
    response = api.client.post(
        "/api/v1/repositories/1/agent/runs", json={"query": "inspect", "trace": True}
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    assert response.json()["tool_execution_attempts"] == 4
    assert (api.repo / "app.py").read_bytes() == before
    assert not (api.repo / "new.py").exists()
    assert not (api.repo / ".pytest_cache").exists()
    assert "steps" not in response.json()
    assert "tool_arguments" not in response.text


def test_read_file_observation_not_exposed_in_response(api):
    api.llm.responses.extend(
        [
            {"action": "tool", "tool_name": "read_file", "tool_arguments": {"path": "app.py"}},
            {"action": "final", "final_answer": "Inspected app.py."},
        ]
    )
    response = api.client.post("/api/v1/repositories/1/agent/runs", json={"query": "inspect"})
    assert response.status_code == 200
    assert response.json()["tool_execution_attempts"] == 1
    assert "return 1" not in response.text


def test_model_failure_has_safe_correlated_failed_trace(api):
    api.llm.responses.append(
        RuntimeError(r"C:\private sk-password postgresql://private:secret@host/db")
    )
    response = api.client.post(
        "/api/v1/repositories/1/agent/runs", json={"query": "inspect", "trace": True}
    )
    assert response.status_code == 500
    error = response.json()["error"]
    assert set(error) == {"code", "message", "trace_run_id"}
    assert "private" not in response.text
    trace = api.client.get(f"/api/v1/runs/{error['trace_run_id']}")
    assert trace.json()["status"] == "failed"
    assert "sk-password" not in trace.text


def test_incomplete_agent_is_not_recorded_as_completed(api):
    api.llm.responses.append(
        {"action": "tool", "tool_name": "read_file", "tool_arguments": {"path": "app.py"}}
    )
    response = api.client.post(
        "/api/v1/repositories/1/agent/runs",
        json={"query": "inspect", "max_iterations": 1, "trace": True},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "max_iterations_reached"
    trace = api.client.get(f"/api/v1/runs/{response.json()['trace_run_id']}").json()
    assert trace["status"] == "failed" and trace["domain_status"] == "max_iterations_reached"


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


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
    git(api.repo, "init", "-b", "main")
    git(api.repo, "config", "user.email", "api-tests@example.invalid")
    git(api.repo, "config", "user.name", "API Tests")
    git(api.repo, "add", ".")
    git(api.repo, "commit", "-m", "temporary fixture baseline")
    return api


def test_coding_real_edit_and_verification_gates(coding_project):
    api = coding_project
    before = (api.repo / "app.py").read_bytes()
    api.llm.responses.extend(
        [
            {
                "action": "tool",
                "tool_name": "replace_text",
                "tool_arguments": {
                    "path": "app.py",
                    "old_text": "return 1",
                    "new_text": "return 2",
                    "expected_sha256": hashlib.sha256(before).hexdigest(),
                },
            },
            {"action": "final", "final_answer": "Updated value()."},
        ]
    )
    response = api.client.post(
        "/api/v1/repositories/1/coding/runs",
        json={
            "objective": "Return two",
            "acceptance_criteria": ["The value test passes"],
            "trace": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["changed_files"] == ["app.py"]
    assert body["workspace_revision"] == 1 and body["completion_attempts"] == 1
    assert body["tests"]["passed"] and body["ruff"]["passed"]
    assert body["tests"]["workspace_revision"] == body["ruff"]["workspace_revision"] == 1
    assert (api.repo / "app.py").read_bytes() == before.replace(b"return 1", b"return 2")
    for private in ("stdout", "stderr", "old_text", "new_text", "diff", "steps", "baseline"):
        assert private not in body
    trace = api.client.get(f"/api/v1/runs/{body['trace_run_id']}").json()
    assert trace["status"] == "completed" and trace["successful_mutations"] == 1
    assert any(e["event_type"] == "final_review.completed" for e in trace["events"])


def test_coding_dirty_preflight_prevents_llm_call(coding_project):
    api = coding_project
    (api.repo / "user.txt").write_text("preserve my work", encoding="utf-8")
    response = api.client.post(
        "/api/v1/repositories/1/coding/runs", json={"objective": "change", "trace": True}
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "coding_precondition_failed"
    assert not api.llm.calls
    assert (api.repo / "user.txt").read_text(encoding="utf-8") == "preserve my work"


def test_coding_final_claim_cannot_bypass_failing_tests(coding_project):
    api = coding_project
    api.llm.responses.append({"action": "final", "final_answer": "All done and passing."})
    response = api.client.post(
        "/api/v1/repositories/1/coding/runs", json={"objective": "Return two", "max_iterations": 1}
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] != "completed"
    assert response.json()["tests"]["passed"] is False
