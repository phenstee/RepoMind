"""HTTP contracts, error privacy and dependency-independent startup."""

import inspect
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from repomind.api import create_app
from repomind.api.dependencies import get_services
from repomind.api.routes import router


def test_health_and_openapi_do_not_build_services():
    app = create_app()

    def forbidden():
        raise AssertionError("Health must not need any service")

    app.dependency_overrides[get_services] = forbidden
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json() == {"status": "ok", "service": "repomind"}
        schema = client.get("/openapi.json").json()
        assert len(schema["paths"]) == 22
        assert "/api/v1/jobs/{job_id}/cancel" in schema["paths"]
        assert schema["paths"]["/api/v1/repositories"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("RepositoryListResponse")
        assert schema["paths"]["/api/v1/repositories/{repository_id}/rag"]["post"]["responses"][
            "200"
        ]["content"]["application/json"]["schema"]["$ref"].endswith("RAGResponse")
        assert schema["paths"]["/api/v1/repositories"]["post"]["responses"]["422"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("ErrorResponse")
        assert schema["paths"]["/api/v1/repositories/{repository_id}/rag/stream"]["post"][
            "responses"
        ]["200"]["content"] == {"text/event-stream": {"schema": {"type": "string"}}}
        assert client.get("/docs").status_code == 200
    assert app.state.container.services is None
    assert app.state.container.engine is None


def test_all_domain_routes_are_sync():
    assert all(not inspect.iscoroutinefunction(route.endpoint) for route in router.routes)


@pytest.mark.parametrize(
    "path,body",
    [
        ("/repositories", {"name": "", "path": "sample"}),
        ("/repositories", {"name": "/host/private", "path": "sample"}),
        ("/repositories/1/rag", {"question": " "}),
        ("/repositories/1/rag", {"question": "a" * 10001}),
        ("/repositories/1/rag", {"question": "q", "top_k": 0}),
        ("/repositories/1/rag", {"question": "q", "top_k": 21}),
        ("/repositories/1/rag", {"question": "q", "top_k": True}),
        ("/repositories/1/rag", {"question": "q", "strategy": "anything"}),
        ("/repositories/1/agent/runs", {"query": "q", "max_iterations": 21}),
        ("/repositories/1/agent/runs", {"query": "q", "tools": ["create_file"]}),
        (
            "/repositories/1/coding/runs",
            {"objective": "q", "verification": {"require_tests": False}},
        ),
        (
            "/repositories/1/coding/runs",
            {"objective": "q", "verification": {"test_paths": ["../outside"]}},
        ),
        (
            "/repositories/1/coding/runs",
            {"objective": "q", "verification": {"ruff_paths": ["--fix"]}},
        ),
        ("/repositories/1/coding/runs", {"objective": "q", "acceptance_criteria": ["a"] * 51}),
        ("/repositories/1/coding/runs", {"objective": "q", "command": "anything"}),
    ],
)
def test_bounded_inputs_and_no_echo(api, path, body):
    response = api.client.post("/api/v1" + path, json=body)
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_request", "message": "Request validation failed."}
    }
    assert not api.llm.calls and not api.embeddings.calls


@pytest.mark.parametrize(
    "path",
    [
        "/runs?limit=101",
        "/runs?limit=0",
        "/runs?status=wrong",
        "/runs?run_type=wrong",
        "/runs/not-a-uuid",
        "/repositories/0",
        "/repositories/-1",
        "/repositories/1/files?limit=101",
        "/repositories/1/files?offset=-1",
    ],
)
def test_query_and_id_validation(api, path):
    assert api.client.get("/api/v1" + path).status_code == 422


def test_safe_unknown_ids_and_exception_errors(api, monkeypatch):
    assert api.client.get("/api/v1/repositories/999").status_code == 404
    assert api.client.get(f"/api/v1/runs/{UUID(int=99)}").status_code == 404

    def fail(*args):
        raise RuntimeError(r"C:\Users\private sk-api-secret postgresql://user:password@host/db")

    monkeypatch.setattr(api.store, "get", fail)
    response = api.client.get("/api/v1/repositories/1")
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "The operation could not be completed."}
    }
    malformed = api.client.post(
        "/api/v1/repositories",
        content='{ "secret": "sk-private",',
        headers={"content-type": "application/json"},
    )
    assert malformed.status_code == 422
    assert "sk-private" not in malformed.text


def test_services_can_be_replaced_at_dependency_boundary(api):
    from types import SimpleNamespace

    from repomind.api.models import RepositoryResponse

    binding = api.store.get(1)
    fake = SimpleNamespace(
        repositories=SimpleNamespace(
            get=lambda repository_id: RepositoryResponse(
                id=88, name="injected", created_at=binding.created_at
            )
        )
    )
    api.app.dependency_overrides[get_services] = lambda: fake
    assert api.client.get("/api/v1/repositories/1").json()["id"] == 88


def test_explicit_local_cors_and_cross_app_state(api):
    response = api.client.options(
        "/api/v1/repositories",
        headers={"origin": "http://localhost:3000", "access-control-request-method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    response = api.client.options(
        "/api/v1/repositories",
        headers={"origin": "https://untrusted.example", "access-control-request-method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers
    assert create_app().state.container is not api.app.state.container


def test_durable_job_submission_is_queued_without_running_a_model(api):
    response = api.client.post(
        "/api/v1/repositories/1/jobs/rag", json={"question": "Where is the entry point?"}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_type"] == "rag"
    detail = api.client.get(f"/api/v1/jobs/{body['job_id']}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "queued"
    assert detail.json()["result"] is None
    assert not api.llm.calls and not api.embeddings.calls


def test_queued_job_cancellation_is_idempotent_and_does_not_expose_request(api):
    queued = api.client.post(
        "/api/v1/repositories/1/jobs/rag",
        json={"question": "private repository question"},
    ).json()

    first = api.client.post(f"/api/v1/jobs/{queued['job_id']}/cancel")
    second = api.client.post(f"/api/v1/jobs/{queued['job_id']}/cancel")

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["status"] == "cancelled"
    assert first.json()["cancel_requested"] is True
    assert first.json()["cancellation_control"] == "cancelled"
    assert "private repository question" not in first.text
    detail = api.client.get(f"/api/v1/jobs/{queued['job_id']}").json()
    assert detail["status"] == "cancelled"
    assert detail["cancelled_at"] is not None
