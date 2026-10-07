"""HTTP contracts, error privacy and dependency-independent startup."""

import inspect
import logging
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from repomind.api import create_app
from repomind.api.dependencies import get_services
from repomind.api.routes import router
from repomind.api.security import request_hostname
from repomind.config import Settings
from repomind.jobs.store import JobNotFoundError


def test_health_and_openapi_do_not_build_services():
    app = create_app()

    def forbidden():
        raise AssertionError("Health must not need any service")

    app.dependency_overrides[get_services] = forbidden
    # Default settings trust loopback Host names only, not TestClient's "testserver".
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
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


TRUSTED_ORIGIN = {"origin": "http://localhost:3000"}


@pytest.mark.parametrize(
    "path", ["/repositories/1/jobs/index", "/repositories/1/index", "/repositories"]
)
def test_cross_site_simple_post_without_client_header_is_rejected(api, path):
    # A hostile page can send a text/plain "simple request" without any preflight.
    browser = TestClient(api.app, raise_server_exceptions=False)
    response = browser.post(
        "/api/v1" + path,
        content=b"",
        headers={"origin": "https://evil.example", "content-type": "text/plain"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "missing_client_header",
            "message": "State-changing requests require the X-RepoMind-Client header.",
        }
    }
    assert "access-control-allow-origin" not in response.headers
    assert not api.app.state.container.get().jobs.store.jobs
    assert not api.embeddings.calls


def test_cross_site_cancel_without_client_header_leaves_job_queued(api):
    job_id = api.client.post("/api/v1/repositories/1/jobs/index").json()["job_id"]

    browser = TestClient(api.app, raise_server_exceptions=False)
    rejected = browser.post(
        f"/api/v1/jobs/{job_id}/cancel", headers={"origin": "https://evil.example"}
    )
    blank = browser.post(f"/api/v1/jobs/{job_id}/cancel", headers={"X-RepoMind-Client": " "})

    assert rejected.status_code == blank.status_code == 403
    assert api.client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "queued"


def test_client_header_rejection_is_readable_by_trusted_frontend_and_reads_need_no_header(api):
    browser = TestClient(api.app, raise_server_exceptions=False)

    rejected = browser.post("/api/v1/repositories/1/jobs/index", headers=TRUSTED_ORIGIN)
    read = browser.get("/api/v1/repositories/1", headers=TRUSTED_ORIGIN)

    assert rejected.status_code == 403
    assert rejected.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert read.status_code == 200


def test_cors_preflight_allows_the_client_header_for_trusted_origins(api):
    response = api.client.options(
        "/api/v1/repositories/1/jobs/index",
        headers={
            **TRUSTED_ORIGIN,
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type,x-repomind-client",
        },
    )

    assert response.status_code == 200
    assert "x-repomind-client" in response.headers["access-control-allow-headers"].lower()


@pytest.mark.parametrize(
    "host", ["localhost:8000", "127.0.0.1:8000", "127.0.0.1", "api:8000", "[::1]:8000"]
)
def test_default_settings_accept_loopback_and_compose_host_names(host):
    with TestClient(create_app(settings=Settings(_env_file=None))) as client:
        assert client.get("/api/v1/health", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["rebind.attacker.example", "testserver", "127.0.0.1.nip.io", ""])
def test_dns_rebinding_host_names_are_rejected(host):
    with TestClient(create_app(settings=Settings(_env_file=None))) as client:
        response = client.get("/api/v1/health", headers={"host": host})
    assert response.status_code == 400
    assert response.json() == {
        "error": {"code": "invalid_host", "message": "Request host is not allowed."}
    }


@pytest.mark.parametrize(
    "header,expected",
    [
        ("localhost:8000", "localhost"),
        ("LOCALHOST", "localhost"),
        ("[::1]:8000", "[::1]"),
        ("[::1]", "[::1]"),
        ("[::1", ""),
    ],
)
def test_request_hostname_handles_ports_case_and_ipv6(header, expected):
    assert request_hostname(header) == expected


def test_unknown_repository_404_carries_cors_headers_without_logging_a_traceback(api, caplog):
    with caplog.at_level(logging.ERROR):
        response = api.client.get("/api/v1/repositories/999", headers=TRUSTED_ORIGIN)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "repository_not_found"
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert not caplog.records


def test_storage_and_job_lookup_failures_carry_cors_headers(api, monkeypatch):
    def unavailable(*args):
        raise OperationalError("SELECT 1", {}, Exception("postgresql://user:password@db"))

    monkeypatch.setattr(api.store, "get", unavailable)
    storage = api.client.get("/api/v1/repositories/1", headers=TRUSTED_ORIGIN)
    assert storage.status_code == 503
    assert storage.json()["error"]["code"] == "storage_unavailable"
    assert "password" not in storage.text
    assert storage.headers["access-control-allow-origin"] == "http://localhost:3000"

    def missing(*args):
        raise JobNotFoundError("Job lease is no longer active")

    monkeypatch.setattr(api.app.state.container.get().jobs, "get", missing)
    job = api.client.get(f"/api/v1/jobs/{UUID(int=7)}", headers=TRUSTED_ORIGIN)
    assert job.status_code == 404
    assert job.json()["error"] == {"code": "job_not_found", "message": "Job not found."}
    assert job.headers["access-control-allow-origin"] == "http://localhost:3000"
