"""API confinement and indexing reuse, with no live providers or test-name collisions."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repomind.api import create_app
from repomind.api.errors import APIError
from repomind.api.services.repositories import WorkspacePolicy
from repomind.config import Settings
from repomind.jobs import JobCancellationRequested


@pytest.mark.parametrize(
    "path",
    [
        "..",
        "../outside",
        "sample/../../outside",
        "sample/../sample",
        "/etc",
        "C:\\Users",
        "C:relative",
        "\\\\host\\share",
        "sample\\..\\outside",
        ".",
        "sample:stream",
        "sample\x00",
        "sample.",
        "sample ",
    ],
)
def test_reject_unsafe_paths(api, path):
    response = api.client.post("/api/v1/repositories", json={"name": "unsafe", "path": path})
    assert response.status_code == 400, response.text
    assert str(api.root) not in response.text
    assert len(api.store.bindings) == 1


def test_register_lookup_idempotence_and_conflict(api):
    response = api.client.post("/api/v1/repositories", json={"name": "sample", "path": "sample"})
    assert response.json()["id"] == 1
    assert set(response.json()) == {"id", "name", "created_at"}
    assert api.client.get("/api/v1/repositories/1").json() == response.json()
    (api.root / "other").mkdir()
    assert (
        api.client.post(
            "/api/v1/repositories", json={"name": "sample", "path": "other"}
        ).status_code
        == 409
    )
    assert (
        api.client.post(
            "/api/v1/repositories", json={"name": "missing", "path": "missing"}
        ).status_code
        == 404
    )


def test_repository_list_is_compact_and_uses_relative_binding(api):
    (api.root / "other").mkdir()
    second = api.client.post(
        "/api/v1/repositories", json={"name": "other", "path": "other"}
    )
    assert second.status_code == 201
    response = api.client.get("/api/v1/repositories?limit=1")
    assert response.status_code == 200
    assert response.json() == {
        "repositories": [
            {
                "id": second.json()["id"],
                "name": "other",
                "workspace_relative_path": "other",
                "created_at": "2026-09-15T00:00:00Z",
            }
        ]
    }


def test_missing_workspace_is_safe_and_health_still_works(api):
    with TestClient(
        create_app(
            settings=Settings(_env_file=None, repomind_workspace_root=None),
            repository_store=api.store,
            trace_store=api.traces,
        ),
        raise_server_exceptions=False,
    ) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert (
            client.post("/api/v1/repositories", json={"name": "x", "path": "x"}).status_code == 503
        )


def test_legacy_unbound_repository_cannot_be_silently_rebound(api):
    api.store.bindings[1] = replace(api.store.bindings[1], workspace_relative_path=None)
    assert (
        api.client.post(
            "/api/v1/repositories", json={"name": "sample", "path": "sample"}
        ).status_code
        == 409
    )
    assert api.client.post("/api/v1/repositories/1/index").status_code == 409
    assert not api.embeddings.calls


@pytest.mark.parametrize("outside", [True, False])
def test_symlinks_rejected_even_when_target_is_inside(api, tmp_path, outside):
    target = tmp_path / "outside" if outside else api.repo
    target.mkdir(exist_ok=True)
    link = api.root / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks require OS privileges")
    assert (
        api.client.post("/api/v1/repositories", json={"name": "link", "path": "link"}).status_code
        == 400
    )
    assert (
        api.client.post(
            "/api/v1/repositories", json={"name": "link", "path": "link/child"}
        ).status_code
        == 400
    )


def test_junction_or_replaced_binding_rechecked(api, monkeypatch):
    # Portable equivalent of Windows's junction predicate; actual junction covered separately.
    original = Path.is_junction
    monkeypatch.setattr(Path, "is_junction", lambda path: path == api.repo or original(path))
    assert api.client.post("/api/v1/repositories/1/index").status_code == 400
    assert (
        api.client.post("/api/v1/repositories/1/agent/runs", json={"query": "q"}).status_code == 400
    )
    assert (
        api.client.post("/api/v1/repositories/1/coding/runs", json={"objective": "q"}).status_code
        == 400
    )
    assert not api.llm.calls and not api.embeddings.calls


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction test")
def test_actual_junction_escape_is_rejected(api, tmp_path):
    import _winapi

    target = tmp_path / "outside"
    target.mkdir()
    link = api.root / "junction"
    _winapi.CreateJunction(str(target), str(link))
    try:
        assert link.is_junction()
        response = api.client.post(
            "/api/v1/repositories", json={"name": "escape", "path": "junction"}
        )
        assert response.status_code == 400, response.text
    finally:
        # Remove only this test-created junction, never the target directory.
        link.rmdir()
    assert target.is_dir()


def test_index_composes_real_ingestion_chunks_and_fake_embeddings(api):
    response = api.client.post("/api/v1/repositories/1/index")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "repository_id": 1,
        "files_indexed": 1,
        "chunks_indexed": 1,
        "embedding_model": "offline-model",
    }
    assert api.store.snapshots[1].name == "sample"
    assert api.store.chunks[1][0].chunk.content == "def value():\r\n    return 1\r\n"
    files = api.client.get("/api/v1/repositories/1/files?limit=1&offset=0").json()
    assert files["files"][0]["relative_path"] == "app.py"
    assert "content" not in files["files"][0]
    assert api.client.get("/api/v1/repositories/1/files?offset=1").json()["files"] == []


def test_failed_embedding_preserves_index_and_size_limits_prevent_calls(api, monkeypatch):
    assert api.client.post("/api/v1/repositories/1/index").status_code == 200
    old = api.store.snapshots[1]

    def fail(chunks):
        raise RuntimeError("sk-secret")

    monkeypatch.setattr(api.embeddings, "embed_chunks", fail)
    assert api.client.post("/api/v1/repositories/1/index").status_code == 500
    assert api.store.snapshots[1] is old
    service = api.app.state.container.get().repositories
    monkeypatch.setattr(service, "MAX_SOURCE_BYTES", 1)
    assert api.client.post("/api/v1/repositories/1/index").status_code == 413


def test_cancellation_after_embedding_preserves_previous_atomic_index(api):
    assert api.client.post("/api/v1/repositories/1/index").status_code == 200
    old_snapshot = api.store.snapshots[1]
    old_chunks = api.store.chunks[1]

    class CancelBeforePersistence:
        def __init__(self) -> None:
            self.checkpoints = 0

        def checkpoint(self) -> None:
            self.checkpoints += 1
            if self.checkpoints == 5:
                raise JobCancellationRequested("cancel before persistence")

        def side_effect_started(self) -> None:
            raise AssertionError("indexing has no coding side effect boundary")

    service = api.app.state.container.get().repositories
    with pytest.raises(JobCancellationRequested):
        service.index(1, api.embeddings, cancellation=CancelBeforePersistence())

    assert api.store.snapshots[1] is old_snapshot
    assert api.store.chunks[1] is old_chunks


def test_nested_workspace_operations_conflict_and_lock_releases(api):
    policy = api.app.state.container.get().repositories.workspace
    with policy.operation(api.repo):
        assert api.client.post("/api/v1/repositories/1/index").status_code == 409
        with pytest.raises(APIError), policy.operation(api.repo / "nested"):
            pytest.fail("must not overlap")
    assert api.client.post("/api/v1/repositories/1/index").status_code == 200


def test_workspace_root_itself_cannot_be_relative_or_drive_root():
    for root in (Path("."), Path(Path.cwd().anchor)):
        with pytest.raises(APIError) as caught:
            WorkspacePolicy(root).resolve("sample")
        assert caught.value.status == 503


def test_empty_repository_index_avoids_embeddings(api):
    (api.root / "empty").mkdir()
    repository_id = api.client.post(
        "/api/v1/repositories", json={"name": "empty", "path": "empty"}
    ).json()["id"]
    response = api.client.post(f"/api/v1/repositories/{repository_id}/index")
    assert response.status_code == 200
    assert response.json()["files_indexed"] == response.json()["chunks_indexed"] == 0
    assert response.json()["embedding_model"] is None
    assert not api.embeddings.calls


@pytest.mark.parametrize("limit", ["MAX_FILES", "MAX_CHUNKS"])
def test_index_limits_checked_before_embedding(api, monkeypatch, limit):
    monkeypatch.setattr(api.app.state.container.get().repositories, limit, 0)
    assert api.client.post("/api/v1/repositories/1/index").status_code == 413
    assert not api.embeddings.calls and not api.store.snapshots


def test_stored_traversal_is_rejected_at_use(api):
    api.store.bindings[1] = replace(api.store.bindings[1], workspace_relative_path="../outside")
    response = api.client.post("/api/v1/repositories/1/agent/runs", json={"query": "inspect"})
    assert response.status_code == 400
    assert not api.llm.calls


def test_workspace_setting_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOMIND_WORKSPACE_ROOT", str(tmp_path))
    assert Settings(_env_file=None).repomind_workspace_root == tmp_path
