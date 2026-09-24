"""Real HTTP/service/persistence composition, with rollback-only DB and fake models."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from repomind.api import create_app
from repomind.api.store import PostgresRepositoryStore
from repomind.config import Settings
from repomind.db import load_chunks, read_index_manifest
from repomind.db.models import RepositoryFileRecord, RepositoryRecord
from repomind.observability import PostgresTraceStore
from repomind.retrieval import EmbeddedChunk, EmbeddingTextStrategy, EmbeddingVector

pytestmark = pytest.mark.postgres


class FakeEmbeddings:
    embedding_model = "offline-api-model"
    embedding_text_strategy = EmbeddingTextStrategy.RAW_SOURCE

    def __init__(self):
        self.embedded_paths = []

    def embed_text(self, text):
        return EmbeddingVector(values=(1.0, 0.0), model=self.embedding_model)

    def embed_chunks(self, chunks):
        self.embedded_paths.extend(c.relative_path.as_posix() for c in chunks)
        return [EmbeddedChunk(chunk=c, embedding=self.embed_text(c.content)) for c in chunks]


class FakeLLM:
    def generate_structured(self, prompt, response_model, **kwargs):
        if "ranked_candidate_ids" in response_model.model_fields:
            return response_model(ranked_candidate_ids=["C1"])
        return response_model(
            answer="value returns one", source_ids=["S1"], insufficient_evidence=False
        )


@pytest.fixture
def client(db_session, tmp_path):
    repo = tmp_path / "sample"
    repo.mkdir()
    (repo / "app.py").write_bytes(b"def value():\r\n    return 1\r\n")
    factory = sessionmaker(
        bind=db_session.connection(),
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    app = create_app(
        settings=Settings(_env_file=None, repomind_workspace_root=tmp_path),
        repository_store=PostgresRepositoryStore(factory),
        trace_store=PostgresTraceStore(factory),
        llm_factory=lambda trace: FakeLLM(),
        embedding_factory=lambda trace: FakeEmbeddings(),
    )
    with TestClient(app, raise_server_exceptions=False) as http:
        yield http


def test_api_registration_index_retrieval_and_trace_history(client, db_session):
    name = "api-" + uuid4().hex
    registered = client.post("/api/v1/repositories", json={"name": name, "path": "sample"})
    assert registered.status_code == 201, registered.text
    repository_id = registered.json()["id"]
    again = client.post("/api/v1/repositories", json={"name": name, "path": "sample"})
    assert again.json()["id"] == repository_id
    record = db_session.scalar(select(RepositoryRecord).where(RepositoryRecord.id == repository_id))
    assert record.workspace_relative_path == "sample"
    indexed = client.post(f"/api/v1/repositories/{repository_id}/index")
    assert indexed.status_code == 200, indexed.text
    assert indexed.json()["chunks_indexed"] == 1
    assert load_chunks(db_session, repository_id)[0].content == "def value():\r\n    return 1\r\n"
    files = client.get(f"/api/v1/repositories/{repository_id}/files").json()["files"]
    assert [f["relative_path"] for f in files] == ["app.py"]
    for strategy in ("semantic", "hybrid", "hybrid_rerank"):
        response = client.post(
            f"/api/v1/repositories/{repository_id}/rag",
            json={"question": "What does value do?", "strategy": strategy, "trace": True},
        )
        assert response.status_code == 200, response.text
        assert response.json()["citations"][0]["relative_path"] == "app.py"
        run_id = response.json()["trace_run_id"]
        detail = client.get(f"/api/v1/runs/{run_id}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["status"] == "completed"
        assert "return 1" not in detail.text
    listing = client.get("/api/v1/runs?run_type=rag&status=completed&limit=3")
    assert listing.status_code == 200
    assert len(listing.json()["runs"]) == 3


def test_api_incremental_reindex_reuses_rows_and_retrieval_sees_current_corpus(
    client, db_session
):
    name = "api-incremental-" + uuid4().hex
    workspace = client.app.state.container.settings.repomind_workspace_root / "sample"
    repository_id = client.post(
        "/api/v1/repositories", json={"name": name, "path": "sample"}
    ).json()["id"]
    (workspace / "keep.py").write_bytes(b"def keep():\r\n    return 9\r\n")
    (workspace / "drop.py").write_bytes(b"def drop():\r\n    return 8\r\n")
    assert client.post(f"/api/v1/repositories/{repository_id}/index").status_code == 200

    kept = db_session.scalar(
        select(RepositoryFileRecord).where(
            RepositoryFileRecord.repository_id == repository_id,
            RepositoryFileRecord.relative_path == "keep.py",
        )
    )
    kept_ids = (kept.id, sorted(chunk.id for chunk in kept.chunks))

    (workspace / "app.py").write_bytes(b"def value():\r\n    return 42\r\n")
    (workspace / "drop.py").unlink()
    second = client.post(f"/api/v1/repositories/{repository_id}/index")

    assert second.status_code == 200, second.text
    # The response still reports totals for the resulting index.
    assert second.json()["files_indexed"] == 2
    assert second.json()["chunks_indexed"] == 2

    db_session.expire_all()
    reused = db_session.scalar(
        select(RepositoryFileRecord).where(
            RepositoryFileRecord.repository_id == repository_id,
            RepositoryFileRecord.relative_path == "keep.py",
        )
    )
    # Unchanged file kept stable row identity, including its chunk/vector rows.
    assert (reused.id, sorted(chunk.id for chunk in reused.chunks)) == kept_ids
    files = client.get(f"/api/v1/repositories/{repository_id}/files").json()["files"]
    assert sorted(f["relative_path"] for f in files) == ["app.py", "keep.py"]
    contents = {chunk.content for chunk in load_chunks(db_session, repository_id)}
    assert "def value():\r\n    return 42\r\n" in contents
    assert not any("return 8" in content for content in contents)


def test_api_index_transaction_rolls_back_on_persistence_failure(client, db_session, monkeypatch):
    name = "api-rollback-" + uuid4().hex
    repository_id = client.post(
        "/api/v1/repositories", json={"name": name, "path": "sample"}
    ).json()["id"]
    assert client.post(f"/api/v1/repositories/{repository_id}/index").status_code == 200
    before = load_chunks(db_session, repository_id)
    before_fingerprint = read_index_manifest(db_session, repository_id).fingerprint

    def fail(*args):
        raise RuntimeError("private database failure")

    # Give the run real work, then fail AFTER file rows and the fingerprint
    # have been rewritten, proving the whole delta rolls back together.
    (client.app.state.container.settings.repomind_workspace_root / "sample" / "added.py").write_bytes(
        b"def added():\r\n    return 2\r\n"
    )
    monkeypatch.setattr("repomind.db.repositories._append_embedded_chunks", fail)
    response = client.post(f"/api/v1/repositories/{repository_id}/index")
    assert response.status_code == 500
    assert "private" not in response.text
    assert load_chunks(db_session, repository_id) == before
    assert read_index_manifest(db_session, repository_id).fingerprint == before_fingerprint


def _sse_events(response):
    import json

    return [
        {
            key: json.loads(value) if key == "data" else value
            for key, value in (line.split(": ", 1) for line in block.splitlines())
        }
        for block in response.text.split("\n\n")
        if block
    ]


def test_streamed_api_retains_postgres_trace_history(client, db_session):
    name = "api-stream-" + uuid4().hex
    repository_id = client.post(
        "/api/v1/repositories", json={"name": name, "path": "sample"}
    ).json()["id"]
    indexed = client.post(f"/api/v1/repositories/{repository_id}/index/stream")
    assert indexed.headers["content-type"].startswith("text/event-stream")
    assert _sse_events(indexed)[-1]["event"] == "result"
    response = client.post(
        f"/api/v1/repositories/{repository_id}/rag/stream",
        json={"question": "What does value do?", "strategy": "hybrid", "top_k": 1},
    )
    events = _sse_events(response)
    assert events[-1]["event"] == "result"
    run_id = events[-1]["data"]["run_id"]
    trace = client.get(f"/api/v1/runs/{run_id}")
    assert trace.status_code == 200, trace.text
    assert trace.json()["run_type"] == "rag"
    assert [event["event_type"] for event in trace.json()["events"]] == [
        event["event"] for event in events[:-1]
    ]
