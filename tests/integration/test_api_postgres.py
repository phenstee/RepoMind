"""Real HTTP/service/persistence composition, with rollback-only DB and fake models."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from repomind.api import create_app
from repomind.api.store import PostgresRepositoryStore
from repomind.config import Settings
from repomind.db import load_chunks
from repomind.db.models import RepositoryRecord
from repomind.observability import PostgresTraceStore
from repomind.retrieval import EmbeddedChunk, EmbeddingVector

pytestmark = pytest.mark.postgres


class FakeEmbeddings:
    def embed_text(self, text):
        return EmbeddingVector(values=(1.0, 0.0), model="offline-api-model")

    def embed_chunks(self, chunks):
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


def test_api_index_transaction_rolls_back_on_persistence_failure(client, db_session, monkeypatch):
    name = "api-rollback-" + uuid4().hex
    repository_id = client.post(
        "/api/v1/repositories", json={"name": name, "path": "sample"}
    ).json()["id"]
    assert client.post(f"/api/v1/repositories/{repository_id}/index").status_code == 200
    before = load_chunks(db_session, repository_id)

    def fail(*args):
        raise RuntimeError("private database failure")

    # Fails AFTER snapshot replacement, proving the whole transaction rolls back.
    monkeypatch.setattr("repomind.api.store.persist_embedded_chunks", fail)
    response = client.post(f"/api/v1/repositories/{repository_id}/index")
    assert response.status_code == 500
    assert "private" not in response.text
    assert load_chunks(db_session, repository_id) == before
