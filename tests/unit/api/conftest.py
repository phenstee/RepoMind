"""Offline API dependencies; real services/domain, fake external boundaries."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from repomind.agent import AgentDecisionResponse
from repomind.api import create_app
from repomind.api.errors import APIError
from repomind.api.models import RepositoryFileResponse
from repomind.api.store import RepositoryBinding
from repomind.coding import CodingPlan, CodingReview
from repomind.config import Settings
from repomind.db.repositories import RepositoryNotFoundError, content_sha256
from repomind.indexing import (
    IndexedFileState,
    IndexManifest,
    IndexSummary,
    validate_index_update,
)
from repomind.llm import OpenAILLMClient
from repomind.retrieval import (
    EmbeddedChunk,
    EmbeddingTextStrategy,
    EmbeddingVector,
    OpenAIEmbeddingClient,
)


class MemoryRepositoryStore:
    """Models per-file persisted index state so incremental bugs are visible.

    A fake that simply replaced the whole index would hide classification,
    reuse, and deletion mistakes, so this one keeps the same per-path state the
    real store keeps: content hash, chunk list, and the index fingerprint.
    Reused files keep their original ``EmbeddedChunk`` objects, which lets a
    test assert reuse by identity.
    """

    def __init__(self):
        self.bindings = {}
        self.snapshots = {}
        self.chunks = {}
        self.indexes = {}
        self.fingerprints = {}
        self.search_calls = []
        self.candidates = []
        self.neighbor_corpus = []

    def register(self, name, path):
        for binding in self.bindings.values():
            if binding.name == name:
                if binding.workspace_relative_path != path:
                    raise APIError(409, "repository_conflict", "Repository name is already bound.")
                return binding
        binding = RepositoryBinding(
            len(self.bindings) + 1, name, path, datetime(2026, 9, 15, tzinfo=UTC)
        )
        self.bindings[binding.id] = binding
        return binding

    def get(self, repository_id):
        if repository_id not in self.bindings:
            raise RepositoryNotFoundError("private database path/password")
        return self.bindings[repository_id]

    def list_repositories(self, limit):
        return list(reversed(list(self.bindings.values())))[:limit]

    def files(self, repository_id, limit, offset):
        index = self.indexes.get(repository_id, {})
        sources = [index[path]["source"] for path in sorted(index)]
        return [
            RepositoryFileResponse(
                relative_path=f.relative_path.as_posix(),
                language=f.language,
                size_bytes=f.size_bytes,
                line_count=f.line_count,
            )
            for f in sources[offset : offset + limit]
        ]

    def index_manifest(self, repository_id):
        index = self.indexes.get(repository_id)
        if index is None:
            return IndexManifest()
        return IndexManifest(
            fingerprint=self.fingerprints.get(repository_id),
            files={
                path: IndexedFileState(
                    content_hash=entry["content_hash"], chunk_count=len(entry["chunks"])
                )
                for path, entry in index.items()
            },
        )

    def apply_index_update(self, repository_id, update):
        # Share the real store's reconciliation invariants so unit tests
        # cannot pass against a fake that tolerates a malformed delta.
        validate_index_update(update, manifest=self.index_manifest(repository_id))
        index = self.indexes.setdefault(repository_id, {})
        if update.full_rebuild:
            index.clear()
        sources = {s.relative_path.as_posix(): s for s in update.snapshot.files}
        for path in update.deleted:
            index.pop(path, None)
        for path in update.upserted:
            source = sources[path]
            index[path] = {
                "source": source,
                "content_hash": content_sha256(source.content),
                "chunks": [],
            }
        for embedded in update.chunks:
            index[embedded.chunk.relative_path.as_posix()]["chunks"].append(embedded)

        self.fingerprints[repository_id] = update.fingerprint
        self.snapshots[repository_id] = update.snapshot
        self.chunks[repository_id] = [
            chunk for path in sorted(index) for chunk in index[path]["chunks"]
        ]
        models = [c.embedding.model for c in self.chunks[repository_id]]
        return IndexSummary(
            file_count=len(index),
            chunk_count=len(self.chunks[repository_id]),
            embedding_model=min(models) if models else None,
        )

    def search(self, repository_id, query, embedding, *, hybrid, top_k, include_symbols=False):
        self.search_calls.append((repository_id, query, hybrid, top_k, include_symbols))
        return self.candidates[:top_k]

    def load_neighbors(self, repository_id, keys):
        wanted = set(keys)
        return [
            chunk
            for chunk in self.neighbor_corpus
            if (chunk.relative_path.as_posix(), chunk.chunk_index) in wanted
        ]


class MemoryTraces:
    def __init__(self):
        self.runs = {}
        self.fail = False

    def persist_run_trace(self, trace):
        if self.fail:
            raise RuntimeError("postgresql://private:password@host/database")
        self.runs[trace.run_id] = trace

    def get_run_trace(self, run_id):
        return self.runs.get(run_id)

    def list_run_traces(self, *, run_type=None, status=None, limit=20):
        return tuple(
            r
            for r in self.runs.values()
            if (run_type is None or r.run_type == run_type)
            and (status is None or r.status == status)
        )[:limit]


class FakeEmbeddings:
    """Counts exactly which chunks were sent for embedding.

    ``embedded_paths`` makes it obvious which files a run actually paid to
    embed, so a test can prove an unchanged file was never re-embedded.
    """

    def __init__(self):
        self.calls = []
        self.embedded_batches = []
        self.values = (1.0, 0.0)
        self.model = "offline-model"
        self.text_strategy = EmbeddingTextStrategy.RAW_SOURCE

    @property
    def vector(self):
        """Always carries the model currently advertised for the fingerprint.

        A fake whose advertised ``embedding_model`` could drift from the model
        stamped on its vectors would let a test "pass" while production stored
        a vector contradicting its own fingerprint.
        """

        return EmbeddingVector(values=self.values, model=self.model)

    @property
    def embedding_model(self):
        return self.model

    @property
    def embedding_text_strategy(self):
        return self.text_strategy

    @property
    def embedded_paths(self):
        return [
            chunk.relative_path.as_posix() for batch in self.embedded_batches for chunk in batch
        ]

    def embed_text(self, text):
        self.calls.append(text)
        return self.vector

    def embed_chunks(self, chunks):
        self.calls.append(chunks)
        self.embedded_batches.append(list(chunks))
        return [EmbeddedChunk(chunk=c, embedding=self.vector) for c in chunks]


def _as_provider_decision(payload: dict) -> dict:
    """Re-encode an internal-shaped decision into the provider wire format.

    API tests author decisions in the readable internal shape, but a real
    provider must send tool arguments as one serialized JSON object string
    because the strict response schema cannot express an open dict. Encoding
    here keeps the fake faithful to the wire format without rewriting every
    scripted payload.
    """

    if "tool_arguments" not in payload:
        return payload
    encoded = dict(payload)
    encoded["tool_arguments_json"] = json.dumps(encoded.pop("tool_arguments"))
    return encoded


class ScriptedLLM:
    def __init__(self):
        self.responses = []
        self.plan_responses = []
        self.review_responses = []
        self.calls = []

    def generate_structured(self, prompt, response_model, **kwargs):
        self.calls.append((prompt, response_model, kwargs))
        if response_model is CodingPlan:
            if self.plan_responses:
                response = self.plan_responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                if callable(response):
                    response = response(prompt)
                return response_model.model_validate(response)
            payload = json.loads(prompt)
            criteria = payload["task"]["acceptance_criteria"]
            return CodingPlan.model_validate(
                {
                    "task_summary": "Implement and verify the requested change.",
                    "steps": [
                        {
                            "step_id": 1,
                            "action": "Inspect, implement, and verify the requested change.",
                            "criterion_indices": list(range(len(criteria))),
                            "verification": ["pytest", "ruff"],
                        }
                    ],
                    "acceptance_coverage": [
                        {"criterion_index": index, "step_ids": [1]}
                        for index in range(len(criteria))
                    ],
                    "verification_plan": ["pytest", "ruff"],
                }
            )
        if response_model is CodingReview:
            if self.review_responses:
                response = self.review_responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                if callable(response):
                    response = response(prompt)
                return response_model.model_validate(response)
            payload = json.loads(prompt)
            criteria = payload["task"]["acceptance_criteria"]
            return CodingReview.model_validate(
                {
                    "verdict": "approve",
                    "workspace_revision": payload["workspace_revision"],
                    "acceptance_results": [
                        {
                            "criterion_index": index,
                            "status": "satisfied",
                            "evidence": "The bounded diff and verification support this criterion.",
                        }
                        for index in range(len(criteria))
                    ],
                }
            )
        if not self.responses:
            raise AssertionError("Unexpected model request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if response_model is AgentDecisionResponse and isinstance(response, dict):
            response = _as_provider_decision(response)
        return response_model.model_validate(response)


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real model requests are forbidden in API tests")

    monkeypatch.setattr(OpenAILLMClient, "generate_structured", forbidden)
    monkeypatch.setattr(OpenAIEmbeddingClient, "embed_text", forbidden)
    monkeypatch.setattr(OpenAIEmbeddingClient, "embed_chunks", forbidden)


@pytest.fixture
def api(tmp_path):
    root = tmp_path / "workspace"
    repo = root / "sample"
    repo.mkdir(parents=True)
    (repo / "app.py").write_bytes(b"def value():\r\n    return 1\r\n")
    store, traces, llm, embeddings = (
        MemoryRepositoryStore(),
        MemoryTraces(),
        ScriptedLLM(),
        FakeEmbeddings(),
    )
    app = create_app(
        settings=Settings(_env_file=None, repomind_workspace_root=root),
        repository_store=store,
        trace_store=traces,
        llm_factory=lambda trace: llm,
        embedding_factory=lambda trace: embeddings,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        result = client.post("/api/v1/repositories", json={"name": "sample", "path": "sample"})
        assert result.status_code == 201, result.text
        yield SimpleNamespace(
            client=client,
            app=app,
            root=root,
            repo=repo,
            store=store,
            traces=traces,
            llm=llm,
            embeddings=embeddings,
        )
