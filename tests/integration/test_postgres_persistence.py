"""Integration tests requiring real PostgreSQL with the pgvector extension."""

from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from pydantic import BaseModel
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from repomind.db import (
    PersistenceError,
    content_sha256,
    load_chunks,
    load_embedded_chunks,
    persist_chunks,
    persist_embedded_chunks,
    persist_repository_snapshot,
    pgvector_semantic_search,
    postgres_hybrid_search,
)
from repomind.db.models import CodeChunkRecord, RepositoryFileRecord, RepositoryRecord
from repomind.ingestion import CodeChunk, RepositorySnapshot, SourceFile
from repomind.rag import (
    answer_repository_question_with_retriever,
    build_repository_context,
)
from repomind.retrieval import (
    EmbeddedChunk,
    EmbeddingVector,
    LLMReranker,
    RankedChunk,
    rank_by_similarity,
)

pytestmark = pytest.mark.postgres


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _source(path: str, content: str, language: str | None = "python") -> SourceFile:
    return SourceFile(
        relative_path=path,
        language=language,
        content=content,
        size_bytes=len(content.encode("utf-8")),
        line_count=len(content.splitlines()),
    )


def _snapshot(
    name: str,
    *sources: SourceFile,
    root: Path = Path("C:/private/local/repository"),
) -> RepositorySnapshot:
    return RepositorySnapshot(
        root=root,
        name=name,
        files=list(sources),
        skipped=[],
        file_count=len(sources),
        total_size_bytes=sum(source.size_bytes for source in sources),
        languages={"python": sum(source.language == "python" for source in sources)},
    )


def _chunk(
    path: str,
    content: str,
    *,
    chunk_index: int,
    start_line: int = 1,
) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=start_line,
        end_line=start_line + max(1, len(content.splitlines())) - 1,
        content=content,
        chunk_index=chunk_index,
    )


def _embedded(
    chunk: CodeChunk,
    values: tuple[float, ...],
    model: str = "model-a",
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=chunk,
        embedding=EmbeddingVector(values=values, model=model),
    )


class _FakeRerankLLM:
    def __init__(self, candidate_ids: list[str]) -> None:
        self.candidate_ids = candidate_ids
        self.prompts: list[str] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        self.prompts.append(prompt)
        return response_model(ranked_candidate_ids=self.candidate_ids)


class _FakeAnswerLLM:
    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        return response_model(
            answer="The retry implementation is in the LLM client.",
            source_ids=["S1"],
            insufficient_evidence=False,
        )


class _StaticRetriever:
    def __init__(self, results: Sequence[RankedChunk]) -> None:
        self.results = results

    def __call__(self, query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return self.results[:top_k]


def test_migration_is_at_head_and_vector_extension_exists(
    postgres_engine: Engine,
) -> None:
    alembic_config = Config("alembic.ini")
    expected_head = ScriptDirectory.from_config(alembic_config).get_current_head()

    with postgres_engine.connect() as connection:
        current_revision = MigrationContext.configure(connection).get_current_revision()
        extension_version = connection.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )

    assert current_revision == expected_head
    assert extension_version is not None


def test_repository_snapshot_persists_files_content_and_hashes(
    db_session: Session,
) -> None:
    content = "def café():\r\n    return '雪'\r\n"
    snapshot = _snapshot(
        _name("snapshot"),
        _source("src/example.py", content),
        _source("README.md", "# Example\n", "markdown"),
    )

    repository = persist_repository_snapshot(db_session, snapshot)
    files = db_session.scalars(
        select(RepositoryFileRecord)
        .where(RepositoryFileRecord.repository_id == repository.id)
        .order_by(RepositoryFileRecord.relative_path)
    ).all()

    assert repository.name == snapshot.name
    assert "root" not in RepositoryRecord.__table__.c
    assert [record.relative_path for record in files] == ["README.md", "src/example.py"]
    source_record = files[1]
    assert source_record.content == content
    assert source_record.content_hash == content_sha256(content)
    assert source_record.size_bytes == len(content.encode("utf-8"))
    assert source_record.line_count == 2


def test_chunk_persistence_preserves_metadata_content_and_hash(
    db_session: Session,
) -> None:
    source = _source("src/example.py", "one\ntwo\nthree\n")
    repository = persist_repository_snapshot(db_session, _snapshot(_name("chunks"), source))
    chunks = [
        _chunk("src/example.py", "one\ntwo\n", chunk_index=0),
        _chunk("src/example.py", "two\nthree\n", chunk_index=1, start_line=2),
    ]

    persist_chunks(db_session, repository.id, chunks)
    records = db_session.scalars(
        select(CodeChunkRecord).order_by(CodeChunkRecord.chunk_index)
    ).all()

    assert [(record.start_line, record.end_line) for record in records] == [(1, 2), (2, 3)]
    assert [record.content for record in records] == [chunk.content for chunk in chunks]
    assert records[0].content_hash == content_sha256(chunks[0].content)
    assert records[0].embedding is None
    assert records[0].embedding_model is None


def test_embedded_chunk_round_trip_preserves_domain_values(db_session: Session) -> None:
    source = _source("src/example.py", "alpha\r\nbeta\r\n")
    repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("round-trip"), source),
    )
    embedded = [
        _embedded(
            _chunk("src/example.py", source.content, chunk_index=0),
            (1.0, 0.25, 0.0),
        )
    ]

    persist_embedded_chunks(db_session, repository.id, embedded)
    loaded = load_embedded_chunks(db_session, repository.id)

    assert loaded == embedded


def test_empty_stored_chunk_index_returns_no_search_results(db_session: Session) -> None:
    repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("empty-search"), _source("src/example.py", "content\n")),
    )

    results = pgvector_semantic_search(
        db_session,
        repository.id,
        EmbeddingVector(values=(1.0, 0.0), model="model-a"),
    )

    assert results == []


def test_chunk_persistence_rejects_unknown_file_before_replacement(
    db_session: Session,
) -> None:
    source = _source("src/known.py", "known\n")
    repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("unknown-file"), source),
    )
    persist_chunks(
        db_session,
        repository.id,
        [_chunk("src/known.py", source.content, chunk_index=0)],
    )

    with pytest.raises(PersistenceError, match="not stored"):
        persist_chunks(
            db_session,
            repository.id,
            [_chunk("src/missing.py", "missing\n", chunk_index=0)],
        )

    remaining = db_session.scalars(select(CodeChunkRecord)).all()
    assert len(remaining) == 1
    assert remaining[0].content == "known\n"


def test_exact_pgvector_search_matches_in_memory_ranking_and_context(
    db_session: Session,
) -> None:
    sources = [
        _source("src/a.py", "A\n"),
        _source("src/b.py", "B\n"),
        _source("src/c.py", "C\n"),
    ]
    repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("search"), *sources),
    )
    embedded = [
        _embedded(_chunk("src/a.py", "A\n", chunk_index=0), (1.0, 0.0, 0.0)),
        _embedded(_chunk("src/b.py", "B\n", chunk_index=0), (0.7, 0.7, 0.0)),
        _embedded(_chunk("src/c.py", "C\n", chunk_index=0), (0.0, 1.0, 0.0)),
    ]
    persist_embedded_chunks(db_session, repository.id, embedded)
    query = EmbeddingVector(values=(1.0, 0.0, 0.0), model="model-a")

    database_results = pgvector_semantic_search(
        db_session,
        repository.id,
        query,
        top_k=10,
    )
    memory_results = rank_by_similarity(query, embedded, top_k=10)
    top_one = pgvector_semantic_search(db_session, repository.id, query, top_k=1)

    assert [result.chunk.relative_path for result in database_results] == [
        result.chunk.relative_path for result in memory_results
    ]
    assert [result.score for result in database_results] == pytest.approx(
        [result.score for result in memory_results],
        abs=1e-6,
    )
    assert [result.chunk.relative_path.as_posix() for result in top_one] == ["src/a.py"]
    context = build_repository_context(database_results)
    assert [source.source_id for source in context.sources] == ["S1", "S2", "S3"]
    assert "src/a.py" in context.text


def test_pgvector_search_isolates_repositories(db_session: Session) -> None:
    source_a = _source("src/a.py", "repository A\n")
    source_b = _source("src/b.py", "repository B\n")
    repository_a = persist_repository_snapshot(
        db_session,
        _snapshot(_name("isolation-a"), source_a),
    )
    repository_b = persist_repository_snapshot(
        db_session,
        _snapshot(_name("isolation-b"), source_b),
    )
    persist_embedded_chunks(
        db_session,
        repository_a.id,
        [_embedded(_chunk("src/a.py", source_a.content, chunk_index=0), (1.0, 0.0))],
    )
    persist_embedded_chunks(
        db_session,
        repository_b.id,
        [_embedded(_chunk("src/b.py", source_b.content, chunk_index=0), (1.0, 0.0))],
    )

    results = pgvector_semantic_search(
        db_session,
        repository_a.id,
        EmbeddingVector(values=(1.0, 0.0), model="model-a"),
    )

    assert [result.chunk.relative_path.as_posix() for result in results] == ["src/a.py"]


def test_pgvector_search_isolates_embedding_models(db_session: Session) -> None:
    sources = [
        _source("src/a.py", "model A\n"),
        _source("src/b.py", "model B\n"),
    ]
    repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("models"), *sources),
    )
    persist_embedded_chunks(
        db_session,
        repository.id,
        [
            _embedded(
                _chunk("src/a.py", sources[0].content, chunk_index=0),
                (1.0, 0.0),
                "model-a",
            ),
            _embedded(
                _chunk("src/b.py", sources[1].content, chunk_index=0),
                (1.0, 0.0, 0.0),
                "model-b",
            ),
        ],
    )

    model_a_results = pgvector_semantic_search(
        db_session,
        repository.id,
        EmbeddingVector(values=(1.0, 0.0), model="model-a"),
    )
    missing_results = pgvector_semantic_search(
        db_session,
        repository.id,
        EmbeddingVector(values=(1.0, 0.0), model="model-c"),
    )
    model_b_results = pgvector_semantic_search(
        db_session,
        repository.id,
        EmbeddingVector(values=(1.0, 0.0, 0.0), model="model-b"),
    )

    assert [result.chunk.relative_path.as_posix() for result in model_a_results] == [
        "src/a.py"
    ]
    assert [result.chunk.relative_path.as_posix() for result in model_b_results] == [
        "src/b.py"
    ]
    assert missing_results == []

    with pytest.raises(PersistenceError, match="stored.*uses 2"):
        pgvector_semantic_search(
            db_session,
            repository.id,
            EmbeddingVector(values=(1.0, 0.0, 0.0), model="model-a"),
        )


def test_pgvector_ties_use_source_metadata_order(db_session: Session) -> None:
    sources = [
        _source("src/z.py", "z\n"),
        _source("src/a.py", "a\n"),
    ]
    repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("ties"), *sources),
    )
    persist_embedded_chunks(
        db_session,
        repository.id,
        [
            _embedded(_chunk("src/z.py", "z\n", chunk_index=0), (1.0, 0.0)),
            _embedded(_chunk("src/a.py", "a\n", chunk_index=0), (1.0, 0.0)),
        ],
    )

    results = pgvector_semantic_search(
        db_session,
        repository.id,
        EmbeddingVector(values=(1.0, 0.0), model="model-a"),
    )

    assert [result.chunk.relative_path.as_posix() for result in results] == [
        "src/a.py",
        "src/z.py",
    ]


def test_repeated_snapshot_replaces_files_and_old_chunks(db_session: Session) -> None:
    name = _name("replace")
    first_source = _source("src/old.py", "old\n")
    repository = persist_repository_snapshot(db_session, _snapshot(name, first_source))
    persist_chunks(
        db_session,
        repository.id,
        [_chunk("src/old.py", "old\n", chunk_index=0)],
    )

    replacement = persist_repository_snapshot(
        db_session,
        _snapshot(name, _source("src/new.py", "new\n")),
    )

    assert replacement.id == repository.id
    assert db_session.scalar(select(func.count()).select_from(RepositoryRecord)) == 1
    assert db_session.scalar(select(func.count()).select_from(RepositoryFileRecord)) == 1
    assert db_session.scalar(select(func.count()).select_from(CodeChunkRecord)) == 0
    stored_path = db_session.scalar(select(RepositoryFileRecord.relative_path))
    assert stored_path == "src/new.py"


def test_repository_delete_cascades_to_files_and_chunks(db_session: Session) -> None:
    source = _source("src/example.py", "content\n")
    repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("cascade"), source),
    )
    persist_embedded_chunks(
        db_session,
        repository.id,
        [_embedded(_chunk("src/example.py", source.content, chunk_index=0), (1.0, 0.0))],
    )

    db_session.delete(repository)
    db_session.flush()

    assert db_session.scalar(select(func.count()).select_from(RepositoryFileRecord)) == 0
    assert db_session.scalar(select(func.count()).select_from(CodeChunkRecord)) == 0


def test_failed_transaction_rolls_back_partial_repository(
    postgres_engine: Engine,
) -> None:
    name = _name("rollback")
    snapshot = _snapshot(name, _source("src/example.py", "content\n"))

    with (
        Session(postgres_engine) as session,
        pytest.raises(IntegrityError),
        session.begin(),
    ):
        persist_repository_snapshot(session, snapshot)
        session.add(RepositoryRecord(name=name))
        session.flush()

    with Session(postgres_engine) as verification_session:
        stored = verification_session.scalar(
            select(RepositoryRecord).where(RepositoryRecord.name == name)
        )
        assert stored is None


def test_postgres_hybrid_search_fuses_persisted_chunks_with_repository_isolation(
    db_session: Session,
) -> None:
    repository_sources = [
        _source("src/retry.py", "bounded exponential backoff\n"),
        _source(
            "src/settings.py",
            'OPENAI_EMBEDDING_MODEL = "text-embedding-test"\n',
        ),
    ]
    repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("hybrid"), *repository_sources),
    )
    persisted = [
        _embedded(
            _chunk("src/retry.py", repository_sources[0].content, chunk_index=0),
            (1.0, 0.0),
        ),
        _embedded(
            _chunk("src/settings.py", repository_sources[1].content, chunk_index=0),
            (0.5, 0.5),
        ),
    ]
    persist_embedded_chunks(db_session, repository.id, persisted)

    other_source = _source(
        "src/other.py",
        'OPENAI_EMBEDDING_MODEL = "must-not-leak"\n',
    )
    other_repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("hybrid-other"), other_source),
    )
    persist_embedded_chunks(
        db_session,
        other_repository.id,
        [
            _embedded(
                _chunk("src/other.py", other_source.content, chunk_index=0),
                (1.0, 0.0),
            )
        ],
    )

    loaded = load_chunks(db_session, repository.id)
    results = postgres_hybrid_search(
        db_session,
        repository.id,
        "OPENAI_EMBEDDING_MODEL",
        EmbeddingVector(values=(1.0, 0.0), model="model-a"),
        top_k=5,
    )

    assert loaded == [embedded.chunk for embedded in persisted]
    assert [result.chunk.relative_path.as_posix() for result in results] == [
        "src/settings.py",
        "src/retry.py",
    ]
    assert results[0].semantic_rank == 2
    assert results[0].lexical_rank == 1
    assert all(result.chunk.relative_path.as_posix() != "src/other.py" for result in results)
    context = build_repository_context(results)
    assert context.sources[0].chunk.relative_path.as_posix() == "src/settings.py"


def test_postgres_hybrid_results_are_reranked_as_domain_chunks(
    db_session: Session,
) -> None:
    sources = [
        _source("README.md", "Retry behavior for failed OpenAI requests.\n", "markdown"),
        _source("src/llm/client.py", "def retry_failed_request(): pass\n"),
    ]
    repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("rerank"), *sources),
    )
    persist_embedded_chunks(
        db_session,
        repository.id,
        [
            _embedded(
                _chunk("README.md", sources[0].content, chunk_index=0),
                (1.0, 0.0),
            ),
            _embedded(
                _chunk("src/llm/client.py", sources[1].content, chunk_index=0),
                (0.8, 0.2),
            ),
        ],
    )

    other_source = _source("src/other.py", "retry_failed_request\n")
    other_repository = persist_repository_snapshot(
        db_session,
        _snapshot(_name("rerank-other"), other_source),
    )
    persist_embedded_chunks(
        db_session,
        other_repository.id,
        [
            _embedded(
                _chunk("src/other.py", other_source.content, chunk_index=0),
                (1.0, 0.0),
            )
        ],
    )

    hybrid_results = postgres_hybrid_search(
        db_session,
        repository.id,
        "Where is retry behavior for failed OpenAI requests implemented?",
        EmbeddingVector(values=(1.0, 0.0), model="model-a"),
        top_k=2,
    )
    llm = _FakeRerankLLM(["C2", "C1"])
    reranked = LLMReranker(llm).rerank(
        "Where is retry behavior for failed OpenAI requests implemented?",
        hybrid_results,
        top_k=2,
    )

    assert all(isinstance(result.chunk, CodeChunk) for result in reranked)
    assert [result.chunk.relative_path.as_posix() for result in reranked] == [
        "src/llm/client.py",
        "README.md",
    ]
    assert [result.original_rank for result in reranked] == [2, 1]
    assert all(result.chunk.relative_path.as_posix() != "src/other.py" for result in reranked)
    assert "src/llm/client.py" in llm.prompts[0]
    context = build_repository_context(reranked)
    assert context.sources[0].chunk.relative_path.as_posix() == "src/llm/client.py"
    answer = answer_repository_question_with_retriever(
        "Where is retry behavior for failed OpenAI requests implemented?",
        _StaticRetriever(reranked),
        _FakeAnswerLLM(),
    )
    assert answer.citations[0].relative_path == Path("src/llm/client.py")
    assert answer.citations[0].start_line == 1
