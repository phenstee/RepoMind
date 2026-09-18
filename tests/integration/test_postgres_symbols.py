"""Real PostgreSQL coverage for Milestone 24 symbol-aware retrieval fusion."""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from repomind.db import (
    find_symbol_candidates,
    persist_chunks,
    persist_embedded_chunks,
    persist_repository_snapshot,
    postgres_hybrid_symbol_search,
)
from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    ChunkKind,
    CodeChunk,
    RepositorySnapshot,
    SourceFile,
    chunk_source_file,
)
from repomind.rag import ContextAssemblyConfig, ContextStrategy, assemble_context
from repomind.rag.assembly import InMemoryNeighborLoader
from repomind.retrieval import (
    EmbeddedChunk,
    EmbeddingVector,
    extract_identifier_candidates,
    symbol_search,
)

pytestmark = pytest.mark.postgres

_STRUCTURAL_SOURCE = '''class UserService:
    def login(self, username, password):
        normalized = username.strip().casefold()
        if not normalized or not password:
            raise ValueError("credentials are required")
        token = _issue_token(normalized)
        return token

    def logout(self, token):
        if not token:
            raise ValueError("token is required")
        self._revoke(token)


class AdminService:
    def login(self, username, password):
        return "admin-token"
'''

_LINE_SOURCE = "".join(f"value_{index} = {index}\n" for index in range(20))


def _persist(session: Session, prefix: str, source: SourceFile, config: ChunkingConfig) -> int:
    name = f"{prefix}-{uuid4().hex}"
    snapshot = RepositorySnapshot(
        root=Path("C:/isolated/symbol-fixture"),
        name=name,
        files=[source],
        skipped=[],
        file_count=1,
        total_size_bytes=source.size_bytes,
        languages={"python": 1},
    )
    repository = persist_repository_snapshot(session, snapshot)
    chunks = chunk_source_file(source, config)
    persist_chunks(session, repository.id, chunks)
    return repository.id


def _structural_source() -> SourceFile:
    return SourceFile(
        relative_path=Path("src/auth/service.py"),
        language="python",
        content=_STRUCTURAL_SOURCE,
        size_bytes=len(_STRUCTURAL_SOURCE.encode()),
        line_count=len(_STRUCTURAL_SOURCE.splitlines()),
    )


def _persist_manual_chunks(
    session: Session, prefix: str, chunks: list[CodeChunk]
) -> int:
    paths = sorted({chunk.relative_path for chunk in chunks}, key=lambda path: path.as_posix())
    sources = [
        SourceFile(
            relative_path=path,
            language="python",
            content="fixture\n",
            size_bytes=len(b"fixture\n"),
            line_count=1,
        )
        for path in paths
    ]
    snapshot = RepositorySnapshot(
        root=Path("C:/isolated/symbol-fixture"),
        name=f"{prefix}-{uuid4().hex}",
        files=sources,
        skipped=[],
        file_count=len(sources),
        total_size_bytes=sum(source.size_bytes for source in sources),
        languages={"python": len(sources)},
    )
    repository = persist_repository_snapshot(session, snapshot)
    persist_chunks(session, repository.id, chunks)
    return repository.id


def _run_chunk(
    index: int,
    qualified_name: str,
    *,
    path: str = "src/runs.py",
    fragment_index: int | None = None,
    fragment_count: int | None = None,
) -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=index + 1,
        end_line=index + 1,
        content=f"# {qualified_name} chunk {index}\n",
        chunk_index=index,
        chunking_strategy=ChunkingStrategy.STRUCTURAL,
        chunk_kind=(
            ChunkKind.STRUCTURAL_FRAGMENT
            if fragment_index is not None
            else ChunkKind.METHOD
        ),
        symbol_name="run",
        qualified_symbol_name=qualified_name,
        fragment_index=fragment_index,
        fragment_count=fragment_count,
    )


def test_qualified_symbol_lookup_finds_exact_chunk_over_real_postgres(db_session: Session) -> None:
    source = _structural_source()
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, max_lines_per_chunk=20, overlap_lines=0)
    repository_id = _persist(db_session, "qualified-lookup", source, config)

    candidates = extract_identifier_candidates("UserService.login")
    results = find_symbol_candidates(db_session, repository_id, candidates, limit=10)

    assert len(results) == 1
    assert results[0].chunk.qualified_symbol_name == "UserService.login"
    assert results[0].matched_identifier == "UserService.login"


def test_simple_symbol_lookup_finds_both_ambiguous_matches(db_session: Session) -> None:
    source = _structural_source()
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, max_lines_per_chunk=20, overlap_lines=0)
    repository_id = _persist(db_session, "simple-lookup", source, config)

    candidates = extract_identifier_candidates("login")
    results = find_symbol_candidates(db_session, repository_id, candidates, limit=10)

    qualified_names = {r.chunk.qualified_symbol_name for r in results}
    assert qualified_names == {"UserService.login", "AdminService.login"}
    assert {result.matched_identifier for result in results} == {"login"}


def test_symbol_lookup_is_isolated_by_repository(db_session: Session) -> None:
    source = _structural_source()
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, max_lines_per_chunk=20, overlap_lines=0)
    repository_a = _persist(db_session, "isolate-a", source, config)
    repository_b = _persist(db_session, "isolate-b", source, config)

    candidates = extract_identifier_candidates("UserService.login")
    results_a = find_symbol_candidates(db_session, repository_a, candidates, limit=10)
    results_b = find_symbol_candidates(db_session, repository_b, candidates, limit=10)

    assert len(results_a) == 1
    assert len(results_b) == 1
    assert results_a[0].chunk.relative_path == results_b[0].chunk.relative_path


def test_symbol_candidate_count_is_bounded_over_many_matches(db_session: Session) -> None:
    # 60 distinct classes each with their own "run" method, so this exercises
    # bounding across many genuinely distinct symbols (not one repeatedly
    # redefined name, which would legitimately collapse to one dedup key).
    content = "".join(f"class Mod{index}:\n    def run(self):\n        return {index}\n\n\n" for index in range(60))
    source = SourceFile(
        relative_path=Path("src/many_runs.py"),
        language="python",
        content=content,
        size_bytes=len(content.encode()),
        line_count=len(content.splitlines()),
    )
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, max_lines_per_chunk=3, overlap_lines=0)
    repository_id = _persist(db_session, "bounded", source, config)

    candidates = extract_identifier_candidates("run")
    results = find_symbol_candidates(db_session, repository_id, candidates, limit=5)

    assert len(results) == 5


def test_large_fragmented_symbol_cannot_starve_distinct_candidates(
    db_session: Session,
) -> None:
    chunks = [
        _run_chunk(
            index,
            "BigService.run",
            fragment_index=index + 1,
            fragment_count=50,
        )
        for index in range(50)
    ]
    chunks.extend(
        _run_chunk(50 + index, f"Worker{index}.run") for index in range(10)
    )
    repository_id = _persist_manual_chunks(db_session, "fragment-starvation", chunks)
    candidates = extract_identifier_candidates("run")

    first = find_symbol_candidates(db_session, repository_id, candidates, limit=5)
    second = find_symbol_candidates(db_session, repository_id, candidates, limit=5)

    assert len(first) == 5
    assert first == second
    assert [result.chunk.qualified_symbol_name for result in first] == [
        "BigService.run",
        "Worker0.run",
        "Worker1.run",
        "Worker2.run",
        "Worker3.run",
    ]
    assert first[0].chunk.fragment_index == 1


def test_several_large_symbols_cannot_hide_later_distinct_candidates(
    db_session: Session,
) -> None:
    chunks: list[CodeChunk] = []
    for symbol_offset, qualified_name in enumerate(("BigA.run", "BigB.run")):
        chunks.extend(
            _run_chunk(
                symbol_offset * 30 + fragment_offset,
                qualified_name,
                fragment_index=fragment_offset + 1,
                fragment_count=30,
            )
            for fragment_offset in range(30)
        )
    chunks.extend(_run_chunk(60 + index, f"Later{index}.run") for index in range(10))
    repository_id = _persist_manual_chunks(db_session, "multi-fragment-starvation", chunks)

    results = find_symbol_candidates(
        db_session,
        repository_id,
        extract_identifier_candidates("run"),
        limit=5,
    )

    assert len(results) == 5
    assert [result.chunk.qualified_symbol_name for result in results] == [
        "BigA.run",
        "BigB.run",
        "Later0.run",
        "Later1.run",
        "Later2.run",
    ]


def test_persisted_lookup_matches_in_memory_identifier_contract(
    db_session: Session,
) -> None:
    chunks = [
        _run_chunk(0, "AdminService.login"),
        _run_chunk(1, "UserService.login"),
    ]
    chunks = [
        chunk.model_copy(update={"symbol_name": "login"})
        for chunk in chunks
    ]
    repository_id = _persist_manual_chunks(db_session, "identifier-parity", chunks)

    for query in ("login", "UserService.login"):
        candidates = extract_identifier_candidates(query)
        in_memory = symbol_search(candidates, chunks, limit=10)
        persisted = find_symbol_candidates(
            db_session, repository_id, candidates, limit=10
        )

        def projection(results):
            return [
                (
                    result.chunk.relative_path,
                    result.chunk.qualified_symbol_name,
                    result.chunk.chunk_index,
                    result.match_tier,
                    result.matched_identifier,
                )
                for result in results
            ]

        assert projection(persisted) == projection(in_memory)


def test_structural_fragments_collapse_to_first_fragment_over_real_postgres(
    db_session: Session,
) -> None:
    body_lines = "\n".join(f"        step_{i} = {i}" for i in range(20))
    content = f"class Big:\n    def login(self):\n{body_lines}\n        return step_0\n"
    source = SourceFile(
        relative_path=Path("src/big.py"),
        language="python",
        content=content,
        size_bytes=len(content.encode()),
        line_count=len(content.splitlines()),
    )
    config = ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURAL, max_lines_per_chunk=3, overlap_lines=0, max_chars_per_chunk=60
    )
    repository_id = _persist(db_session, "fragments", source, config)

    candidates = extract_identifier_candidates("Big.login")
    results = find_symbol_candidates(db_session, repository_id, candidates, limit=10)

    assert len(results) == 1
    assert results[0].chunk.fragment_index in (None, 1)


def test_line_v1_repository_returns_no_symbol_candidates_without_error(
    db_session: Session,
) -> None:
    source = SourceFile(
        relative_path=Path("src/vectors.py"),
        language="python",
        content=_LINE_SOURCE,
        size_bytes=len(_LINE_SOURCE.encode()),
        line_count=20,
    )
    config = ChunkingConfig(max_lines_per_chunk=4, overlap_lines=0)
    repository_id = _persist(db_session, "line-empty", source, config)

    candidates = extract_identifier_candidates("value_1")
    results = find_symbol_candidates(db_session, repository_id, candidates, limit=10)

    assert results == []


def test_symbol_lookup_uses_exactly_one_bounded_query_regardless_of_candidates(
    db_session: Session,
) -> None:
    source = _structural_source()
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, max_lines_per_chunk=20, overlap_lines=0)
    repository_id = _persist(db_session, "single-query", source, config)
    candidates = extract_identifier_candidates(
        "UserService.login AdminService.login run get set test save load login logout"
    )

    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(db_session.get_bind(), "before_cursor_execute", _record)
    try:
        find_symbol_candidates(db_session, repository_id, candidates, limit=10)
    finally:
        event.remove(db_session.get_bind(), "before_cursor_execute", _record)

    select_statements = [s for s in statements if s.strip().upper().startswith("SELECT")]
    # One bounded, tier-ordered SELECT regardless of candidate count - never
    # one query per candidate. (The repository-existence check resolves from
    # the session's identity map here since _persist already loaded it.)
    assert len(select_statements) == 1
    assert "LIMIT" in select_statements[0].upper()


def test_symbol_fused_retrieval_composes_normally_with_context_assembler(
    db_session: Session,
) -> None:
    source = _structural_source()
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, max_lines_per_chunk=20, overlap_lines=0)
    name = f"assembler-fusion-{uuid4().hex}"
    snapshot = RepositorySnapshot(
        root=Path("C:/isolated/symbol-fixture"),
        name=name,
        files=[source],
        skipped=[],
        file_count=1,
        total_size_bytes=source.size_bytes,
        languages={"python": 1},
    )
    repository = persist_repository_snapshot(db_session, snapshot)
    chunks = chunk_source_file(source, config)
    persist_chunks(db_session, repository.id, chunks)

    candidates = extract_identifier_candidates("UserService.login")
    symbol_results = find_symbol_candidates(db_session, repository.id, candidates, limit=10)
    assert symbol_results

    result = assemble_context(
        symbol_results,
        InMemoryNeighborLoader(chunks),
        ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1),
    )

    packed_identities = [(c.chunk.relative_path, c.chunk.start_line, c.chunk.end_line) for c in result.chunks]
    assert len(packed_identities) == len(set(packed_identities))
    assert result.chunks[0].chunk.qualified_symbol_name == "UserService.login"


def test_symbol_name_and_qualified_symbol_name_indexes_exist(db_session: Session) -> None:
    rows = db_session.execute(
        text(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'code_chunks' "
            "AND indexname IN ('ix_code_chunks_symbol_name', "
            "'ix_code_chunks_qualified_symbol_name')"
        )
    ).scalars().all()

    assert set(rows) == {"ix_code_chunks_symbol_name", "ix_code_chunks_qualified_symbol_name"}


def test_hybrid_symbol_search_fuses_with_persisted_semantic_and_bm25(db_session: Session) -> None:
    source = _structural_source()
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, max_lines_per_chunk=20, overlap_lines=0)
    name = f"fusion-{uuid4().hex}"
    snapshot = RepositorySnapshot(
        root=Path("C:/isolated/symbol-fixture"),
        name=name,
        files=[source],
        skipped=[],
        file_count=1,
        total_size_bytes=source.size_bytes,
        languages={"python": 1},
    )
    repository = persist_repository_snapshot(db_session, snapshot)
    chunks = chunk_source_file(source, config)
    model = "fusion-test-model"
    embedded = [
        EmbeddedChunk(
            chunk=chunk,
            embedding=EmbeddingVector(values=(float(index) + 1.0, 0.0), model=model),
        )
        for index, chunk in enumerate(chunks)
    ]
    persist_embedded_chunks(db_session, repository.id, embedded)

    query_embedding = EmbeddingVector(values=(0.0, 1.0), model=model)
    results = postgres_hybrid_symbol_search(
        db_session,
        repository.id,
        "UserService.login",
        query_embedding,
        top_k=5,
    )

    matched = next((r for r in results if r.chunk.qualified_symbol_name == "UserService.login"), None)
    assert matched is not None
    assert matched.symbol_rank is not None


def test_hybrid_symbol_search_selects_behavioral_fragment_over_real_postgres(
    db_session: Session,
) -> None:
    source = _structural_source()
    name = f"fragment-fusion-{uuid4().hex}"
    snapshot = RepositorySnapshot(
        root=Path("C:/isolated/symbol-fixture"),
        name=name,
        files=[source],
        skipped=[],
        file_count=1,
        total_size_bytes=source.size_bytes,
        languages={"python": 1},
    )
    repository = persist_repository_snapshot(db_session, snapshot)

    def fragment(
        index: int,
        content: str,
        *,
        qualified_name: str = "UserService.login",
        fragment_index: int | None = None,
    ) -> CodeChunk:
        return CodeChunk(
            relative_path=source.relative_path,
            language="python",
            start_line=index * 3 + 1,
            end_line=index * 3 + 2,
            content=content,
            chunk_index=index,
            chunking_strategy=ChunkingStrategy.STRUCTURAL,
            chunk_kind=(
                ChunkKind.STRUCTURAL_FRAGMENT
                if fragment_index is not None
                else ChunkKind.METHOD
            ),
            symbol_name=qualified_name.rsplit(".", 1)[-1],
            qualified_symbol_name=qualified_name,
            fragment_index=fragment_index,
            fragment_count=3 if fragment_index is not None else None,
        )

    declaration = fragment(
        0,
        "def login(self, username, password):\n    normalized = username.strip()\n",
        fragment_index=1,
    )
    validation = fragment(
        1,
        "if not normalized or not password:\n    raise ValueError('credentials are required')\n",
        fragment_index=2,
    )
    token = fragment(
        2,
        "token = issue_token(normalized)\nreturn token\n",
        fragment_index=3,
    )
    distraction = fragment(3, "def unrelated(self):\n    return None\n", qualified_name="Other.unrelated")
    admin = fragment(
        4,
        "def login(self, username, password):\n    return 'admin-token'\n",
        qualified_name="AdminService.login",
    )
    model = "fragment-fusion-test-model"
    embedded = [
        EmbeddedChunk(
            chunk=chunk,
            embedding=EmbeddingVector(values=values, model=model),
        )
        for chunk, values in (
            (declaration, (0.0, 1.0)),
            (validation, (0.8, 0.2)),
            (token, (-1.0, 0.0)),
            (distraction, (1.0, 0.0)),
            (admin, (-1.0, 0.0)),
        )
    ]
    persist_embedded_chunks(db_session, repository.id, embedded)
    query_embedding = EmbeddingVector(values=(1.0, 0.0), model=model)

    natural = postgres_hybrid_symbol_search(
        db_session,
        repository.id,
        "How does UserService.login validate credentials?",
        query_embedding,
        top_k=3,
        semantic_candidates=2,
        lexical_candidates=2,
    )
    exact = postgres_hybrid_symbol_search(
        db_session,
        repository.id,
        "UserService.login",
        query_embedding,
        top_k=3,
        semantic_candidates=5,
        lexical_candidates=5,
    )

    assert natural[0].chunk == validation
    assert natural[0].source_ranks["symbol"] == 1
    assert exact[0].chunk == declaration
    assert exact[0].chunk.qualified_symbol_name == "UserService.login"
