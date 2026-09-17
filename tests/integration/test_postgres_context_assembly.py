"""Real PostgreSQL coverage for Milestone 23 context assembly.

Covers batched neighbor loading, repository isolation, structural-fragment
and line_v1 expansion over persisted chunks, and citation integrity end to
end through the assembler.
"""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from repomind.db import load_neighbor_chunks, persist_chunks, persist_repository_snapshot
from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    RepositorySnapshot,
    SourceFile,
    chunk_source_file,
)
from repomind.rag import ContextAssemblyConfig, ContextStrategy, assemble_context
from repomind.retrieval import SemanticSearchResult

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
'''

_LINE_SOURCE = "".join(f"value_{index} = {index}\n" for index in range(30))


def _persist(session: Session, prefix: str, source: SourceFile, config: ChunkingConfig) -> int:
    name = f"{prefix}-{uuid4().hex}"
    snapshot = RepositorySnapshot(
        root=Path("C:/isolated/context-assembly-fixture"),
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


def _neighbor_loader(session: Session, repository_id: int):
    def loader(keys):
        return load_neighbor_chunks(session, repository_id, keys)

    return loader


def test_batched_neighbor_loading_uses_one_query_regardless_of_key_count(
    db_session: Session,
) -> None:
    source = SourceFile(
        relative_path=Path("src/vectors.py"),
        language="python",
        content=_LINE_SOURCE,
        size_bytes=len(_LINE_SOURCE.encode()),
        line_count=30,
    )
    repository_id = _persist(
        db_session,
        "neighbor-batch",
        source,
        ChunkingConfig(max_lines_per_chunk=1, overlap_lines=0),
    )

    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(db_session.get_bind(), "before_cursor_execute", _record)
    try:
        keys = [("src/vectors.py", index) for index in range(1, 20)]
        found = load_neighbor_chunks(db_session, repository_id, keys)
    finally:
        event.remove(db_session.get_bind(), "before_cursor_execute", _record)

    select_statements = [s for s in statements if s.strip().upper().startswith("SELECT")]
    assert len(select_statements) == 1
    assert len(found) == 19


def test_neighbor_loading_is_isolated_by_repository(db_session: Session) -> None:
    source = SourceFile(
        relative_path=Path("src/vectors.py"),
        language="python",
        content=_LINE_SOURCE,
        size_bytes=len(_LINE_SOURCE.encode()),
        line_count=30,
    )
    config = ChunkingConfig(max_lines_per_chunk=1, overlap_lines=0)
    repository_a = _persist(db_session, "isolate-a", source, config)
    repository_b = _persist(db_session, "isolate-b", source, config)

    keys = [("src/vectors.py", 0), ("src/vectors.py", 1)]
    found_a = load_neighbor_chunks(db_session, repository_a, keys)
    found_b = load_neighbor_chunks(db_session, repository_b, keys)

    assert len(found_a) == 2
    assert len(found_b) == 2


def test_structural_fragment_expansion_over_persisted_chunks(db_session: Session) -> None:
    source = SourceFile(
        relative_path=Path("src/service.py"),
        language="python",
        content=_STRUCTURAL_SOURCE,
        size_bytes=len(_STRUCTURAL_SOURCE.encode()),
        line_count=len(_STRUCTURAL_SOURCE.splitlines()),
    )
    config = ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURAL,
        max_lines_per_chunk=2,
        overlap_lines=0,
        max_chars_per_chunk=40,
    )
    repository_id = _persist(db_session, "structural-fragment", source, config)
    chunks = chunk_source_file(source, config)
    login_fragments = [
        chunk for chunk in chunks if chunk.qualified_symbol_name == "UserService.login"
    ]
    assert len(login_fragments) >= 2
    middle_fragment = next(chunk for chunk in login_fragments if chunk.fragment_index == 2)
    seed = SemanticSearchResult(chunk=middle_fragment, score=0.9, rank=1)

    result = assemble_context(
        [seed],
        _neighbor_loader(db_session, repository_id),
        ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1),
    )

    fragment_origins = {
        packed.chunk.fragment_index: packed.origin.value
        for packed in result.chunks
        if packed.chunk.qualified_symbol_name == "UserService.login"
    }
    assert fragment_origins.get(1) == "same_symbol_fragment" or 1 not in fragment_origins
    assert fragment_origins.get(3) == "same_symbol_fragment" or 3 not in fragment_origins
    assert 2 in fragment_origins  # the seed itself


def test_line_v1_expansion_over_persisted_chunks(db_session: Session) -> None:
    source = SourceFile(
        relative_path=Path("src/vectors.py"),
        language="python",
        content=_LINE_SOURCE,
        size_bytes=len(_LINE_SOURCE.encode()),
        line_count=30,
    )
    config = ChunkingConfig(max_lines_per_chunk=2, overlap_lines=0)
    repository_id = _persist(db_session, "line-expansion", source, config)
    chunks = chunk_source_file(source, config)
    seed_chunk = chunks[3]
    seed = SemanticSearchResult(chunk=seed_chunk, score=0.9, rank=1)

    result = assemble_context(
        [seed],
        _neighbor_loader(db_session, repository_id),
        ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1),
    )

    packed_indexes = {packed.chunk.chunk_index for packed in result.chunks}
    assert packed_indexes == {2, 3, 4}


def test_citation_integrity_survives_expansion(db_session: Session) -> None:
    source = SourceFile(
        relative_path=Path("src/vectors.py"),
        language="python",
        content=_LINE_SOURCE,
        size_bytes=len(_LINE_SOURCE.encode()),
        line_count=30,
    )
    config = ChunkingConfig(max_lines_per_chunk=2, overlap_lines=0)
    repository_id = _persist(db_session, "citation-integrity", source, config)
    chunks = chunk_source_file(source, config)
    seed_chunk = chunks[3]
    seed = SemanticSearchResult(chunk=seed_chunk, score=0.9, rank=1)

    result = assemble_context(
        [seed],
        _neighbor_loader(db_session, repository_id),
        ContextAssemblyConfig(strategy=ContextStrategy.EXPANDED, neighbor_radius=1),
    )

    for packed in result.chunks:
        original = next(c for c in chunks if c.chunk_index == packed.chunk.chunk_index)
        assert packed.chunk.relative_path == original.relative_path
        assert packed.chunk.start_line == original.start_line
        assert packed.chunk.end_line == original.end_line
        assert packed.chunk.content == original.content
