"""Persistence adapter: existing repository rows, transactions and retrieval APIs."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from repomind.api.errors import APIError
from repomind.api.models import RepositoryFileResponse
from repomind.db import (
    apply_index_update,
    load_neighbor_chunks,
    pgvector_semantic_search,
    postgres_hybrid_search,
    postgres_hybrid_symbol_search,
    read_index_manifest,
    session_scope,
)
from repomind.db.models import RepositoryFileRecord, RepositoryRecord
from repomind.db.repositories import RepositoryNotFoundError
from repomind.indexing import IndexManifest, IndexSummary, IndexUpdate
from repomind.ingestion import CodeChunk
from repomind.retrieval import (
    DEFAULT_SYMBOL_CANDIDATE_LIMIT,
    EmbeddingVector,
    RankedChunk,
    SemanticSearchMode,
)


@dataclass(frozen=True)
class RepositoryBinding:
    """Detached projection of the existing persisted repository, not a second identity."""

    id: int
    name: str
    workspace_relative_path: str | None
    created_at: datetime


class RepositoryStore(Protocol):
    def register(self, name: str, path: str) -> RepositoryBinding: ...
    def get(self, repository_id: int) -> RepositoryBinding: ...
    def list_repositories(self, limit: int) -> Sequence[RepositoryBinding]: ...
    def files(
        self, repository_id: int, limit: int, offset: int
    ) -> list[RepositoryFileResponse]: ...
    def index_manifest(self, repository_id: int) -> IndexManifest: ...
    def apply_index_update(self, repository_id: int, update: IndexUpdate) -> IndexSummary: ...
    def search(
        self,
        repository_id: int,
        query: str,
        embedding: EmbeddingVector,
        *,
        hybrid: bool,
        top_k: int,
        semantic_mode: SemanticSearchMode = SemanticSearchMode.EXACT,
        include_symbols: bool = False,
    ) -> Sequence[RankedChunk]: ...
    def load_neighbors(
        self, repository_id: int, keys: Sequence[tuple[str, int]]
    ) -> Sequence[CodeChunk]: ...


def _binding(record: RepositoryRecord) -> RepositoryBinding:
    return RepositoryBinding(
        record.id, record.name, record.workspace_relative_path, record.created_at
    )


class PostgresRepositoryStore:
    def __init__(self, factory: sessionmaker[Session]):
        self.factory = factory

    def register(self, name: str, path: str) -> RepositoryBinding:
        with session_scope(self.factory) as session:
            record = session.scalar(select(RepositoryRecord).where(RepositoryRecord.name == name))
            if record is not None and record.workspace_relative_path != path:
                raise APIError(409, "repository_conflict", "Repository name is already bound.")
            if record is None:
                record = RepositoryRecord(name=name, workspace_relative_path=path)
                session.add(record)
                session.flush()
            return _binding(record)

    def get(self, repository_id: int) -> RepositoryBinding:
        with self.factory() as session:
            record = session.get(RepositoryRecord, repository_id)
            if record is None:
                raise RepositoryNotFoundError("Repository not found")
            return _binding(record)

    def list_repositories(self, limit: int) -> list[RepositoryBinding]:
        with self.factory() as session:
            records = session.scalars(
                select(RepositoryRecord).order_by(RepositoryRecord.created_at.desc()).limit(limit)
            )
            return [_binding(record) for record in records]

    def files(self, repository_id: int, limit: int, offset: int) -> list[RepositoryFileResponse]:
        with self.factory() as session:
            rows = session.execute(
                select(
                    RepositoryFileRecord.relative_path,
                    RepositoryFileRecord.language,
                    RepositoryFileRecord.size_bytes,
                    RepositoryFileRecord.line_count,
                )
                .where(RepositoryFileRecord.repository_id == repository_id)
                .order_by(RepositoryFileRecord.relative_path)
                .limit(limit)
                .offset(offset)
            )
            return [
                RepositoryFileResponse(
                    relative_path=row.relative_path,
                    language=row.language,
                    size_bytes=row.size_bytes,
                    line_count=row.line_count,
                )
                for row in rows
            ]

    def index_manifest(self, repository_id: int) -> IndexManifest:
        with self.factory() as session:
            return read_index_manifest(session, repository_id)

    def apply_index_update(self, repository_id: int, update: IndexUpdate) -> IndexSummary:
        """Apply the whole delta in one transaction: all of it lands, or none of it."""

        with session_scope(self.factory) as session:
            record = session.get(RepositoryRecord, repository_id)
            if record is None:
                raise RepositoryNotFoundError("Repository not found")
            if record.name != update.snapshot.name:
                raise APIError(409, "repository_conflict", "Repository binding changed.")
            return apply_index_update(session, repository_id, update)

    def search(
        self,
        repository_id: int,
        query: str,
        embedding: EmbeddingVector,
        *,
        hybrid: bool,
        top_k: int,
        semantic_mode: SemanticSearchMode = SemanticSearchMode.EXACT,
        include_symbols: bool = False,
    ) -> Sequence[RankedChunk]:
        with self.factory() as session:
            if include_symbols:
                return postgres_hybrid_symbol_search(
                    session,
                    repository_id,
                    query,
                    embedding,
                    top_k=top_k,
                    symbol_candidate_limit=DEFAULT_SYMBOL_CANDIDATE_LIMIT,
                    semantic_mode=semantic_mode,
                )
            if hybrid:
                return postgres_hybrid_search(
                    session,
                    repository_id,
                    query,
                    embedding,
                    top_k=top_k,
                    semantic_mode=semantic_mode,
                )
            return pgvector_semantic_search(
                session,
                repository_id,
                embedding,
                top_k=top_k,
                mode=semantic_mode,
            )

    def load_neighbors(
        self, repository_id: int, keys: Sequence[tuple[str, int]]
    ) -> Sequence[CodeChunk]:
        with self.factory() as session:
            return load_neighbor_chunks(session, repository_id, keys)
