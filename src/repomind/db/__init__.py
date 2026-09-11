"""PostgreSQL persistence and exact pgvector retrieval APIs."""

from repomind.db.base import Base
from repomind.db.repositories import (
    PersistenceError,
    RepositoryNotFoundError,
    content_sha256,
    load_embedded_chunks,
    persist_chunks,
    persist_embedded_chunks,
    persist_repository_snapshot,
    pgvector_semantic_search,
)
from repomind.db.session import (
    create_database_engine,
    create_session_factory,
    session_scope,
)

__all__ = [
    "Base",
    "PersistenceError",
    "RepositoryNotFoundError",
    "content_sha256",
    "create_database_engine",
    "create_session_factory",
    "load_embedded_chunks",
    "persist_chunks",
    "persist_embedded_chunks",
    "persist_repository_snapshot",
    "pgvector_semantic_search",
    "session_scope",
]
