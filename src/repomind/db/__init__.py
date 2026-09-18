"""PostgreSQL persistence plus semantic and hybrid retrieval APIs."""

from repomind.db.base import Base
from repomind.db.hybrid import postgres_hybrid_search, postgres_hybrid_symbol_search
from repomind.db.repositories import (
    PersistenceError,
    RepositoryNotFoundError,
    content_sha256,
    find_symbol_candidates,
    load_chunks,
    load_embedded_chunks,
    load_neighbor_chunks,
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
    "find_symbol_candidates",
    "load_chunks",
    "load_embedded_chunks",
    "load_neighbor_chunks",
    "persist_chunks",
    "persist_embedded_chunks",
    "persist_repository_snapshot",
    "pgvector_semantic_search",
    "postgres_hybrid_search",
    "postgres_hybrid_symbol_search",
    "session_scope",
]
