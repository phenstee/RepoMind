"""Correctness-first hybrid retrieval over a persisted repository."""

from sqlalchemy.orm import Session

from repomind.db.repositories import load_chunks, pgvector_semantic_search
from repomind.retrieval import (
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_RRF_K,
    BM25Index,
    EmbeddingVector,
    HybridSearchError,
    HybridSearchResult,
    SemanticSearchMode,
    reciprocal_rank_fusion,
)


def _candidate_depth(candidate_count: int | None, top_k: int, name: str) -> int:
    if candidate_count is None:
        return top_k * DEFAULT_CANDIDATE_MULTIPLIER
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
    ):
        raise HybridSearchError(f"{name} must be a positive integer")
    return candidate_count


def postgres_hybrid_search(
    session: Session,
    repository_id: int,
    query: str,
    query_embedding: EmbeddingVector,
    *,
    top_k: int = 5,
    semantic_candidates: int | None = None,
    lexical_candidates: int | None = None,
    rrf_k: float = DEFAULT_RRF_K,
    semantic_mode: SemanticSearchMode = SemanticSearchMode.EXACT,
) -> list[HybridSearchResult]:
    """Combine selected pgvector ranking with BM25 rebuilt from persisted chunks."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise HybridSearchError("top_k must be a positive integer")
    if not isinstance(query, str) or not query.strip():
        raise HybridSearchError("query must not be empty or whitespace-only")

    semantic_depth = _candidate_depth(
        semantic_candidates,
        top_k,
        "semantic_candidates",
    )
    lexical_depth = _candidate_depth(
        lexical_candidates,
        top_k,
        "lexical_candidates",
    )
    chunks = load_chunks(session, repository_id)
    semantic_results = pgvector_semantic_search(
        session,
        repository_id,
        query_embedding,
        top_k=semantic_depth,
        mode=semantic_mode,
    )
    lexical_results = BM25Index.from_chunks(chunks).search(
        query,
        top_k=lexical_depth,
    )
    return reciprocal_rank_fusion(
        semantic_results,
        lexical_results,
        top_k=top_k,
        rrf_k=rrf_k,
    )
