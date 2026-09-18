"""Correctness-first hybrid retrieval over a persisted repository."""

from sqlalchemy.orm import Session

from repomind.db.repositories import find_symbol_candidates, load_chunks, pgvector_semantic_search
from repomind.retrieval import (
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_RRF_K,
    DEFAULT_SYMBOL_CANDIDATE_LIMIT,
    BM25Index,
    EmbeddingVector,
    FusedSearchResult,
    HybridSearchError,
    HybridSearchResult,
    SemanticSearchMode,
    extract_identifier_candidates,
    fuse_ranked_sources,
    reciprocal_rank_fusion,
)
from repomind.retrieval.symbols import (
    is_exact_identifier_query,
    select_evidence_backed_symbol_fragments,
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


def postgres_hybrid_symbol_search(
    session: Session,
    repository_id: int,
    query: str,
    query_embedding: EmbeddingVector,
    *,
    top_k: int = 5,
    semantic_candidates: int | None = None,
    lexical_candidates: int | None = None,
    symbol_candidate_limit: int = DEFAULT_SYMBOL_CANDIDATE_LIMIT,
    rrf_k: float = DEFAULT_RRF_K,
    semantic_mode: SemanticSearchMode = SemanticSearchMode.EXACT,
) -> list[FusedSearchResult]:
    """Fuse selected pgvector ranking, rebuilt BM25, and bounded symbol matches.

    Symbol candidates come from one bounded, repository-scoped SQL query
    (:func:`repomind.db.repositories.find_symbol_candidates`); BM25 still
    rebuilds from every persisted chunk exactly as :func:`postgres_hybrid_search`
    already does, a known scalability limitation this milestone does not
    change. The original query text is embedded and tokenized unmodified;
    identifier extraction runs alongside it, never rewriting it.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise HybridSearchError("top_k must be a positive integer")
    if not isinstance(query, str) or not query.strip():
        raise HybridSearchError("query must not be empty or whitespace-only")

    semantic_depth = _candidate_depth(semantic_candidates, top_k, "semantic_candidates")
    lexical_depth = _candidate_depth(lexical_candidates, top_k, "lexical_candidates")
    chunks = load_chunks(session, repository_id)
    semantic_results = pgvector_semantic_search(
        session,
        repository_id,
        query_embedding,
        top_k=semantic_depth,
        mode=semantic_mode,
    )
    lexical_results = BM25Index.from_chunks(chunks).search(query, top_k=lexical_depth)
    candidates = extract_identifier_candidates(query)
    symbol_results = find_symbol_candidates(
        session, repository_id, candidates, limit=symbol_candidate_limit
    )
    if symbol_results and not is_exact_identifier_query(query, candidates):
        evidence_results = fuse_ranked_sources(
            {"semantic": semantic_results, "lexical": lexical_results},
            top_k=max(1, len(semantic_results) + len(lexical_results)),
            rrf_k=rrf_k,
        )
        symbol_results = select_evidence_backed_symbol_fragments(
            symbol_results, evidence_results
        )
    return fuse_ranked_sources(
        {"semantic": semantic_results, "lexical": lexical_results, "symbol": symbol_results},
        top_k=top_k,
        rrf_k=rrf_k,
    )
