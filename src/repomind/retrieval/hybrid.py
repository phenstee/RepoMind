"""Rank-based fusion of semantic and lexical retrieval results."""

from collections.abc import Sequence
from math import isfinite

from repomind.ingestion import CodeChunk
from repomind.retrieval.bm25 import BM25Index
from repomind.retrieval.models import (
    BM25SearchResult,
    EmbeddedChunk,
    HybridSearchResult,
    SemanticSearchResult,
)
from repomind.retrieval.semantic_search import EmbeddingProvider, semantic_search
from repomind.retrieval.similarity import RetrievalError

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 4

ChunkIdentity = tuple[str, int, int, int]


class HybridSearchError(RetrievalError):
    """Raised when hybrid-search or fusion inputs are invalid."""


def chunk_identity(chunk: CodeChunk) -> ChunkIdentity:
    """Return a stable source-domain identity that never depends on a DB ID."""

    return (
        chunk.relative_path.as_posix(),
        chunk.chunk_index,
        chunk.start_line,
        chunk.end_line,
    )


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HybridSearchError(f"{name} must be a positive integer")


def _candidate_depth(candidate_count: int | None, top_k: int, name: str) -> int:
    if candidate_count is None:
        return top_k * DEFAULT_CANDIDATE_MULTIPLIER
    _validate_positive_integer(candidate_count, name)
    return candidate_count


def _validate_rrf_k(rrf_k: float) -> None:
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, (int, float)):
        raise HybridSearchError("rrf_k must be a positive finite number")
    if not isfinite(rrf_k) or rrf_k <= 0:
        raise HybridSearchError("rrf_k must be a positive finite number")


def reciprocal_rank_fusion(
    semantic_results: Sequence[SemanticSearchResult],
    lexical_results: Sequence[BM25SearchResult],
    *,
    top_k: int,
    rrf_k: float = DEFAULT_RRF_K,
) -> list[HybridSearchResult]:
    """Fuse rankings with ``sum(1 / (rrf_k + rank))`` per chunk."""

    _validate_positive_integer(top_k, "top_k")
    _validate_rrf_k(rrf_k)

    chunks: dict[ChunkIdentity, CodeChunk] = {}
    semantic_ranks: dict[ChunkIdentity, int] = {}
    lexical_ranks: dict[ChunkIdentity, int] = {}
    scores: dict[ChunkIdentity, float] = {}

    def remember_chunk(identity: ChunkIdentity, chunk: CodeChunk) -> None:
        existing = chunks.get(identity)
        if existing is not None and existing != chunk:
            raise HybridSearchError(f"Conflicting chunks share source identity: {identity!r}")
        chunks[identity] = chunk

    for result in semantic_results:
        identity = chunk_identity(result.chunk)
        if identity in semantic_ranks:
            raise HybridSearchError(f"Duplicate semantic chunk identity: {identity!r}")
        remember_chunk(identity, result.chunk)
        semantic_ranks[identity] = result.rank
        scores[identity] = scores.get(identity, 0.0) + 1.0 / (rrf_k + result.rank)

    for result in lexical_results:
        identity = chunk_identity(result.chunk)
        if identity in lexical_ranks:
            raise HybridSearchError(f"Duplicate lexical chunk identity: {identity!r}")
        remember_chunk(identity, result.chunk)
        lexical_ranks[identity] = result.rank
        scores[identity] = scores.get(identity, 0.0) + 1.0 / (rrf_k + result.rank)

    def sort_key(identity: ChunkIdentity) -> tuple[float, int, ChunkIdentity]:
        individual_ranks = (
            rank
            for rank in (semantic_ranks.get(identity), lexical_ranks.get(identity))
            if rank is not None
        )
        return (-scores[identity], min(individual_ranks), identity)

    ordered = sorted(scores, key=sort_key)[:top_k]
    return [
        HybridSearchResult(
            chunk=chunks[identity],
            rank=rank,
            fusion_score=scores[identity],
            semantic_rank=semantic_ranks.get(identity),
            lexical_rank=lexical_ranks.get(identity),
        )
        for rank, identity in enumerate(ordered, start=1)
    ]


def hybrid_search(
    query: str,
    embedded_chunks: Sequence[EmbeddedChunk],
    bm25_index: BM25Index,
    embedding_provider: EmbeddingProvider,
    *,
    top_k: int = 5,
    semantic_candidates: int | None = None,
    lexical_candidates: int | None = None,
    rrf_k: float = DEFAULT_RRF_K,
) -> list[HybridSearchResult]:
    """Embed a query once, retrieve both rankings, then combine them with RRF."""

    _validate_positive_integer(top_k, "top_k")
    _validate_rrf_k(rrf_k)
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
    semantic_results = semantic_search(
        query,
        embedded_chunks,
        embedding_provider,
        top_k=semantic_depth,
    )
    lexical_results = bm25_index.search(query, top_k=lexical_depth)
    return reciprocal_rank_fusion(
        semantic_results,
        lexical_results,
        top_k=top_k,
        rrf_k=rrf_k,
    )
