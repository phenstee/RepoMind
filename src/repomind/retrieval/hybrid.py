"""Rank-based fusion of semantic, lexical, and other retrieval results."""

from collections.abc import Mapping, Sequence
from math import isfinite

from repomind.ingestion import CodeChunk
from repomind.retrieval.bm25 import BM25Index
from repomind.retrieval.identifiers import extract_identifier_candidates
from repomind.retrieval.models import (
    BM25SearchResult,
    EmbeddedChunk,
    FusedSearchResult,
    HybridSearchResult,
    RankedChunk,
    SemanticSearchResult,
)
from repomind.retrieval.semantic_search import EmbeddingProvider, semantic_search
from repomind.retrieval.similarity import RetrievalError
from repomind.retrieval.symbols import (
    DEFAULT_SYMBOL_CANDIDATE_LIMIT,
    is_exact_identifier_query,
    select_evidence_backed_symbol_fragments,
    symbol_search,
)

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


def fuse_ranked_sources(
    sources: Mapping[str, Sequence[RankedChunk]],
    *,
    top_k: int,
    rrf_k: float = DEFAULT_RRF_K,
) -> list[FusedSearchResult]:
    """Fuse an arbitrary number of named rankings with ``sum(1 / (rrf_k + rank))``.

    Each source is identified by name (for example ``"semantic"``,
    ``"lexical"``, ``"symbol"``) so callers can recover which sources
    contributed to a given result through ``FusedSearchResult.source_ranks``.
    A source with no results (or an absent key) simply contributes nothing;
    fusing with an empty additional source reproduces the result of fusing
    without it, identity for identity, order for order.
    """

    _validate_positive_integer(top_k, "top_k")
    _validate_rrf_k(rrf_k)
    if not sources:
        raise HybridSearchError("fuse_ranked_sources requires at least one named source")

    chunks: dict[ChunkIdentity, CodeChunk] = {}
    ranks_by_source: dict[str, dict[ChunkIdentity, int]] = {name: {} for name in sources}
    scores: dict[ChunkIdentity, float] = {}

    def remember_chunk(identity: ChunkIdentity, chunk: CodeChunk) -> None:
        existing = chunks.get(identity)
        if existing is not None and existing != chunk:
            raise HybridSearchError(f"Conflicting chunks share source identity: {identity!r}")
        chunks[identity] = chunk

    for name, results in sources.items():
        source_ranks = ranks_by_source[name]
        for result in results:
            identity = chunk_identity(result.chunk)
            if identity in source_ranks:
                raise HybridSearchError(f"Duplicate {name} chunk identity: {identity!r}")
            remember_chunk(identity, result.chunk)
            source_ranks[identity] = result.rank
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (rrf_k + result.rank)

    def sort_key(identity: ChunkIdentity) -> tuple[float, int, ChunkIdentity]:
        individual_ranks = (
            source_ranks[identity]
            for source_ranks in ranks_by_source.values()
            if identity in source_ranks
        )
        return (-scores[identity], min(individual_ranks), identity)

    ordered = sorted(scores, key=sort_key)[:top_k]
    return [
        FusedSearchResult(
            chunk=chunks[identity],
            rank=rank,
            fusion_score=scores[identity],
            source_ranks={
                name: source_ranks[identity]
                for name, source_ranks in ranks_by_source.items()
                if identity in source_ranks
            },
        )
        for rank, identity in enumerate(ordered, start=1)
    ]


def reciprocal_rank_fusion(
    semantic_results: Sequence[SemanticSearchResult],
    lexical_results: Sequence[BM25SearchResult],
    *,
    top_k: int,
    rrf_k: float = DEFAULT_RRF_K,
) -> list[HybridSearchResult]:
    """Fuse rankings with ``sum(1 / (rrf_k + rank))`` per chunk.

    A thin, behavior-preserving projection of :func:`fuse_ranked_sources` onto
    the original two-named-source, two-fixed-field result shape.
    """

    fused = fuse_ranked_sources(
        {"semantic": semantic_results, "lexical": lexical_results},
        top_k=top_k,
        rrf_k=rrf_k,
    )
    return [
        HybridSearchResult(
            chunk=result.chunk,
            rank=result.rank,
            fusion_score=result.fusion_score,
            semantic_rank=result.semantic_rank,
            lexical_rank=result.lexical_rank,
        )
        for result in fused
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


def hybrid_symbol_search(
    query: str,
    embedded_chunks: Sequence[EmbeddedChunk],
    bm25_index: BM25Index,
    embedding_provider: EmbeddingProvider,
    chunks: Sequence[CodeChunk],
    *,
    top_k: int = 5,
    semantic_candidates: int | None = None,
    lexical_candidates: int | None = None,
    symbol_candidate_limit: int | None = None,
    rrf_k: float = DEFAULT_RRF_K,
) -> list[FusedSearchResult]:
    """Fuse semantic, BM25, and bounded symbol candidates for one exact query.

    ``chunks`` is the same in-memory corpus already backing ``bm25_index``;
    it is only scanned up to ``symbol_candidate_limit`` matches, never used to
    rebuild a second full-corpus index. The original query text is embedded
    and tokenized exactly as in :func:`hybrid_search`; identifier extraction
    runs in parallel and never rewrites it.
    """

    _validate_positive_integer(top_k, "top_k")
    _validate_rrf_k(rrf_k)
    if not isinstance(query, str) or not query.strip():
        raise HybridSearchError("query must not be empty or whitespace-only")

    semantic_depth = _candidate_depth(semantic_candidates, top_k, "semantic_candidates")
    lexical_depth = _candidate_depth(lexical_candidates, top_k, "lexical_candidates")
    resolved_symbol_limit = (
        DEFAULT_SYMBOL_CANDIDATE_LIMIT if symbol_candidate_limit is None else symbol_candidate_limit
    )

    semantic_results = semantic_search(query, embedded_chunks, embedding_provider, top_k=semantic_depth)
    lexical_results = bm25_index.search(query, top_k=lexical_depth)
    candidates = extract_identifier_candidates(query)
    symbol_results = symbol_search(candidates, chunks, limit=resolved_symbol_limit)
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
