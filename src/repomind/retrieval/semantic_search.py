"""In-memory semantic ranking over already embedded source chunks."""

from collections.abc import Sequence
from typing import Protocol

from repomind.retrieval.models import (
    EmbeddedChunk,
    EmbeddingVector,
    SemanticSearchResult,
)
from repomind.retrieval.similarity import RetrievalError, SimilarityError, cosine_similarity


class SemanticSearchError(RetrievalError):
    """Raised when semantic-search inputs are incompatible or invalid."""


class EmbeddingProvider(Protocol):
    """Smallest interface needed to embed a semantic-search query."""

    def embed_text(self, text: str) -> EmbeddingVector:
        """Return an embedding for exact input text."""


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise SemanticSearchError("top_k must be a positive integer")


def rank_by_similarity(
    query_embedding: EmbeddingVector,
    chunks: Sequence[EmbeddedChunk],
    *,
    top_k: int,
) -> list[SemanticSearchResult]:
    """Rank pre-embedded chunks by cosine score without making API calls.

    Python's stable sort preserves original corpus order when scores tie.
    """

    _validate_top_k(top_k)
    for embedded_chunk in chunks:
        path = embedded_chunk.chunk.relative_path.as_posix()
        if embedded_chunk.embedding.model != query_embedding.model:
            raise SemanticSearchError(
                f"Embedding model mismatch for {path!r}: query uses "
                f"{query_embedding.model!r}, but chunk uses "
                f"{embedded_chunk.embedding.model!r}"
            )
        if embedded_chunk.embedding.dimensions != query_embedding.dimensions:
            raise SimilarityError(
                f"Embedding vectors must have the same dimensions for {path!r}: "
                f"query has {query_embedding.dimensions}, but chunk has "
                f"{embedded_chunk.embedding.dimensions}"
            )

    scored: list[tuple[EmbeddedChunk, float]] = []
    for embedded_chunk in chunks:
        score = cosine_similarity(
            query_embedding.values,
            embedded_chunk.embedding.values,
        )
        scored.append((embedded_chunk, score))

    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
    return [
        SemanticSearchResult(chunk=item.chunk, score=score, rank=rank)
        for rank, (item, score) in enumerate(ranked, start=1)
    ]


def semantic_search(
    query: str,
    embedded_chunks: Sequence[EmbeddedChunk],
    embedding_provider: EmbeddingProvider,
    *,
    top_k: int = 5,
) -> list[SemanticSearchResult]:
    """Embed an exact query once, then rank already embedded chunks."""

    _validate_top_k(top_k)
    if not isinstance(query, str) or not query.strip():
        raise SemanticSearchError("query must not be empty or whitespace-only")
    if not embedded_chunks:
        return []

    query_embedding = embedding_provider.embed_text(query)
    return rank_by_similarity(query_embedding, embedded_chunks, top_k=top_k)
