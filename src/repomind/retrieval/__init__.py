"""Embedding generation primitives for future retrieval milestones."""

from repomind.retrieval.embeddings import EmbeddingError, OpenAIEmbeddingClient
from repomind.retrieval.models import (
    EmbeddedChunk,
    EmbeddingBatchResult,
    EmbeddingConfig,
    EmbeddingUsage,
    EmbeddingVector,
    SemanticSearchResult,
)
from repomind.retrieval.semantic_search import (
    EmbeddingProvider,
    SemanticSearchError,
    rank_by_similarity,
    semantic_search,
)
from repomind.retrieval.similarity import RetrievalError, SimilarityError, cosine_similarity

__all__ = [
    "EmbeddedChunk",
    "EmbeddingBatchResult",
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingUsage",
    "EmbeddingVector",
    "OpenAIEmbeddingClient",
    "RetrievalError",
    "SemanticSearchError",
    "SemanticSearchResult",
    "SimilarityError",
    "cosine_similarity",
    "rank_by_similarity",
    "semantic_search",
]
