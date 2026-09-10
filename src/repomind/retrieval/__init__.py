"""Embedding generation primitives for future retrieval milestones."""

from repomind.retrieval.embeddings import EmbeddingError, OpenAIEmbeddingClient
from repomind.retrieval.models import (
    EmbeddedChunk,
    EmbeddingBatchResult,
    EmbeddingConfig,
    EmbeddingUsage,
    EmbeddingVector,
)

__all__ = [
    "EmbeddedChunk",
    "EmbeddingBatchResult",
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingUsage",
    "EmbeddingVector",
    "OpenAIEmbeddingClient",
]
