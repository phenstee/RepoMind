"""Semantic, lexical, and hybrid repository retrieval primitives."""

from repomind.retrieval.bm25 import BM25Error, BM25Index
from repomind.retrieval.embeddings import (
    EmbeddingError,
    OpenAIEmbeddingClient,
    embedding_text_for_chunk,
)
from repomind.retrieval.hybrid import (
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_RRF_K,
    ChunkIdentity,
    HybridSearchError,
    chunk_identity,
    hybrid_search,
    reciprocal_rank_fusion,
)
from repomind.retrieval.models import (
    BM25Config,
    BM25SearchResult,
    EmbeddedChunk,
    EmbeddingBatchResult,
    EmbeddingConfig,
    EmbeddingTextStrategy,
    EmbeddingUsage,
    EmbeddingVector,
    HybridSearchResult,
    RankedChunk,
    RerankedSearchResult,
    RerankingConfig,
    SemanticSearchMode,
    SemanticSearchResult,
)
from repomind.retrieval.reranking import (
    LLMReranker,
    Reranker,
    RerankingError,
    hybrid_search_with_reranking,
)
from repomind.retrieval.semantic_search import (
    EmbeddingProvider,
    SemanticSearchError,
    rank_by_similarity,
    semantic_search,
)
from repomind.retrieval.similarity import RetrievalError, SimilarityError, cosine_similarity
from repomind.retrieval.tokenization import tokenize_code

__all__ = [
    "DEFAULT_CANDIDATE_MULTIPLIER",
    "DEFAULT_RRF_K",
    "BM25Config",
    "BM25Error",
    "BM25Index",
    "BM25SearchResult",
    "ChunkIdentity",
    "EmbeddedChunk",
    "EmbeddingBatchResult",
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingTextStrategy",
    "EmbeddingUsage",
    "EmbeddingVector",
    "HybridSearchError",
    "HybridSearchResult",
    "LLMReranker",
    "OpenAIEmbeddingClient",
    "RankedChunk",
    "RerankedSearchResult",
    "Reranker",
    "RerankingConfig",
    "RerankingError",
    "RetrievalError",
    "SemanticSearchError",
    "SemanticSearchMode",
    "SemanticSearchResult",
    "SimilarityError",
    "chunk_identity",
    "cosine_similarity",
    "embedding_text_for_chunk",
    "hybrid_search",
    "hybrid_search_with_reranking",
    "rank_by_similarity",
    "reciprocal_rank_fusion",
    "semantic_search",
    "tokenize_code",
]
