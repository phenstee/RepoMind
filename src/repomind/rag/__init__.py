"""Configurable retrieval-augmented generation for repository questions."""

from repomind.rag.assembly import (
    ContextAssemblyError,
    InMemoryNeighborLoader,
    NeighborLoader,
    assemble_context,
)
from repomind.rag.context import RAGError, RankedChunk, build_repository_context
from repomind.rag.models import (
    AssembledContextChunk,
    AssembledContextResult,
    BuiltRepositoryContext,
    ContextAssemblyConfig,
    ContextOrigin,
    ContextSource,
    ContextStrategy,
    RAGConfig,
    RepositoryAnswer,
    SourceCitation,
)
from repomind.rag.pipeline import (
    Retriever,
    StructuredLLMProvider,
    answer_repository_question,
    answer_repository_question_with_retriever,
    build_rag_context,
)
from repomind.rag.tokens import estimate_tokens

__all__ = [
    "AssembledContextChunk",
    "AssembledContextResult",
    "BuiltRepositoryContext",
    "ContextAssemblyConfig",
    "ContextAssemblyError",
    "ContextOrigin",
    "ContextSource",
    "ContextStrategy",
    "InMemoryNeighborLoader",
    "NeighborLoader",
    "RAGConfig",
    "RAGError",
    "RankedChunk",
    "RepositoryAnswer",
    "Retriever",
    "SourceCitation",
    "StructuredLLMProvider",
    "answer_repository_question",
    "answer_repository_question_with_retriever",
    "assemble_context",
    "build_rag_context",
    "build_repository_context",
    "estimate_tokens",
]
