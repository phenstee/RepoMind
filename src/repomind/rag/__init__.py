"""Configurable retrieval-augmented generation for repository questions."""

from repomind.rag.context import RAGError, RankedChunk, build_repository_context
from repomind.rag.models import (
    BuiltRepositoryContext,
    ContextSource,
    RAGConfig,
    RepositoryAnswer,
    SourceCitation,
)
from repomind.rag.pipeline import (
    Retriever,
    StructuredLLMProvider,
    answer_repository_question,
    answer_repository_question_with_retriever,
)

__all__ = [
    "BuiltRepositoryContext",
    "ContextSource",
    "RAGConfig",
    "RAGError",
    "RankedChunk",
    "RepositoryAnswer",
    "Retriever",
    "SourceCitation",
    "StructuredLLMProvider",
    "answer_repository_question",
    "answer_repository_question_with_retriever",
    "build_repository_context",
]
