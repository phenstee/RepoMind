"""Bounded navigation over RepoMind's persisted retrieval stack.

This module deliberately owns no database session, embedding client, or
FastAPI state: it only defines the small callable boundary the application
layer composes an implementation against, plus the handler that turns
whatever that implementation returns into bounded navigation metadata.

Indexed results are navigation hints, not authoritative evidence. The
persisted index may not reflect the current working tree, so the handler
never returns chunk content - only location metadata a caller can pass to
``read_file`` to observe current source.
"""

from collections.abc import Sequence
from typing import Protocol

from repomind.ingestion import CodeChunk
from repomind.tools.models import (
    IndexedCodeLocation,
    IndexedCodeSearchInput,
    IndexedCodeSearchOutput,
)


class RankedLocation(Protocol):
    """Minimal shape this handler needs from any retrieval result."""

    chunk: CodeChunk
    rank: int


class IndexedRetriever(Protocol):
    """Bounded ranked-location lookup backing the indexed navigation tool."""

    def __call__(self, query: str, *, top_k: int) -> Sequence[RankedLocation]:
        """Return ranked candidate locations for the exact query, most relevant first."""


def _sources(result: RankedLocation) -> tuple[str, ...]:
    # Duck-typed on purpose: FusedSearchResult exposes these as properties,
    # but a plain SemanticSearchResult/BM25SearchResult/SymbolSearchResult
    # (or a test fake) does not, and that is not an error.
    labels = (
        ("semantic", getattr(result, "semantic_rank", None)),
        ("bm25", getattr(result, "lexical_rank", None)),
        ("symbol", getattr(result, "symbol_rank", None)),
    )
    found = tuple(name for name, rank in labels if rank is not None)
    return found


def indexed_code_search(
    retriever: IndexedRetriever,
    arguments: IndexedCodeSearchInput,
) -> IndexedCodeSearchOutput:
    """Locate candidate locations; the embedding call happens only here, lazily."""

    results = retriever(arguments.query, top_k=arguments.max_results)
    locations = [
        IndexedCodeLocation(
            relative_path=result.chunk.relative_path,
            start_line=result.chunk.start_line,
            end_line=result.chunk.end_line,
            rank=result.rank,
            chunk_kind=result.chunk.chunk_kind.value,
            qualified_symbol_name=result.chunk.qualified_symbol_name,
            sources=_sources(result),
        )
        for result in results
    ]
    return IndexedCodeSearchOutput(
        query=arguments.query,
        locations=locations,
        result_count=len(locations),
    )
