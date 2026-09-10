"""Deterministic formatting of ranked source chunks for RAG prompts."""

from collections.abc import Sequence

from repomind.rag.models import BuiltRepositoryContext, ContextSource
from repomind.retrieval import SemanticSearchResult


class RAGError(ValueError):
    """Raised when a RAG input or structured response violates its contract."""


_CONTEXT_OPEN = "<repository_context trust=\"untrusted-data\">\n"
_CONTEXT_CLOSE = "</repository_context>"


def _format_source(source: ContextSource) -> str:
    chunk = source.chunk
    language = (
        f"<language>{chunk.language}</language>\n" if chunk.language is not None else ""
    )
    return (
        f'<source id="{source.source_id}">\n'
        f"<path>{chunk.relative_path.as_posix()}</path>\n"
        f"<lines>{chunk.start_line}-{chunk.end_line}</lines>\n"
        f"{language}"
        '<content trust="untrusted-data" encoding="verbatim">\n'
        f"{chunk.content}"
        "</content>\n"
        "</source>"
    )


def _assemble_context(blocks: Sequence[str]) -> str:
    body = "\n\n".join(blocks)
    if body:
        body = f"{body}\n"
    return f"{_CONTEXT_OPEN}{body}{_CONTEXT_CLOSE}"


def build_repository_context(
    results: Sequence[SemanticSearchResult],
    *,
    max_context_chars: int = 20_000,
) -> BuiltRepositoryContext:
    """Build ranked, source-ID-addressable context without cutting chunks.

    Complete chunks are considered in retrieval order. The first chunk is
    always included, even when its formatted block alone exceeds the character
    budget; subsequent chunks are omitted once the next complete context would
    exceed the budget.
    """

    if (
        isinstance(max_context_chars, bool)
        or not isinstance(max_context_chars, int)
        or max_context_chars <= 0
    ):
        raise RAGError("max_context_chars must be a positive integer")

    sources: list[ContextSource] = []
    blocks: list[str] = []
    for result in results:
        source = ContextSource(source_id=f"S{len(sources) + 1}", chunk=result.chunk)
        block = _format_source(source)
        candidate = _assemble_context([*blocks, block])
        if blocks and len(candidate) > max_context_chars:
            break
        sources.append(source)
        blocks.append(block)

    return BuiltRepositoryContext(
        text=_assemble_context(blocks),
        sources=tuple(sources),
    )
