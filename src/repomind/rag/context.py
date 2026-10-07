"""Deterministic formatting of ranked source chunks for RAG prompts."""

from collections.abc import Sequence
from html import escape

from repomind.ingestion import split_source_lines
from repomind.rag.models import BuiltRepositoryContext, ContextSource
from repomind.retrieval.models import RankedChunk


class RAGError(ValueError):
    """Raised when a RAG input or structured response violates its contract."""


_CONTEXT_OPEN = "<repository_context trust=\"untrusted-data\">\n"
_CONTEXT_CLOSE = "</repository_context>"
# A lone first block is wrapped as OPEN + block + "\n" + CLOSE.
_SINGLE_BLOCK_OVERHEAD = len(_CONTEXT_OPEN) + 1 + len(_CONTEXT_CLOSE)
_CONTENT_OPEN = '<content trust="untrusted-data" encoding="xml-escaped">\n'
_TRUNCATED_CONTENT_OPEN = (
    '<content trust="untrusted-data" encoding="xml-escaped" truncated="true">\n'
)
_ESCAPED_LENGTH = {"&": len("&amp;"), "<": len("&lt;"), ">": len("&gt;")}


def escape_untrusted_text(text: str) -> str:
    """Escape ``&``, ``<``, and ``>`` so repository text cannot close its element.

    Without this, a file containing ``</content></source></repository_context>``
    would end the untrusted block early and let the rest of the file pose as
    trusted prompt text. Quotes are left alone because untrusted text is only
    ever placed in element bodies, never in attribute values.
    """

    return escape(text, quote=False)


def format_source_block(source: ContextSource) -> str:
    """Format one source exactly as it appears inside repository context.

    Every repository-controlled value (path, language, content) is XML-escaped.
    A truncated source is marked on its content element, and its line range
    already covers only the text that is shown.
    """

    chunk = source.chunk
    language = (
        f"<language>{escape_untrusted_text(chunk.language)}</language>\n"
        if chunk.language is not None
        else ""
    )
    content_open = _TRUNCATED_CONTENT_OPEN if source.truncated else _CONTENT_OPEN
    return (
        f'<source id="{source.source_id}">\n'
        f"<path>{escape_untrusted_text(chunk.relative_path.as_posix())}</path>\n"
        f"<lines>{chunk.start_line}-{chunk.end_line}</lines>\n"
        f"{language}"
        f"{content_open}"
        f"{escape_untrusted_text(chunk.content)}"
        "</content>\n"
        "</source>"
    )


def _escaped_prefix_length(content: str, max_escaped_chars: int) -> int:
    used = 0
    for index, character in enumerate(content):
        used += _ESCAPED_LENGTH.get(character, 1)
        if used > max_escaped_chars:
            return index
    return len(content)


def truncate_source_to_fit(source: ContextSource, max_block_chars: int) -> ContextSource:
    """Cut a source's content so its formatted block fits ``max_block_chars``.

    Truncation keeps a prefix of the exact source text, preferring whole lines,
    and shrinks ``end_line`` to the last line actually shown, so a citation of
    the truncated source never claims lines the model did not see. At least
    one character is always kept; only a budget smaller than the block's own
    markup can therefore be exceeded. A source that already fits is returned
    unchanged.
    """

    if len(format_source_block(source)) <= max_block_chars:
        return source
    chunk = source.chunk
    if not chunk.content:
        return source

    # The line range can only shrink, so the empty block bounds the markup.
    markup = format_source_block(
        source.model_copy(
            update={"chunk": chunk.model_copy(update={"content": ""}), "truncated": True}
        )
    )
    keep = max(1, _escaped_prefix_length(chunk.content, max_block_chars - len(markup)))
    prefix = chunk.content[:keep]
    lines = split_source_lines(prefix)
    if len(lines) > 1 and not lines[-1].endswith(("\n", "\r")):
        prefix = prefix[: len(prefix) - len(lines[-1])]
        lines.pop()
    end_line = min(chunk.end_line, chunk.start_line + len(lines) - 1)
    return source.model_copy(
        update={
            "chunk": chunk.model_copy(update={"content": prefix, "end_line": end_line}),
            "truncated": True,
        }
    )


def _assemble_context(blocks: Sequence[str]) -> str:
    body = "\n\n".join(blocks)
    if body:
        body = f"{body}\n"
    return f"{_CONTEXT_OPEN}{body}{_CONTEXT_CLOSE}"


def build_repository_context(
    results: Sequence[RankedChunk],
    *,
    max_context_chars: int = 20_000,
) -> BuiltRepositoryContext:
    """Build ranked, source-ID-addressable context within a character budget.

    Complete chunks are considered in retrieval order. The first chunk is
    always included; when its formatted block alone would exceed the budget,
    it is truncated (and marked as such) so the whole context fits.
    Subsequent chunks are never cut: they are omitted once the next complete
    context would exceed the budget.
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
        source = ContextSource(
            source_id=f"S{len(sources) + 1}",
            chunk=result.chunk,
            # Assembled context may already have truncated this chunk to fit
            # a token budget; keep that marker visible to the model.
            truncated=bool(getattr(result, "truncated", False)),
        )
        block = format_source_block(source)
        candidate = _assemble_context([*blocks, block])
        if len(candidate) > max_context_chars:
            if blocks:
                break
            source = truncate_source_to_fit(
                source, max_context_chars - _SINGLE_BLOCK_OVERHEAD
            )
            block = format_source_block(source)
        sources.append(source)
        blocks.append(block)

    return BuiltRepositoryContext(
        text=_assemble_context(blocks),
        sources=tuple(sources),
    )
