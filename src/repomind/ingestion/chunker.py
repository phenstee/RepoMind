"""Versioned deterministic source chunking.

Chunking turns a complete :class:`SourceFile` into smaller ``CodeChunk``
objects so later retrieval stages can work with useful, citation-friendly
units. The line algorithm remains the default while Python structural chunking
is dispatched as an explicit, independently benchmarkable strategy. Both
strategies bound every chunk by ``max_chars_per_chunk`` so a single minified or
generated line can never become an unembeddable chunk.
"""

from __future__ import annotations

from repomind.ingestion.lines import split_oversized_line, split_source_lines
from repomind.ingestion.models import (
    ChunkingConfig,
    ChunkingStrategy,
    ChunkKind,
    CodeChunk,
    RepositorySnapshot,
    SourceFile,
)


def _line_chunk(
    source_file: SourceFile,
    *,
    start_line: int,
    end_line: int,
    content: str,
    chunk_index: int,
) -> CodeChunk:
    return CodeChunk(
        relative_path=source_file.relative_path,
        language=source_file.language,
        start_line=start_line,
        end_line=end_line,
        content=content,
        chunk_index=chunk_index,
        chunking_strategy=ChunkingStrategy.LINE,
        chunk_kind=ChunkKind.LINE,
    )


def _line_chunks(
    source_file: SourceFile,
    config: ChunkingConfig,
) -> list[CodeChunk]:
    """Apply the line-v1 windows, bounded by ``max_chars_per_chunk``.

    A window holds at most ``max_lines_per_chunk`` lines and, whenever whole
    lines allow it, at most ``max_chars_per_chunk`` characters. A window cut
    short by the character cap keeps the configured overlap ratio rather than
    the absolute overlap, so dense files are not duplicated many times over. A
    single line longer than the cap (minified JavaScript, generated data) is
    emitted as consecutive exact character pieces that all cite that one line.
    Files whose original line-v1 windows already fit the cap keep exactly the
    original boundaries.
    """

    lines = split_source_lines(source_file.content)
    if not lines:
        return []

    max_lines = config.max_lines_per_chunk
    max_chars = config.max_chars_per_chunk
    chunks: list[CodeChunk] = []
    start_index = 0

    while start_index < len(lines):
        end_index = start_index
        chars = 0
        while (
            end_index < len(lines)
            and end_index - start_index < max_lines
            and chars + len(lines[end_index]) <= max_chars
        ):
            chars += len(lines[end_index])
            end_index += 1

        if end_index == start_index:
            # One line alone exceeds the character cap. Overlap would only
            # duplicate pieces of it, so the next window starts after it.
            for piece in split_oversized_line(lines[start_index], max_chars):
                chunks.append(
                    _line_chunk(
                        source_file,
                        start_line=start_index + 1,
                        end_line=start_index + 1,
                        content=piece,
                        chunk_index=len(chunks),
                    )
                )
            start_index += 1
            continue

        chunks.append(
            _line_chunk(
                source_file,
                start_line=start_index + 1,
                end_line=end_index,
                content="".join(lines[start_index:end_index]),
                chunk_index=len(chunks),
            )
        )
        if end_index == len(lines):
            break
        # A full window overlaps by exactly overlap_lines (the original line-v1
        # step); a character-capped window overlaps proportionally. Because
        # ChunkingConfig guarantees overlap_lines < max_lines_per_chunk, the
        # overlap is always smaller than the window, so the scan progresses.
        overlap = (end_index - start_index) * config.overlap_lines // max_lines
        # The overlap must also leave room for the first new line; otherwise
        # the next window would only repeat lines this window already holds.
        next_line_chars = len(lines[end_index])
        overlap_chars = sum(len(line) for line in lines[end_index - overlap : end_index])
        while overlap and overlap_chars + next_line_chars > max_chars:
            overlap_chars -= len(lines[end_index - overlap])
            overlap -= 1
        start_index = end_index - overlap

    return chunks


def chunk_source_file(
    source_file: SourceFile,
    config: ChunkingConfig | None = None,
) -> list[CodeChunk]:
    """Split one source file using the explicitly selected strategy.

    An empty file produces an empty list. Both strategies preserve source text
    exactly; structural parsing failures fall back to the line-v1 boundaries.
    """

    chunking_config = config or ChunkingConfig()
    if chunking_config.strategy is ChunkingStrategy.STRUCTURAL:
        from repomind.ingestion.structural import chunk_python_source

        return chunk_python_source(source_file, chunking_config)
    return _line_chunks(source_file, chunking_config)


def chunk_repository(
    repository: RepositorySnapshot,
    config: ChunkingConfig | None = None,
) -> list[CodeChunk]:
    """Chunk every file in a repository snapshot in stable order.

    Files are processed in the order already present in ``repository.files``.
    ``chunk_index`` restarts at zero for each file.
    """

    chunks: list[CodeChunk] = []
    for source_file in repository.files:
        chunks.extend(chunk_source_file(source_file, config))
    return chunks
