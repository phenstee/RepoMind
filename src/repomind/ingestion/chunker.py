"""Versioned deterministic source chunking.

Chunking turns a complete :class:`SourceFile` into smaller ``CodeChunk``
objects so later retrieval stages can work with useful, citation-friendly
units. The original line algorithm remains the default while Python structural
chunking is dispatched as an explicit, independently benchmarkable strategy.
"""

from __future__ import annotations

from repomind.ingestion.models import (
    ChunkingConfig,
    ChunkingStrategy,
    ChunkKind,
    CodeChunk,
    RepositorySnapshot,
    SourceFile,
)


def _line_chunks(
    source_file: SourceFile,
    config: ChunkingConfig,
) -> list[CodeChunk]:
    """Apply the preserved line-v1 algorithm."""

    chunking_config = config
    lines = source_file.content.splitlines(keepends=True)

    if not lines:
        return []

    # A positive step is guaranteed by ChunkingConfig validation.
    step = chunking_config.max_lines_per_chunk - chunking_config.overlap_lines
    chunks: list[CodeChunk] = []
    start_index = 0

    while start_index < len(lines):
        end_index = min(
            start_index + chunking_config.max_lines_per_chunk,
            len(lines),
        )
        chunks.append(
            CodeChunk(
                relative_path=source_file.relative_path,
                language=source_file.language,
                start_line=start_index + 1,
                end_line=end_index,
                content="".join(lines[start_index:end_index]),
                chunk_index=len(chunks),
                chunking_strategy=ChunkingStrategy.LINE,
                chunk_kind=ChunkKind.LINE,
            )
        )

        if end_index == len(lines):
            break
        start_index += step

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
