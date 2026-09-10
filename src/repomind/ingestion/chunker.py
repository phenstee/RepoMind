"""Deterministic line-based code chunking.

Chunking turns a complete :class:`SourceFile` into smaller ``CodeChunk``
objects so later retrieval stages can work with useful, citation-friendly
units. The implementation is intentionally simple and deterministic; more
sophisticated syntax-aware chunking can be added later as a separate strategy.
"""

from __future__ import annotations

from repomind.ingestion.models import (
    ChunkingConfig,
    CodeChunk,
    RepositorySnapshot,
    SourceFile,
)


def chunk_source_file(
    source_file: SourceFile,
    config: ChunkingConfig | None = None,
) -> list[CodeChunk]:
    """Split one source file into ordered, deterministic chunks.

    An empty file produces an empty list because there is no useful text to
    embed or retrieve. Newline characters are preserved exactly by using
    ``splitlines(keepends=True)``.
    """

    chunking_config = config or ChunkingConfig()
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
            )
        )

        if end_index == len(lines):
            break
        start_index += step

    return chunks


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
