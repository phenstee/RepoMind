"""Repository discovery, filtering, and safe source-file loading."""

from repomind.ingestion.chunker import chunk_repository, chunk_source_file
from repomind.ingestion.language import (
    LANGUAGE_BY_EXTENSION,
    SUPPORTED_EXTENSIONS,
    is_supported_source_file,
    language_for_path,
)
from repomind.ingestion.models import (
    ChunkingConfig,
    CodeChunk,
    IngestionConfig,
    RepositorySnapshot,
    SkippedFile,
    SourceFile,
    validate_repository_relative_path,
)
from repomind.ingestion.repository import (
    InvalidRepositoryRootError,
    PathOutsideRepositoryError,
    RepositoryIngestionError,
    find_source_files,
    ingest_repository,
    load_source_file,
    resolve_repository_path,
    validate_repository_root,
)

__all__ = [
    "LANGUAGE_BY_EXTENSION",
    "SUPPORTED_EXTENSIONS",
    "ChunkingConfig",
    "CodeChunk",
    "IngestionConfig",
    "InvalidRepositoryRootError",
    "PathOutsideRepositoryError",
    "RepositoryIngestionError",
    "RepositorySnapshot",
    "SkippedFile",
    "SourceFile",
    "chunk_repository",
    "chunk_source_file",
    "find_source_files",
    "ingest_repository",
    "is_supported_source_file",
    "language_for_path",
    "load_source_file",
    "resolve_repository_path",
    "validate_repository_relative_path",
    "validate_repository_root",
]
