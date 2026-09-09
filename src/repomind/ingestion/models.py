"""Pydantic models used by repository ingestion."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

DEFAULT_IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".uv-cache",
        "node_modules",
        ".next",
        "dist",
        "build",
        "coverage",
        "target",
        "out",
        ".cache",
        ".idea",
        ".vscode",
    }
)

DEFAULT_MAX_FILE_SIZE_BYTES = 1_048_576


class IngestionConfig(BaseModel):
    """Configuration for repository traversal and source-file loading."""

    ignored_directories: frozenset[str] = Field(
        default_factory=lambda: DEFAULT_IGNORED_DIRECTORIES,
    )
    ignored_file_patterns: tuple[str, ...] = Field(default_factory=tuple)
    max_file_size_bytes: int = Field(default=DEFAULT_MAX_FILE_SIZE_BYTES, gt=0)
    binary_sample_bytes: int = Field(default=8192, gt=0)
    fallback_encoding: str = Field(default="cp1252", min_length=1)

    @field_validator("ignored_directories")
    @classmethod
    def _normalize_directories(cls, value: frozenset[str]) -> frozenset[str]:
        return frozenset(name.casefold() for name in value)

    @field_validator("ignored_file_patterns")
    @classmethod
    def _normalize_file_patterns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(pattern.casefold() for pattern in value)


class SourceFile(BaseModel):
    """A successfully loaded source file.

    ``relative_path`` is intentionally the only path exposed here. Absolute
    paths are confined to ingestion internals so environment-specific
    information does not leak into later stages.
    """

    relative_path: Path
    language: str | None
    content: str
    size_bytes: int
    line_count: int

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: Path) -> Path:
        if value.is_absolute():
            raise ValueError("relative_path must be relative to the repository root")
        return value


class SkippedFile(BaseModel):
    """A lightweight explanation for a file that was excluded from a snapshot."""

    relative_path: Path
    reason: str


class RepositorySnapshot(BaseModel):
    """A deterministic representation of an ingested repository."""

    root: Path
    name: str
    files: list[SourceFile]
    skipped: list[SkippedFile]
    file_count: int
    total_size_bytes: int
    languages: dict[str, int]
