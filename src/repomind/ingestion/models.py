"""Pydantic models used by repository ingestion."""

from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel, Field, field_validator, model_validator

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


def _repository_relative_path(value: Path) -> Path:
    """Validate a repository-relative metadata path without touching the filesystem."""

    raw_path = str(value)
    posix_path = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if (
        not raw_path
        or value == Path(".")
        or value.is_absolute()
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError("path must be a safe repository-relative identifier")
    return value


class IngestionConfig(BaseModel):
    """Configuration for repository traversal and source-file loading."""

    ignored_directories: frozenset[str] = Field(
        default_factory=lambda: DEFAULT_IGNORED_DIRECTORIES,
    )
    ignored_file_patterns: tuple[str, ...] = Field(default_factory=tuple)
    max_file_size_bytes: int = Field(default=DEFAULT_MAX_FILE_SIZE_BYTES, gt=0)
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
    size_bytes: int = Field(ge=0)
    line_count: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: Path) -> Path:
        return _repository_relative_path(value)


class SkippedFile(BaseModel):
    """A lightweight explanation for a file that was excluded from a snapshot."""

    relative_path: Path
    reason: str

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: Path) -> Path:
        return _repository_relative_path(value)


class RepositorySnapshot(BaseModel):
    """A deterministic representation of an ingested repository."""

    root: Path
    name: str
    files: list[SourceFile]
    skipped: list[SkippedFile]
    file_count: int = Field(ge=0)
    total_size_bytes: int = Field(ge=0)
    languages: dict[str, int]

    @field_validator("languages")
    @classmethod
    def _validate_language_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("language counts must be nonnegative")
        return value


class ChunkingConfig(BaseModel):
    """Configuration for deterministic line-based source chunking.

    ``max_lines_per_chunk`` is the maximum number of logical source lines in a
    chunk. ``overlap_lines`` is the number of lines repeated between adjacent
    chunks, which helps preserve context across chunk boundaries.
    """

    max_lines_per_chunk: int = Field(default=120, gt=0)
    overlap_lines: int = Field(default=20, ge=0)

    @model_validator(mode="after")
    def _validate_overlap(self) -> "ChunkingConfig":
        if self.overlap_lines >= self.max_lines_per_chunk:
            raise ValueError("overlap_lines must be less than max_lines_per_chunk")
        return self


class CodeChunk(BaseModel):
    """A deterministic slice of a source file.

    ``start_line`` and ``end_line`` are 1-based and inclusive. ``chunk_index``
    restarts at zero for each source file.
    """

    relative_path: Path
    language: str | None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str
    chunk_index: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: Path) -> Path:
        return _repository_relative_path(value)

    @model_validator(mode="after")
    def _validate_line_range(self) -> "CodeChunk":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self
