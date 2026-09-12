"""Pydantic inputs, outputs, and limits for RepoMind's read-only tools."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from repomind.ingestion import validate_repository_relative_path, validate_repository_root

DEFAULT_MAX_TOOL_FILE_BYTES = 1_048_576
DEFAULT_MAX_DIRECTORY_ENTRIES = 500
DEFAULT_MAX_SEARCH_RESULTS = 50
DEFAULT_MAX_DIFF_CHARS = 20_000


def _validate_tool_path(path: Path, *, allow_root: bool) -> Path:
    if allow_root and path == Path("."):
        return path
    return validate_repository_relative_path(path)


class ToolConfig(BaseModel):
    """Centralized hard limits for operations that can produce large observations."""

    model_config = ConfigDict(frozen=True)

    max_file_bytes: int = Field(default=DEFAULT_MAX_TOOL_FILE_BYTES, gt=0, strict=True)
    max_directory_entries: int = Field(
        default=DEFAULT_MAX_DIRECTORY_ENTRIES,
        gt=0,
        strict=True,
    )
    max_search_results: int = Field(
        default=DEFAULT_MAX_SEARCH_RESULTS,
        gt=0,
        strict=True,
    )
    max_diff_chars: int = Field(default=DEFAULT_MAX_DIFF_CHARS, gt=0, strict=True)


class ToolContext(BaseModel):
    """Immutable workspace boundary shared by all tools in one registry."""

    model_config = ConfigDict(frozen=True)

    repository_root: Path

    @field_validator("repository_root")
    @classmethod
    def _resolve_repository_root(cls, value: Path) -> Path:
        return validate_repository_root(value)


class ToolInput(BaseModel):
    """Base for schemas that reject undeclared future-LLM arguments."""

    model_config = ConfigDict(extra="forbid")


class ReadFileInput(ToolInput):
    path: Path
    start_line: int | None = Field(default=None, ge=1, strict=True)
    end_line: int | None = Field(default=None, ge=1, strict=True)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path) -> Path:
        return _validate_tool_path(value, allow_root=False)

    @model_validator(mode="after")
    def _validate_range(self) -> "ReadFileInput":
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.start_line > self.end_line
        ):
            raise ValueError("start_line must be less than or equal to end_line")
        return self


class ReadFileOutput(BaseModel):
    path: Path
    requested_start_line: int | None
    requested_end_line: int | None
    start_line: int = Field(ge=0)
    end_line: int = Field(ge=0)
    content: str
    total_lines: int = Field(ge=0)


class ListDirectoryInput(ToolInput):
    path: Path = Field(default_factory=lambda: Path("."))
    recursive: bool = False

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path) -> Path:
        return _validate_tool_path(value, allow_root=True)


class DirectoryEntry(BaseModel):
    path: Path
    type: Literal["file", "directory", "symlink"]
    size_bytes: int | None = Field(default=None, ge=0)


class ListDirectoryOutput(BaseModel):
    path: Path
    recursive: bool
    entries: list[DirectoryEntry]
    truncated: bool


class SearchCodeInput(ToolInput):
    query: str = Field(min_length=1, max_length=1_000)
    path: Path = Field(default_factory=lambda: Path("."))
    max_results: int = Field(default=DEFAULT_MAX_SEARCH_RESULTS, gt=0, strict=True)
    case_sensitive: bool = False

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return value

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path) -> Path:
        return _validate_tool_path(value, allow_root=True)


class SearchCodeMatch(BaseModel):
    path: Path
    line_number: int = Field(ge=1)
    line: str


class SearchCodeOutput(BaseModel):
    query: str
    matches: list[SearchCodeMatch]
    truncated: bool


class FindSymbolInput(ToolInput):
    symbol: str = Field(min_length=1, max_length=256)
    path: Path = Field(default_factory=lambda: Path("."))
    max_results: int = Field(default=25, gt=0, strict=True)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("symbol must not be empty or whitespace-only")
        return value

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path) -> Path:
        return _validate_tool_path(value, allow_root=True)


class SymbolMatch(BaseModel):
    path: Path
    line_number: int = Field(ge=1)
    line: str
    kind: str | None = None


class FindSymbolOutput(BaseModel):
    symbol: str
    matches: list[SymbolMatch]
    truncated: bool


class GitStatusInput(ToolInput):
    """The status tool accepts no user-controlled command arguments."""


class GitChangedFile(BaseModel):
    path: Path
    status: str = Field(min_length=1, max_length=2)


class GitStatusOutput(BaseModel):
    branch: str | None
    changed_files: list[GitChangedFile]
    clean: bool


class GitDiffInput(ToolInput):
    staged: bool = False
    path: Path | None = None
    max_chars: int | None = Field(default=None, gt=0, strict=True)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return _validate_tool_path(value, allow_root=True)


class GitDiffOutput(BaseModel):
    content: str
    truncated: bool
    staged: bool
    path: Path | None
