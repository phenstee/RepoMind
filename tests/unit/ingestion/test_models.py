"""Tests for ingestion model invariants."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.ingestion.models import (
    CodeChunk,
    RepositorySnapshot,
    SkippedFile,
    SourceFile,
)


def _model_with_path(model_type: type, path: str):
    if model_type is SourceFile:
        return SourceFile(
            relative_path=path,
            language="python",
            content="x\n",
            size_bytes=2,
            line_count=1,
        )
    if model_type is CodeChunk:
        return CodeChunk(
            relative_path=path,
            language="python",
            start_line=1,
            end_line=1,
            content="x\n",
            chunk_index=0,
        )
    return SkippedFile(relative_path=path, reason="test")


@pytest.mark.parametrize("model_type", [SourceFile, CodeChunk, SkippedFile])
@pytest.mark.parametrize(
    "path",
    [
        "src/app.py",
        "frontend/page.tsx",
        ".github/workflows/ci.yml",
    ],
)
def test_repository_relative_paths_accept_safe_nested_paths(model_type: type, path: str) -> None:
    model = _model_with_path(model_type, path)

    assert model.relative_path == Path(path)


@pytest.mark.parametrize("model_type", [SourceFile, CodeChunk, SkippedFile])
@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "../outside.py",
        "src/../../outside.py",
        r"..\outside.py",
        "C:outside.py",
        r"C:\outside.py",
        "/outside.py",
        r"\outside.py",
        r"\\server\share\outside.py",
    ],
)
def test_repository_relative_paths_reject_unsafe_paths(model_type: type, path: str) -> None:
    with pytest.raises(ValidationError):
        _model_with_path(model_type, path)


@pytest.mark.parametrize("field", ["size_bytes", "line_count"])
def test_source_file_counts_must_be_nonnegative(field: str) -> None:
    values = {
        "relative_path": "src/app.py",
        "language": "python",
        "content": "",
        "size_bytes": 0,
        "line_count": 0,
    }
    values[field] = -1

    with pytest.raises(ValidationError):
        SourceFile(**values)


@pytest.mark.parametrize("field", ["file_count", "total_size_bytes"])
def test_repository_snapshot_counts_must_be_nonnegative(field: str) -> None:
    values = {
        "root": Path("repo"),
        "name": "repo",
        "files": [],
        "skipped": [],
        "file_count": 0,
        "total_size_bytes": 0,
        "languages": {},
    }
    values[field] = -1

    with pytest.raises(ValidationError):
        RepositorySnapshot(**values)


def test_repository_snapshot_language_counts_must_be_nonnegative() -> None:
    with pytest.raises(ValidationError):
        RepositorySnapshot(
            root=Path("repo"),
            name="repo",
            files=[],
            skipped=[],
            file_count=0,
            total_size_bytes=0,
            languages={"python": -1},
        )


def test_repository_snapshot_accepts_zero_language_count() -> None:
    snapshot = RepositorySnapshot(
        root=Path("repo"),
        name="repo",
        files=[],
        skipped=[],
        file_count=0,
        total_size_bytes=0,
        languages={"python": 0},
    )

    assert snapshot.languages == {"python": 0}
