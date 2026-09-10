"""Tests for deterministic line-based source chunking."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.ingestion.chunker import chunk_repository, chunk_source_file
from repomind.ingestion.models import (
    ChunkingConfig,
    RepositorySnapshot,
    SourceFile,
)


def _source(
    content: str,
    *,
    path: str = "src/app.py",
    language: str = "python",
) -> SourceFile:
    return SourceFile(
        relative_path=Path(path),
        language=language,
        content=content,
        size_bytes=len(content.encode("utf-8")),
        line_count=len(content.splitlines()),
    )


def test_default_chunking_config() -> None:
    config = ChunkingConfig()

    assert config.max_lines_per_chunk == 120
    assert config.overlap_lines == 20


def test_chunking_config_rejects_invalid_max_size() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(max_lines_per_chunk=0)
    with pytest.raises(ValidationError):
        ChunkingConfig(max_lines_per_chunk=-1)


def test_chunking_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(overlap_lines=-1)
    with pytest.raises(ValidationError):
        ChunkingConfig(max_lines_per_chunk=5, overlap_lines=5)
    with pytest.raises(ValidationError):
        ChunkingConfig(max_lines_per_chunk=5, overlap_lines=6)


def test_empty_source_file_returns_no_chunks() -> None:
    assert chunk_source_file(_source("")) == []


def test_single_line_file() -> None:
    chunks = chunk_source_file(
        _source("a"),
        ChunkingConfig(max_lines_per_chunk=5, overlap_lines=0),
    )

    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 1
    assert chunks[0].content == "a"


def test_file_below_limit_and_exact_limit() -> None:
    config = ChunkingConfig(max_lines_per_chunk=5, overlap_lines=0)

    below = chunk_source_file(_source("a\nb\nc\n"), config)
    exact = chunk_source_file(_source("1\n2\n3\n4\n5\n"), config)

    assert [(chunk.start_line, chunk.end_line) for chunk in below] == [(1, 3)]
    assert [(chunk.start_line, chunk.end_line) for chunk in exact] == [(1, 5)]


def test_multi_chunk_known_example() -> None:
    content = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n"
    chunks = chunk_source_file(
        _source(content),
        ChunkingConfig(max_lines_per_chunk=5, overlap_lines=2),
    )

    assert len(chunks) == 3
    assert [chunk.start_line for chunk in chunks] == [1, 4, 7]
    assert [chunk.end_line for chunk in chunks] == [5, 8, 10]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.content for chunk in chunks] == [
        "1\n2\n3\n4\n5\n",
        "4\n5\n6\n7\n8\n",
        "7\n8\n9\n10\n",
    ]


def test_zero_overlap_does_not_duplicate_or_skip_lines() -> None:
    content = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n"
    chunks = chunk_source_file(
        _source(content),
        ChunkingConfig(max_lines_per_chunk=5, overlap_lines=0),
    )

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 5), (6, 10)]
    assert "".join(chunk.content for chunk in chunks) == content


def test_high_overlap_progresses_and_terminates() -> None:
    content = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n"
    chunks = chunk_source_file(
        _source(content),
        ChunkingConfig(max_lines_per_chunk=4, overlap_lines=3),
    )

    assert len(chunks) == 7
    assert [chunk.start_line for chunk in chunks] == [1, 2, 3, 4, 5, 6, 7]
    assert [chunk.end_line for chunk in chunks] == [4, 5, 6, 7, 8, 9, 10]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("a", [("a", 1, 1)]),
        ("a\n", [("a\n", 1, 1)]),
        ("a\nb", [("a\n", 1, 1), ("b", 2, 2)]),
        ("a\nb\n", [("a\n", 1, 1), ("b\n", 2, 2)]),
    ],
)
def test_newline_preservation(content: str, expected: list[tuple[str, int, int]]) -> None:
    chunks = chunk_source_file(
        _source(content),
        ChunkingConfig(max_lines_per_chunk=1, overlap_lines=0),
    )

    assert [
        (chunk.content, chunk.start_line, chunk.end_line)
        for chunk in chunks
    ] == expected


def test_crlf_content_is_preserved() -> None:
    content = "a\r\nb\r\nc\r\n"
    chunks = chunk_source_file(
        _source(content),
        ChunkingConfig(max_lines_per_chunk=2, overlap_lines=1),
    )

    assert [chunk.content for chunk in chunks] == [
        "a\r\nb\r\n",
        "b\r\nc\r\n",
    ]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 2), (2, 3)]


def test_unicode_content_is_unchanged() -> None:
    content = 'def 你好():\n    return "世界"\n'
    chunks = chunk_source_file(
        _source(content),
        ChunkingConfig(max_lines_per_chunk=1, overlap_lines=0),
    )

    assert [chunk.content for chunk in chunks] == [
        "def 你好():\n",
        '    return "世界"\n',
    ]


def test_chunk_metadata_is_preserved() -> None:
    content = "1\n2\n3\n4\n5\n"
    chunks = chunk_source_file(
        _source(content, path="frontend/page.tsx", language="typescript"),
        ChunkingConfig(max_lines_per_chunk=3, overlap_lines=1),
    )

    assert all(chunk.relative_path == Path("frontend/page.tsx") for chunk in chunks)
    assert all(chunk.language == "typescript" for chunk in chunks)
    assert [chunk.start_line for chunk in chunks] == [1, 3]
    assert [chunk.end_line for chunk in chunks] == [3, 5]


def test_chunk_repository_preserves_file_and_chunk_order() -> None:
    first = _source("a1\na2\na3\n", path="src/a.py")
    second = _source("b1\nb2\n", path="src/b.py")
    repository = RepositorySnapshot(
        root=Path("repo"),
        name="repo",
        files=[first, second],
        skipped=[],
        file_count=2,
        total_size_bytes=first.size_bytes + second.size_bytes,
        languages={"python": 2},
    )

    chunks = chunk_repository(
        repository,
        ChunkingConfig(max_lines_per_chunk=2, overlap_lines=0),
    )

    assert [chunk.relative_path.as_posix() for chunk in chunks] == [
        "src/a.py",
        "src/a.py",
        "src/b.py",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 0]


@pytest.mark.parametrize(
    ("content", "config"),
    [
        ("a\nb\nc\nd\ne\n", ChunkingConfig(max_lines_per_chunk=2, overlap_lines=0)),
        ("a\nb\nc\nd\ne\n", ChunkingConfig(max_lines_per_chunk=3, overlap_lines=1)),
        ("a\r\nb\r\nc\r\n", ChunkingConfig(max_lines_per_chunk=2, overlap_lines=1)),
    ],
)
def test_chunk_invariants(content: str, config: ChunkingConfig) -> None:
    source = _source(content)
    chunks = chunk_source_file(source, config)

    assert chunks
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert 1 <= chunk.start_line <= chunk.end_line <= source.line_count

    if config.overlap_lines == 0:
        assert "".join(chunk.content for chunk in chunks) == content
    else:
        covered_lines = {
            line
            for chunk in chunks
            for line in range(chunk.start_line, chunk.end_line + 1)
        }
        assert covered_lines == set(range(1, source.line_count + 1))
