"""Tests for bounded repository filesystem inspection."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.tools import (
    ListDirectoryInput,
    ReadFileInput,
    ToolConfig,
    ToolContext,
    ToolExecutionError,
    list_directory,
    read_file,
)


def test_read_file_preserves_unicode_crlf_and_full_range(tmp_path: Path) -> None:
    content = "café\r\n雪\r\n"
    (tmp_path / "sample.py").write_bytes(content.encode())

    output = read_file(ToolContext(repository_root=tmp_path), ReadFileInput(path="sample.py"))

    assert output.content == content
    assert (output.start_line, output.end_line, output.total_lines) == (1, 2, 2)
    assert output.model_dump(mode="json")["path"] == "sample.py"
    assert output.sha256 == hashlib.sha256(content.encode()).hexdigest()


def test_read_file_returns_bounded_range_and_clamps_end_to_eof(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_bytes(b"one\ntwo\nthree")

    output = read_file(
        ToolContext(repository_root=tmp_path),
        ReadFileInput(path="sample.py", start_line=2, end_line=99),
    )

    assert output.content == "two\nthree"
    assert (output.start_line, output.end_line) == (2, 3)
    assert (output.requested_start_line, output.requested_end_line) == (2, 99)
    assert output.sha256 == hashlib.sha256(b"one\ntwo\nthree").hexdigest()


def test_read_file_handles_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.py").write_bytes(b"")
    output = read_file(ToolContext(repository_root=tmp_path), ReadFileInput(path="empty.py"))
    assert (output.content, output.start_line, output.end_line, output.total_lines) == (
        "",
        0,
        0,
        0,
    )


def test_read_file_rejects_start_beyond_eof(tmp_path: Path) -> None:
    (tmp_path / "short.py").write_bytes(b"one\ntwo\n")
    with pytest.raises(ToolExecutionError, match="exceeds"):
        read_file(
            ToolContext(repository_root=tmp_path),
            ReadFileInput(path="short.py", start_line=3),
        )


@pytest.mark.parametrize("values", [{"start_line": 0}, {"end_line": 0}, {"start_line": 3, "end_line": 2}])
def test_read_file_rejects_invalid_ranges(values: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        ReadFileInput(path="file.py", **values)


def test_read_file_rejects_missing_directory_binary_and_oversized(tmp_path: Path) -> None:
    context = ToolContext(repository_root=tmp_path)
    (tmp_path / "folder").mkdir()
    (tmp_path / "binary.py").write_bytes(b"abc\x00def")
    (tmp_path / "large.py").write_bytes(b"12345")

    for name, message, config in (
        ("missing.py", "does not exist", None),
        ("folder", "not a regular file", None),
        ("binary.py", "binary", None),
        ("large.py", "byte tool limit", ToolConfig(max_file_bytes=4)),
    ):
        with pytest.raises(ToolExecutionError, match=message):
            read_file(context, ReadFileInput(path=name), config=config)


def test_list_directory_is_sorted_structured_and_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "z.py").write_text("z", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "nested.py").write_text("n", encoding="utf-8")

    output = list_directory(ToolContext(repository_root=tmp_path), ListDirectoryInput())

    assert [(item.path.as_posix(), item.type) for item in output.entries] == [
        ("a", "directory"),
        ("z.py", "file"),
    ]
    assert output.entries[1].size_bytes == 1
    assert not output.truncated


def test_list_directory_recurses_prunes_defaults_and_reports_truncation(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "secret").write_text("x", encoding="utf-8")

    output = list_directory(
        ToolContext(repository_root=tmp_path),
        ListDirectoryInput(recursive=True),
        config=ToolConfig(max_directory_entries=2),
    )

    assert [item.path.as_posix() for item in output.entries] == ["src", "src/a.py"]
    assert output.truncated
    assert all(".git" not in item.path.parts for item in output.entries)


def test_list_directory_rejects_missing_and_file_targets(tmp_path: Path) -> None:
    context = ToolContext(repository_root=tmp_path)
    (tmp_path / "file.py").write_text("x", encoding="utf-8")
    with pytest.raises(ToolExecutionError, match="does not exist"):
        list_directory(context, ListDirectoryInput(path="missing"))
    with pytest.raises(ToolExecutionError, match="not a directory"):
        list_directory(context, ListDirectoryInput(path="file.py"))


def test_list_directory_accepts_a_nested_scope(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x", encoding="utf-8")
    output = list_directory(
        ToolContext(repository_root=tmp_path),
        ListDirectoryInput(path="src"),
    )
    assert [item.path.as_posix() for item in output.entries] == ["src/app.py"]


def test_tools_do_not_follow_symlink_escape_when_available(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret", encoding="utf-8")
    link = root / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("File symlink creation is not available")

    context = ToolContext(repository_root=root)
    listing = list_directory(context, ListDirectoryInput())
    assert [(item.path.as_posix(), item.type) for item in listing.entries] == [
        ("linked.py", "symlink")
    ]
    with pytest.raises(ToolExecutionError, match="Unsafe repository path"):
        read_file(context, ReadFileInput(path="linked.py"))


def test_tools_reject_simulated_junction_component(tmp_path: Path, monkeypatch) -> None:
    junction = tmp_path / "junction"
    junction.mkdir()
    (junction / "escaped.py").write_text("secret", encoding="utf-8")
    original_is_junction = Path.is_junction

    def fake_is_junction(path: Path) -> bool:
        return path.name == "junction" or original_is_junction(path)

    monkeypatch.setattr(Path, "is_junction", fake_is_junction)
    context = ToolContext(repository_root=tmp_path)

    with pytest.raises(ToolExecutionError, match="Unsafe repository path"):
        read_file(context, ReadFileInput(path="junction/escaped.py"))


@pytest.mark.parametrize(
    "path",
    [".git/config", ".GIT/config", "sub/.git/config", ".git", ".git./config"],
)
def test_read_file_rejects_git_metadata_paths(tmp_path: Path, path: str) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="Git metadata"):
        ReadFileInput(path=path)
    # The execution-time boundary holds even for unvalidated model instances.
    with pytest.raises(ToolExecutionError, match="Git metadata"):
        read_file(
            ToolContext(repository_root=tmp_path),
            ReadFileInput.model_construct(path=Path(path), start_line=None, end_line=None),
        )


def test_list_directory_never_lists_or_enters_git_metadata(tmp_path: Path) -> None:
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    context = ToolContext(repository_root=tmp_path)

    flat = list_directory(context, ListDirectoryInput())
    recursive = list_directory(context, ListDirectoryInput(recursive=True))

    assert [entry.path.as_posix() for entry in flat.entries] == ["app.py", "pkg"]
    assert [entry.path.as_posix() for entry in recursive.entries] == ["app.py", "pkg"]
    with pytest.raises(ValidationError, match="Git metadata"):
        ListDirectoryInput(path=".git")


def test_read_file_numbers_lines_like_ast_and_index_citations(tmp_path: Path) -> None:
    # Form feeds, vertical tabs, and U+2028 are ordinary characters to ast,
    # editors, and Git; only \r\n, \r, and \n end a line.
    content = "first = 1\n\x0c\nsecond = ' '\x0b\r\nthird = 3\rfourth = 4\n"
    (tmp_path / "sample.py").write_bytes(content.encode("utf-8"))
    context = ToolContext(repository_root=tmp_path)

    whole = read_file(context, ReadFileInput(path="sample.py"))
    third_line = read_file(context, ReadFileInput(path="sample.py", start_line=3, end_line=3))
    tail = read_file(context, ReadFileInput(path="sample.py", start_line=4))

    assert whole.total_lines == 5
    assert whole.content == content
    assert third_line.content == "second = ' '\x0b\r\n"
    assert tail.content == "third = 3\rfourth = 4\n"
