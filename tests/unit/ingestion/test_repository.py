"""Tests for repository discovery and source-file loading."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from repomind.ingestion.models import IngestionConfig
from repomind.ingestion.repository import (
    BinarySourceFileError,
    FileTooLargeError,
    InvalidRepositoryRootError,
    PathOutsideRepositoryError,
    SourceDecodeError,
    SourceReadError,
    SymlinkSourceFileError,
    UnsupportedSourceFileError,
    find_source_files,
    ingest_repository,
    load_source_file,
)


def _write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.write_bytes(data)


def test_find_source_files_discovers_supported_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "a")
    _write(root / "sub" / "b.ts", "b")
    _write(root / "sub" / "notes.txt", "txt")
    _write(root / "node_modules" / "dependency.js", "ignored")
    _write(root / ".venv" / "ignored.py", "ignored")
    _write(root / ".github" / "workflows" / "ci.yml", "name: CI\n")

    paths = find_source_files(root)
    relative = [path.relative_to(root).as_posix() for path in paths]

    assert relative == [
        ".github/workflows/ci.yml",
        "a.py",
        "sub/b.ts",
    ]


def test_find_source_files_accepts_str_and_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "app.py", "print('ok')\n")

    assert find_source_files(root) == find_source_files(str(root))


def test_find_source_files_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "z.py", "z")
    _write(root / "a.py", "a")
    _write(root / "m.py", "m")

    paths = find_source_files(root)
    relative = [path.relative_to(root).as_posix() for path in paths]

    assert relative == ["a.py", "m.py", "z.py"]
    assert find_source_files(root) == paths


def test_find_source_files_rejects_invalid_root(tmp_path: Path) -> None:
    with pytest.raises(InvalidRepositoryRootError):
        find_source_files(tmp_path / "missing")

    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("x")
    with pytest.raises(InvalidRepositoryRootError):
        find_source_files(not_a_directory)


def test_load_source_file_extracts_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    content = "print('hello')\n"
    _write(root / "src" / "app.py", content)

    source = load_source_file(root, root / "src" / "app.py")

    assert source.relative_path == Path("src") / "app.py"
    assert source.language == "python"
    assert source.content == content
    assert source.size_bytes == len(content.encode("utf-8"))
    assert source.line_count == 1


def test_load_source_file_strips_utf8_bom(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "bom.py", "print('x')\n".encode("utf-8-sig"))

    source = load_source_file(root, root / "bom.py")

    assert source.content == "print('x')\n"
    assert source.line_count == 1


def test_load_source_file_handles_empty_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "empty.py", b"")

    source = load_source_file(root, root / "empty.py")

    assert source.content == ""
    assert source.size_bytes == 0
    assert source.line_count == 0


def test_load_source_file_rejects_unsupported_extension(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "notes.txt", "hello")

    with pytest.raises(UnsupportedSourceFileError):
        load_source_file(root, root / "notes.txt")


def test_load_source_file_rejects_oversized_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "large.py", b"123456")
    config = IngestionConfig(max_file_size_bytes=4)

    with pytest.raises(FileTooLargeError):
        load_source_file(root, root / "large.py", config)


def test_load_source_file_uses_a_bounded_read(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    path = root / "growing.py"
    _write(path, b"")
    config = IngestionConfig(max_file_size_bytes=4)
    source = MagicMock()
    source.__enter__.return_value = source
    source.read.return_value = b"12345"
    open_mock = MagicMock(return_value=source)
    monkeypatch.setattr(Path, "open", open_mock)

    with pytest.raises(FileTooLargeError):
        load_source_file(root, path, config)

    open_mock.assert_called_once_with("rb")
    source.read.assert_called_once_with(5)


def test_load_source_file_rejects_binary_content(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "fake.py", b"print('x')\x00")

    with pytest.raises(BinarySourceFileError):
        load_source_file(root, root / "fake.py")


def test_load_source_file_rejects_null_byte_after_old_sample(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "late-null.py", b"a" * 8192 + b"\x00")

    with pytest.raises(BinarySourceFileError):
        load_source_file(root, root / "late-null.py")


def test_load_source_file_rejects_undecodable_content(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "bad.py", b"\xff\xfe")
    config = IngestionConfig(fallback_encoding="ascii")

    with pytest.raises(SourceDecodeError):
        load_source_file(root, root / "bad.py", config)


def test_load_source_file_rejects_path_outside_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside.py"
    root.mkdir()
    _write(outside, "print('outside')\n")

    with pytest.raises(PathOutsideRepositoryError):
        load_source_file(root, outside)


@pytest.mark.parametrize(
    ("content", "expected_line_count"),
    [
        ("", 0),
        ("a", 1),
        ("a\n", 1),
        ("a\nb", 2),
        ("a\nb\n", 2),
    ],
)
def test_line_count_semantics(
    tmp_path: Path,
    content: str,
    expected_line_count: int,
) -> None:
    root = tmp_path / "repo"
    _write(root / "sample.py", content.encode("utf-8"))

    source = load_source_file(root, root / "sample.py")

    assert source.line_count == expected_line_count


def test_custom_ignored_directory(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "generated" / "code.py", "x")
    config = IngestionConfig(ignored_directories={"generated"})

    assert find_source_files(root, config) == []


def test_custom_ignored_file_pattern(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "keep.py", "x")
    _write(root / "skip.generated.py", "x")
    config = IngestionConfig(ignored_file_patterns=("*.generated.py",))

    paths = find_source_files(root, config)
    assert [path.name for path in paths] == ["keep.py"]

    snapshot = ingest_repository(root, config)
    assert [item.reason for item in snapshot.skipped] == ["ignored_file_pattern"]


def test_symlink_files_are_skipped_when_available(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside.py"
    _write(outside, "print('outside')\n")
    root.mkdir()

    try:
        (root / "linked.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not available in this environment")

    assert find_source_files(root) == []

    snapshot = ingest_repository(root)
    assert [(item.relative_path.as_posix(), item.reason) for item in snapshot.skipped] == [
        ("linked.py", "symlink")
    ]


def test_symlink_directories_are_not_traversed_when_available(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    _write(outside / "escaped.py", "print('outside')\n")

    try:
        (root / "linked").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlink creation is not available in this environment")

    assert find_source_files(root) == []
    with pytest.raises(SymlinkSourceFileError):
        load_source_file(root, root / "linked" / "escaped.py")


def test_junction_directories_are_pruned(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    _write(root / "junction" / "escaped.py", "print('outside')\n")
    original_is_junction = Path.is_junction

    def fake_is_junction(path: Path) -> bool:
        return path.name == "junction" or original_is_junction(path)

    monkeypatch.setattr(Path, "is_junction", fake_is_junction)

    assert find_source_files(root) == []


@pytest.mark.parametrize(
    "read_error",
    [
        FileNotFoundError("test disappearance"),
        PermissionError("test permission failure"),
        OSError("test read failure"),
    ],
)
def test_ingest_repository_records_read_errors_and_continues(
    tmp_path: Path,
    monkeypatch,
    read_error: OSError,
) -> None:
    root = tmp_path / "repo"
    broken = root / "broken.py"
    healthy = root / "healthy.py"
    _write(broken, "broken = True\n")
    _write(healthy, "healthy = True\n")
    original_open = Path.open

    def fail_read(path: Path, *args, **kwargs):
        if path.name == "broken.py":
            raise read_error
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_read)

    snapshot = ingest_repository(root)

    assert [source.relative_path.as_posix() for source in snapshot.files] == ["healthy.py"]
    assert [(item.relative_path.as_posix(), item.reason) for item in snapshot.skipped] == [
        ("broken.py", SourceReadError.reason)
    ]


def test_ingest_repository_builds_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "fake_repo"
    _write(root / "README.md", "# Example\n")
    _write(root / "pyproject.toml", "[project]\n")
    _write(root / "src" / "app.py", "print('app')\n")
    _write(root / "src" / "service.py", "print('service')\n")
    _write(root / "frontend" / "page.tsx", "export const Page = () => null;\n")
    _write(root / "tests" / "test_app.py", "def test_app():\n    pass\n")
    _write(root / "node_modules" / "dependency.js", "ignored")
    _write(root / ".venv" / "ignored.py", "ignored")
    _write(root / "image.png", b"\x89PNG\r\n")

    snapshot = ingest_repository(root)
    relative_paths = {source.relative_path.as_posix() for source in snapshot.files}

    assert relative_paths == {
        "README.md",
        "frontend/page.tsx",
        "pyproject.toml",
        "src/app.py",
        "src/service.py",
        "tests/test_app.py",
    }
    assert snapshot.file_count == 6
    assert snapshot.name == "fake_repo"
    assert snapshot.languages == {
        "markdown": 1,
        "python": 3,
        "toml": 1,
        "typescript": 1,
    }
    assert snapshot.total_size_bytes == sum(source.size_bytes for source in snapshot.files)

    skipped = {(item.relative_path.as_posix(), item.reason) for item in snapshot.skipped}
    assert ("image.png", "unsupported_extension") in skipped
