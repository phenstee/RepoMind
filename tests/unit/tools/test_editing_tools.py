"""Tests for precise atomic repository mutation tools."""

import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.tools import (
    CreateFileInput,
    ReplaceTextInput,
    ToolConfig,
    ToolContext,
    ToolExecutionError,
    create_file,
    read_file,
    replace_text,
)
from repomind.tools.models import ReadFileInput


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _context(root: Path) -> ToolContext:
    return ToolContext(repository_root=root)


def test_create_file_preserves_unicode_crlf_and_reports_hash(tmp_path: Path) -> None:
    content = "café\r\n雪\r\n"
    output = create_file(
        _context(tmp_path), CreateFileInput(path="new.py", content=content)
    )

    expected = content.encode("utf-8")
    assert (tmp_path / "new.py").read_bytes() == expected
    assert output.path == Path("new.py")
    assert output.bytes_written == len(expected)
    assert output.sha256 == _hash(expected)


def test_create_file_rejects_existing_missing_parent_binary_and_oversize(
    tmp_path: Path,
) -> None:
    (tmp_path / "existing.py").write_text("original", encoding="utf-8")
    context = _context(tmp_path)

    cases = (
        (CreateFileInput(path="existing.py", content="new"), None, "already exists"),
        (CreateFileInput(path="missing/new.py", content="new"), None, "Parent directory"),
        (CreateFileInput(path="binary.py", content="a\x00b"), None, "binary"),
        (
            CreateFileInput(path="large.py", content="12345"),
            ToolConfig(max_write_bytes=4),
            "write limit",
        ),
    )
    for arguments, config, message in cases:
        with pytest.raises(ToolExecutionError, match=message):
            create_file(context, arguments, config=config)

    assert (tmp_path / "existing.py").read_text(encoding="utf-8") == "original"
    assert not (tmp_path / "large.py").exists()


@pytest.mark.parametrize(
    "path",
    ["../outside.py", "C:outside.py", "C:\\Windows\\system.ini", "/root/test.py"],
)
def test_mutation_inputs_reject_hostile_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        CreateFileInput(path=path, content="x")
    with pytest.raises(ValidationError):
        ReplaceTextInput(
            path=path,
            old_text="x",
            new_text="y",
            expected_sha256="0" * 64,
        )


def test_create_file_failure_leaves_no_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_link(source: Path, target: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr("repomind.tools.editing.os.link", fail_link)
    with pytest.raises(ToolExecutionError, match="atomically"):
        create_file(_context(tmp_path), CreateFileInput(path="new.py", content="complete"))

    assert not (tmp_path / "new.py").exists()
    assert list(tmp_path.iterdir()) == []


def test_create_file_rejects_symlink_parent_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    try:
        os.symlink(outside, tmp_path / "linked", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ToolExecutionError, match="Unsafe repository path"):
        create_file(
            _context(tmp_path), CreateFileInput(path="linked/escape.py", content="x")
        )
    assert not (outside / "escape.py").exists()


def test_replace_text_exactly_once_preserves_lf_unicode_and_hashes(tmp_path: Path) -> None:
    before = "greeting = 'café'\nvalue = '雪'\n".encode()
    target = tmp_path / "app.py"
    target.write_bytes(before)

    output = replace_text(
        _context(tmp_path),
        ReplaceTextInput(
            path="app.py",
            old_text="value = '雪'\n",
            new_text="value = '月'\n",
            expected_sha256=_hash(before),
        ),
    )

    after = "greeting = 'café'\nvalue = '月'\n".encode()
    assert target.read_bytes() == after
    assert output.replacements == 1
    assert (output.before_sha256, output.after_sha256) == (_hash(before), _hash(after))
    assert (output.bytes_before, output.bytes_after) == (len(before), len(after))


def test_replace_text_preserves_crlf_and_bytes_outside_replacement(tmp_path: Path) -> None:
    before = b"first\r\nold value\r\nlast\r\n"
    target = tmp_path / "app.py"
    target.write_bytes(before)

    replace_text(
        _context(tmp_path),
        ReplaceTextInput(
            path="app.py",
            old_text="old value",
            new_text="new value",
            expected_sha256=_hash(before),
        ),
    )

    assert target.read_bytes() == b"first\r\nnew value\r\nlast\r\n"


def test_replace_text_preserves_cp1252_encoding(tmp_path: Path) -> None:
    before = "price = '10€'\r\n".encode("cp1252")
    target = tmp_path / "legacy.py"
    target.write_bytes(before)

    replace_text(
        _context(tmp_path),
        ReplaceTextInput(
            path="legacy.py",
            old_text="10€",
            new_text="20€",
            expected_sha256=_hash(before),
        ),
    )

    assert target.read_bytes() == "price = '20€'\r\n".encode("cp1252")


def test_replace_text_rejects_symlink_target(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-external.py"
    outside.write_bytes(b"outside")
    try:
        os.symlink(outside, tmp_path / "linked.py")
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(ToolExecutionError, match="Unsafe repository path"):
        replace_text(
            _context(tmp_path),
            ReplaceTextInput(
                path="linked.py",
                old_text="outside",
                new_text="changed",
                expected_sha256=_hash(b"outside"),
            ),
        )
    assert outside.read_bytes() == b"outside"


@pytest.mark.parametrize(
    ("content", "old_text", "message"),
    [(b"alpha\n", "missing", "found 0"), (b"same same", "same", "found 2")],
)
def test_replace_text_rejects_zero_or_ambiguous_matches(
    tmp_path: Path, content: bytes, old_text: str, message: str
) -> None:
    target = tmp_path / "app.py"
    target.write_bytes(content)
    with pytest.raises(ToolExecutionError, match=message):
        replace_text(
            _context(tmp_path),
            ReplaceTextInput(
                path="app.py",
                old_text=old_text,
                new_text="new",
                expected_sha256=_hash(content),
            ),
        )
    assert target.read_bytes() == content


def test_replace_text_rejects_stale_hash_and_preserves_external_change(
    tmp_path: Path,
) -> None:
    target = tmp_path / "app.py"
    target.write_bytes(b"observed\n")
    observation = read_file(_context(tmp_path), ReadFileInput(path="app.py"))
    target.write_bytes(b"external change\n")

    with pytest.raises(ToolExecutionError, match="changed since it was read"):
        replace_text(
            _context(tmp_path),
            ReplaceTextInput(
                path="app.py",
                old_text="observed",
                new_text="agent edit",
                expected_sha256=observation.sha256,
            ),
        )
    assert target.read_bytes() == b"external change\n"


def test_replace_text_rejects_oversized_result_without_modification(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_bytes(b"a")
    with pytest.raises(ToolExecutionError, match="Updated file exceeds"):
        replace_text(
            _context(tmp_path),
            ReplaceTextInput(
                path="app.py",
                old_text="a",
                new_text="12345",
                expected_sha256=_hash(b"a"),
            ),
            config=ToolConfig(max_write_bytes=4),
        )
    assert target.read_bytes() == b"a"


def test_replace_failure_preserves_original_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "app.py"
    target.write_bytes(b"old")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("repomind.tools.editing.os.replace", fail_replace)
    with pytest.raises(ToolExecutionError, match="atomically"):
        replace_text(
            _context(tmp_path),
            ReplaceTextInput(
                path="app.py",
                old_text="old",
                new_text="new",
                expected_sha256=_hash(b"old"),
            ),
        )

    assert target.read_bytes() == b"old"
    assert [path.name for path in tmp_path.iterdir()] == ["app.py"]
