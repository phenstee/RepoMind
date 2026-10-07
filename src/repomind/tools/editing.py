"""Precise, bounded repository mutations with atomic write semantics."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from repomind.tools.filesystem import (
    _decode_text,
    _read_bounded_bytes,
    _resolve_workspace_path,
)
from repomind.tools.models import (
    CreateFileInput,
    CreateFileOutput,
    ReplaceTextInput,
    ReplaceTextOutput,
    ToolConfig,
    ToolContext,
)
from repomind.tools.registry import ToolExecutionError

# Files that configure, or are imported by, the fixed pytest/Ruff verifiers.
# Letting the editing agent author them would let it change what "verification
# passed" means (for example a conftest.py that forces pytest's exit status to
# zero) instead of fixing the code under test.
_PROTECTED_FILE_NAMES = frozenset(
    {
        "conftest.py",
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
        "tox.ini",
        "setup.cfg",
        "pyproject.toml",
        "ruff.toml",
        ".ruff.toml",
        "sitecustomize.py",
        "usercustomize.py",
        "pytest.py",
        "ruff.py",
    }
)
_PROTECTED_FILE_SUFFIXES = (".pth",)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_protected_path(path: Path) -> None:
    """Refuse mutations of verification configuration and import-hook files."""

    # Windows ignores trailing dots and spaces, so "conftest.py." is conftest.py.
    name = path.name.rstrip(" .").casefold()
    if name in _PROTECTED_FILE_NAMES or name.endswith(_PROTECTED_FILE_SUFFIXES):
        raise ToolExecutionError(
            f"Refusing to modify protected file: {path.as_posix()}. Verification "
            "configuration and import-hook files (conftest.py, pytest.ini, tox.ini, "
            "setup.cfg, pyproject.toml, ruff.toml, sitecustomize.py, "
            "usercustomize.py, pytest.py, ruff.py, *.pth) control how the fixed "
            "pytest and Ruff checks run, so they cannot be created or edited; "
            "change the code under test instead."
        )


def _write_temporary_file(parent: Path, name: str, data: bytes) -> Path:
    temporary_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            dir=parent,
            prefix=f".{name}.",
            suffix=".tmp",
        )
        temporary_path = Path(raw_path)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        return temporary_path
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ToolExecutionError("Could not prepare atomic repository write") from exc


def _atomic_create(target: Path, data: bytes) -> None:
    temporary_path = _write_temporary_file(target.parent, target.name, data)
    try:
        # A same-directory hard link publishes the complete temporary file while
        # retaining no-overwrite semantics if another process creates the target.
        os.link(temporary_path, target)
    except FileExistsError as exc:
        raise ToolExecutionError(f"File already exists: {target.name}") from exc
    except OSError as exc:
        raise ToolExecutionError("Could not create repository file atomically") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_replace(target: Path, data: bytes) -> None:
    temporary_path = _write_temporary_file(target.parent, target.name, data)
    try:
        shutil.copymode(target, temporary_path)
        os.replace(temporary_path, target)
    except OSError as exc:
        raise ToolExecutionError("Could not replace repository file atomically") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def create_file(
    context: ToolContext,
    arguments: CreateFileInput,
    *,
    config: ToolConfig | None = None,
) -> CreateFileOutput:
    """Create one new UTF-8 file without overwriting an existing path."""

    resolved_config = config or ToolConfig()
    _reject_protected_path(arguments.path)
    target = _resolve_workspace_path(context, arguments.path, allow_root=False)
    if not target.parent.exists() or not target.parent.is_dir():
        raise ToolExecutionError(
            f"Parent directory does not exist: {arguments.path.parent.as_posix()}"
        )
    if target.exists():
        raise ToolExecutionError(f"File already exists: {arguments.path.as_posix()}")
    if "\x00" in arguments.content:
        raise ToolExecutionError("File content appears to be binary")

    data = arguments.content.encode("utf-8")
    if len(data) > resolved_config.max_write_bytes:
        raise ToolExecutionError(
            f"File exceeds the {resolved_config.max_write_bytes}-byte write limit"
        )
    _atomic_create(target, data)
    return CreateFileOutput(
        path=arguments.path,
        sha256=_sha256(data),
        bytes_written=len(data),
    )


def _text_encoding(data: bytes) -> str:
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError:
        _decode_text(data)
        return "cp1252"
    return "utf-8"


def replace_text(
    context: ToolContext,
    arguments: ReplaceTextInput,
    *,
    config: ToolConfig | None = None,
) -> ReplaceTextOutput:
    """Replace one exact literal occurrence when the expected file hash is current."""

    resolved_config = config or ToolConfig()
    if max(len(arguments.old_text), len(arguments.new_text)) > (
        resolved_config.max_replacement_chars
    ):
        raise ToolExecutionError(
            "Replacement text exceeds the configured character limit"
        )

    _reject_protected_path(arguments.path)
    target = _resolve_workspace_path(context, arguments.path, allow_root=False)
    if not target.exists():
        raise ToolExecutionError(f"File does not exist: {arguments.path.as_posix()}")
    if not target.is_file():
        raise ToolExecutionError(
            f"Path is not a regular file: {arguments.path.as_posix()}"
        )

    data = _read_bounded_bytes(target, resolved_config.max_write_bytes)
    before_sha256 = _sha256(data)
    if before_sha256 != arguments.expected_sha256:
        raise ToolExecutionError(
            "File changed since it was read; inspect the current file before editing."
        )

    content = _decode_text(data)
    occurrences = content.count(arguments.old_text)
    if occurrences != 1:
        raise ToolExecutionError(
            f"old_text must appear exactly once; found {occurrences} occurrences"
        )

    encoding = _text_encoding(data)
    try:
        old_bytes = arguments.old_text.encode(encoding)
        new_bytes = arguments.new_text.encode(encoding)
    except UnicodeEncodeError as exc:
        raise ToolExecutionError(
            f"Replacement cannot be represented in the file's {encoding} encoding"
        ) from exc
    updated = data.replace(old_bytes, new_bytes, 1)
    if len(updated) > resolved_config.max_write_bytes:
        raise ToolExecutionError(
            f"Updated file exceeds the {resolved_config.max_write_bytes}-byte write limit"
        )

    _atomic_replace(target, updated)
    return ReplaceTextOutput(
        path=arguments.path,
        replacements=1,
        before_sha256=before_sha256,
        after_sha256=_sha256(updated),
        bytes_before=len(data),
        bytes_after=len(updated),
    )
