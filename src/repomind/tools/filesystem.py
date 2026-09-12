"""Safe, bounded filesystem inspection tools."""

from __future__ import annotations

from pathlib import Path

from repomind.ingestion import (
    RepositoryIngestionError,
    resolve_repository_path,
)
from repomind.ingestion.models import DEFAULT_IGNORED_DIRECTORIES
from repomind.tools.models import (
    DirectoryEntry,
    ListDirectoryInput,
    ListDirectoryOutput,
    ReadFileInput,
    ReadFileOutput,
    ToolConfig,
    ToolContext,
)
from repomind.tools.registry import ToolExecutionError


def _resolve_workspace_path(
    context: ToolContext,
    path: Path,
    *,
    allow_root: bool,
) -> Path:
    try:
        return resolve_repository_path(
            context.repository_root,
            path,
            allow_root=allow_root,
        )
    except RepositoryIngestionError as exc:
        raise ToolExecutionError(f"Unsafe repository path: {path.as_posix()}") from exc


def _is_link_or_junction(path: Path) -> bool:
    try:
        return path.is_symlink() or path.is_junction()
    except OSError:
        return True


def _read_text(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as source:
            data = source.read(max_bytes + 1)
    except OSError as exc:
        raise ToolExecutionError("Could not read repository file") from exc

    if len(data) > max_bytes:
        raise ToolExecutionError(f"File exceeds the {max_bytes}-byte tool limit")
    if b"\x00" in data:
        raise ToolExecutionError("File appears to contain binary data")

    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return data.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError("File could not be decoded as repository text") from exc


def read_file(
    context: ToolContext,
    arguments: ReadFileInput,
    *,
    config: ToolConfig | None = None,
) -> ReadFileOutput:
    """Read all or a 1-based inclusive range of a repository text file."""

    resolved_config = config or ToolConfig()
    path = _resolve_workspace_path(context, arguments.path, allow_root=False)
    if not path.exists():
        raise ToolExecutionError(f"File does not exist: {arguments.path.as_posix()}")
    if not path.is_file():
        raise ToolExecutionError(f"Path is not a regular file: {arguments.path.as_posix()}")

    content = _read_text(path, resolved_config.max_file_bytes)
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        return ReadFileOutput(
            path=arguments.path,
            requested_start_line=arguments.start_line,
            requested_end_line=arguments.end_line,
            start_line=0,
            end_line=0,
            content="",
            total_lines=0,
        )

    start_line = arguments.start_line or 1
    if start_line > total_lines:
        raise ToolExecutionError(
            f"start_line {start_line} exceeds the file's {total_lines} lines"
        )
    end_line = min(arguments.end_line or total_lines, total_lines)
    return ReadFileOutput(
        path=arguments.path,
        requested_start_line=arguments.start_line,
        requested_end_line=arguments.end_line,
        start_line=start_line,
        end_line=end_line,
        content="".join(lines[start_line - 1 : end_line]),
        total_lines=total_lines,
    )


def _entry(root: Path, path: Path) -> DirectoryEntry | None:
    relative = path.relative_to(root)
    if _is_link_or_junction(path):
        return DirectoryEntry(path=relative, type="symlink")
    try:
        if path.is_dir():
            return DirectoryEntry(path=relative, type="directory")
        if path.is_file():
            return DirectoryEntry(path=relative, type="file", size_bytes=path.stat().st_size)
    except OSError as exc:
        raise ToolExecutionError(
            f"Could not inspect repository entry: {relative.as_posix()}"
        ) from exc
    return None


def list_directory(
    context: ToolContext,
    arguments: ListDirectoryInput,
    *,
    config: ToolConfig | None = None,
) -> ListDirectoryOutput:
    """List a repository directory without following links or junctions."""

    resolved_config = config or ToolConfig()
    target = _resolve_workspace_path(context, arguments.path, allow_root=True)
    if not target.exists():
        raise ToolExecutionError(f"Directory does not exist: {arguments.path.as_posix()}")
    if not target.is_dir():
        raise ToolExecutionError(f"Path is not a directory: {arguments.path.as_posix()}")

    entries: list[DirectoryEntry] = []
    pending = [target]
    truncated = False
    while pending:
        current = pending.pop()
        try:
            children = sorted(
                current.iterdir(),
                key=lambda child: (child.name.casefold(), child.name),
            )
        except OSError as exc:
            raise ToolExecutionError("Could not list repository directory") from exc

        child_directories: list[Path] = []
        for child in children:
            is_link = _is_link_or_junction(child)
            if (
                arguments.recursive
                and not is_link
                and child.is_dir()
                and child.name.casefold() in DEFAULT_IGNORED_DIRECTORIES
            ):
                continue
            item = _entry(context.repository_root, child)
            if item is not None:
                entries.append(item)
            if len(entries) > resolved_config.max_directory_entries:
                truncated = True
                break
            if arguments.recursive and item is not None and item.type == "directory":
                child_directories.append(child)
        if truncated:
            break
        pending.extend(reversed(child_directories))
        if not arguments.recursive:
            break

    entries = sorted(
        entries[: resolved_config.max_directory_entries],
        key=lambda item: (item.path.as_posix().casefold(), item.path.as_posix()),
    )
    return ListDirectoryOutput(
        path=arguments.path,
        recursive=arguments.recursive,
        entries=entries,
        truncated=truncated,
    )
