"""Safe, deterministic repository discovery and source-file loading."""

from __future__ import annotations

import fnmatch
import os
from collections import Counter
from pathlib import Path

from repomind.ingestion.language import is_supported_source_file, language_for_path
from repomind.ingestion.models import (
    IngestionConfig,
    RepositorySnapshot,
    SkippedFile,
    SourceFile,
)


class RepositoryIngestionError(ValueError):
    """Base exception for expected repository-ingestion failures."""

    reason = "ingestion_error"


class InvalidRepositoryRootError(RepositoryIngestionError):
    reason = "invalid_root"


class PathOutsideRepositoryError(RepositoryIngestionError):
    reason = "path_outside_repository"


class UnsupportedSourceFileError(RepositoryIngestionError):
    reason = "unsupported_extension"


class SymlinkSourceFileError(RepositoryIngestionError):
    reason = "symlink"


class FileTooLargeError(RepositoryIngestionError):
    reason = "file_too_large"


class BinarySourceFileError(RepositoryIngestionError):
    reason = "binary_content"


class SourceDecodeError(RepositoryIngestionError):
    reason = "decode_error"


def validate_repository_root(root: str | Path) -> Path:
    """Resolve and validate a repository root."""

    candidate = Path(root).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise InvalidRepositoryRootError(f"Repository root does not exist: {root}") from exc

    if not resolved.is_dir():
        raise InvalidRepositoryRootError(f"Repository root is not a directory: {root}")
    return resolved


def _relative_path(root: Path, path: Path) -> Path:
    return path.relative_to(root)


def _relative_sort_key(root: Path, path: Path) -> str:
    return _relative_path(root, path).as_posix().casefold()


def _is_path_within(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_ignored_directory_name(name: str, config: IngestionConfig) -> bool:
    return name.casefold() in config.ignored_directories


def _is_ignored_file(path: Path, config: IngestionConfig) -> bool:
    name = path.name.casefold()
    return any(fnmatch.fnmatch(name, pattern) for pattern in config.ignored_file_patterns)


def _iter_repository_files(root: Path, config: IngestionConfig):
    """Yield every file below ``root`` without following directory symlinks.

    Ignored directories are pruned before traversal. File symlinks are yielded
    so callers can decide whether to record them as skipped.
    """

    for current_root, dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current_root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not _is_ignored_directory_name(name, config)
            and not (current_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            yield current_path / filename


def find_source_files(
    root: str | Path,
    config: IngestionConfig | None = None,
) -> list[Path]:
    """Return supported, non-symlink source files beneath ``root`` in stable order."""

    resolved_root = validate_repository_root(root)
    resolved_config = config or IngestionConfig()

    paths = [
        path
        for path in _iter_repository_files(resolved_root, resolved_config)
        if not path.is_symlink()
        and path.is_file()
        and is_supported_source_file(path)
        and not _is_ignored_file(path, resolved_config)
    ]
    return sorted(paths, key=lambda path: _relative_sort_key(resolved_root, path))


def _decode_source(data: bytes, fallback_encoding: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return data.decode(fallback_encoding)
        except (UnicodeDecodeError, LookupError) as fallback_error:
            raise SourceDecodeError(
                "File is not valid UTF-8 and could not be decoded with "
                f"{fallback_encoding!r}"
            ) from fallback_error


def _count_lines(content: str) -> int:
    """Return the number of logical lines using Python ``splitlines()`` semantics."""

    return len(content.splitlines())


def load_source_file(
    root: str | Path,
    path: str | Path,
    config: IngestionConfig | None = None,
) -> SourceFile:
    """Load and validate one source file, returning its public representation."""

    resolved_root = validate_repository_root(root)
    resolved_config = config or IngestionConfig()
    candidate = Path(path).expanduser()

    if not candidate.is_absolute():
        candidate = resolved_root / candidate

    if candidate.is_symlink():
        raise SymlinkSourceFileError(f"Symlinks are not ingested: {candidate}")

    resolved_path = candidate.resolve(strict=False)
    if not _is_path_within(resolved_root, resolved_path):
        raise PathOutsideRepositoryError(
            f"Source file is outside the repository root: {resolved_path}"
        )
    if not resolved_path.exists():
        raise RepositoryIngestionError(f"Source file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise RepositoryIngestionError(f"Source path is not a file: {resolved_path}")
    if not is_supported_source_file(resolved_path):
        raise UnsupportedSourceFileError(
            f"Unsupported source-file extension: {resolved_path.suffix}"
        )
    if _is_ignored_file(resolved_path, resolved_config):
        raise RepositoryIngestionError(
            f"Source file matches an ignored file pattern: {resolved_path}"
        )

    size_bytes = resolved_path.stat().st_size
    if size_bytes > resolved_config.max_file_size_bytes:
        raise FileTooLargeError(
            f"Source file exceeds {resolved_config.max_file_size_bytes} bytes: "
            f"{resolved_path}"
        )

    data = resolved_path.read_bytes()
    if b"\x00" in data[: resolved_config.binary_sample_bytes]:
        raise BinarySourceFileError(f"Source file appears to be binary: {resolved_path}")

    content = _decode_source(data, resolved_config.fallback_encoding)
    return SourceFile(
        relative_path=_relative_path(resolved_root, resolved_path),
        language=language_for_path(resolved_path),
        content=content,
        size_bytes=size_bytes,
        line_count=_count_lines(content),
    )


def ingest_repository(
    root: str | Path,
    config: IngestionConfig | None = None,
) -> RepositorySnapshot:
    """Discover and load a repository into a deterministic snapshot."""

    resolved_root = validate_repository_root(root)
    resolved_config = config or IngestionConfig()
    files: list[SourceFile] = []
    skipped: list[SkippedFile] = []

    for path in _iter_repository_files(resolved_root, resolved_config):
        relative = _relative_path(resolved_root, path)

        if path.is_symlink():
            skipped.append(SkippedFile(relative_path=relative, reason="symlink"))
            continue
        if not path.is_file():
            continue
        if not is_supported_source_file(path):
            skipped.append(
                SkippedFile(relative_path=relative, reason="unsupported_extension")
            )
            continue
        if _is_ignored_file(path, resolved_config):
            skipped.append(
                SkippedFile(relative_path=relative, reason="ignored_file_pattern")
            )
            continue

        try:
            files.append(load_source_file(resolved_root, path, resolved_config))
        except RepositoryIngestionError as exc:
            skipped.append(SkippedFile(relative_path=relative, reason=exc.reason))

    files.sort(key=lambda source: source.relative_path.as_posix().casefold())
    skipped.sort(key=lambda item: item.relative_path.as_posix().casefold())
    languages = dict(
        sorted(Counter(source.language or "unknown" for source in files).items())
    )

    return RepositorySnapshot(
        root=resolved_root,
        name=resolved_root.name,
        files=files,
        skipped=skipped,
        file_count=len(files),
        total_size_bytes=sum(source.size_bytes for source in files),
        languages=languages,
    )
