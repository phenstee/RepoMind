"""Centralized source-file extension and language mapping."""

from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".go",
        ".java",
        ".rs",
        ".cpp",
        ".cc",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".sql",
        ".sh",
        ".ps1",
    }
)

LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sql": "sql",
    ".sh": "shell",
    ".ps1": "powershell",
}


def _extension(path: Path) -> str:
    return path.suffix.casefold()


def is_supported_source_file(path: Path) -> bool:
    """Return whether ``path`` has a supported source-file extension."""

    return _extension(path) in SUPPORTED_EXTENSIONS


def language_for_path(path: Path) -> str | None:
    """Return a stable language name for ``path`` or ``None`` if unsupported."""

    return LANGUAGE_BY_EXTENSION.get(_extension(path))
