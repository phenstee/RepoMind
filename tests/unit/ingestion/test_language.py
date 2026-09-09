"""Tests for centralized source-file language mapping."""

from pathlib import Path

from repomind.ingestion.language import (
    LANGUAGE_BY_EXTENSION,
    SUPPORTED_EXTENSIONS,
    is_supported_source_file,
    language_for_path,
)


def test_minimum_supported_extensions_are_present() -> None:
    required = {
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

    assert required <= SUPPORTED_EXTENSIONS


def test_language_mapping_examples() -> None:
    assert LANGUAGE_BY_EXTENSION[".py"] == "python"
    assert LANGUAGE_BY_EXTENSION[".ts"] == "typescript"
    assert LANGUAGE_BY_EXTENSION[".tsx"] == "typescript"
    assert LANGUAGE_BY_EXTENSION[".js"] == "javascript"
    assert LANGUAGE_BY_EXTENSION[".go"] == "go"
    assert LANGUAGE_BY_EXTENSION[".java"] == "java"
    assert LANGUAGE_BY_EXTENSION[".rs"] == "rust"
    assert LANGUAGE_BY_EXTENSION[".sql"] == "sql"


def test_extension_matching_is_case_insensitive() -> None:
    assert is_supported_source_file(Path("app.PY"))
    assert language_for_path(Path("service.Py")) == "python"
    assert language_for_path(Path("README.MD")) == "markdown"


def test_unknown_extension_is_unsupported() -> None:
    assert not is_supported_source_file(Path("notes.txt"))
    assert language_for_path(Path("image.png")) is None
