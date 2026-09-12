"""Tests for literal code search and lightweight symbol location."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.tools import (
    FindSymbolInput,
    SearchCodeInput,
    ToolConfig,
    ToolContext,
    find_symbol,
    search_code,
)


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)


def test_search_code_is_literal_case_insensitive_and_deterministic(tmp_path: Path) -> None:
    _write(tmp_path / "z.py", "Needle once\nneedle twice\n")
    _write(tmp_path / "a.py", "NEEDLE first\n")

    output = search_code(
        ToolContext(repository_root=tmp_path),
        SearchCodeInput(query="needle"),
    )

    assert [(match.path.as_posix(), match.line_number, match.line) for match in output.matches] == [
        ("a.py", 1, "NEEDLE first"),
        ("z.py", 1, "Needle once"),
        ("z.py", 2, "needle twice"),
    ]
    assert not output.truncated


def test_search_code_case_sensitive_scoped_unicode_and_no_matches(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "one.py", "café\nCAFÉ\n")
    _write(tmp_path / "other.py", "café\n")
    context = ToolContext(repository_root=tmp_path)

    output = search_code(
        context,
        SearchCodeInput(query="café", path="src", case_sensitive=True),
    )
    assert [(match.path.as_posix(), match.line_number) for match in output.matches] == [
        ("src/one.py", 1)
    ]
    assert search_code(context, SearchCodeInput(query="missing")).matches == []

    file_scope = search_code(
        context,
        SearchCodeInput(query="CAFÉ", path="src/one.py", case_sensitive=True),
    )
    assert [(match.path.as_posix(), match.line_number) for match in file_scope.matches] == [
        ("src/one.py", 2)
    ]


def test_search_code_honors_request_and_global_limits(tmp_path: Path) -> None:
    _write(tmp_path / "many.py", "hit\nhit\nhit\n")
    context = ToolContext(repository_root=tmp_path)

    requested = search_code(context, SearchCodeInput(query="hit", max_results=1))
    configured = search_code(
        context,
        SearchCodeInput(query="hit", max_results=10),
        config=ToolConfig(max_search_results=2),
    )

    assert len(requested.matches) == 1 and requested.truncated
    assert len(configured.matches) == 2 and configured.truncated


def test_search_code_excludes_unsupported_binary_and_ignored_files(tmp_path: Path) -> None:
    _write(tmp_path / "valid.py", "target\n")
    _write(tmp_path / "binary.py", b"target\x00data")
    _write(tmp_path / "notes.txt", "target\n")
    _write(tmp_path / "image.png", b"target\x00data")
    _write(tmp_path / "node_modules" / "package.js", "target\n")

    output = search_code(
        ToolContext(repository_root=tmp_path),
        SearchCodeInput(query="target"),
    )
    assert [match.path.as_posix() for match in output.matches] == ["valid.py"]


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_search_code_rejects_empty_queries(query: str) -> None:
    with pytest.raises(ValidationError):
        SearchCodeInput(query=query)


def test_find_symbol_matches_exact_python_and_javascript_declarations(tmp_path: Path) -> None:
    _write(
        tmp_path / "app.py",
        "def target():\n    pass\nclass Target:\n    pass\ndef target_extra():\n    pass\n",
    )
    _write(
        tmp_path / "web.ts",
        "export function target() {}\nclass target_extra {}\nconst target = () => {};\n",
    )
    context = ToolContext(repository_root=tmp_path)

    functions = find_symbol(context, FindSymbolInput(symbol="target"))
    classes = find_symbol(context, FindSymbolInput(symbol="Target"))

    assert [(match.path.as_posix(), match.line_number, match.kind) for match in functions.matches] == [
        ("app.py", 1, "function"),
        ("web.ts", 1, "function"),
        ("web.ts", 3, "variable"),
    ]
    assert [(match.path.as_posix(), match.kind) for match in classes.matches] == [
        ("app.py", "class")
    ]


def test_find_symbol_scopes_results_and_does_not_fall_back_to_references(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "decl.py", "class Service:\n    pass\n")
    _write(tmp_path / "tests" / "reference.py", "value = Service()\n")
    context = ToolContext(repository_root=tmp_path)

    assert len(find_symbol(context, FindSymbolInput(symbol="Service", path="src")).matches) == 1
    assert find_symbol(context, FindSymbolInput(symbol="Service", path="tests")).matches == []
    assert find_symbol(context, FindSymbolInput(symbol="Missing")).matches == []
