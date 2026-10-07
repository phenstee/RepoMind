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
from repomind.tools.models import MAX_SEARCH_LINE_CHARS


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


def test_search_results_bound_long_lines_and_keep_the_match_visible(tmp_path: Path) -> None:
    minified = "var a='" + "A" * 900_000 + "';var NEEDLE=1;" + "B" * 50_000 + "\n"
    _write(tmp_path / "min.js", minified)
    _write(tmp_path / "short.py", "NEEDLE = 2\n")
    context = ToolContext(repository_root=tmp_path)

    output = search_code(context, SearchCodeInput(query="needle"))

    long_match, short_match = output.matches
    assert long_match.path == Path("min.js")
    assert long_match.line_truncated
    assert len(long_match.line) == MAX_SEARCH_LINE_CHARS
    assert "NEEDLE" in long_match.line
    assert short_match.line == "NEEDLE = 2" and not short_match.line_truncated
    assert len(output.model_dump_json()) < 5_000


def test_find_symbol_bounds_long_declaration_lines(tmp_path: Path) -> None:
    _write(tmp_path / "gen.py", "def generated(" + "x, " * 2_000 + "):\n    pass\n")

    output = find_symbol(ToolContext(repository_root=tmp_path), FindSymbolInput(symbol="generated"))

    (match,) = output.matches
    assert match.line_truncated
    assert match.line.startswith("def generated(")
    assert len(match.line) == MAX_SEARCH_LINE_CHARS


def test_search_never_descends_into_git_metadata(tmp_path: Path) -> None:
    _write(tmp_path / ".git" / "hooks" / "helper.py", "def leaked_symbol():\n    pass\n")
    _write(tmp_path / "app.py", "x = 1\n")
    context = ToolContext(repository_root=tmp_path)

    assert search_code(context, SearchCodeInput(query="leaked_symbol")).matches == []
    assert find_symbol(context, FindSymbolInput(symbol="leaked_symbol")).matches == []
    with pytest.raises(ValidationError, match="Git metadata"):
        SearchCodeInput(query="x", path=".git")


def test_search_line_numbers_match_newline_based_numbering(tmp_path: Path) -> None:
    content = "\x0c\nmarker_one = 1\x0c\n#   separator\nmarker_two = 2\r\ndef target():\n"
    _write(tmp_path / "sample.py", content)
    context = ToolContext(repository_root=tmp_path)

    matches = search_code(context, SearchCodeInput(query="marker_")).matches
    (symbol,) = find_symbol(context, FindSymbolInput(symbol="target")).matches

    assert [(match.line_number, match.line) for match in matches] == [
        (2, "marker_one = 1\x0c"),
        (4, "marker_two = 2"),
    ]
    assert (symbol.line_number, symbol.line) == (5, "def target():")
