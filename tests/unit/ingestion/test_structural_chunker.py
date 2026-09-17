"""Focused tests for deterministic Python structural chunking."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    ChunkKind,
    CodeChunk,
    SourceFile,
    chunk_source_file,
)


def _source(content: str, *, language: str | None = "python") -> SourceFile:
    return SourceFile(
        relative_path=Path("src/service.py"),
        language=language,
        content=content,
        size_bytes=len(content.encode()),
        line_count=len(content.splitlines()),
    )


def _structural(**updates: object) -> ChunkingConfig:
    return ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURAL,
        max_lines_per_chunk=updates.get("max_lines_per_chunk", 120),
        overlap_lines=updates.get("overlap_lines", 0),
        max_chars_per_chunk=updates.get("max_chars_per_chunk", 12_000),
    )


def test_python_functions_classes_methods_and_decorators_are_structural() -> None:
    content = (
        "import os\n"
        "\n"
        "@cached\n"
        "def first():\n"
        "    \"\"\"Keep this docstring.\"\"\"\n"
        "    return os.name\n"
        "\n"
        "class Service:\n"
        "    VALUE = 1\n"
        "\n"
        "    @classmethod\n"
        "    async def login(cls):\n"
        "        # Keep comments inside the exact slice.\n"
        "        return cls.VALUE\n"
    )

    chunks = chunk_source_file(_source(content), _structural())

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.chunk_kind for chunk in chunks] == [
        ChunkKind.MODULE,
        ChunkKind.FUNCTION,
        ChunkKind.CLASS,
        ChunkKind.METHOD,
    ]
    function = chunks[1]
    assert function.content.startswith("@cached\ndef first")
    assert function.qualified_symbol_name == "first"
    assert (function.start_line, function.end_line) == (3, 6)
    method = chunks[3]
    assert method.content.startswith("    @classmethod\n    async def login")
    assert method.symbol_name == "login"
    assert method.qualified_symbol_name == "Service.login"
    assert method.parent_symbol == "Service"
    assert (method.start_line, method.end_line) == (11, 14)


def test_nested_definitions_have_deterministic_qualified_names() -> None:
    content = (
        "def outer():\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner()\n"
    )

    first = chunk_source_file(_source(content), _structural())
    second = chunk_source_file(_source(content), _structural())

    assert first == second
    assert [chunk.qualified_symbol_name for chunk in first] == ["outer", "outer.inner"]
    assert first[0].content == content
    assert first[1].content == "    def inner():\n        return 1\n"
    assert first[1].parent_symbol == "outer"


def test_decorated_class_context_starts_at_decorator() -> None:
    content = "@registry.register\nclass Handler:\n    enabled = True\n"

    chunks = chunk_source_file(_source(content), _structural())

    assert len(chunks) == 1
    assert chunks[0].chunk_kind is ChunkKind.CLASS
    assert chunks[0].content == content
    assert chunks[0].start_line == 1
    assert chunks[0].qualified_symbol_name == "Handler"


def test_invalid_python_falls_back_to_preserved_line_boundaries() -> None:
    content = "def broken(:\r\n    pass\r\nthird\r\n"
    structural_config = _structural(max_lines_per_chunk=2, overlap_lines=1)
    line_config = structural_config.model_copy(update={"strategy": ChunkingStrategy.LINE})

    structural = chunk_source_file(_source(content), structural_config)
    line = chunk_source_file(_source(content), line_config)

    assert [(chunk.start_line, chunk.end_line, chunk.content) for chunk in structural] == [
        (chunk.start_line, chunk.end_line, chunk.content) for chunk in line
    ]
    assert all(chunk.chunking_strategy is ChunkingStrategy.STRUCTURAL for chunk in structural)
    assert all(chunk.chunk_kind is ChunkKind.LINE_FALLBACK for chunk in structural)
    assert "\r\n" in structural[0].content


def test_unsupported_language_uses_structural_line_fallback() -> None:
    chunks = chunk_source_file(
        _source("export const one = 1;\n", language="typescript"),
        _structural(),
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_kind is ChunkKind.LINE_FALLBACK
    assert chunks[0].content == "export const one = 1;\n"


def test_oversized_function_splits_into_bounded_exact_fragments() -> None:
    content = "def large():\n" + "".join(
        f"    value_{index} = {index}\n" for index in range(10)
    )
    chunks = chunk_source_file(
        _source(content),
        _structural(max_lines_per_chunk=4, max_chars_per_chunk=70),
    )

    assert len(chunks) > 1
    assert "".join(chunk.content for chunk in chunks) == content
    assert all(chunk.chunk_kind is ChunkKind.STRUCTURAL_FRAGMENT for chunk in chunks)
    assert all(chunk.qualified_symbol_name == "large" for chunk in chunks)
    assert all(len(chunk.content) <= 70 for chunk in chunks)
    assert all(chunk.end_line - chunk.start_line + 1 <= 4 for chunk in chunks)
    assert [chunk.fragment_index for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.fragment_count == len(chunks) for chunk in chunks)


def test_single_oversized_line_is_split_without_rewriting_source() -> None:
    content = "VALUE = '" + ("x" * 90) + "'\n"
    chunks = chunk_source_file(
        _source(content),
        _structural(max_lines_per_chunk=5, max_chars_per_chunk=25),
    )

    assert "".join(chunk.content for chunk in chunks) == content
    assert all(len(chunk.content) <= 25 for chunk in chunks)
    assert all((chunk.start_line, chunk.end_line) == (1, 1) for chunk in chunks)


def test_fragment_kind_and_position_metadata_must_agree() -> None:
    values = {
        "relative_path": "src/service.py",
        "language": "python",
        "start_line": 1,
        "end_line": 1,
        "content": "value = 1\n",
        "chunk_index": 0,
        "chunking_strategy": ChunkingStrategy.STRUCTURAL,
    }

    with pytest.raises(ValidationError, match="fragment metadata"):
        CodeChunk(**values, fragment_index=1, fragment_count=2)
    with pytest.raises(ValidationError, match="fragment metadata"):
        CodeChunk(**values, chunk_kind=ChunkKind.STRUCTURAL_FRAGMENT)


def test_structural_chunks_preserve_crlf_source_slices_and_citation_lines() -> None:
    content = "VALUE = 1\r\n\r\ndef answer():\r\n    return VALUE\r\n"
    chunks = chunk_source_file(_source(content), _structural())

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 2), (3, 4)]
    assert chunks[0].content == "VALUE = 1\r\n\r\n"
    assert chunks[1].content == "def answer():\r\n    return VALUE\r\n"
    assert all("\r\n" in chunk.content for chunk in chunks)
