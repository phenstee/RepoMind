"""Deterministic Python AST chunking that preserves exact source slices."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import ceil

from repomind.ingestion.models import (
    ChunkingConfig,
    ChunkingStrategy,
    ChunkKind,
    CodeChunk,
    SourceFile,
)

_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


@dataclass(frozen=True)
class _StructuralUnit:
    start_line: int
    end_line: int
    kind: ChunkKind
    symbol_name: str | None = None
    qualified_symbol_name: str | None = None
    parent_symbol: str | None = None
    preferred_breaks: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Piece:
    text: str
    line: int


def _node_start(node: _Definition) -> int:
    decorators = getattr(node, "decorator_list", ())
    return min((decorator.lineno for decorator in decorators), default=node.lineno)


def _qualified(parent: str | None, name: str) -> str:
    return f"{parent}.{name}" if parent else name


def _statement_breaks(node: ast.AST) -> tuple[int, ...]:
    return tuple(
        statement.end_lineno
        for statement in getattr(node, "body", ())
        if getattr(statement, "end_lineno", None) is not None
    )


def _nested_definitions(statements: list[ast.stmt]) -> list[_Definition]:
    """Return nested definitions without walking into their own scopes."""

    found: list[_Definition] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.append(child)
            else:
                visit(child)

    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append(statement)
        else:
            visit(statement)
    return sorted(found, key=lambda item: (_node_start(item), item.end_lineno or item.lineno))


def _function_units(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    parent: str | None,
    method: bool,
) -> list[_StructuralUnit]:
    qualified = _qualified(parent, node.name)
    units = [
        _StructuralUnit(
            start_line=_node_start(node),
            end_line=node.end_lineno or node.lineno,
            kind=ChunkKind.METHOD if method else ChunkKind.FUNCTION,
            symbol_name=node.name,
            qualified_symbol_name=qualified,
            parent_symbol=parent,
            preferred_breaks=_statement_breaks(node),
        )
    ]
    for nested in _nested_definitions(node.body):
        units.extend(_definition_units(nested, parent=qualified, method=False))
    return units


def _class_units(node: ast.ClassDef, *, parent: str | None) -> list[_StructuralUnit]:
    qualified = _qualified(parent, node.name)
    children = [
        statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not children:
        return [
            _StructuralUnit(
                start_line=_node_start(node),
                end_line=node.end_lineno or node.lineno,
                kind=ChunkKind.CLASS,
                symbol_name=node.name,
                qualified_symbol_name=qualified,
                parent_symbol=parent,
                preferred_breaks=_statement_breaks(node),
            )
        ]

    units: list[_StructuralUnit] = []
    cursor = _node_start(node)
    class_end = node.end_lineno or node.lineno
    for child in children:
        child_start = _node_start(child)
        if cursor < child_start:
            units.append(
                _StructuralUnit(
                    start_line=cursor,
                    end_line=child_start - 1,
                    kind=ChunkKind.CLASS,
                    symbol_name=node.name,
                    qualified_symbol_name=qualified,
                    parent_symbol=parent,
                )
            )
        units.extend(
            _definition_units(
                child,
                parent=qualified,
                method=isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)),
            )
        )
        cursor = (child.end_lineno or child.lineno) + 1
    if cursor <= class_end:
        units.append(
            _StructuralUnit(
                start_line=cursor,
                end_line=class_end,
                kind=ChunkKind.CLASS,
                symbol_name=node.name,
                qualified_symbol_name=qualified,
                parent_symbol=parent,
            )
        )
    return units


def _definition_units(
    node: _Definition,
    *,
    parent: str | None,
    method: bool = False,
) -> list[_StructuralUnit]:
    if isinstance(node, ast.ClassDef):
        return _class_units(node, parent=parent)
    return _function_units(node, parent=parent, method=method)


def _module_units(tree: ast.Module, line_count: int) -> list[_StructuralUnit]:
    definitions = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    units: list[_StructuralUnit] = []
    cursor = 1
    for node in definitions:
        start = _node_start(node)
        if cursor < start:
            units.append(_StructuralUnit(cursor, start - 1, ChunkKind.MODULE))
        units.extend(_definition_units(node, parent=None))
        cursor = (node.end_lineno or node.lineno) + 1
    if cursor <= line_count:
        units.append(_StructuralUnit(cursor, line_count, ChunkKind.MODULE))
    return units


def _pieces_for_range(
    lines: list[str],
    start_line: int,
    end_line: int,
    max_chars: int,
) -> list[_Piece]:
    pieces: list[_Piece] = []
    for line_number in range(start_line, end_line + 1):
        text = lines[line_number - 1]
        if not text:
            continue
        piece_count = ceil(len(text) / max_chars)
        piece_size, larger_pieces = divmod(len(text), piece_count)
        offset = 0
        for piece_index in range(piece_count):
            end = offset + piece_size + (piece_index < larger_pieces)
            pieces.append(_Piece(text=text[offset:end], line=line_number))
            offset = end
    return pieces


def _bounded_parts(
    lines: list[str],
    unit: _StructuralUnit,
    config: ChunkingConfig,
) -> list[tuple[int, int, str]]:
    pieces = _pieces_for_range(
        lines,
        unit.start_line,
        unit.end_line,
        config.max_chars_per_chunk,
    )
    if not pieces:
        return []

    parts: list[tuple[int, int, str]] = []
    cursor = 0
    preferred = set(unit.preferred_breaks)
    while cursor < len(pieces):
        end = cursor
        chars = 0
        start_line = pieces[cursor].line
        preferred_end: int | None = None
        while end < len(pieces):
            piece = pieces[end]
            line_span = piece.line - start_line + 1
            if chars + len(piece.text) > config.max_chars_per_chunk:
                break
            if line_span > config.max_lines_per_chunk:
                break
            chars += len(piece.text)
            end += 1
            next_line = pieces[end].line if end < len(pieces) else None
            if piece.line in preferred and next_line != piece.line:
                preferred_end = end

        if end == cursor:
            # Each atomic piece is already no larger than max_chars_per_chunk.
            end += 1
        elif end < len(pieces) and preferred_end is not None:
            end = preferred_end

        selected = pieces[cursor:end]
        parts.append(
            (selected[0].line, selected[-1].line, "".join(piece.text for piece in selected))
        )
        cursor = end
    return parts


def _fallback_chunks(source_file: SourceFile, config: ChunkingConfig) -> list[CodeChunk]:
    # Imported lazily so the public dispatcher can import this module without a cycle.
    from repomind.ingestion.chunker import _line_chunks

    chunks = _line_chunks(
        source_file,
        config.model_copy(update={"strategy": ChunkingStrategy.LINE}),
    )
    return [
        chunk.model_copy(
            update={
                "chunking_strategy": ChunkingStrategy.STRUCTURAL,
                "chunk_kind": ChunkKind.LINE_FALLBACK,
            }
        )
        for chunk in chunks
    ]


def chunk_python_source(source_file: SourceFile, config: ChunkingConfig) -> list[CodeChunk]:
    """Chunk valid Python by source structure, falling back without failing indexing."""

    if source_file.language != "python":
        return _fallback_chunks(source_file, config)
    try:
        tree = ast.parse(source_file.content)
    except (SyntaxError, ValueError, MemoryError, OverflowError, RecursionError):
        return _fallback_chunks(source_file, config)

    lines = source_file.content.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    for unit in _module_units(tree, len(lines)):
        parts = _bounded_parts(lines, unit, config)
        if not parts or not any(content.strip() for _, _, content in parts):
            continue
        fragment_count = len(parts)
        for fragment_index, (start_line, end_line, content) in enumerate(parts, start=1):
            fragmented = fragment_count > 1
            chunks.append(
                CodeChunk(
                    relative_path=source_file.relative_path,
                    language=source_file.language,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    chunk_index=len(chunks),
                    chunking_strategy=ChunkingStrategy.STRUCTURAL,
                    chunk_kind=(
                        ChunkKind.STRUCTURAL_FRAGMENT if fragmented else unit.kind
                    ),
                    symbol_name=unit.symbol_name,
                    qualified_symbol_name=unit.qualified_symbol_name,
                    parent_symbol=unit.parent_symbol,
                    fragment_index=fragment_index if fragmented else None,
                    fragment_count=fragment_count if fragmented else None,
                )
            )
    return chunks
