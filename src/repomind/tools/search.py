"""Deterministic literal and lightweight declaration search tools."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from repomind.ingestion import IngestionConfig, find_source_files, is_supported_source_file
from repomind.tools.filesystem import _read_text, _resolve_workspace_path
from repomind.tools.models import (
    FindSymbolInput,
    FindSymbolOutput,
    SearchCodeInput,
    SearchCodeMatch,
    SearchCodeOutput,
    SymbolMatch,
    ToolConfig,
    ToolContext,
)
from repomind.tools.registry import ToolExecutionError


def _source_paths(context: ToolContext, scope: Path, config: ToolConfig) -> list[Path]:
    target = _resolve_workspace_path(context, scope, allow_root=True)
    if not target.exists():
        raise ToolExecutionError(f"Search path does not exist: {scope.as_posix()}")
    if target.is_file():
        return [target] if is_supported_source_file(target) else []
    if not target.is_dir():
        raise ToolExecutionError(f"Search path is not a file or directory: {scope.as_posix()}")

    paths = find_source_files(
        target,
        IngestionConfig(max_file_size_bytes=config.max_file_bytes),
    )
    return sorted(
        paths,
        key=lambda path: (
            path.relative_to(context.repository_root).as_posix().casefold(),
            path.relative_to(context.repository_root).as_posix(),
        ),
    )


def _source_lines(
    context: ToolContext,
    scope: Path,
    config: ToolConfig,
) -> Iterator[tuple[Path, int, str]]:
    for path in _source_paths(context, scope, config):
        try:
            content = _read_text(path, config.max_file_bytes)
        except ToolExecutionError:
            continue
        relative = path.relative_to(context.repository_root)
        for line_number, line in enumerate(content.splitlines(), start=1):
            yield relative, line_number, line


def search_code(
    context: ToolContext,
    arguments: SearchCodeInput,
    *,
    config: ToolConfig | None = None,
) -> SearchCodeOutput:
    """Search supported repository source files for a literal substring."""

    resolved_config = config or ToolConfig()
    result_limit = min(arguments.max_results, resolved_config.max_search_results)
    needle = arguments.query if arguments.case_sensitive else arguments.query.casefold()
    matches: list[SearchCodeMatch] = []
    truncated = False
    for path, line_number, line in _source_lines(
        context,
        arguments.path,
        resolved_config,
    ):
        haystack = line if arguments.case_sensitive else line.casefold()
        if needle not in haystack:
            continue
        if len(matches) == result_limit:
            truncated = True
            break
        matches.append(SearchCodeMatch(path=path, line_number=line_number, line=line))

    return SearchCodeOutput(
        query=arguments.query,
        matches=matches,
        truncated=truncated,
    )


def _symbol_patterns(symbol: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    escaped = re.escape(symbol)
    identifier_end = r"(?![A-Za-z0-9_$])"
    return (
        (
            "function",
            re.compile(rf"^\s*(?:async\s+)?def\s+{escaped}{identifier_end}\s*\("),
        ),
        (
            "function",
            re.compile(
                rf"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?"
                rf"function\s+{escaped}{identifier_end}\s*\("
            ),
        ),
        (
            "class",
            re.compile(
                rf"^\s*(?:export\s+(?:default\s+)?)?class\s+"
                rf"{escaped}{identifier_end}(?:\s*[(:]|\s*$)"
            ),
        ),
        (
            "variable",
            re.compile(
                rf"^\s*(?:export\s+)?(?:const|let)\s+{escaped}{identifier_end}\s*="
            ),
        ),
    )


def find_symbol(
    context: ToolContext,
    arguments: FindSymbolInput,
    *,
    config: ToolConfig | None = None,
) -> FindSymbolOutput:
    """Locate declaration-like matches without claiming full AST resolution."""

    resolved_config = config or ToolConfig()
    result_limit = min(arguments.max_results, resolved_config.max_search_results)
    patterns = _symbol_patterns(arguments.symbol)
    matches: list[SymbolMatch] = []
    truncated = False
    for path, line_number, line in _source_lines(
        context,
        arguments.path,
        resolved_config,
    ):
        kind = next((kind for kind, pattern in patterns if pattern.search(line)), None)
        if kind is None:
            continue
        if len(matches) == result_limit:
            truncated = True
            break
        matches.append(
            SymbolMatch(path=path, line_number=line_number, line=line, kind=kind)
        )

    return FindSymbolOutput(
        symbol=arguments.symbol,
        matches=matches,
        truncated=truncated,
    )
