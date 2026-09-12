"""Safe read-only tools for direct inspection of a repository workspace."""

from functools import partial

from repomind.tools.filesystem import list_directory, read_file
from repomind.tools.git import git_diff, git_status
from repomind.tools.models import (
    DirectoryEntry,
    FindSymbolInput,
    FindSymbolOutput,
    GitChangedFile,
    GitDiffInput,
    GitDiffOutput,
    GitStatusInput,
    GitStatusOutput,
    ListDirectoryInput,
    ListDirectoryOutput,
    ReadFileInput,
    ReadFileOutput,
    SearchCodeInput,
    SearchCodeMatch,
    SearchCodeOutput,
    SymbolMatch,
    ToolConfig,
    ToolContext,
)
from repomind.tools.registry import (
    ToolDefinition,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
    ToolValidationError,
)
from repomind.tools.search import find_symbol, search_code


def create_default_tool_registry(
    context: ToolContext,
    *,
    config: ToolConfig | None = None,
) -> ToolRegistry:
    """Create an isolated registry containing RepoMind's six read-only tools."""

    resolved_config = config or ToolConfig()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read all or part of a text file inside the repository.",
            input_model=ReadFileInput,
            output_model=ReadFileOutput,
            handler=partial(read_file, context, config=resolved_config),
        )
    )
    registry.register(
        ToolDefinition(
            name="list_directory",
            description="Inspect files and directories inside the repository.",
            input_model=ListDirectoryInput,
            output_model=ListDirectoryOutput,
            handler=partial(list_directory, context, config=resolved_config),
        )
    )
    registry.register(
        ToolDefinition(
            name="search_code",
            description="Search repository source files for an exact text substring.",
            input_model=SearchCodeInput,
            output_model=SearchCodeOutput,
            handler=partial(search_code, context, config=resolved_config),
        )
    )
    registry.register(
        ToolDefinition(
            name="find_symbol",
            description="Locate likely declarations of a named code symbol.",
            input_model=FindSymbolInput,
            output_model=FindSymbolOutput,
            handler=partial(find_symbol, context, config=resolved_config),
        )
    )
    registry.register(
        ToolDefinition(
            name="git_status",
            description="Inspect the current repository branch and working-tree changes.",
            input_model=GitStatusInput,
            output_model=GitStatusOutput,
            handler=partial(git_status, context, config=resolved_config),
        )
    )
    registry.register(
        ToolDefinition(
            name="git_diff",
            description="Inspect staged or unstaged Git changes without modifying the repository.",
            input_model=GitDiffInput,
            output_model=GitDiffOutput,
            handler=partial(git_diff, context, config=resolved_config),
        )
    )
    return registry


__all__ = [
    "DirectoryEntry",
    "FindSymbolInput",
    "FindSymbolOutput",
    "GitChangedFile",
    "GitDiffInput",
    "GitDiffOutput",
    "GitStatusInput",
    "GitStatusOutput",
    "ListDirectoryInput",
    "ListDirectoryOutput",
    "ReadFileInput",
    "ReadFileOutput",
    "SearchCodeInput",
    "SearchCodeMatch",
    "SearchCodeOutput",
    "SymbolMatch",
    "ToolConfig",
    "ToolContext",
    "ToolDefinition",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolValidationError",
    "create_default_tool_registry",
    "find_symbol",
    "git_diff",
    "git_status",
    "list_directory",
    "read_file",
    "search_code",
]
