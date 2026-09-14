"""Explicit read-only and controlled-editing repository tool registries."""

from functools import partial

from repomind.tools.editing import create_file, replace_text
from repomind.tools.filesystem import list_directory, read_file
from repomind.tools.git import git_diff, git_status
from repomind.tools.models import (
    CreateFileInput,
    CreateFileOutput,
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
    ReplaceTextInput,
    ReplaceTextOutput,
    RunRuffInput,
    RunRuffOutput,
    RunTestsInput,
    RunTestsOutput,
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
from repomind.tools.verification import run_ruff, run_tests


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


def create_editing_tool_registry(
    context: ToolContext,
    *,
    config: ToolConfig | None = None,
) -> ToolRegistry:
    """Explicitly opt into precise mutation and fixed local verification tools."""

    resolved_config = config or ToolConfig()
    registry = create_default_tool_registry(context, config=resolved_config)
    registry.register(
        ToolDefinition(
            name="create_file",
            description="Create one bounded UTF-8 file in an existing repository directory.",
            input_model=CreateFileInput,
            output_model=CreateFileOutput,
            handler=partial(create_file, context, config=resolved_config),
        )
    )
    registry.register(
        ToolDefinition(
            name="replace_text",
            description=(
                "Replace one exact literal occurrence in a repository text file using "
                "an expected SHA-256 precondition."
            ),
            input_model=ReplaceTextInput,
            output_model=ReplaceTextOutput,
            handler=partial(replace_text, context, config=resolved_config),
        )
    )
    registry.register(
        ToolDefinition(
            name="run_tests",
            description="Run fixed, bounded pytest verification on validated test paths.",
            input_model=RunTestsInput,
            output_model=RunTestsOutput,
            handler=partial(run_tests, context, config=resolved_config),
        )
    )
    registry.register(
        ToolDefinition(
            name="run_ruff",
            description="Run fixed, bounded Ruff checks on validated repository paths.",
            input_model=RunRuffInput,
            output_model=RunRuffOutput,
            handler=partial(run_ruff, context, config=resolved_config),
        )
    )
    return registry


__all__ = [
    "CreateFileInput",
    "CreateFileOutput",
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
    "ReplaceTextInput",
    "ReplaceTextOutput",
    "RunRuffInput",
    "RunRuffOutput",
    "RunTestsInput",
    "RunTestsOutput",
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
    "create_editing_tool_registry",
    "create_file",
    "find_symbol",
    "git_diff",
    "git_status",
    "list_directory",
    "read_file",
    "replace_text",
    "run_ruff",
    "run_tests",
    "search_code",
]
