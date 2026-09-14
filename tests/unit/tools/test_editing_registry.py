"""Capability-boundary tests for read-only and editing registries."""

from pathlib import Path

from repomind.tools import (
    ToolContext,
    create_default_tool_registry,
    create_editing_tool_registry,
)

READ_ONLY_TOOLS = {
    "find_symbol",
    "git_diff",
    "git_status",
    "list_directory",
    "read_file",
    "search_code",
}
EDITING_TOOLS = READ_ONLY_TOOLS | {
    "create_file",
    "replace_text",
    "run_ruff",
    "run_tests",
}


def test_default_registry_remains_strictly_read_only(tmp_path: Path) -> None:
    names = {
        tool.name
        for tool in create_default_tool_registry(
            ToolContext(repository_root=tmp_path)
        ).list_tools()
    }
    assert names == READ_ONLY_TOOLS
    assert not {"create_file", "replace_text", "run_tests", "run_ruff"} & names


def test_editing_registry_contains_only_intended_capabilities(tmp_path: Path) -> None:
    names = {
        tool.name
        for tool in create_editing_tool_registry(
            ToolContext(repository_root=tmp_path)
        ).list_tools()
    }
    assert names == EDITING_TOOLS
    assert not {
        "delete_file",
        "rename_file",
        "shell",
        "run_command",
        "python_exec",
        "git_commit",
        "git_push",
    } & names
