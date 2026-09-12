"""Tests for deterministic schema validation and tool dispatch."""

from typing import Any

import pytest
from pydantic import BaseModel

from repomind.tools import (
    ToolContext,
    ToolDefinition,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
    ToolValidationError,
    create_default_tool_registry,
)


class _Input(BaseModel):
    value: int


class _Output(BaseModel):
    doubled: int


def _tool(name: str, calls: list[int] | None = None) -> ToolDefinition:
    def handler(arguments: BaseModel) -> BaseModel:
        validated = _Input.model_validate(arguments)
        if calls is not None:
            calls.append(validated.value)
        return _Output(doubled=validated.value * 2)

    return ToolDefinition(name, "Double a number.", _Input, _Output, handler)


def test_registry_registers_gets_lists_and_executes_deterministically() -> None:
    calls: list[int] = []
    registry = ToolRegistry()
    registry.register(_tool("zeta", calls))
    registry.register(_tool("alpha", calls))

    assert registry.get("zeta").name == "zeta"
    assert [tool.name for tool in registry.list_tools()] == ["alpha", "zeta"]
    assert registry.execute("zeta", {"value": 4}) == _Output(doubled=8)
    assert calls == [4]


def test_registry_rejects_duplicate_and_unknown_names() -> None:
    registry = ToolRegistry()
    registry.register(_tool("same"))

    with pytest.raises(ToolValidationError, match="already registered"):
        registry.register(_tool("same"))
    with pytest.raises(ToolNotFoundError, match="Unknown tool"):
        registry.get("missing")


def test_invalid_input_is_rejected_before_handler() -> None:
    calls: list[int] = []
    registry = ToolRegistry()
    registry.register(_tool("double", calls))

    with pytest.raises(ToolValidationError, match="Invalid arguments"):
        registry.execute("double", {"value": "not-an-integer"})
    assert calls == []


def test_handler_failure_is_normalized_and_chained() -> None:
    def fail(arguments: BaseModel) -> BaseModel:
        del arguments
        raise OSError("private detail")

    registry = ToolRegistry()
    registry.register(ToolDefinition("fail", "Fail.", _Input, _Output, fail))

    with pytest.raises(ToolExecutionError, match="Tool fail failed") as error_info:
        registry.execute("fail", {"value": 1})
    assert isinstance(error_info.value.__cause__, OSError)


def test_tool_input_json_schema_is_openai_independent() -> None:
    schema: dict[str, Any] = _tool("double").input_json_schema()
    assert schema["properties"]["value"]["type"] == "integer"
    assert schema["required"] == ["value"]


def test_default_registry_has_six_tools_and_rejects_undeclared_arguments(tmp_path) -> None:
    registry = create_default_tool_registry(ToolContext(repository_root=tmp_path))
    assert [tool.name for tool in registry.list_tools()] == [
        "find_symbol",
        "git_diff",
        "git_status",
        "list_directory",
        "read_file",
        "search_code",
    ]
    with pytest.raises(ToolValidationError):
        registry.execute("git_status", {"command": "status; remove-everything"})
