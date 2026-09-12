"""Deterministic validation and dispatch for RepoMind tools."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


class ToolError(RuntimeError):
    """Base exception for predictable tool failures."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""


class ToolValidationError(ToolError):
    """Raised before execution when tool arguments violate their schema."""


class ToolExecutionError(ToolError):
    """Raised when a validated tool operation cannot complete safely."""


ToolHandler = Callable[[BaseModel], BaseModel]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One named tool with explicit input/output schemas and implementation."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: ToolHandler

    def input_json_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()


class ToolRegistry:
    """A small, OpenAI-independent registry with deterministic ordering."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if not tool.name or not tool.name.replace("_", "a").isalnum():
            raise ToolValidationError("tool name must contain letters, digits, or underscores")
        if tool.name in self._tools:
            raise ToolValidationError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Unknown tool: {name}") from exc

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def execute(self, name: str, arguments: Mapping[str, Any]) -> BaseModel:
        tool = self.get(name)
        try:
            validated_input = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(f"Invalid arguments for tool {name}") from exc

        try:
            result = tool.handler(validated_input)
            return tool.output_model.model_validate(result)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool {name} failed") from exc
