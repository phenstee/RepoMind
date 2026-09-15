"""Deterministic validation and dispatch for RepoMind tools."""

from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from repomind.observability import TraceContext
from repomind.observability.sanitization import (
    sanitize_error,
    sanitize_tool_arguments,
    sanitize_tool_output,
)


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

    def __init__(self, *, trace: TraceContext | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self.trace = trace

    def with_trace(self, trace: TraceContext) -> "ToolRegistry":
        """Bind a run to an independent registry view, sharing immutable definitions."""
        registry = copy(self)
        registry.trace = trace
        registry._tools = self._tools.copy()
        return registry

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

    def execute(
        self, name: str, arguments: Mapping[str, Any], *, trace: TraceContext | None = None
    ) -> BaseModel:
        trace = trace if trace is not None else self.trace
        if trace is None or trace.run_id is None:
            return self._execute(name, arguments)
        started = trace.now()
        metadata = {
            "tool": name,
            **trace.project(lambda: sanitize_tool_arguments(name, arguments)),
            "workspace_revision": trace.workspace_revision,
        }
        trace.emit("tool.started", **metadata)
        verification = name in {"run_tests", "run_ruff"}
        if verification:
            trace.emit("verification.started", **metadata)
        try:
            result = self._execute(name, arguments)
        except BaseException as exc:
            failure_kind = (
                "validation"
                if isinstance(exc, (ToolValidationError, ToolNotFoundError))
                else "execution"
            )
            failure = {**metadata, **sanitize_error(exc), "failure_kind": failure_kind}
            trace.emit("tool.failed", duration_ms=trace.elapsed(started), **failure)
            if verification:
                trace.emit("verification.failed", duration_ms=trace.elapsed(started), **failure)
            raise
        output = trace.project(lambda: sanitize_tool_output(name, result.model_dump()))
        trace.emit(
            "tool.completed",
            duration_ms=trace.elapsed(started),
            **(metadata | output | {"output_type": type(result).__name__}),
        )
        if verification:
            trace.emit(
                "verification.completed", duration_ms=trace.elapsed(started), **(metadata | output)
            )
        if name in {"create_file", "replace_text"}:
            trace.workspace_revision += 1
            trace.emit(
                "file.mutated",
                **output,
                mutation_type=name,
                workspace_revision=trace.workspace_revision,
            )
        return result

    def _execute(self, name: str, arguments: Mapping[str, Any]) -> BaseModel:
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
