"""Explicit finite decision/tool/observation loop for read-only analysis."""

import json
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from repomind.agent.models import (
    AgentConfig,
    AgentDecision,
    AgentRun,
    AgentRunStatus,
    AgentStep,
    ToolObservation,
)
from repomind.agent.prompts import READ_ONLY_AGENT_SYSTEM_PROMPT, build_agent_prompt
from repomind.llm import LLMError
from repomind.tools import ToolError, ToolRegistry

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)
MAX_IDENTICAL_TOOL_EXECUTIONS = 2


class AgentError(RuntimeError):
    """Raised when the LLM or internal agent contract fails unrecoverably."""


class StructuredAgentLLM(Protocol):
    """Smallest structured-generation surface required by the agent loop."""

    def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredModelT],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> StructuredModelT:
        """Generate and validate one structured decision."""


def _tool_call_identity(tool_name: str, arguments: Mapping[str, Any]) -> str:
    payload = {"arguments": arguments, "tool_name": tool_name}
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AgentError("Tool arguments are not JSON-serializable") from exc


def _execute_tool(
    registry: ToolRegistry,
    decision: AgentDecision,
) -> ToolObservation:
    if decision.tool_name is None or decision.tool_arguments is None:
        raise AgentError("Validated tool decision is missing tool fields")
    try:
        output = registry.execute(decision.tool_name, decision.tool_arguments)
    except ToolError as exc:
        return ToolObservation(
            tool_name=decision.tool_name,
            arguments=decision.tool_arguments,
            success=False,
            error=str(exc),
        )
    return ToolObservation(
        tool_name=decision.tool_name,
        arguments=decision.tool_arguments,
        success=True,
        output=output.model_dump(mode="json"),
    )


def run_read_only_agent(
    query: str,
    llm_provider: StructuredAgentLLM,
    tool_registry: ToolRegistry,
    *,
    config: AgentConfig | None = None,
) -> AgentRun:
    """Run sequential decisions until a final answer or hard iteration limit."""

    if not isinstance(query, str) or not query.strip():
        raise AgentError("query must not be empty or whitespace-only")
    resolved_config = config or AgentConfig()
    steps: list[AgentStep] = []
    tool_call_counts: dict[str, int] = {}
    tool_calls = 0

    for iteration in range(1, resolved_config.max_iterations + 1):
        prompt = build_agent_prompt(
            query,
            tool_registry,
            steps,
            max_history_chars=resolved_config.max_history_chars,
        )
        try:
            decision = llm_provider.generate_structured(
                prompt,
                AgentDecision,
                system_prompt=READ_ONLY_AGENT_SYSTEM_PROMPT,
                temperature=0.0,
            )
        except (LLMError, ValidationError) as exc:
            raise AgentError("Structured agent decision failed") from exc
        if not isinstance(decision, AgentDecision):
            raise AgentError("LLM returned an unexpected agent decision model")

        if decision.action == "final":
            step = AgentStep(iteration=iteration, decision=decision)
            steps.append(step)
            return AgentRun(
                query=query,
                status=AgentRunStatus.COMPLETED,
                final_answer=decision.final_answer,
                steps=tuple(steps),
                iterations=iteration,
                llm_calls=iteration,
                tool_calls=tool_calls,
            )

        if decision.tool_name is None or decision.tool_arguments is None:
            raise AgentError("Validated tool decision is missing tool fields")
        identity = _tool_call_identity(decision.tool_name, decision.tool_arguments)
        prior_executions = tool_call_counts.get(identity, 0)
        if prior_executions >= MAX_IDENTICAL_TOOL_EXECUTIONS:
            observation = ToolObservation(
                tool_name=decision.tool_name,
                arguments=decision.tool_arguments,
                success=False,
                error="Repeated identical tool call; choose a different action.",
            )
        else:
            tool_call_counts[identity] = prior_executions + 1
            tool_calls += 1
            observation = _execute_tool(tool_registry, decision)
        steps.append(
            AgentStep(
                iteration=iteration,
                decision=decision,
                observation=observation,
            )
        )

    return AgentRun(
        query=query,
        status=AgentRunStatus.MAX_ITERATIONS,
        steps=tuple(steps),
        iterations=resolved_config.max_iterations,
        llm_calls=resolved_config.max_iterations,
        tool_calls=tool_calls,
    )
