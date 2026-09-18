"""Explicit finite decision/tool/observation loop for repository agents."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from repomind.agent.models import (
    AgentConfig,
    AgentDecision,
    AgentRun,
    AgentRunStatus,
    AgentStep,
    EditingAgentConfig,
    ToolObservation,
    WorkflowFeedback,
)
from repomind.agent.prompts import (
    EDITING_AGENT_SYSTEM_PROMPT,
    READ_ONLY_AGENT_SYSTEM_PROMPT,
    build_agent_prompt,
)
from repomind.jobs.control import CooperativeCancellation, NoCancellation
from repomind.llm import LLMError
from repomind.observability import TraceContext, TraceRecorder
from repomind.observability.instrumentation import generate_structured, traced_run
from repomind.tools import ToolError, ToolRegistry

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)
MAX_IDENTICAL_TOOL_EXECUTIONS = 2


@dataclass(frozen=True, slots=True)
class _FinalDecisionControl:
    """Private completion interception result used by the coding workflow."""

    feedback: WorkflowFeedback | None = None
    stop: bool = False


_ObservationHandler = Callable[[AgentStep], None]
_FinalDecisionHandler = Callable[
    [AgentDecision, tuple[AgentStep, ...]],
    _FinalDecisionControl,
]


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
    trace: TraceContext | None = None,
) -> ToolObservation:
    if decision.tool_name is None or decision.tool_arguments is None:
        raise AgentError("Validated tool decision is missing tool fields")
    try:
        output = registry.execute(decision.tool_name, decision.tool_arguments, trace=trace)
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


def _run_agent(
    query: str,
    llm_provider: StructuredAgentLLM,
    tool_registry: ToolRegistry,
    *,
    config: AgentConfig,
    system_prompt: str,
    mutation_tool_names: frozenset[str] = frozenset(),
    max_mutations: int | None = None,
    observation_handler: _ObservationHandler | None = None,
    final_decision_handler: _FinalDecisionHandler | None = None,
    trace: TraceContext | None = None,
    cancellation: CooperativeCancellation | None = None,
) -> AgentRun:
    """Run shared sequential loop mechanics with application-chosen capabilities."""

    if not isinstance(query, str) or not query.strip():
        raise AgentError("query must not be empty or whitespace-only")
    trace = trace if trace is not None else TraceContext()
    cancellation = cancellation or NoCancellation()
    steps: list[AgentStep] = []
    tool_call_counts: dict[str, int] = {}
    tool_calls = 0
    successful_mutations = 0

    for iteration in range(1, config.max_iterations + 1):
        cancellation.checkpoint()
        prompt = build_agent_prompt(
            query,
            tool_registry,
            steps,
            max_history_chars=config.max_history_chars,
        )
        try:
            decision = generate_structured(
                llm_provider,
                prompt,
                AgentDecision,
                system_prompt=system_prompt,
                temperature=0.0,
                trace=trace,
            )
        except (LLMError, ValidationError) as exc:
            raise AgentError("Structured agent decision failed") from exc
        if not isinstance(decision, AgentDecision):
            raise AgentError("LLM returned an unexpected agent decision model")
        cancellation.checkpoint()

        trace.emit(
            "agent.decision",
            iteration=iteration,
            action=decision.action,
            tool=decision.tool_name,
            answer_chars=len(decision.final_answer or ""),
        )
        if decision.action == "final":
            control = (
                final_decision_handler(decision, tuple(steps))
                if final_decision_handler is not None
                else _FinalDecisionControl()
            )
            step = AgentStep(
                iteration=iteration,
                decision=decision,
                workflow_feedback=control.feedback,
            )
            steps.append(step)
            if control.feedback is not None:
                if control.stop:
                    return AgentRun(
                        query=query,
                        status=AgentRunStatus.WORKFLOW_STOPPED,
                        steps=tuple(steps),
                        iterations=iteration,
                        llm_calls=iteration,
                        tool_calls=tool_calls,
                    )
                continue
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
        is_mutation = decision.tool_name in mutation_tool_names
        if is_mutation and max_mutations is not None and successful_mutations >= max_mutations:
            trace.emit("tool.blocked", tool=decision.tool_name, failure_kind="mutation_limit")
            observation = ToolObservation(
                tool_name=decision.tool_name,
                arguments=decision.tool_arguments,
                success=False,
                error="Successful mutation budget exhausted; do not modify another file.",
            )
        elif prior_executions >= MAX_IDENTICAL_TOOL_EXECUTIONS:
            trace.emit("tool.blocked", tool=decision.tool_name, failure_kind="repeated_call")
            observation = ToolObservation(
                tool_name=decision.tool_name,
                arguments=decision.tool_arguments,
                success=False,
                error="Repeated identical tool call; choose a different action.",
            )
        else:
            tool_call_counts[identity] = prior_executions + 1
            tool_calls += 1
            cancellation.checkpoint()
            observation = _execute_tool(tool_registry, decision, trace)
            if is_mutation and observation.success:
                cancellation.side_effect_started()
                successful_mutations += 1
            cancellation.checkpoint()
        steps.append(
            AgentStep(
                iteration=iteration,
                decision=decision,
                observation=observation,
            )
        )
        if observation_handler is not None:
            observation_handler(steps[-1])

    trace.emit(
        "agent.stopped",
        domain_status="max_iterations_reached",
        blocker_codes=["agent_limit"],
        iteration=config.max_iterations,
    )
    return AgentRun(
        query=query,
        status=AgentRunStatus.MAX_ITERATIONS,
        steps=tuple(steps),
        iterations=config.max_iterations,
        llm_calls=config.max_iterations,
        tool_calls=tool_calls,
    )


@traced_run("read_only_agent")
def run_read_only_agent(
    query: str,
    llm_provider: StructuredAgentLLM,
    tool_registry: ToolRegistry,
    *,
    config: AgentConfig | None = None,
    system_prompt: str | None = None,
    recorder: TraceRecorder | None = None,
    trace: TraceContext | None = None,
    cancellation: CooperativeCancellation | None = None,
) -> AgentRun:
    """Run with caller-supplied capabilities under the read-only agent contract.

    ``system_prompt`` defaults to the baseline filesystem-only instructions.
    Callers composing a registry with additional tools (for example indexed
    navigation) may supply a matching prompt; the loop itself never inspects
    which tools are registered.
    """

    return _run_read_only_agent_controlled(
        query,
        llm_provider,
        tool_registry,
        config=config or AgentConfig(),
        system_prompt=system_prompt or READ_ONLY_AGENT_SYSTEM_PROMPT,
        trace=trace,
        cancellation=cancellation,
    )


def _run_read_only_agent_controlled(
    query: str,
    llm_provider: StructuredAgentLLM,
    tool_registry: ToolRegistry,
    *,
    config: AgentConfig,
    system_prompt: str,
    observation_handler: _ObservationHandler | None = None,
    final_decision_handler: _FinalDecisionHandler | None = None,
    trace: TraceContext | None = None,
    cancellation: CooperativeCancellation | None = None,
) -> AgentRun:
    """Run read-only mechanics with private composition-policy hooks."""

    return _run_agent(
        query,
        llm_provider,
        tool_registry,
        config=config,
        system_prompt=system_prompt,
        observation_handler=observation_handler,
        final_decision_handler=final_decision_handler,
        trace=trace,
        cancellation=cancellation,
    )


@traced_run("editing_agent")
def run_editing_agent(
    query: str,
    llm_provider: StructuredAgentLLM,
    tool_registry: ToolRegistry,
    *,
    config: EditingAgentConfig | None = None,
    recorder: TraceRecorder | None = None,
    trace: TraceContext | None = None,
    cancellation: CooperativeCancellation | None = None,
) -> AgentRun:
    """Run an explicitly provisioned editing registry with a mutation budget."""

    return _run_editing_agent_controlled(
        query,
        llm_provider,
        tool_registry,
        config=config or EditingAgentConfig(),
        trace=trace,
        cancellation=cancellation,
    )


def _run_editing_agent_controlled(
    query: str,
    llm_provider: StructuredAgentLLM,
    tool_registry: ToolRegistry,
    *,
    config: EditingAgentConfig,
    observation_handler: _ObservationHandler | None = None,
    final_decision_handler: _FinalDecisionHandler | None = None,
    trace: TraceContext | None = None,
    cancellation: CooperativeCancellation | None = None,
) -> AgentRun:
    """Run editing mechanics with private workflow lifecycle hooks."""

    return _run_agent(
        query,
        llm_provider,
        tool_registry,
        config=config,
        system_prompt=EDITING_AGENT_SYSTEM_PROMPT,
        mutation_tool_names=frozenset({"create_file", "replace_text"}),
        max_mutations=config.max_mutations_per_run,
        observation_handler=observation_handler,
        final_decision_handler=final_decision_handler,
        trace=trace,
        cancellation=cancellation,
    )
