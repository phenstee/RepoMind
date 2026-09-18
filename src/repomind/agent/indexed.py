"""Runtime grounding policy for opt-in indexed repository navigation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repomind.agent.loop import (
    StructuredAgentLLM,
    _FinalDecisionControl,
    _run_read_only_agent_controlled,
)
from repomind.agent.models import (
    AgentConfig,
    AgentDecision,
    AgentRun,
    AgentStep,
    WorkflowFeedback,
)
from repomind.agent.prompts import INDEXED_READ_ONLY_AGENT_SYSTEM_PROMPT
from repomind.ingestion import validate_repository_relative_path
from repomind.jobs.control import CooperativeCancellation
from repomind.observability import TraceContext, TraceRecorder
from repomind.observability.instrumentation import traced_run
from repomind.tools import ToolRegistry

_INDEXED_SEARCH_TOOL = "indexed_code_search"
_READ_FILE_TOOL = "read_file"
_GROUNDING_BLOCKER = "current indexed navigation requires a matching read_file"
_GROUNDING_MESSAGE = (
    "Indexed search results are navigation hints from the persisted index. "
    "Read the current source at a returned location before finalizing repository claims."
)


def _path_identity(value: Any) -> str | None:
    if not isinstance(value, (str, Path)):
        return None
    try:
        path = validate_repository_relative_path(Path(value))
    except (TypeError, ValueError):
        return None
    return path.as_posix()


@dataclass(slots=True)
class _IndexedGroundingPolicy:
    """Track only the latest unverified non-empty indexed result paths."""

    pending_paths: frozenset[str] = field(default_factory=frozenset)

    def observe(self, step: AgentStep) -> None:
        observation = step.observation
        if observation is None or not observation.success or observation.output is None:
            return

        if observation.tool_name == _INDEXED_SEARCH_TOOL:
            locations = observation.output.get("locations")
            if not isinstance(locations, list):
                return
            paths = frozenset(
                path
                for location in locations
                if isinstance(location, dict)
                if (path := _path_identity(location.get("relative_path"))) is not None
            )
            if paths:
                self.pending_paths = paths
            return

        if observation.tool_name == _READ_FILE_TOOL:
            path = _path_identity(observation.output.get("path"))
            if path is not None and path in self.pending_paths:
                self.pending_paths = frozenset()

    def handle_final(
        self,
        decision: AgentDecision,
        steps: tuple[AgentStep, ...],
    ) -> _FinalDecisionControl:
        del decision, steps
        if not self.pending_paths:
            return _FinalDecisionControl()
        return _FinalDecisionControl(
            feedback=WorkflowFeedback(
                message=_GROUNDING_MESSAGE,
                blockers=(_GROUNDING_BLOCKER,),
            )
        )


@traced_run("read_only_agent")
def run_indexed_read_only_agent(
    query: str,
    llm_provider: StructuredAgentLLM,
    tool_registry: ToolRegistry,
    *,
    config: AgentConfig | None = None,
    recorder: TraceRecorder | None = None,
    trace: TraceContext | None = None,
    cancellation: CooperativeCancellation | None = None,
) -> AgentRun:
    """Run the read-only agent with current-file confirmation for index hints."""

    policy = _IndexedGroundingPolicy()
    return _run_read_only_agent_controlled(
        query,
        llm_provider,
        tool_registry,
        config=config or AgentConfig(),
        system_prompt=INDEXED_READ_ONLY_AGENT_SYSTEM_PROMPT,
        observation_handler=policy.observe,
        final_decision_handler=policy.handle_final,
        trace=trace,
        cancellation=cancellation,
    )
