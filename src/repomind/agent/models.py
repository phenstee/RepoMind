"""Inspectable state models for RepoMind's handwritten agents."""

import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentConfig(BaseModel):
    """Hard limits for one finite agent run."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(default=8, gt=0, strict=True)
    max_history_chars: int = Field(default=60_000, gt=0, strict=True)


class EditingAgentConfig(AgentConfig):
    """Finite run limits including a successful-mutation budget."""

    max_mutations_per_run: int = Field(default=8, gt=0, strict=True)


class AgentDecision(BaseModel):
    """Exactly one structured next action selected by the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["tool", "final"]
    tool_name: str | None = Field(default=None, max_length=128)
    tool_arguments: dict[str, Any] | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def _validate_action_fields(self) -> "AgentDecision":
        if self.action == "tool":
            if self.tool_name is None or not self.tool_name.strip():
                raise ValueError("tool action requires a non-empty tool_name")
            if self.tool_arguments is None:
                raise ValueError("tool action requires tool_arguments")
            if self.final_answer is not None:
                raise ValueError("tool action must not include final_answer")
            return self

        if self.final_answer is None or not self.final_answer.strip():
            raise ValueError("final action requires a non-empty final_answer")
        if self.tool_name is not None or self.tool_arguments is not None:
            raise ValueError("final action must not include tool fields")
        return self


def _decode_tool_arguments(encoded: str) -> dict[str, Any]:
    """Decode one serialized JSON object of tool arguments, bounding failures.

    The messages deliberately describe the contract instead of echoing the
    payload: the encoded string is untrusted model output, and a raw decoder
    traceback is not a useful API-level failure.
    """

    try:
        arguments = json.loads(encoded)
    except ValueError as exc:
        raise ValueError("tool_arguments_json must contain valid JSON") from exc
    if not isinstance(arguments, dict):
        # ValueError, not TypeError: Pydantic only converts ValueError raised
        # inside a validator into a ValidationError, which the agent loop
        # already turns into a bounded AgentError.
        raise ValueError("tool_arguments_json must decode to a JSON object")  # noqa: TRY004
    return arguments


class AgentDecisionResponse(BaseModel):
    """Provider-facing decision envelope compatible with strict schemas.

    ``AgentDecision.tool_arguments`` is an open mapping whose shape depends on
    which registry tool the model selected, and strict Structured Outputs
    cannot express an open object - every object must close with
    ``additionalProperties: false``. This envelope therefore carries the
    arguments as one serialized JSON object string, which RepoMind decodes
    locally into the internal :class:`AgentDecision`. Tool-specific argument
    validation stays where it belongs: ``ToolRegistry.execute`` remains the
    authority on whether the decoded arguments suit the selected tool, so this
    envelope never learns any individual tool's schema.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["tool", "final"]
    tool_name: str | None = Field(default=None, max_length=128)
    tool_arguments_json: str | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def _validate_action_fields(self) -> "AgentDecisionResponse":
        if self.action == "tool":
            if self.tool_name is None or not self.tool_name.strip():
                raise ValueError("tool action requires a non-empty tool_name")
            if self.tool_arguments_json is None:
                raise ValueError("tool action requires tool_arguments_json")
            if self.final_answer is not None:
                raise ValueError("tool action must not include final_answer")
            _decode_tool_arguments(self.tool_arguments_json)
            return self

        if self.final_answer is None or not self.final_answer.strip():
            raise ValueError("final action requires a non-empty final_answer")
        if self.tool_name is not None or self.tool_arguments_json is not None:
            raise ValueError("final action must not include tool fields")
        return self

    def to_decision(self) -> AgentDecision:
        """Convert validated provider output into the internal decision model."""

        if self.action == "final":
            return AgentDecision(action="final", final_answer=self.final_answer)
        return AgentDecision(
            action="tool",
            tool_name=self.tool_name,
            tool_arguments=_decode_tool_arguments(self.tool_arguments_json or ""),
            final_answer=None,
        )


class ToolObservation(BaseModel):
    """JSON-compatible result of one requested tool action."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    arguments: dict[str, Any]
    success: bool
    output: dict[str, Any] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _validate_result_fields(self) -> "ToolObservation":
        if self.success:
            if self.output is None or self.error is not None:
                raise ValueError("successful observation requires output and no error")
        elif self.output is not None or self.error is None or not self.error.strip():
            raise ValueError("failed observation requires an error and no output")
        return self


class WorkflowFeedback(BaseModel):
    """Trusted completion guidance with separately untrusted supporting evidence."""

    model_config = ConfigDict(frozen=True)

    message: str = Field(min_length=1)
    blockers: tuple[str, ...]
    evidence: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_text(self) -> "WorkflowFeedback":
        if not self.message.strip():
            raise ValueError("workflow feedback message must not be blank")
        if any(not blocker.strip() for blocker in self.blockers):
            raise ValueError("workflow feedback blockers must not be blank")
        return self


class AgentStep(BaseModel):
    """One model decision and its optional resulting tool observation."""

    model_config = ConfigDict(frozen=True)

    iteration: int = Field(ge=1, strict=True)
    decision: AgentDecision
    observation: ToolObservation | None = None
    workflow_feedback: WorkflowFeedback | None = None

    @model_validator(mode="after")
    def _validate_observation(self) -> "AgentStep":
        if self.decision.action == "tool":
            if self.observation is None:
                raise ValueError("tool decision requires an observation")
            if self.workflow_feedback is not None:
                raise ValueError("tool decision must not include workflow feedback")
        elif self.observation is not None:
            raise ValueError("final decision must not include an observation")
        return self


class AgentRunStatus(StrEnum):
    """Terminal state of a returned agent run."""

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations_reached"
    WORKFLOW_STOPPED = "workflow_stopped"


class AgentRun(BaseModel):
    """Immutable, inspectable result of a complete or bounded agent run."""

    model_config = ConfigDict(frozen=True)

    query: str
    status: AgentRunStatus
    final_answer: str | None = None
    steps: tuple[AgentStep, ...]
    iterations: int = Field(ge=0, strict=True)
    llm_calls: int = Field(ge=0, strict=True)
    tool_calls: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _validate_terminal_state(self) -> "AgentRun":
        if self.iterations != len(self.steps) or self.llm_calls != self.iterations:
            raise ValueError("iteration, step, and LLM-call counts must agree")
        if self.status is AgentRunStatus.COMPLETED:
            if self.final_answer is None or not self.final_answer.strip():
                raise ValueError("completed run requires a final answer")
        elif self.final_answer is not None:
            raise ValueError("non-completed run must not claim a final answer")
        return self
