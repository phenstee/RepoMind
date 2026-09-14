"""Inspectable state models for RepoMind's handwritten agents."""

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


class AgentStep(BaseModel):
    """One model decision and its optional resulting tool observation."""

    model_config = ConfigDict(frozen=True)

    iteration: int = Field(ge=1, strict=True)
    decision: AgentDecision
    observation: ToolObservation | None = None

    @model_validator(mode="after")
    def _validate_observation(self) -> "AgentStep":
        if self.decision.action == "tool" and self.observation is None:
            raise ValueError("tool decision requires an observation")
        if self.decision.action == "final" and self.observation is not None:
            raise ValueError("final decision must not include an observation")
        return self


class AgentRunStatus(StrEnum):
    """Terminal state of a returned agent run."""

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations_reached"


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
            raise ValueError("max-iteration run must not claim a final answer")
        return self
