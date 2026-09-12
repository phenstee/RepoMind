"""Invariant tests for inspectable agent state models."""

import pytest
from pydantic import ValidationError

from repomind.agent import (
    AgentConfig,
    AgentDecision,
    AgentRun,
    AgentRunStatus,
    AgentStep,
    ToolObservation,
)


@pytest.mark.parametrize(
    "values",
    [
        {"action": "tool", "tool_arguments": {}},
        {"action": "tool", "tool_name": "read_file"},
        {
            "action": "tool",
            "tool_name": "read_file",
            "tool_arguments": {},
            "final_answer": "not allowed",
        },
        {"action": "final"},
        {"action": "final", "final_answer": "   "},
        {"action": "final", "final_answer": "done", "tool_name": "read_file"},
        {"action": "final", "final_answer": "done", "tool_arguments": {}},
    ],
)
def test_agent_decision_rejects_inconsistent_actions(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AgentDecision(**values)  # type: ignore[arg-type]


def test_agent_decision_accepts_exactly_one_tool_or_final_action() -> None:
    tool = AgentDecision(action="tool", tool_name="git_status", tool_arguments={})
    final = AgentDecision(action="final", final_answer="  Preserve this formatting.  ")

    assert tool.tool_arguments == {}
    assert final.final_answer == "  Preserve this formatting.  "
    assert "reasoning" not in AgentDecision.model_fields
    assert "thoughts" not in AgentDecision.model_fields


@pytest.mark.parametrize(
    "values",
    [
        {"max_iterations": 0},
        {"max_iterations": True},
        {"max_history_chars": -1},
        {"max_history_chars": 2.5},
    ],
)
def test_agent_config_requires_strict_positive_limits(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AgentConfig(**values)  # type: ignore[arg-type]


def test_tool_observation_enforces_success_and_failure_shapes() -> None:
    success = ToolObservation(
        tool_name="git_status",
        arguments={},
        success=True,
        output={"clean": True},
    )
    failure = ToolObservation(
        tool_name="read_file",
        arguments={"path": "missing.py"},
        success=False,
        error="missing",
    )
    assert success.error is None
    assert failure.output is None

    with pytest.raises(ValidationError):
        ToolObservation(tool_name="x", arguments={}, success=True, error="bad")
    with pytest.raises(ValidationError):
        ToolObservation(tool_name="x", arguments={}, success=False)


def test_agent_step_and_run_are_consistent_and_json_serializable() -> None:
    decision = AgentDecision(action="final", final_answer="done")
    step = AgentStep(iteration=1, decision=decision)
    run = AgentRun(
        query="question",
        status=AgentRunStatus.COMPLETED,
        final_answer="done",
        steps=(step,),
        iterations=1,
        llm_calls=1,
        tool_calls=0,
    )

    assert run.model_dump(mode="json")["status"] == "completed"
    with pytest.raises(ValidationError):
        AgentStep(iteration=1, decision=AgentDecision(action="tool", tool_name="x", tool_arguments={}))
    with pytest.raises(ValidationError):
        AgentRun(
            query="question",
            status=AgentRunStatus.MAX_ITERATIONS,
            final_answer="false completion",
            steps=(),
            iterations=0,
            llm_calls=0,
            tool_calls=0,
        )
