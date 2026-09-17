"""Structured planner/reviewer contracts and bounded evidence tests."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from repomind.coding import (
    CodingPlan,
    CodingReview,
    CodingTask,
    FinalChangeReview,
    PlanningError,
    VerificationPolicy,
    WorkspaceBaseline,
    bounded_review_diff,
    generate_coding_plan,
)
from repomind.observability import TraceContext
from repomind.tools import GitDiffOutput, GitStatusOutput


def _plan() -> dict[str, object]:
    return {
        "task_summary": "Make the bounded requested change.",
        "steps": [
            {
                "step_id": 1,
                "action": "Inspect the implementation and update its tests.",
                "likely_paths": ["src/app.py"],
                "criterion_indices": [0],
                "verification": ["pytest", "ruff"],
            }
        ],
        "acceptance_coverage": [{"criterion_index": 0, "step_ids": [1]}],
        "verification_plan": ["pytest", "ruff"],
    }


class _Planner:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, type[BaseModel], dict[str, object]]] = []

    def generate_structured(self, prompt, response_model, **kwargs):
        self.calls.append((prompt, response_model, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return response_model.model_validate(self.response)


def test_plan_contract_is_bounded_and_references_are_valid() -> None:
    task = CodingTask(objective="Change app.", acceptance_criteria=("Tests pass.",))
    assert CodingPlan.model_validate(_plan()).validate_for_task(task).steps[0].step_id == 1

    invalid = _plan() | {"acceptance_coverage": [{"criterion_index": 0, "step_ids": [2]}]}
    with pytest.raises(ValidationError, match="unknown plan step"):
        CodingPlan.model_validate(invalid)
    disagreement = _plan()
    disagreement["steps"] = [
        {"step_id": 1, "action": "Inspect the implementation.", "criterion_indices": []}
    ]
    with pytest.raises(ValidationError, match="must agree"):
        CodingPlan.model_validate(disagreement)
    with pytest.raises(ValidationError):
        CodingPlan.model_validate(_plan() | {"unknown": True})
    with pytest.raises(ValidationError):
        CodingPlan.model_validate(_plan() | {"task_summary": "x" * 1001})


def test_planner_uses_structured_output_and_compact_repository_metadata() -> None:
    provider = _Planner(_plan())
    task = CodingTask(objective="Change app.", acceptance_criteria=("Tests pass.",))
    result = generate_coding_plan(
        task,
        WorkspaceBaseline(
            git_status=GitStatusOutput(branch="main", changed_files=[], clean=True),
            changed_files=(),
            clean=True,
        ),
        VerificationPolicy(),
        provider,
        trace=TraceContext(),
    )

    prompt, response_model, kwargs = provider.calls[0]
    payload = json.loads(prompt)
    assert result.steps[0].criterion_indices == (0,)
    assert response_model is CodingPlan
    assert payload["task"]["acceptance_criteria"] == ["Tests pass."]
    assert "repository_diff" not in payload
    assert "scratchpad" in kwargs["system_prompt"]


def test_planner_failure_is_wrapped_without_model_text() -> None:
    provider = _Planner(RuntimeError("sk-test-secret"))
    task = CodingTask(objective="Change app.")
    baseline = WorkspaceBaseline(
        git_status=GitStatusOutput(branch="main", changed_files=[], clean=True),
        changed_files=(),
        clean=True,
    )
    with pytest.raises(PlanningError, match="structured planning failed") as raised:
        generate_coding_plan(
            task,
            baseline,
            VerificationPolicy(),
            provider,
            trace=TraceContext(),
        )
    assert "secret" not in str(raised.value)


def test_review_contract_and_diff_bound_reject_truncated_evidence() -> None:
    review = CodingReview.model_validate(
        {
            "verdict": "approve",
            "workspace_revision": 1,
            "acceptance_results": [
                {"criterion_index": 0, "status": "satisfied", "evidence": "Tests pass."}
            ],
        }
    )
    assert (
        review.validate_for_task(
            CodingTask(objective="Change app.", acceptance_criteria=("Tests pass.",)), 1
        ).verdict
        == "approve"
    )

    final = FinalChangeReview(
        workspace_revision=1,
        git_status=GitStatusOutput(
            branch="main", changed_files=[{"path": Path("app.py"), "status": "M"}], clean=False
        ),
        unstaged_diff=GitDiffOutput(
            content="diff --git a/app.py b/app.py",
            truncated=True,
            staged=False,
            path=None,
        ),
        changed_files=(Path("app.py"),),
        baseline_changed_files=(),
        workflow_changed_files=(Path("app.py"),),
        unexpected_changed_files=(),
        diff_truncated=True,
    )
    assert bounded_review_diff(final, max_chars=1000) is None

    with pytest.raises(ValidationError, match="approval requires"):
        CodingReview.model_validate(
            {
                "verdict": "approve",
                "workspace_revision": 1,
                "acceptance_results": [
                    {
                        "criterion_index": 0,
                        "status": "not_satisfied",
                        "evidence": "Missing test.",
                    }
                ],
            }
        )
