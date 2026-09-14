"""Validation tests for coding-task workflow contracts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.coding import CodingTask, CodingWorkflowConfig, VerificationPolicy


def test_coding_task_preserves_ordered_human_acceptance_criteria() -> None:
    task = CodingTask(
        objective="Fix async retry handling.",
        acceptance_criteria=("Remain nonblocking.", "Cover retry then success."),
    )
    assert task.objective == "Fix async retry handling."
    assert task.acceptance_criteria == (
        "Remain nonblocking.",
        "Cover retry then success.",
    )


@pytest.mark.parametrize(
    "values",
    [
        {"objective": ""},
        {"objective": "   "},
        {"objective": "valid", "acceptance_criteria": ("",)},
        {"objective": "valid", "acceptance_criteria": ("  ",)},
    ],
)
def test_coding_task_rejects_blank_contract_entries(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CodingTask(**values)  # type: ignore[arg-type]


def test_verification_policy_preserves_exact_safe_scopes() -> None:
    policy = VerificationPolicy(
        test_paths=(Path("tests/unit"), Path("tests/integration")),
        ruff_paths=(Path("src"), Path("tests")),
    )
    assert policy.test_paths == (Path("tests/unit"), Path("tests/integration"))
    assert policy.ruff_paths == (Path("src"), Path("tests"))


@pytest.mark.parametrize(
    "values",
    [
        {"require_tests": True, "test_paths": ()},
        {"require_ruff": True, "ruff_paths": ()},
        {"test_paths": (Path("../tests"),)},
        {"test_paths": (Path("-q"),)},
        {"ruff_paths": (Path("../../outside"),)},
        {"ruff_paths": (Path("--fix"),)},
    ],
)
def test_verification_policy_rejects_missing_or_unsafe_scopes(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        VerificationPolicy(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_workflow_config_requires_strict_positive_attempt_limit(value: object) -> None:
    with pytest.raises(ValidationError):
        CodingWorkflowConfig(max_completion_attempts=value)  # type: ignore[arg-type]
