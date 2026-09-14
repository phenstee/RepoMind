"""Typed contracts and evidence for autonomous coding-task workflows."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repomind.agent import AgentRun
from repomind.tools import (
    GitDiffOutput,
    GitStatusOutput,
    RunRuffInput,
    RunRuffOutput,
    RunTestsInput,
    RunTestsOutput,
)


class CodingTask(BaseModel):
    """Human-authored objective and ordered semantic acceptance criteria."""

    model_config = ConfigDict(frozen=True)

    objective: str = Field(min_length=1, max_length=10_000)
    acceptance_criteria: tuple[str, ...] = Field(default=(), max_length=50)

    @model_validator(mode="after")
    def _validate_text(self) -> "CodingTask":
        if not self.objective.strip():
            raise ValueError("objective must not be blank")
        if any(not criterion.strip() for criterion in self.acceptance_criteria):
            raise ValueError("acceptance criteria must not contain blank entries")
        return self


class VerificationPolicy(BaseModel):
    """Application-controlled mechanical requirements for task completion."""

    model_config = ConfigDict(frozen=True)

    require_tests: bool = True
    test_paths: tuple[Path, ...] = (Path("tests"),)
    require_ruff: bool = True
    ruff_paths: tuple[Path, ...] = (Path("."),)
    require_final_git_status: bool = True
    require_final_diff: bool = True

    @model_validator(mode="after")
    def _validate_scopes(self) -> "VerificationPolicy":
        if self.require_tests and not self.test_paths:
            raise ValueError("required tests need at least one configured path")
        if self.require_ruff and not self.ruff_paths:
            raise ValueError("required Ruff needs at least one configured path")
        if self.test_paths:
            RunTestsInput(paths=list(self.test_paths))
        if self.ruff_paths:
            RunRuffInput(paths=list(self.ruff_paths))
        return self


class CodingWorkflowConfig(BaseModel):
    """Small set of application-owned workflow lifecycle controls."""

    model_config = ConfigDict(frozen=True)

    require_clean_worktree: bool = True
    max_completion_attempts: int = Field(default=3, gt=0, strict=True)


class WorkspaceBaseline(BaseModel):
    """Git state captured before the LLM receives the coding task."""

    model_config = ConfigDict(frozen=True)

    git_status: GitStatusOutput
    changed_files: tuple[Path, ...]
    clean: bool


class VerificationReport(BaseModel):
    """Latest structured verification evidence and its logical revision."""

    model_config = ConfigDict(frozen=True)

    tests_required: bool
    tests_passed: bool | None = None
    tests_revision: int | None = Field(default=None, ge=0)
    tests_result: RunTestsOutput | None = None
    tests_execution_error: str | None = None
    ruff_required: bool
    ruff_passed: bool | None = None
    ruff_revision: int | None = Field(default=None, ge=0)
    ruff_result: RunRuffOutput | None = None
    ruff_execution_error: str | None = None
    current_workspace_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_evidence(self) -> "VerificationReport":
        for result, error, passed, revision, label in (
            (
                self.tests_result,
                self.tests_execution_error,
                self.tests_passed,
                self.tests_revision,
                "tests",
            ),
            (
                self.ruff_result,
                self.ruff_execution_error,
                self.ruff_passed,
                self.ruff_revision,
                "Ruff",
            ),
        ):
            if result is not None and error is not None:
                raise ValueError(f"{label} result and execution error are exclusive")
            if result is not None:
                if passed is not result.passed or revision is None:
                    raise ValueError(
                        f"{label} summary must match its result and include a revision"
                    )
            elif error is not None:
                if passed is not None or revision is None:
                    raise ValueError(
                        f"{label} execution error needs a revision and no pass claim"
                    )
            elif passed is not None or revision is not None:
                raise ValueError(f"{label} summary requires result or execution error")
        return self


class FinalChangeReview(BaseModel):
    """Fresh repository-owned Git evidence captured before completion."""

    model_config = ConfigDict(frozen=True)

    workspace_revision: int = Field(ge=0)
    git_status: GitStatusOutput
    unstaged_diff: GitDiffOutput | None = None
    unstaged_diff_error: str | None = None
    staged_diff: GitDiffOutput | None = None
    staged_diff_error: str | None = None
    changed_files: tuple[Path, ...]
    baseline_changed_files: tuple[Path, ...]
    workflow_changed_files: tuple[Path, ...]
    unexpected_changed_files: tuple[Path, ...]
    diff_truncated: bool


class CodingTaskStatus(StrEnum):
    """Terminal deterministic status of a coding workflow."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    PRECONDITION_FAILED = "precondition_failed"
    AGENT_LIMIT_REACHED = "agent_limit_reached"
    VERIFICATION_FAILED = "verification_failed"
    ERROR = "error"


class CodingTaskResult(BaseModel):
    """Complete task contract, evidence, counters, and terminal decision."""

    model_config = ConfigDict(frozen=True)

    task: CodingTask
    status: CodingTaskStatus
    baseline: WorkspaceBaseline | None
    agent_run: AgentRun | None
    final_answer: str | None
    changed_files: tuple[Path, ...]
    workspace_revision: int = Field(ge=0)
    successful_mutations: int = Field(ge=0)
    verification: VerificationReport
    final_review: FinalChangeReview | None
    blockers: tuple[str, ...]
    completion_attempts: int = Field(ge=0)
    verification_executions: int = Field(ge=0)
