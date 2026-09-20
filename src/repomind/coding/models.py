"""Typed contracts and evidence for autonomous coding-task workflows."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from repomind.agent import AgentRun
from repomind.ingestion import validate_repository_relative_path
from repomind.tools import (
    GitDiffOutput,
    GitStatusOutput,
    RunRuffInput,
    RunRuffOutput,
    RunTestsInput,
    RunTestsOutput,
)

ConciseText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]

# A repository-relative path that stays a ``pathlib.Path`` internally while
# describing itself to providers as a plain JSON string. Pydantic renders a
# bare ``Path`` as ``{"type": "string", "format": "path"}``, and strict
# Structured Outputs reject that format. This annotation changes only the
# generated schema: parsing, the repository-relative validation below, and
# JSON serialization are unaffected.
RepositoryPath = Annotated[Path, WithJsonSchema({"type": "string"})]


class CodingPlanStep(BaseModel):
    """One bounded advisory action; it never executes capabilities directly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: int = Field(ge=1, le=12, strict=True)
    action: ConciseText
    likely_paths: tuple[RepositoryPath, ...] = Field(default=(), max_length=8)
    criterion_indices: tuple[int, ...] = Field(default=(), max_length=50)
    verification: tuple[Literal["pytest", "ruff"], ...] = Field(default=(), max_length=2)

    @field_validator("likely_paths")
    @classmethod
    def _relative_paths(cls, values: tuple[Path, ...]) -> tuple[Path, ...]:
        return tuple(validate_repository_relative_path(value) for value in values)

    @model_validator(mode="after")
    def _unique_references(self) -> "CodingPlanStep":
        if len(set(self.likely_paths)) != len(self.likely_paths):
            raise ValueError("plan step paths must be unique")
        if len(set(self.criterion_indices)) != len(self.criterion_indices):
            raise ValueError("plan step criterion references must be unique")
        if any(index < 0 for index in self.criterion_indices):
            raise ValueError("plan step criterion references must be non-negative")
        if len(set(self.verification)) != len(self.verification):
            raise ValueError("plan step verification mechanisms must be unique")
        return self


class PlanAcceptanceCoverage(BaseModel):
    """Explicit mapping from one task criterion to steps or an uncertainty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion_index: int = Field(ge=0, le=49, strict=True)
    step_ids: tuple[int, ...] = Field(default=(), max_length=12)
    uncertainty: ConciseText | None = None

    @model_validator(mode="after")
    def _mapped_or_uncertain(self) -> "PlanAcceptanceCoverage":
        if not self.step_ids and self.uncertainty is None:
            raise ValueError("criterion coverage needs a planned step or uncertainty")
        if len(set(self.step_ids)) != len(self.step_ids):
            raise ValueError("criterion coverage step references must be unique")
        return self


class CodingPlan(BaseModel):
    """Compact explicit planning output, not hidden model reasoning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_summary: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
    ]
    relevant_areas: tuple[ConciseText, ...] = Field(default=(), max_length=12)
    steps: tuple[CodingPlanStep, ...] = Field(min_length=1, max_length=12)
    acceptance_coverage: tuple[PlanAcceptanceCoverage, ...] = Field(default=(), max_length=50)
    verification_plan: tuple[Literal["pytest", "ruff"], ...] = Field(default=(), max_length=2)
    risks: tuple[ConciseText, ...] = Field(default=(), max_length=8)
    uncertainties: tuple[ConciseText, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def _consistent_references(self) -> "CodingPlan":
        step_ids = tuple(step.step_id for step in self.steps)
        if step_ids != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("plan step IDs must be contiguous and start at one")
        coverage_indices = [item.criterion_index for item in self.acceptance_coverage]
        if len(set(coverage_indices)) != len(coverage_indices):
            raise ValueError("each acceptance criterion may appear only once")
        valid_steps = set(step_ids)
        if any(not set(item.step_ids).issubset(valid_steps) for item in self.acceptance_coverage):
            raise ValueError("acceptance coverage references an unknown plan step")
        steps_by_id = {step.step_id: step for step in self.steps}
        if any(
            item.criterion_index not in steps_by_id[step_id].criterion_indices
            for item in self.acceptance_coverage
            for step_id in item.step_ids
        ):
            raise ValueError("acceptance coverage and plan step criteria must agree")
        if len(set(self.verification_plan)) != len(self.verification_plan):
            raise ValueError("plan verification mechanisms must be unique")
        return self

    def validate_for_task(self, task: "CodingTask") -> "CodingPlan":
        expected = set(range(len(task.acceptance_criteria)))
        actual = {item.criterion_index for item in self.acceptance_coverage}
        if actual != expected:
            raise ValueError("plan must cover every acceptance criterion exactly once")
        if any(not set(step.criterion_indices).issubset(expected) for step in self.steps):
            raise ValueError("plan step references an unknown acceptance criterion")
        return self


class ReviewVerdict(StrEnum):
    APPROVE = "approve"
    CHANGES_REQUIRED = "changes_required"


class AcceptanceReviewStatus(StrEnum):
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    UNCERTAIN = "uncertain"


class AcceptanceReview(BaseModel):
    """Concise reviewer judgment for one human acceptance criterion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion_index: int = Field(ge=0, le=49, strict=True)
    status: AcceptanceReviewStatus
    evidence: ConciseText


class CodingReview(BaseModel):
    """Bounded independent review tied to one logical workspace revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: ReviewVerdict
    workspace_revision: int = Field(ge=0, strict=True)
    acceptance_results: tuple[AcceptanceReview, ...] = Field(default=(), max_length=50)
    findings: tuple[ConciseText, ...] = Field(default=(), max_length=8)
    required_corrections: tuple[ConciseText, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def _consistent_verdict(self) -> "CodingReview":
        indices = [item.criterion_index for item in self.acceptance_results]
        if len(set(indices)) != len(indices):
            raise ValueError("each acceptance criterion may be reviewed only once")
        all_satisfied = all(
            item.status is AcceptanceReviewStatus.SATISFIED for item in self.acceptance_results
        )
        if self.verdict is ReviewVerdict.APPROVE and (
            not all_satisfied or self.required_corrections
        ):
            raise ValueError("approval requires satisfied criteria and no corrections")
        if self.verdict is ReviewVerdict.CHANGES_REQUIRED and not (
            self.required_corrections or not all_satisfied
        ):
            raise ValueError("changes-required review needs a correction or unmet criterion")
        return self

    def validate_for_task(self, task: "CodingTask", workspace_revision: int) -> "CodingReview":
        expected = set(range(len(task.acceptance_criteria)))
        actual = {item.criterion_index for item in self.acceptance_results}
        if actual != expected:
            raise ValueError("review must evaluate every acceptance criterion exactly once")
        if self.workspace_revision != workspace_revision:
            raise ValueError("review workspace revision does not match current revision")
        return self


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
    max_review_diff_chars: int = Field(default=20_000, gt=0, le=100_000, strict=True)


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
    plan: CodingPlan | None = None
    review: CodingReview | None = None
    review_attempts: int = Field(default=0, ge=0)
    review_blocks: int = Field(default=0, ge=0)
