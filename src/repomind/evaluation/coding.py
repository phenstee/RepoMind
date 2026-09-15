"""Isolated coding-workflow evaluation with read-only hidden oracles."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from repomind.agent import AgentStep
from repomind.coding import CodingTask, CodingTaskResult, CodingTaskStatus, VerificationPolicy
from repomind.evaluation.models import (
    CodingBenchmarkCase,
    CodingBenchmarkCaseResult,
    CodingBenchmarkSuite,
    CodingEvaluationReport,
    CodingOracleResult,
    EvaluationMode,
    FileTextExpectation,
    OracleCheckResult,
)
from repomind.tools import GitStatusOutput, ToolContext, create_default_tool_registry

_MUTATION_TOOLS = frozenset({"create_file", "replace_text"})
_VERIFICATION_TOOLS = frozenset({"run_tests", "run_ruff"})


class CodingTaskRunner(Protocol):
    """Run only the visible task contract in one isolated copied workspace."""

    def __call__(
        self,
        task: CodingTask,
        workspace: Path,
        verification_policy: VerificationPolicy,
    ) -> CodingTaskResult:
        """Return the existing coding workflow's structured terminal result."""


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(path))


def _resolve_oracle_path(workspace: Path, relative_path: Path) -> Path:
    root = workspace.resolve(strict=True)
    candidate = root / relative_path
    current = root
    for part in relative_path.parts:
        current /= part
        if _is_link_or_junction(current):
            raise ValueError(f"oracle path crosses a link or junction: {relative_path.as_posix()}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"oracle path escapes the fixture workspace: {relative_path.as_posix()}")
    return candidate


def _check_path_state(
    workspace: Path,
    path: Path,
    *,
    should_exist: bool,
) -> OracleCheckResult:
    target = _resolve_oracle_path(workspace, path)
    actual = target.exists()
    passed = actual is should_exist
    label = "file_exists" if should_exist else "file_not_exists"
    expectation = "exist" if should_exist else "not exist"
    return OracleCheckResult(
        check=label,
        path=path,
        passed=passed,
        message=f"Expected {path.as_posix()} to {expectation}; observed exists={actual}.",
    )


def _check_text(
    workspace: Path,
    expectation: FileTextExpectation,
    *,
    should_contain: bool,
) -> OracleCheckResult:
    target = _resolve_oracle_path(workspace, expectation.path)
    label = "file_contains" if should_contain else "file_not_contains"
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return OracleCheckResult(
            check=label,
            path=expectation.path,
            passed=False,
            message=f"Could not read {expectation.path.as_posix()} as UTF-8 text.",
        )
    contains = expectation.text in content
    passed = contains is should_contain
    expectation_label = "contain" if should_contain else "not contain"
    return OracleCheckResult(
        check=label,
        path=expectation.path,
        passed=passed,
        message=(
            f"Expected {expectation.path.as_posix()} to {expectation_label} the "
            f"configured evaluator substring; observed contains={contains}."
        ),
    )


def evaluate_coding_oracle(
    case: CodingBenchmarkCase,
    workspace: Path,
    changed_files: Sequence[Path],
) -> CodingOracleResult:
    """Inspect final fixture state without modifying it or executing oracle code."""

    changed = set(changed_files)
    checks: list[OracleCheckResult] = []
    for path in case.required_changed_paths:
        passed = path in changed
        checks.append(
            OracleCheckResult(
                check="required_changed_path",
                path=path,
                passed=passed,
                message=f"Required changed path present={passed}: {path.as_posix()}.",
            )
        )
    if case.allowed_changed_paths is not None:
        allowed = set(case.allowed_changed_paths)
        for path in sorted(changed, key=lambda item: item.as_posix()):
            passed = path in allowed
            checks.append(
                OracleCheckResult(
                    check="allowed_changed_path",
                    path=path,
                    passed=passed,
                    message=f"Changed path allowed={passed}: {path.as_posix()}.",
                )
            )
    checks.extend(
        _check_path_state(workspace, path, should_exist=True) for path in case.file_exists
    )
    checks.extend(
        _check_path_state(workspace, path, should_exist=False)
        for path in case.file_not_exists
    )
    checks.extend(
        _check_text(workspace, expectation, should_contain=True)
        for expectation in case.file_contains
    )
    checks.extend(
        _check_text(workspace, expectation, should_contain=False)
        for expectation in case.file_not_contains
    )
    return CodingOracleResult(
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def _verification_passed(result: CodingTaskResult) -> bool:
    report = result.verification
    tests_passed = not report.tests_required or report.tests_passed is True
    ruff_passed = not report.ruff_required or report.ruff_passed is True
    return tests_passed and ruff_passed


def _is_failure_step(step: AgentStep) -> bool:
    observation = step.observation
    if observation is not None and observation.tool_name in _VERIFICATION_TOOLS:
        if not observation.success:
            return True
        if observation.output is not None and observation.output.get("passed") is False:
            return True
    feedback = step.workflow_feedback
    return feedback is not None and any(
        "verification did not pass" in blocker.casefold()
        or "verification could not execute" in blocker.casefold()
        for blocker in feedback.blockers
    )


def _recovery_observed(result: CodingTaskResult, expects_recovery: bool) -> bool:
    """Detect failure, then a successful mutation, then verified completion.

    Failure and mutation order comes from immutable ``AgentStep`` events. The
    final successful verification and completion come from ``CodingTaskResult``.
    No model reasoning or hidden chain-of-thought is inspected.
    """

    if not expects_recovery or result.agent_run is None:
        return False
    failure_iterations = [
        step.iteration for step in result.agent_run.steps if _is_failure_step(step)
    ]
    successful_mutation_iterations = [
        step.iteration
        for step in result.agent_run.steps
        if step.observation is not None
        and step.observation.tool_name in _MUTATION_TOOLS
        and step.observation.success
    ]
    ordered_recovery = any(
        failure < mutation
        for failure in failure_iterations
        for mutation in successful_mutation_iterations
    )
    return (
        ordered_recovery
        and result.status is CodingTaskStatus.COMPLETED
        and _verification_passed(result)
    )


def _evaluate_case(
    case: CodingBenchmarkCase,
    runner: CodingTaskRunner,
) -> CodingBenchmarkCaseResult:
    source = case.fixture_repository.resolve(strict=True)
    if not source.is_dir() or not (source / ".git").exists():
        raise ValueError(
            f"coding fixture must be a Git repository directory: {source}"
        )
    status = create_default_tool_registry(
        ToolContext(repository_root=source)
    ).execute("git_status", {})
    if not GitStatusOutput.model_validate(status.model_dump()).clean:
        raise ValueError(f"coding fixture Git repository must be clean: {source}")
    with TemporaryDirectory(prefix="repomind-coding-eval-") as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(source, workspace, symlinks=True)
        result = runner(case.task, workspace, case.verification_policy)
        oracle = evaluate_coding_oracle(case, workspace, result.changed_files)

    completed = result.status is CodingTaskStatus.COMPLETED
    verification_passed = _verification_passed(result)
    task_success = completed and oracle.passed
    agent_run = result.agent_run
    failures = tuple(check.message for check in oracle.checks if not check.passed)
    return CodingBenchmarkCaseResult(
        case_id=case.id,
        workflow_status=result.status,
        oracle=oracle,
        oracle_passed=oracle.passed,
        task_success=task_success,
        false_positive_completion=completed and not oracle.passed,
        final_verification_passed=verification_passed,
        recovery_observed=_recovery_observed(result, case.expects_recovery),
        llm_calls=agent_run.llm_calls if agent_run is not None else 0,
        tool_calls=agent_run.tool_calls if agent_run is not None else 0,
        successful_mutations=result.successful_mutations,
        agent_iterations=agent_run.iterations if agent_run is not None else 0,
        completion_attempts=result.completion_attempts,
        changed_files=result.changed_files,
        oracle_failures=failures,
        workflow_blockers=result.blockers,
    )


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values)


def evaluate_coding_suite(
    suite: CodingBenchmarkSuite,
    runner: CodingTaskRunner,
    *,
    mode: EvaluationMode = EvaluationMode.OFFLINE_SCRIPTED,
) -> CodingEvaluationReport:
    """Run coding cases sequentially in disposable copies and aggregate results."""

    resolved_mode = EvaluationMode(mode)
    if resolved_mode is EvaluationMode.OFFLINE_FIXTURE:
        raise ValueError("coding evaluation mode must be offline_scripted or live_model")
    case_results = tuple(_evaluate_case(case, runner) for case in suite.cases)
    count = len(case_results)
    recovery_results = [
        result.recovery_observed
        for case, result in zip(suite.cases, case_results, strict=True)
        if case.expects_recovery
    ]
    return CodingEvaluationReport(
        benchmark_version=suite.version,
        mode=resolved_mode,
        case_results=case_results,
        case_count=count,
        workflow_completion_rate=sum(
            result.workflow_status is CodingTaskStatus.COMPLETED
            for result in case_results
        )
        / count,
        task_success_rate=sum(result.task_success for result in case_results) / count,
        false_positive_completion_rate=sum(
            result.false_positive_completion for result in case_results
        )
        / count,
        verification_pass_rate=sum(
            result.final_verification_passed for result in case_results
        )
        / count,
        recovery_rate=(
            sum(recovery_results) / len(recovery_results) if recovery_results else None
        ),
        mean_llm_calls=_mean([result.llm_calls for result in case_results]),
        mean_tool_calls=_mean([result.tool_calls for result in case_results]),
        mean_successful_mutations=_mean(
            [result.successful_mutations for result in case_results]
        ),
        mean_agent_iterations=_mean(
            [result.agent_iterations for result in case_results]
        ),
        mean_completion_attempts=_mean(
            [result.completion_attempts for result in case_results]
        ),
    )
