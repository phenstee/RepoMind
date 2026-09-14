"""Direct tests for deterministic, revision-aware completion gates."""

from pathlib import Path

from repomind.coding.completion import _evaluate_completion
from repomind.coding.models import (
    FinalChangeReview,
    VerificationPolicy,
    VerificationReport,
)
from repomind.tools import GitDiffOutput, GitStatusOutput, RunRuffOutput, RunTestsOutput


def _tests(*, passed: bool = True, timed_out: bool = False) -> RunTestsOutput:
    return RunTestsOutput(
        paths=[Path("tests")],
        exit_code=0 if passed else None if timed_out else 1,
        passed=passed,
        stdout="",
        stderr="",
        truncated=False,
        timed_out=timed_out,
        duration_seconds=0.1,
    )


def _ruff() -> RunRuffOutput:
    return RunRuffOutput(
        paths=[Path(".")],
        exit_code=0,
        passed=True,
        stdout="All checks passed!",
        stderr="",
        truncated=False,
        timed_out=False,
        duration_seconds=0.1,
    )


def _review(revision: int, *, unexpected: tuple[Path, ...] = ()) -> FinalChangeReview:
    return FinalChangeReview(
        workspace_revision=revision,
        git_status=GitStatusOutput(branch="main", changed_files=[], clean=True),
        unstaged_diff=GitDiffOutput(
            content="", truncated=False, staged=False, path=None
        ),
        changed_files=(),
        baseline_changed_files=(),
        workflow_changed_files=(),
        unexpected_changed_files=unexpected,
        diff_truncated=False,
    )


def _report(revision: int, *, test_revision: int | None = None) -> VerificationReport:
    return VerificationReport(
        tests_required=True,
        tests_passed=True,
        tests_revision=revision if test_revision is None else test_revision,
        tests_result=_tests(),
        ruff_required=True,
        ruff_passed=True,
        ruff_revision=revision,
        ruff_result=_ruff(),
        current_workspace_revision=revision,
    )


def test_completion_accepts_fresh_required_evidence() -> None:
    decision = _evaluate_completion(
        agent_requested_completion=True,
        workspace_revision=2,
        verification=_report(2),
        final_review=_review(2),
        policy=VerificationPolicy(),
    )
    assert decision.completed
    assert decision.blockers == ()


def test_completion_rejects_stale_test_and_final_review_evidence() -> None:
    decision = _evaluate_completion(
        agent_requested_completion=True,
        workspace_revision=2,
        verification=_report(2, test_revision=1),
        final_review=_review(1),
        policy=VerificationPolicy(),
    )
    assert not decision.completed
    assert any("pytest verification is stale" in blocker for blocker in decision.blockers)
    assert any("Git review is stale" in blocker for blocker in decision.blockers)


def test_completion_rejects_passing_but_narrower_test_scope() -> None:
    report = _report(1).model_copy(
        update={
            "tests_result": _tests().model_copy(
                update={"paths": [Path("tests/unit/test_one.py")]}
            )
        }
    )
    decision = _evaluate_completion(
        agent_requested_completion=True,
        workspace_revision=1,
        verification=report,
        final_review=_review(1),
        policy=VerificationPolicy(),
    )
    assert not decision.completed
    assert "scope does not match" in decision.blockers[0]


def test_timeout_and_execution_error_never_count_as_verification() -> None:
    timed_out = VerificationReport(
        tests_required=True,
        tests_passed=False,
        tests_revision=0,
        tests_result=_tests(passed=False, timed_out=True),
        ruff_required=False,
        current_workspace_revision=0,
    )
    execution_error = timed_out.model_copy(
        update={"tests_result": None, "tests_execution_error": "could not start"}
    )
    for report in (timed_out, execution_error):
        decision = _evaluate_completion(
            agent_requested_completion=True,
            workspace_revision=0,
            verification=report,
            final_review=_review(0),
            policy=VerificationPolicy(require_ruff=False),
        )
        assert not decision.completed


def test_unexpected_changes_block_completion_without_git_cleanup() -> None:
    decision = _evaluate_completion(
        agent_requested_completion=True,
        workspace_revision=0,
        verification=_report(0),
        final_review=_review(0, unexpected=(Path("surprise.py"),)),
        policy=VerificationPolicy(),
    )
    assert not decision.completed
    assert decision.blockers == ("Unexpected changed files detected: surprise.py.",)
