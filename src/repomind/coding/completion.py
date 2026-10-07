"""Deterministic completion gates for coding-task evidence."""

from dataclasses import dataclass

from repomind.coding.models import (
    CodingReview,
    FinalChangeReview,
    ReviewVerdict,
    VerificationPolicy,
    VerificationReport,
)


@dataclass(frozen=True, slots=True)
class _CompletionDecision:
    completed: bool
    blockers: tuple[str, ...]


def _verification_blockers(
    report: VerificationReport,
    policy: VerificationPolicy,
    workspace_revision: int,
) -> list[str]:
    blockers: list[str] = []
    if policy.require_tests:
        if report.tests_execution_error is not None:
            blockers.append(
                f"Required pytest verification could not execute: "
                f"{report.tests_execution_error}"
            )
        elif (
            report.tests_result is None
            or not report.tests_result.passed
            or report.tests_passed is not True
        ):
            blockers.append("Required pytest verification did not pass.")
        elif tuple(report.tests_result.paths) != policy.test_paths:
            blockers.append("Required pytest verification scope does not match policy.")
        elif report.tests_revision != workspace_revision:
            blockers.append(
                "Required pytest verification is stale: passed at workspace "
                f"revision {report.tests_revision}, current revision is "
                f"{workspace_revision}."
            )
    if policy.require_ruff:
        if report.ruff_execution_error is not None:
            blockers.append(
                f"Required Ruff verification could not execute: "
                f"{report.ruff_execution_error}"
            )
        elif (
            report.ruff_result is None
            or not report.ruff_result.passed
            or report.ruff_passed is not True
        ):
            blockers.append("Required Ruff verification did not pass.")
        elif tuple(report.ruff_result.paths) != policy.ruff_paths:
            blockers.append("Required Ruff verification scope does not match policy.")
        elif report.ruff_revision != workspace_revision:
            blockers.append(
                "Required Ruff verification is stale: passed at workspace "
                f"revision {report.ruff_revision}, current revision is "
                f"{workspace_revision}."
            )
    return blockers


def _missing_change_evidence(review: FinalChangeReview, policy: VerificationPolicy) -> bool:
    """Whether workflow-changed files exist but the captured diff shows nothing.

    A change the diff cannot show (for example a file Git does not render) must
    never reach review as an empty diff that looks like "nothing changed".
    """

    if not policy.require_final_diff or not review.workflow_changed_files:
        return False
    if review.unstaged_diff_error is not None or review.staged_diff_error is not None:
        return False
    return not any(
        diff is not None and diff.content.strip()
        for diff in (review.unstaged_diff, review.staged_diff)
    )


def _review_blockers(
    review: FinalChangeReview | None,
    policy: VerificationPolicy,
    workspace_revision: int,
) -> list[str]:
    if review is None:
        if policy.require_final_git_status or policy.require_final_diff:
            return ["Required final Git review is unavailable."]
        return []

    blockers: list[str] = []
    if review.workspace_revision != workspace_revision:
        blockers.append(
            "Final Git review is stale: captured at workspace revision "
            f"{review.workspace_revision}, current revision is {workspace_revision}."
        )
    if policy.require_final_diff:
        if review.unstaged_diff is None:
            blockers.append("Required unstaged Git diff is unavailable.")
        if review.unstaged_diff_error is not None:
            blockers.append(
                f"Required unstaged Git diff failed: {review.unstaged_diff_error}"
            )
        if review.staged_diff_error is not None:
            blockers.append(f"Required staged Git diff failed: {review.staged_diff_error}")
    if _missing_change_evidence(review, policy):
        paths = ", ".join(path.as_posix() for path in review.workflow_changed_files)
        blockers.append(
            f"Final Git diff is empty although the workflow changed files: {paths}."
        )
    if review.git_status.truncated:
        blockers.append(
            "Final Git status is truncated; not every changed file could be reviewed."
        )
    if review.unexpected_changed_files:
        paths = ", ".join(path.as_posix() for path in review.unexpected_changed_files)
        blockers.append(f"Unexpected changed files detected: {paths}.")
    return blockers


def _independent_review_blockers(
    review: CodingReview | None,
    workspace_revision: int,
) -> list[str]:
    if review is None:
        return ["Required independent coding review is unavailable."]
    if review.workspace_revision != workspace_revision:
        return [
            (
                "Independent coding review is stale: captured at workspace revision "
                f"{review.workspace_revision}, current revision is {workspace_revision}."
            )
        ]
    if review.verdict is ReviewVerdict.CHANGES_REQUIRED:
        return ["Independent coding review requires changes."]
    return []


def _evaluate_completion(
    *,
    agent_requested_completion: bool,
    workspace_revision: int,
    verification: VerificationReport,
    final_review: FinalChangeReview | None,
    coding_review: CodingReview | None,
    policy: VerificationPolicy,
    require_coding_review: bool = True,
) -> _CompletionDecision:
    blockers: list[str] = []
    if not agent_requested_completion:
        blockers.append("The agent did not request completion.")
    blockers.extend(_verification_blockers(verification, policy, workspace_revision))
    blockers.extend(_review_blockers(final_review, policy, workspace_revision))
    if require_coding_review:
        blockers.extend(_independent_review_blockers(coding_review, workspace_revision))
    return _CompletionDecision(completed=not blockers, blockers=tuple(blockers))
