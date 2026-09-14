"""Optional confirmed live run of the autonomous coding-task workflow."""

import argparse
from pathlib import Path

from pydantic import ValidationError

from repomind.agent import EditingAgentConfig
from repomind.coding import CodingTask, VerificationPolicy, run_coding_task
from repomind.config import get_settings
from repomind.ingestion import RepositoryIngestionError
from repomind.llm import OpenAILLMClient
from repomind.tools import ToolContext, create_editing_tool_registry


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one autonomous RepoMind coding task with completion gates.",
    )
    parser.add_argument("task", help="Coding objective")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Explicit clean repository root that the workflow may modify",
    )
    parser.add_argument(
        "--acceptance",
        action="append",
        default=[],
        help="Ordered semantic acceptance criterion; repeat as needed",
    )
    parser.add_argument(
        "--max-iterations",
        type=_positive_integer,
        default=8,
        help="Maximum LLM decisions (default: 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.task.strip():
        print("Task must not be empty. No mutation or API request was made.")
        return

    settings = get_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set. No mutation or API request was made.")
        return

    try:
        context = ToolContext(repository_root=args.root)
        task = CodingTask(
            objective=args.task,
            acceptance_criteria=tuple(args.acceptance),
        )
        policy = VerificationPolicy()
    except (RepositoryIngestionError, ValidationError) as exc:
        print(f"Invalid coding-task configuration: {exc}")
        return

    print("AUTONOMOUS EDITING ENABLED")
    print(f"Repository: {context.repository_root}")
    print("Completion policy:")
    print(f"- tests: {', '.join(path.as_posix() for path in policy.test_paths)}")
    print(f"- Ruff: {', '.join(path.as_posix() for path in policy.ruff_paths)}")
    print("- clean worktree required: yes")
    print("The agent can modify files.")
    print("Git commit/push are unavailable.")
    if input("Continue? [y/N] ").strip().casefold() != "y":
        print("Cancelled. No mutation or API request was made.")
        return

    result = run_coding_task(
        task,
        OpenAILLMClient(settings),
        create_editing_tool_registry(context),
        verification_policy=policy,
        agent_config=EditingAgentConfig(max_iterations=args.max_iterations),
    )

    print(f"Status: {result.status.value.upper()}")
    print("Files changed:")
    for path in result.changed_files:
        print(f"- {path.as_posix()}")
    print(f"Tests: {'PASS' if result.verification.tests_passed else 'NOT PASSED'}")
    print(f"Ruff: {'PASS' if result.verification.ruff_passed else 'NOT PASSED'}")
    print("Final diff:")
    if result.final_review is not None and result.final_review.unstaged_diff is not None:
        print(result.final_review.unstaged_diff.content or "(empty)")
        if result.final_review.diff_truncated:
            print("(diff output truncated)")
    else:
        print("(unavailable)")
    print("Agent answer:")
    print(result.final_answer or "(none)")
    if result.blockers:
        print("Blockers:")
        for blocker in result.blockers:
            print(f"- {blocker}")


if __name__ == "__main__":
    main()
