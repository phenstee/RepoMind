"""Optional, explicitly confirmed live check for the controlled editing agent.

Run only against a disposable or version-controlled test repository::

    uv run python scripts/manual_editing_agent_check.py \
        "Add a short docstring to foo()" --root path/to/test/repository
"""

import argparse
from pathlib import Path

from pydantic import ValidationError

from repomind.agent import AgentError, EditingAgentConfig, run_editing_agent
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
        description="Run one bounded live task through RepoMind's EDITING agent.",
    )
    parser.add_argument("task", help="Coding task for the editing agent")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Explicit repository root that the agent may modify",
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
    except (RepositoryIngestionError, ValidationError) as exc:
        print(f"Invalid repository root: {exc}")
        return

    print("THIS AGENT CAN MODIFY FILES IN THE SELECTED REPOSITORY.")
    print(f"Repository root: {context.repository_root}")
    print("Mutation capability enabled: create_file, replace_text")
    print("Test execution capability enabled: run_tests, run_ruff")
    print("No arbitrary shell is available.")
    print("No Git commit or push is available.")
    if input("Continue? [y/N] ").strip().casefold() != "y":
        print("Cancelled. No mutation or API request was made.")
        return

    try:
        run = run_editing_agent(
            args.task,
            OpenAILLMClient(settings),
            create_editing_tool_registry(context),
            config=EditingAgentConfig(max_iterations=args.max_iterations),
        )
    except AgentError as exc:
        print(f"Editing agent check failed: {exc}")
        return

    print("EDITING AGENT RUN")
    for step in run.steps:
        if step.decision.action == "final":
            print(f"{step.iteration}: final")
        elif step.observation is not None:
            outcome = "ok" if step.observation.success else f"error: {step.observation.error}"
            print(f"{step.iteration}: {step.decision.tool_name} -> {outcome}")
    print(f"Status: {run.status.value}")
    print(f"LLM calls: {run.llm_calls}; tool executions: {run.tool_calls}")
    if run.final_answer is not None:
        print("\nFinal answer:\n")
        print(run.final_answer)


if __name__ == "__main__":
    main()
