"""Optional one-query live check for the handwritten READ-ONLY AGENT.

Run with::

    uv run python scripts/manual_agent_check.py \
        "Where is async retry behavior implemented?"

The script exits before constructing an OpenAI client when ``OPENAI_API_KEY``
is absent. Its registry contains only RepoMind's six read-only inspection tools.
"""

import argparse
from pathlib import Path

from pydantic import ValidationError

from repomind.agent import AgentConfig, AgentError, run_read_only_agent
from repomind.config import get_settings
from repomind.ingestion import RepositoryIngestionError
from repomind.llm import OpenAILLMClient
from repomind.tools import ToolContext, create_default_tool_registry


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
        description="Run one bounded live query through the READ-ONLY RepoMind agent.",
    )
    parser.add_argument("question", help="Repository question for the read-only agent")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root to inspect (default: current directory)",
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
    if not args.question.strip():
        print("Question must not be empty. No API request was made.")
        return

    settings = get_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set. No agent API request was made.")
        return

    try:
        registry = create_default_tool_registry(ToolContext(repository_root=args.root))
        run = run_read_only_agent(
            args.question,
            OpenAILLMClient(settings),
            registry,
            config=AgentConfig(max_iterations=args.max_iterations),
        )
    except (AgentError, RepositoryIngestionError, ValidationError) as exc:
        print(f"Read-only agent check failed: {exc}")
        return

    print("READ-ONLY AGENT RUN")
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
