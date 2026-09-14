"""Offline deterministic tests for the controlled editing agent."""

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from repomind.agent import (
    AgentDecision,
    AgentRunStatus,
    EditingAgentConfig,
    run_editing_agent,
)
from repomind.agent.prompts import EDITING_AGENT_SYSTEM_PROMPT
from repomind.tools import ToolContext, create_editing_tool_registry


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tool(name: str, **arguments: object) -> AgentDecision:
    return AgentDecision(action="tool", tool_name=name, tool_arguments=arguments)


def _final(answer: str = "done") -> AgentDecision:
    return AgentDecision(action="final", final_answer=answer)


class _ScriptedLLM:
    def __init__(
        self,
        responses: list[AgentDecision | Callable[[str], AgentDecision]],
    ) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        response = self.responses.pop(0)
        return response(prompt) if callable(response) else response


def _registry(root: Path):
    return create_editing_tool_registry(ToolContext(repository_root=root))


def _write_passing_project(root: Path) -> None:
    (root / "app.py").write_bytes(b"value = 1\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_app.py").write_text(
        "from app import value\n\ndef test_value():\n    assert value == 1\n",
        encoding="utf-8",
    )


def test_basic_edit_diff_test_final_flow(tmp_path: Path) -> None:
    _write_passing_project(tmp_path)
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True, capture_output=True)
    before = b"value = 1\n"
    llm = _ScriptedLLM(
        [
            _tool("read_file", path="app.py"),
            _tool(
                "replace_text",
                path="app.py",
                old_text="value = 1\n",
                new_text="value = 1  # retained\n",
                expected_sha256=_hash(before),
            ),
            _tool("git_diff", path="app.py"),
            _tool("run_tests", paths=["tests/test_app.py"]),
            _final("Changed app.py and observed passing tests."),
        ]
    )

    run = run_editing_agent("Add a clarifying comment", llm, _registry(tmp_path))

    assert run.status is AgentRunStatus.COMPLETED
    assert (run.iterations, run.tool_calls) == (5, 4)
    assert (tmp_path / "app.py").read_bytes() == b"value = 1  # retained\n"
    assert "+value = 1  # retained" in run.steps[2].observation.output["content"]
    assert run.steps[3].observation.output["passed"] is True
    assert all(call["system_prompt"] == EDITING_AGENT_SYSTEM_PROMPT for call in llm.calls)


def test_failed_test_is_observed_then_second_edit_passes(tmp_path: Path) -> None:
    before = b"def value():\n    return 2\n"
    incorrect = b"def value():\n    return 3\n"
    corrected = b"def value():\n    return 2\n"
    (tmp_path / "app.py").write_bytes(before)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(
        "from app import value\n\ndef test_value():\n    assert value() == 2\n",
        encoding="utf-8",
    )
    llm = _ScriptedLLM(
        [
            _tool("read_file", path="app.py"),
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 2",
                new_text="return 3",
                expected_sha256=_hash(before),
            ),
            _tool("run_tests", paths=["tests/test_app.py"]),
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 3",
                new_text="return 2",
                expected_sha256=_hash(incorrect),
            ),
            _tool("run_tests", paths=["tests/test_app.py"]),
            _final("Corrected app.py after observing one failing test run."),
        ]
    )

    run = run_editing_agent("Keep value returning two", llm, _registry(tmp_path))

    assert run.steps[2].observation.success
    assert run.steps[2].observation.output["passed"] is False
    assert run.steps[4].observation.output["passed"] is True
    assert (tmp_path / "app.py").read_bytes() == corrected
    assert [step.decision.tool_name for step in run.steps].count("run_tests") == 2


def test_stale_hash_failure_can_be_recovered_by_rereading(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    original = b"value = 1\n"
    external = b"value = 2\n"
    target.write_bytes(original)

    def external_change(_: str) -> AgentDecision:
        target.write_bytes(external)
        return _tool(
            "replace_text",
            path="app.py",
            old_text="value = 1",
            new_text="value = 3",
            expected_sha256=_hash(original),
        )

    llm = _ScriptedLLM(
        [
            _tool("read_file", path="app.py"),
            external_change,
            _tool("read_file", path="app.py"),
            _tool(
                "replace_text",
                path="app.py",
                old_text="value = 2",
                new_text="value = 3",
                expected_sha256=_hash(external),
            ),
            _final("Re-read the externally changed file, then edited it."),
        ]
    )

    run = run_editing_agent("Update the value safely", llm, _registry(tmp_path))

    assert not run.steps[1].observation.success
    assert "changed since it was read" in run.steps[1].observation.error
    assert run.steps[2].observation.output["sha256"] == _hash(external)
    assert run.steps[3].observation.success
    assert target.read_bytes() == b"value = 3\n"


def test_create_file_inspect_and_test_flow(tmp_path: Path) -> None:
    _write_passing_project(tmp_path)
    content = "\"\"\"Small helper.\"\"\"\n"
    llm = _ScriptedLLM(
        [
            _tool("list_directory", path="."),
            _tool("create_file", path="helper.py", content=content),
            _tool("read_file", path="helper.py"),
            _tool("run_tests", paths=["tests"]),
            _final("Created helper.py and observed passing tests."),
        ]
    )

    run = run_editing_agent("Create a helper module", llm, _registry(tmp_path))

    assert (tmp_path / "helper.py").read_text(encoding="utf-8") == content
    assert run.steps[1].observation.output["sha256"] == _hash(content.encode())
    assert run.steps[2].observation.output["content"] == content
    assert run.steps[3].observation.output["passed"] is True


def test_successful_mutation_limit_blocks_later_write(tmp_path: Path) -> None:
    llm = _ScriptedLLM(
        [
            _tool("create_file", path="first.py", content="first\n"),
            _tool("create_file", path="second.py", content="second\n"),
            _final("Stopped at the configured mutation budget."),
        ]
    )

    run = run_editing_agent(
        "Create files",
        llm,
        _registry(tmp_path),
        config=EditingAgentConfig(max_iterations=3, max_mutations_per_run=1),
    )

    assert (tmp_path / "first.py").exists()
    assert not (tmp_path / "second.py").exists()
    assert not run.steps[1].observation.success
    assert "mutation budget exhausted" in run.steps[1].observation.error
    assert run.tool_calls == 1
