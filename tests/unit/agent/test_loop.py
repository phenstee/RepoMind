"""Offline scripted tests for the handwritten read-only agent loop."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from repomind.agent import (
    AgentConfig,
    AgentDecision,
    AgentError,
    AgentRunStatus,
    run_read_only_agent,
)
from repomind.llm import LLMError
from repomind.tools import ToolContext, create_default_tool_registry


class _UnexpectedResponse(BaseModel):
    value: str = "unexpected"


class _ScriptedLLM:
    def __init__(self, responses: list[BaseModel | Exception]) -> None:
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
        self.calls.append(
            {
                "prompt": prompt,
                "response_model": response_model,
                "system_prompt": system_prompt,
                "temperature": temperature,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _tool(name: str, **arguments: object) -> AgentDecision:
    return AgentDecision(action="tool", tool_name=name, tool_arguments=arguments)


def _final(answer: str = "done") -> AgentDecision:
    return AgentDecision(action="final", final_answer=answer)


def _registry(root: Path):
    return create_default_tool_registry(ToolContext(repository_root=root))


def test_basic_one_tool_run_records_observation_and_completes(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("raise RepositoryIngestionError()\n", encoding="utf-8")
    llm = _ScriptedLLM(
        [_tool("search_code", query="RepositoryIngestionError"), _final("Found in app.py:1")]
    )

    run = run_read_only_agent("Find the error", llm, _registry(tmp_path))

    assert run.status is AgentRunStatus.COMPLETED
    assert run.final_answer == "Found in app.py:1"
    assert (run.iterations, run.llm_calls, run.tool_calls) == (2, 2, 1)
    assert run.steps[0].observation is not None
    assert run.steps[0].observation.success
    assert run.steps[0].observation.output["matches"][0]["path"] == "app.py"
    assert "RepositoryIngestionError" in llm.calls[1]["prompt"]
    assert llm.calls[0]["temperature"] == 0.0


def test_multi_tool_run_executes_exact_sequential_order(tmp_path: Path) -> None:
    source = tmp_path / "src" / "client.py"
    source.parent.mkdir()
    source.write_text("class OpenAILLMClient:\n    pass\n", encoding="utf-8")
    llm = _ScriptedLLM(
        [
            _tool("list_directory", path="src"),
            _tool("search_code", query="OpenAILLMClient", path="src"),
            _tool("read_file", path="src/client.py", start_line=1, end_line=2),
            _final("The class is in src/client.py:1-2."),
        ]
    )

    run = run_read_only_agent("Locate the client", llm, _registry(tmp_path))

    assert [step.decision.tool_name for step in run.steps[:-1]] == [
        "list_directory",
        "search_code",
        "read_file",
    ]
    assert run.iterations == 4 and run.tool_calls == 3
    assert run.steps[2].observation.output["content"].startswith("class OpenAILLMClient")


def test_tool_failure_and_invalid_arguments_become_recoverable_observations(
    tmp_path: Path,
) -> None:
    (tmp_path / "valid.py").write_text("answer\n", encoding="utf-8")
    llm = _ScriptedLLM(
        [
            _tool("read_file", path="missing.py"),
            _tool("read_file", path=123, imaginary_argument="bad"),
            _tool("read_file", path="valid.py"),
            _final(),
        ]
    )

    run = run_read_only_agent("Recover from bad reads", llm, _registry(tmp_path))

    observations = [step.observation for step in run.steps[:-1]]
    assert not observations[0].success and "does not exist" in observations[0].error
    assert not observations[1].success and "Invalid arguments" in observations[1].error
    assert observations[2].success
    assert run.status is AgentRunStatus.COMPLETED
    assert "does not exist" in llm.calls[1]["prompt"]
    assert "Invalid arguments" in llm.calls[2]["prompt"]


def test_unknown_tool_is_an_observation_and_model_can_recover(tmp_path: Path) -> None:
    llm = _ScriptedLLM(
        [_tool("magic_delete"), _tool("list_directory", path="."), _final()]
    )
    run = run_read_only_agent("Inspect safely", llm, _registry(tmp_path))

    assert not run.steps[0].observation.success
    assert run.steps[0].observation.error == "Unknown tool: magic_delete"
    assert run.steps[1].observation.success
    assert run.tool_calls == 2


def test_max_iterations_stops_without_an_extra_llm_call(tmp_path: Path) -> None:
    llm = _ScriptedLLM(
        [
            _tool("list_directory", path="."),
            _tool("list_directory", path=".", recursive=True),
            _tool("list_directory", path="."),
            _final("must not be reached"),
        ]
    )
    run = run_read_only_agent(
        "Keep inspecting",
        llm,
        _registry(tmp_path),
        config=AgentConfig(max_iterations=3),
    )

    assert run.status is AgentRunStatus.MAX_ITERATIONS
    assert run.final_answer is None
    assert (run.iterations, run.llm_calls, run.tool_calls) == (3, 3, 3)
    assert len(llm.calls) == 3


def test_repeated_identical_tool_call_executes_twice_then_is_blocked(tmp_path: Path) -> None:
    repeated = [
        AgentDecision(
            action="tool",
            tool_name="list_directory",
            tool_arguments={"path": ".", "recursive": False},
        ),
        AgentDecision(
            action="tool",
            tool_name="list_directory",
            tool_arguments={"recursive": False, "path": "."},
        ),
        _tool("list_directory", path=".", recursive=False),
        _tool("list_directory", recursive=False, path="."),
    ]
    llm = _ScriptedLLM([*repeated, _final()])
    run = run_read_only_agent(
        "Do not loop forever",
        llm,
        _registry(tmp_path),
        config=AgentConfig(max_iterations=5),
    )

    assert run.steps[0].observation.success
    assert run.steps[1].observation.success
    assert not run.steps[2].observation.success
    assert run.steps[2].observation.error == (
        "Repeated identical tool call; choose a different action."
    )
    assert not run.steps[3].observation.success
    assert run.tool_calls == 2


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_empty_query_is_rejected_before_llm_call(tmp_path: Path, query: str) -> None:
    llm = _ScriptedLLM([_final()])
    with pytest.raises(AgentError, match="must not be empty"):
        run_read_only_agent(query, llm, _registry(tmp_path))
    assert llm.calls == []


def test_final_first_stops_without_tool_or_extra_llm_calls(tmp_path: Path) -> None:
    llm = _ScriptedLLM([_final("immediate"), _tool("list_directory")])
    run = run_read_only_agent("Answer directly", llm, _registry(tmp_path))
    assert run.final_answer == "immediate"
    assert (run.llm_calls, run.tool_calls, len(llm.calls)) == (1, 0, 1)


def test_llm_failure_and_unexpected_model_raise_chained_agent_errors(tmp_path: Path) -> None:
    llm_error = LLMError("provider failed")
    with pytest.raises(AgentError, match="decision failed") as error_info:
        run_read_only_agent("question", _ScriptedLLM([llm_error]), _registry(tmp_path))
    assert error_info.value.__cause__ is llm_error

    with pytest.raises(AgentError, match="unexpected"):
        run_read_only_agent(
            "question",
            _ScriptedLLM([_UnexpectedResponse()]),
            _registry(tmp_path),
        )


def test_agent_tools_observe_files_created_after_registry_construction(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    (tmp_path / "live.py").write_bytes(b"LIVE_WORKTREE_VALUE\n")
    llm = _ScriptedLLM([_tool("read_file", path="live.py"), _final()])

    run = run_read_only_agent("Read the live file", llm, registry)
    assert run.steps[0].observation.output["content"] == "LIVE_WORKTREE_VALUE\n"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")
def test_read_search_and_git_outputs_are_json_compatible_observations(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "live.py").write_text("SERIAL_VALUE\n", encoding="utf-8")
    llm = _ScriptedLLM(
        [
            _tool("read_file", path="live.py"),
            _tool("search_code", query="SERIAL_VALUE"),
            _tool("git_status"),
            _final(),
        ]
    )

    run = run_read_only_agent("Collect observations", llm, _registry(tmp_path))
    serialized = json.dumps(run.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    assert "SERIAL_VALUE" in serialized
    assert '"branch": "main"' in serialized


def test_default_agent_registry_is_strictly_read_only(tmp_path: Path) -> None:
    names = [tool.name for tool in _registry(tmp_path).list_tools()]
    assert names == [
        "find_symbol",
        "git_diff",
        "git_status",
        "list_directory",
        "read_file",
        "search_code",
    ]
    assert not {"write_file", "apply_patch", "run_tests", "shell", "run_command"} & set(names)
