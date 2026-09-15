"""Offline scripted coding benchmarks with hidden, read-only file oracles."""

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from repomind.agent import AgentDecision, EditingAgentConfig
from repomind.coding import (
    CodingTask,
    CodingTaskResult,
    CodingWorkflowConfig,
    VerificationPolicy,
    run_coding_task,
)
from repomind.evaluation import (
    CodingBenchmarkCase,
    CodingBenchmarkSuite,
    EvaluationMode,
    FileTextExpectation,
    evaluate_coding_oracle,
    evaluate_coding_suite,
)
from repomind.tools import ToolContext, create_editing_tool_registry

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tool(name: str, **arguments: object) -> AgentDecision:
    return AgentDecision(action="tool", tool_name=name, tool_arguments=arguments)


def _final(message: str = "Ready for deterministic completion.") -> AgentDecision:
    return AgentDecision(action="final", final_answer=message)


class _ScriptedLLM:
    def __init__(self, responses: list[AgentDecision]) -> None:
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
        return self.responses.pop(0)


class _ScriptedWorkflowRunner:
    def __init__(
        self,
        script: Callable[[Path], list[AgentDecision]],
        *,
        max_iterations: int = 5,
        max_completion_attempts: int = 3,
    ) -> None:
        self.script = script
        self.max_iterations = max_iterations
        self.max_completion_attempts = max_completion_attempts
        self.starting_contents: list[bytes] = []
        self.llms: list[_ScriptedLLM] = []
        self.results: list[CodingTaskResult] = []
        self.visible_tasks: list[CodingTask] = []

    def __call__(
        self,
        task: CodingTask,
        workspace: Path,
        verification_policy: VerificationPolicy,
    ) -> CodingTaskResult:
        self.visible_tasks.append(task)
        self.starting_contents.append((workspace / "app.py").read_bytes())
        llm = _ScriptedLLM(self.script(workspace))
        self.llms.append(llm)
        result = run_coding_task(
            task,
            llm,
            create_editing_tool_registry(ToolContext(repository_root=workspace)),
            verification_policy=verification_policy,
            agent_config=EditingAgentConfig(max_iterations=self.max_iterations),
            workflow_config=CodingWorkflowConfig(
                max_completion_attempts=self.max_completion_attempts
            ),
        )
        self.results.append(result)
        return result


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _fixture_repository(root: Path, source: bytes) -> Path:
    root.mkdir()
    (root / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n.ruff_cache/\n*.pyc\n",
        encoding="utf-8",
    )
    (root / "app.py").write_bytes(source)
    (root / "tests").mkdir()
    (root / "tests" / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "evaluation@example.invalid")
    _git(root, "config", "user.name", "RepoMind Evaluation")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture baseline")
    return root


def _case(
    fixture: Path,
    case_id: str,
    *,
    contains: str,
    expects_recovery: bool = False,
    required_change: bool = True,
) -> CodingBenchmarkCase:
    return CodingBenchmarkCase(
        id=case_id,
        fixture_repository=fixture,
        task=CodingTask(objective="Make add return the sum of its two inputs."),
        required_changed_paths=(Path("app.py"),) if required_change else (),
        allowed_changed_paths=(Path("app.py"),),
        file_contains=(FileTextExpectation(path="app.py", text=contains),),
        expects_recovery=expects_recovery,
    )


def _replace_script(old: bytes, old_text: str, new_text: str) -> list[AgentDecision]:
    return [
        _tool(
            "replace_text",
            path="app.py",
            old_text=old_text,
            new_text=new_text,
            expected_sha256=_hash(old),
        ),
        _final(),
    ]


def test_completed_workflow_and_passing_hidden_oracle_is_true_success(
    tmp_path: Path,
) -> None:
    original = b"def add(a, b):\n    return a - b\n"
    fixture = _fixture_repository(tmp_path / "template", original)
    runner = _ScriptedWorkflowRunner(
        lambda workspace: _replace_script(original, "return a - b", "return a + b")
    )

    report = evaluate_coding_suite(
        CodingBenchmarkSuite(cases=(_case(fixture, "simple-bug", contains="a + b"),)),
        runner,
    )
    result = report.case_results[0]

    assert report.mode is EvaluationMode.OFFLINE_SCRIPTED
    assert result.workflow_status.value == "completed"
    assert result.oracle_passed is True
    assert result.task_success is True
    assert result.false_positive_completion is False
    assert result.final_verification_passed is True
    assert result.llm_calls == 2
    assert result.tool_calls == 1
    assert result.successful_mutations == 1
    assert report.task_success_rate == 1.0


def test_completed_workflow_can_be_hidden_oracle_false_positive(tmp_path: Path) -> None:
    original = b"def add(a, b):\n    return a - b\n"
    fixture = _fixture_repository(tmp_path / "template", original)
    runner = _ScriptedWorkflowRunner(
        lambda workspace: _replace_script(original, "return a - b", "return 5")
    )

    report = evaluate_coding_suite(
        CodingBenchmarkSuite(cases=(_case(fixture, "hard-coded", contains="a + b"),)),
        runner,
    )
    result = report.case_results[0]

    assert result.workflow_status.value == "completed"
    assert result.final_verification_passed is True
    assert result.oracle_passed is False
    assert result.task_success is False
    assert result.false_positive_completion is True
    assert report.workflow_completion_rate == 1.0
    assert report.task_success_rate == 0.0
    assert report.false_positive_completion_rate == 1.0
    assert any("configured evaluator substring" in item for item in result.oracle_failures)


def test_verification_failure_is_not_false_positive_completion(tmp_path: Path) -> None:
    original = b"def add(a, b):\n    return a - b\n"
    fixture = _fixture_repository(tmp_path / "template", original)
    runner = _ScriptedWorkflowRunner(
        lambda workspace: [_final(), _final("Still requesting completion.")],
        max_iterations=2,
        max_completion_attempts=2,
    )

    report = evaluate_coding_suite(
        CodingBenchmarkSuite(
            cases=(
                _case(
                    fixture,
                    "verification-failure",
                    contains="a + b",
                    required_change=False,
                ),
            )
        ),
        runner,
    )
    result = report.case_results[0]

    assert result.workflow_status.value == "verification_failed"
    assert result.final_verification_passed is False
    assert result.task_success is False
    assert result.false_positive_completion is False
    assert report.workflow_completion_rate == 0.0


def test_failure_then_correction_is_measured_as_recovery(tmp_path: Path) -> None:
    original = b"def add(a, b):\n    return a + b\n"
    incorrect = b"def add(a, b):\n    return a - b\n"
    fixture = _fixture_repository(tmp_path / "template", original)

    def recovery_script(workspace: Path) -> list[AgentDecision]:
        return [
            _tool(
                "replace_text",
                path="app.py",
                old_text="return a + b",
                new_text="return a - b",
                expected_sha256=_hash(original),
            ),
            _final("The first edit should be ready."),
            _tool(
                "replace_text",
                path="app.py",
                old_text="return a - b",
                new_text="return a + b",
                expected_sha256=_hash(incorrect),
            ),
            _final("Corrected after verification feedback."),
        ]

    runner = _ScriptedWorkflowRunner(recovery_script, max_iterations=4)
    case = _case(
        fixture,
        "self-correction",
        contains="a + b",
        expects_recovery=True,
        required_change=False,
    )

    report = evaluate_coding_suite(CodingBenchmarkSuite(cases=(case,)), runner)
    result = report.case_results[0]

    assert result.task_success is True
    assert result.recovery_observed is True
    assert result.successful_mutations == 2
    assert result.completion_attempts == 2
    assert report.recovery_rate == 1.0


def test_hidden_oracle_fields_never_reach_task_prompt_or_agent_events(tmp_path: Path) -> None:
    original = b"def add(a, b):\n    return a + b\n"
    fixture = _fixture_repository(tmp_path / "template", original)
    hidden = "EVALUATOR_ONLY_SENTINEL"
    runner = _ScriptedWorkflowRunner(lambda workspace: [_final("No edit needed.")])
    case = CodingBenchmarkCase(
        id="leakage",
        fixture_repository=fixture,
        task=CodingTask(objective="Confirm current addition behavior."),
        required_changed_paths=(Path("app.py"),),
        file_contains=(FileTextExpectation(path="app.py", text=hidden),),
    )

    evaluate_coding_suite(CodingBenchmarkSuite(cases=(case,)), runner)

    agent_result = runner.results[0]
    visible_data = json.dumps(
        {
            "task": runner.visible_tasks[0].model_dump(mode="json"),
            "prompts": runner.llms[0].calls,
            "agent_query": agent_result.agent_run.query,
            "steps": [
                step.model_dump(mode="json") for step in agent_result.agent_run.steps
            ],
        },
        sort_keys=True,
    )
    assert hidden not in visible_data
    assert "file_contains" not in visible_data
    assert "required_changed_paths" not in visible_data


def test_repeated_cases_start_from_pristine_template(tmp_path: Path) -> None:
    original = b"def add(a, b):\n    return a - b\n"
    fixture = _fixture_repository(tmp_path / "template", original)
    runner = _ScriptedWorkflowRunner(
        lambda workspace: _replace_script(original, "return a - b", "return a + b")
    )
    suite = CodingBenchmarkSuite(
        cases=(
            _case(fixture, "first-run", contains="a + b"),
            _case(fixture, "second-run", contains="a + b"),
        )
    )

    report = evaluate_coding_suite(suite, runner)

    assert runner.starting_contents == [original, original]
    assert (fixture / "app.py").read_bytes() == original
    assert [result.task_success for result in report.case_results] == [True, True]


def test_dirty_fixture_is_rejected_before_runner_receives_task(tmp_path: Path) -> None:
    original = b"def add(a, b):\n    return a + b\n"
    fixture = _fixture_repository(tmp_path / "template", original)
    (fixture / "app.py").write_text("dirty\n", encoding="utf-8")
    runner = _ScriptedWorkflowRunner(lambda workspace: [_final()])
    case = _case(
        fixture,
        "dirty-template",
        contains="a + b",
        required_change=False,
    )

    with pytest.raises(ValueError, match="must be clean"):
        evaluate_coding_suite(CodingBenchmarkSuite(cases=(case,)), runner)

    assert runner.visible_tasks == []


def test_coding_oracle_paths_reject_traversal() -> None:
    with pytest.raises(ValidationError, match="repository-relative"):
        CodingBenchmarkCase(
            id="unsafe",
            fixture_repository=Path("fixture"),
            task=CodingTask(objective="Unsafe oracle should fail validation."),
            file_contains=(FileTextExpectation(path="../outside", text="secret"),),
        )


def test_file_state_oracles_are_read_only_and_structured(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    before = target.read_bytes()
    case = CodingBenchmarkCase(
        id="file-oracles",
        fixture_repository=tmp_path,
        task=CodingTask(objective="Evaluate final file state."),
        file_exists=(Path("app.py"),),
        file_not_exists=(Path("removed.py"),),
        file_contains=(FileTextExpectation(path="app.py", text="value = 1"),),
        file_not_contains=(FileTextExpectation(path="app.py", text="secret"),),
    )

    oracle = evaluate_coding_oracle(case, workspace, ())

    assert oracle.passed is True
    assert [check.check for check in oracle.checks] == [
        "file_exists",
        "file_not_exists",
        "file_contains",
        "file_not_contains",
    ]
    assert target.read_bytes() == before
