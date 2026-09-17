"""End-to-end offline tests for autonomous coding-task completion gates."""

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from repomind.agent import AgentDecision, AgentRunStatus, EditingAgentConfig
from repomind.coding import (
    CodingPlan,
    CodingReview,
    CodingTask,
    CodingTaskStatus,
    CodingWorkflowConfig,
    VerificationPolicy,
    run_coding_task,
)
from repomind.jobs import JobCancellationRequested
from repomind.tools import (
    ToolConfig,
    ToolContext,
    ToolDefinition,
    ToolExecutionError,
    ToolRegistry,
    create_default_tool_registry,
    create_editing_tool_registry,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tool(name: str, **arguments: object) -> AgentDecision:
    return AgentDecision(action="tool", tool_name=name, tool_arguments=arguments)


def _final(answer: str = "Ready for completion.") -> AgentDecision:
    return AgentDecision(action="final", final_answer=answer)


class _ScriptedLLM:
    def __init__(
        self,
        responses: list[AgentDecision | Callable[[str], AgentDecision]],
        *,
        plan_response: object | None = None,
        review_responses: list[object] | None = None,
    ) -> None:
        self.responses = responses
        self.plan_response = plan_response
        self.review_responses = review_responses or []
        self.calls: list[dict[str, Any]] = []
        self.planning_calls: list[dict[str, Any]] = []
        self.review_calls: list[dict[str, Any]] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        call = {"prompt": prompt, "system_prompt": system_prompt}
        if response_model is CodingPlan:
            payload = json.loads(prompt)
            self.planning_calls.append(call)
            if self.plan_response is not None:
                if isinstance(self.plan_response, Exception):
                    raise self.plan_response
                response = (
                    self.plan_response(prompt)
                    if callable(self.plan_response)
                    else self.plan_response
                )
                return response_model.model_validate(response)
            criteria = payload["task"]["acceptance_criteria"]
            return CodingPlan.model_validate(
                {
                    "task_summary": "Implement the requested bounded change.",
                    "steps": [
                        {
                            "step_id": 1,
                            "action": "Inspect, implement, and verify the requested change.",
                            "criterion_indices": list(range(len(criteria))),
                            "verification": ["pytest", "ruff"],
                        }
                    ],
                    "acceptance_coverage": [
                        {"criterion_index": index, "step_ids": [1]}
                        for index in range(len(criteria))
                    ],
                    "verification_plan": ["pytest", "ruff"],
                }
            )
        if response_model is CodingReview:
            payload = json.loads(prompt)
            self.review_calls.append(call)
            if self.review_responses:
                scripted = self.review_responses.pop(0)
                if isinstance(scripted, Exception):
                    raise scripted
                response = scripted(prompt) if callable(scripted) else scripted
                return response_model.model_validate(response)
            criteria = payload["task"]["acceptance_criteria"]
            return CodingReview.model_validate(
                {
                    "verdict": "approve",
                    "workspace_revision": payload["workspace_revision"],
                    "acceptance_results": [
                        {
                            "criterion_index": index,
                            "status": "satisfied",
                            "evidence": "The bounded diff and verification support this criterion.",
                        }
                        for index in range(len(criteria))
                    ],
                }
            )
        self.calls.append(call)
        response = self.responses.pop(0)
        return response(prompt) if callable(response) else response


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _project(
    root: Path,
    *,
    source: bytes = b"def value():\n    return 1\n",
    expected: int = 2,
) -> None:
    (root / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n.ruff_cache/\n*.pyc\n",
        encoding="utf-8",
    )
    (root / "app.py").write_bytes(source)
    (root / "tests").mkdir()
    (root / "tests" / "test_app.py").write_text(
        f"from app import value\n\n\ndef test_value():\n    assert value() == {expected}\n",
        encoding="utf-8",
    )
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "RepoMind Tests")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "baseline")


def _registry(root: Path, config: ToolConfig | None = None) -> ToolRegistry:
    return create_editing_tool_registry(ToolContext(repository_root=root), config=config)


def test_successful_task_runs_automatic_gates_and_returns_git_evidence(
    tmp_path: Path,
) -> None:
    before = b"def value():\n    return 1\n"
    _project(tmp_path, source=before, expected=2)
    task = CodingTask(
        objective="Return two from value().",
        acceptance_criteria=("The value test passes.", "Keep the change scoped."),
    )
    llm = _ScriptedLLM(
        [
            _tool("read_file", path="app.py"),
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 1",
                new_text="return 2",
                expected_sha256=_hash(before),
            ),
            _final("Changed app.py to return two."),
        ]
    )

    result = run_coding_task(task, llm, _registry(tmp_path))

    assert result.status is CodingTaskStatus.COMPLETED
    assert result.workspace_revision == 1
    assert result.successful_mutations == 1
    assert result.changed_files == (Path("app.py"),)
    assert result.verification.tests_passed is True
    assert result.verification.tests_revision == 1
    assert result.verification.ruff_passed is True
    assert result.verification.ruff_revision == 1
    assert result.verification_executions == 2
    assert result.final_review is not None
    assert result.final_review.workflow_changed_files == (Path("app.py"),)
    assert "return 2" in result.final_review.unstaged_diff.content
    assert "The value test passes." in llm.calls[0]["prompt"]
    assert "only a request" in llm.calls[0]["prompt"]
    assert '<advisory_plan trust="model-authored-advice">' in llm.calls[0]["prompt"]
    assert "Inspect, implement, and verify the requested change." in llm.calls[0]["prompt"]
    assert len(llm.planning_calls) == 1
    assert len(llm.review_calls) == 1


def test_cancellation_after_first_mutation_is_deferred_through_verification(
    tmp_path: Path,
) -> None:
    before = b"def value():\n    return 1\n"
    _project(tmp_path, source=before, expected=2)
    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 1",
                new_text="return 2",
                expected_sha256=_hash(before),
            ),
            _final("Changed app.py safely."),
        ]
    )

    class DeferredCancellation:
        def __init__(self) -> None:
            self.side_effects = 0
            self.deferred_checkpoints = 0

        def checkpoint(self) -> None:
            if self.side_effects:
                self.deferred_checkpoints += 1

        def side_effect_started(self) -> None:
            self.side_effects += 1

    cancellation = DeferredCancellation()

    result = run_coding_task(
        CodingTask(objective="Return two."),
        llm,
        _registry(tmp_path),
        cancellation=cancellation,
    )

    assert result.status is CodingTaskStatus.COMPLETED
    assert result.verification.tests_passed is True
    assert result.verification.ruff_passed is True
    assert cancellation.side_effects == 1
    assert cancellation.deferred_checkpoints > 0


def test_cancellation_before_first_mutation_leaves_repository_unchanged(
    tmp_path: Path,
) -> None:
    before = b"def value():\n    return 1\n"
    _project(tmp_path, source=before, expected=1)
    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 1",
                new_text="return 2",
                expected_sha256=_hash(before),
            )
        ]
    )

    class CancelBeforeAgent:
        def __init__(self) -> None:
            self.checkpoints = 0

        def checkpoint(self) -> None:
            self.checkpoints += 1
            if self.checkpoints == 4:
                raise JobCancellationRequested("cancelled")

        def side_effect_started(self) -> None:
            raise AssertionError("mutation must not start")

    with pytest.raises(JobCancellationRequested):
        run_coding_task(
            CodingTask(objective="Change value."),
            llm,
            _registry(tmp_path),
            cancellation=CancelBeforeAgent(),
        )

    assert (tmp_path / "app.py").read_bytes() == before
    assert llm.calls == []


def test_tests_are_rerun_when_later_edit_makes_agent_evidence_stale(
    tmp_path: Path,
) -> None:
    original = b"def value():\n    return 0\n"
    first_edit = b"def value():\n    return 1\n"
    _project(tmp_path, source=original, expected=1)
    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 0",
                new_text="return 1",
                expected_sha256=_hash(original),
            ),
            _tool("run_tests", paths=["tests"]),
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 1",
                new_text="return 1  # fresh edit",
                expected_sha256=_hash(first_edit),
            ),
            _final(),
        ]
    )
    policy = VerificationPolicy(require_ruff=False)

    result = run_coding_task(
        CodingTask(objective="Update value safely."),
        llm,
        _registry(tmp_path),
        verification_policy=policy,
    )

    assert result.status is CodingTaskStatus.COMPLETED
    assert result.workspace_revision == 2
    assert result.verification.tests_revision == 2
    assert result.verification_executions == 2


def test_ruff_is_rerun_when_later_edit_makes_agent_evidence_stale(
    tmp_path: Path,
) -> None:
    original = b"value = 0\n"
    first_edit = b"value = 1\n"
    _project(tmp_path, source=original, expected=2)
    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="value = 0",
                new_text="value = 1",
                expected_sha256=_hash(original),
            ),
            _tool("run_ruff", paths=["."]),
            _tool(
                "replace_text",
                path="app.py",
                old_text="value = 1",
                new_text="value = 2",
                expected_sha256=_hash(first_edit),
            ),
            _final(),
        ]
    )
    policy = VerificationPolicy(require_tests=False)

    result = run_coding_task(
        CodingTask(objective="Update a module constant."),
        llm,
        _registry(tmp_path),
        verification_policy=policy,
    )

    assert result.status is CodingTaskStatus.COMPLETED
    assert result.workspace_revision == 2
    assert result.verification.ruff_revision == 2
    assert result.verification_executions == 2


def test_failed_final_tests_feed_back_into_same_agent_then_correction_completes(
    tmp_path: Path,
) -> None:
    original = b"def value():\n    return 2\n"
    incorrect = b"def value():\n    return 3\n"
    _project(tmp_path, source=original, expected=2)
    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 2",
                new_text="return 3",
                expected_sha256=_hash(original),
            ),
            _final("The edit should be ready."),
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 3",
                new_text="return 2",
                expected_sha256=_hash(incorrect),
            ),
            _final("Corrected the failing change."),
        ]
    )

    result = run_coding_task(
        CodingTask(objective="Keep value returning two."),
        llm,
        _registry(tmp_path),
        agent_config=EditingAgentConfig(max_iterations=4),
    )

    assert result.status is CodingTaskStatus.COMPLETED
    assert result.completion_attempts == 2
    assert result.workspace_revision == 2
    assert result.verification.tests_passed is True
    assert result.verification.ruff_passed is True
    assert result.verification_executions == 4
    assert result.agent_run.steps[1].workflow_feedback is not None
    assert "Required pytest verification did not pass" in llm.calls[2]["prompt"]
    assert '<workflow_feedback trust="trusted-workflow-instruction">' in llm.calls[2]["prompt"]
    assert (tmp_path / "app.py").read_bytes() == original


def test_completion_attempt_limit_stops_repeated_failing_requests(
    tmp_path: Path,
) -> None:
    _project(tmp_path, expected=2)
    llm = _ScriptedLLM([_final("done"), _final("still done")])

    result = run_coding_task(
        CodingTask(objective="Make the failing test pass."),
        llm,
        _registry(tmp_path),
        agent_config=EditingAgentConfig(max_iterations=4),
        workflow_config=CodingWorkflowConfig(max_completion_attempts=2),
    )

    assert result.status is CodingTaskStatus.VERIFICATION_FAILED
    assert result.completion_attempts == 2
    assert result.agent_run.status is AgentRunStatus.WORKFLOW_STOPPED
    assert any("pytest verification did not pass" in item for item in result.blockers)
    assert len(llm.calls) == 2


def test_dirty_worktree_default_fails_before_llm_or_mutation(tmp_path: Path) -> None:
    _project(tmp_path, expected=1)
    target = tmp_path / "app.py"
    target.write_bytes(b"user change\n")
    llm = _ScriptedLLM([_final()])

    result = run_coding_task(
        CodingTask(objective="Do not touch existing work."), llm, _registry(tmp_path)
    )

    assert result.status is CodingTaskStatus.PRECONDITION_FAILED
    assert result.baseline is not None and not result.baseline.clean
    assert result.changed_files == (Path("app.py"),)
    assert llm.calls == []
    assert target.read_bytes() == b"user change\n"


def test_dirty_worktree_opt_in_distinguishes_baseline_and_workflow_changes(
    tmp_path: Path,
) -> None:
    original = b"def value():\n    return 1\n"
    _project(tmp_path, source=original, expected=1)
    (tmp_path / "notes.txt").write_text("pre-existing\n", encoding="utf-8")
    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 1",
                new_text="return 1  # clarified",
                expected_sha256=_hash(original),
            ),
            _final(),
        ]
    )
    policy = VerificationPolicy(require_tests=False, require_ruff=False)

    result = run_coding_task(
        CodingTask(objective="Clarify app.py."),
        llm,
        _registry(tmp_path),
        verification_policy=policy,
        workflow_config=CodingWorkflowConfig(require_clean_worktree=False),
    )

    assert result.status is CodingTaskStatus.COMPLETED
    assert result.final_review.baseline_changed_files == (Path("notes.txt"),)
    assert result.final_review.workflow_changed_files == (Path("app.py"),)
    assert result.final_review.unexpected_changed_files == ()
    assert result.changed_files == (Path("app.py"), Path("notes.txt"))


def test_dirty_opt_in_final_review_captures_staged_diff(tmp_path: Path) -> None:
    _project(tmp_path, expected=1)
    (tmp_path / "app.py").write_bytes(b"def value():\n    return 1  # staged\n")
    _git(tmp_path, "add", "app.py")

    result = run_coding_task(
        CodingTask(objective="Confirm the staged user change."),
        _ScriptedLLM([_final("No additional change required.")]),
        _registry(tmp_path),
        verification_policy=VerificationPolicy(require_tests=False, require_ruff=False),
        workflow_config=CodingWorkflowConfig(require_clean_worktree=False),
    )

    assert result.status is CodingTaskStatus.COMPLETED
    assert result.final_review.staged_diff is not None
    assert result.final_review.staged_diff.staged
    assert "# staged" in result.final_review.staged_diff.content
    assert result.final_review.baseline_changed_files == (Path("app.py"),)


def test_external_unexpected_file_blocks_completion_and_is_not_cleaned(
    tmp_path: Path,
) -> None:
    _project(tmp_path, expected=1)

    def external_change(_: str) -> AgentDecision:
        (tmp_path / "surprise.py").write_text("external = True\n", encoding="utf-8")
        return _final()

    llm = _ScriptedLLM([external_change])

    result = run_coding_task(
        CodingTask(objective="Inspect current behavior."),
        llm,
        _registry(tmp_path),
        workflow_config=CodingWorkflowConfig(max_completion_attempts=1),
    )

    assert result.status is CodingTaskStatus.VERIFICATION_FAILED
    assert result.final_review.unexpected_changed_files == (Path("surprise.py"),)
    assert (tmp_path / "surprise.py").exists()


def test_no_change_task_can_complete_with_fresh_checks_and_clean_review(
    tmp_path: Path,
) -> None:
    _project(tmp_path, expected=1)
    result = run_coding_task(
        CodingTask(objective="Confirm the existing behavior is already correct."),
        _ScriptedLLM([_final("No source change is necessary.")]),
        _registry(tmp_path),
    )

    assert result.status is CodingTaskStatus.COMPLETED
    assert result.workspace_revision == 0
    assert result.changed_files == ()
    assert result.final_review.git_status.clean


def test_read_only_registry_is_rejected_before_llm_call(tmp_path: Path) -> None:
    _project(tmp_path, expected=1)
    llm = _ScriptedLLM([_final()])
    registry = create_default_tool_registry(ToolContext(repository_root=tmp_path))

    result = run_coding_task(CodingTask(objective="Edit code."), llm, registry)

    assert result.status is CodingTaskStatus.PRECONDITION_FAILED
    assert "Editing-capable registry" in result.blockers[0]
    assert llm.calls == []


def test_narrow_agent_test_scope_does_not_prove_broader_policy_scope(
    tmp_path: Path,
) -> None:
    _project(tmp_path, expected=1)
    llm = _ScriptedLLM([_tool("run_tests", paths=["tests/test_app.py"]), _final()])

    result = run_coding_task(
        CodingTask(objective="Verify the repository."), llm, _registry(tmp_path)
    )

    assert result.status is CodingTaskStatus.COMPLETED
    assert tuple(result.verification.tests_result.paths) == (Path("tests"),)
    assert result.verification_executions == 3


def test_final_review_preserves_diff_truncation(tmp_path: Path) -> None:
    original = b"value = 'short'\n"
    _project(tmp_path, source=original, expected=2)
    replacement = "value = '" + "x" * 200 + "'\n"
    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="value = 'short'\n",
                new_text=replacement,
                expected_sha256=_hash(original),
            ),
            _final(),
        ]
    )
    policy = VerificationPolicy(require_tests=False, require_ruff=False)

    result = run_coding_task(
        CodingTask(objective="Expand the constant."),
        llm,
        _registry(tmp_path, ToolConfig(max_diff_chars=40)),
        verification_policy=policy,
        workflow_config=CodingWorkflowConfig(max_completion_attempts=1),
    )

    assert result.status is CodingTaskStatus.VERIFICATION_FAILED
    assert result.final_review.diff_truncated
    assert result.final_review.unstaged_diff.truncated
    assert result.review is None
    assert result.review_attempts == 1
    assert "too large or truncated" in result.blockers[0]


def test_planner_failure_stops_before_any_mutation(tmp_path: Path) -> None:
    before = b"def value():\n    return 1\n"
    _project(tmp_path, source=before, expected=1)
    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 1",
                new_text="return 2",
                expected_sha256=_hash(before),
            )
        ],
        plan_response=RuntimeError("sk-test-secret"),
    )

    result = run_coding_task(CodingTask(objective="Return two."), llm, _registry(tmp_path))

    assert result.status is CodingTaskStatus.ERROR
    assert result.successful_mutations == 0
    assert (tmp_path / "app.py").read_bytes() == before
    assert result.blockers == ("Required structured planning could not be completed.",)
    assert llm.calls == []


def test_cancellation_requested_during_planning_stops_before_executor(
    tmp_path: Path,
) -> None:
    before = b"def value():\n    return 1\n"
    _project(tmp_path, source=before, expected=1)

    class CancellationDuringPlan:
        requested = False
        side_effects = 0

        def checkpoint(self) -> None:
            if self.requested:
                raise JobCancellationRequested("cancelled after planner returned")

        def side_effect_started(self) -> None:
            self.side_effects += 1

    cancellation = CancellationDuringPlan()

    def plan_response(prompt: str) -> dict[str, object]:
        cancellation.requested = True
        return {
            "task_summary": "Inspect and implement the change.",
            "steps": [{"step_id": 1, "action": "Inspect and implement the change."}],
            "acceptance_coverage": [],
        }

    llm = _ScriptedLLM([_final()], plan_response=plan_response)

    with pytest.raises(JobCancellationRequested):
        run_coding_task(
            CodingTask(objective="Return two."),
            llm,
            _registry(tmp_path),
            cancellation=cancellation,
        )

    assert cancellation.side_effects == 0
    assert llm.calls == []
    assert (tmp_path / "app.py").read_bytes() == before


def test_review_block_triggers_fresh_edit_verification_and_review(
    tmp_path: Path,
) -> None:
    before = b"def value():\n    return 1\n"
    first = before.replace(b"return 1", b"return 2")
    _project(tmp_path, source=before, expected=2)

    def corrective_edit(prompt: str) -> AgentDecision:
        assert "Add a focused implementation note." in prompt
        return _tool(
            "replace_text",
            path="app.py",
            old_text="def value():\n",
            new_text="def value():\n    # focused implementation note\n",
            expected_sha256=_hash(first),
        )

    def review(verdict: str, revision: int) -> Callable[[str], dict[str, object]]:
        def response(prompt: str) -> dict[str, object]:
            payload = json.loads(prompt)
            assert payload["workspace_revision"] == revision
            satisfied = verdict == "approve"
            return {
                "verdict": verdict,
                "workspace_revision": revision,
                "acceptance_results": [
                    {
                        "criterion_index": 0,
                        "status": "satisfied" if satisfied else "not_satisfied",
                        "evidence": "The current bounded diff was independently reviewed.",
                    }
                ],
                "findings": [] if satisfied else ["The implementation note is missing."],
                "required_corrections": [] if satisfied else ["Add a focused implementation note."],
            }

        return response

    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 1",
                new_text="return 2",
                expected_sha256=_hash(before),
            ),
            _final("Initial implementation complete."),
            corrective_edit,
            _final("Corrective implementation complete."),
        ],
        review_responses=[review("changes_required", 1), review("approve", 2)],
    )

    result = run_coding_task(
        CodingTask(
            objective="Return two with a focused note.",
            acceptance_criteria=("The implementation is tested and documented.",),
        ),
        llm,
        _registry(tmp_path),
    )

    assert result.status is CodingTaskStatus.COMPLETED
    assert result.workspace_revision == 2
    assert result.verification.tests_revision == 2
    assert result.verification.ruff_revision == 2
    assert result.verification_executions == 4
    assert result.review is not None and result.review.workspace_revision == 2
    assert result.review.verdict.value == "approve"
    assert result.review_attempts == 2
    assert result.review_blocks == 1
    assert len(llm.review_calls) == 2


def test_reviewer_failure_cannot_false_complete(tmp_path: Path) -> None:
    before = b"def value():\n    return 1\n"
    _project(tmp_path, source=before, expected=2)
    llm = _ScriptedLLM(
        [
            _tool(
                "replace_text",
                path="app.py",
                old_text="return 1",
                new_text="return 2",
                expected_sha256=_hash(before),
            ),
            _final(),
        ],
        review_responses=[RuntimeError("Bearer private-review-secret")],
    )

    result = run_coding_task(
        CodingTask(objective="Return two."),
        llm,
        _registry(tmp_path),
        workflow_config=CodingWorkflowConfig(max_completion_attempts=1),
    )

    assert result.status is CodingTaskStatus.VERIFICATION_FAILED
    assert result.review is None
    assert result.review_attempts == 1
    assert result.blockers == ("Required independent coding review could not be completed.",)
    assert "private-review-secret" not in result.model_dump_json()


def test_reviewer_is_not_called_when_deterministic_verification_fails(
    tmp_path: Path,
) -> None:
    _project(tmp_path, expected=2)
    llm = _ScriptedLLM([_final("This is complete.")])

    result = run_coding_task(
        CodingTask(objective="Return two."),
        llm,
        _registry(tmp_path),
        workflow_config=CodingWorkflowConfig(max_completion_attempts=1),
    )

    assert result.status is CodingTaskStatus.VERIFICATION_FAILED
    assert result.verification.tests_passed is False
    assert llm.review_calls == []


def test_reviewer_cannot_override_failed_ruff(tmp_path: Path) -> None:
    _project(
        tmp_path,
        source=b"import os\n\n\ndef value():\n    return 1\n",
        expected=1,
    )
    llm = _ScriptedLLM([_final("This is complete.")])

    result = run_coding_task(
        CodingTask(objective="Keep value working."),
        llm,
        _registry(tmp_path),
        workflow_config=CodingWorkflowConfig(max_completion_attempts=1),
    )

    assert result.status is CodingTaskStatus.VERIFICATION_FAILED
    assert result.verification.tests_passed is True
    assert result.verification.ruff_passed is False
    assert llm.review_calls == []


def test_verification_start_error_is_distinct_and_blocks_completion(
    tmp_path: Path,
) -> None:
    _project(tmp_path, expected=1)
    original = _registry(tmp_path)
    registry = ToolRegistry()

    def fail_tests(arguments: BaseModel) -> BaseModel:
        raise ToolExecutionError("simulated pytest startup failure")

    for definition in original.list_tools():
        if definition.name == "run_tests":
            definition = ToolDefinition(
                name=definition.name,
                description=definition.description,
                input_model=definition.input_model,
                output_model=definition.output_model,
                handler=fail_tests,
            )
        registry.register(definition)

    result = run_coding_task(
        CodingTask(objective="Verify current code."),
        _ScriptedLLM([_final()]),
        registry,
        verification_policy=VerificationPolicy(require_ruff=False),
        workflow_config=CodingWorkflowConfig(max_completion_attempts=1),
    )

    assert result.status is CodingTaskStatus.VERIFICATION_FAILED
    assert result.verification.tests_result is None
    assert "startup failure" in result.verification.tests_execution_error
    assert any("could not execute" in blocker for blocker in result.blockers)
