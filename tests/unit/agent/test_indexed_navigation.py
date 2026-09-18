"""Deterministic runtime-policy tests for opt-in indexed investigation."""

from pathlib import Path

from pydantic import BaseModel

from repomind.agent import (
    AgentConfig,
    AgentDecision,
    AgentRunStatus,
    run_indexed_read_only_agent,
    run_read_only_agent,
)
from repomind.ingestion import CodeChunk
from repomind.retrieval import SemanticSearchResult
from repomind.tools import (
    ToolContext,
    create_default_tool_registry,
    create_investigation_tool_registry,
)


class _ScriptedLLM:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = list(decisions)
        self.calls: list[str] = []

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        self.calls.append(prompt)
        return self.decisions.pop(0)


class _Retriever:
    def __init__(self, results_by_query: dict[str, list[SemanticSearchResult]]) -> None:
        self.results_by_query = results_by_query

    def __call__(self, query: str, *, top_k: int):
        return self.results_by_query.get(query, [])[:top_k]


class _FailingRetriever:
    def __call__(self, query: str, *, top_k: int):
        raise RuntimeError("index unavailable")


def _tool(name: str, **arguments: object) -> AgentDecision:
    return AgentDecision(action="tool", tool_name=name, tool_arguments=arguments)


def _final(answer: str = "done") -> AgentDecision:
    return AgentDecision(action="final", final_answer=answer)


def _result(path: str) -> SemanticSearchResult:
    return SemanticSearchResult(
        chunk=CodeChunk(
            relative_path=path,
            language="python",
            start_line=1,
            end_line=2,
            content="",
            chunk_index=0,
        ),
        score=0.9,
        rank=1,
    )


def _indexed_run(
    root: Path,
    decisions: list[AgentDecision],
    retriever,
):
    llm = _ScriptedLLM(decisions)
    registry = create_investigation_tool_registry(
        ToolContext(repository_root=root), retriever
    )
    run = run_indexed_read_only_agent(
        "Inspect the repository",
        llm,
        registry,
        config=AgentConfig(max_iterations=len(decisions)),
    )
    return run, llm


def _write(root: Path, relative_path: str) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def current():\n    return True\n", encoding="utf-8")


def test_nonempty_indexed_result_blocks_immediate_final(tmp_path: Path) -> None:
    retriever = _Retriever({"jobs": [_result("src/jobs/store.py")]})

    run, _ = _indexed_run(
        tmp_path,
        [_tool("indexed_code_search", query="jobs"), _final("stale claim")],
        retriever,
    )

    assert run.status is AgentRunStatus.MAX_ITERATIONS
    assert run.final_answer is None
    feedback = run.steps[-1].workflow_feedback
    assert feedback is not None
    assert "Read the current source" in feedback.message


def test_matching_successful_read_allows_final(tmp_path: Path) -> None:
    _write(tmp_path, "src/jobs/store.py")
    retriever = _Retriever({"jobs": [_result("src/jobs/store.py")]})

    run, _ = _indexed_run(
        tmp_path,
        [
            _tool("indexed_code_search", query="jobs"),
            _tool("read_file", path="src/jobs/store.py"),
            _final("grounded claim"),
        ],
        retriever,
    )

    assert run.status is AgentRunStatus.COMPLETED
    assert run.final_answer == "grounded claim"


def test_unrelated_read_stays_blocked_until_returned_path_is_read(tmp_path: Path) -> None:
    _write(tmp_path, "README.md")
    _write(tmp_path, "src/jobs/store.py")
    retriever = _Retriever({"jobs": [_result("src/jobs/store.py")]})

    run, llm = _indexed_run(
        tmp_path,
        [
            _tool("indexed_code_search", query="jobs"),
            _tool("read_file", path="README.md"),
            _final("unsupported"),
            _tool("read_file", path="src/jobs/store.py"),
            _final("supported"),
        ],
        retriever,
    )

    assert run.status is AgentRunStatus.COMPLETED
    assert run.final_answer == "supported"
    assert run.steps[2].workflow_feedback is not None
    assert 'trust="trusted-workflow-instruction"' in llm.calls[3]


def test_failed_matching_read_does_not_satisfy_grounding(tmp_path: Path) -> None:
    retriever = _Retriever({"jobs": [_result("src/jobs/store.py")]})

    run, _ = _indexed_run(
        tmp_path,
        [
            _tool("indexed_code_search", query="jobs"),
            _tool("read_file", path="src/jobs/store.py"),
            _final("unsupported"),
        ],
        retriever,
    )

    assert run.steps[1].observation is not None
    assert run.steps[1].observation.success is False
    assert run.status is AgentRunStatus.MAX_ITERATIONS
    assert run.steps[2].workflow_feedback is not None


def test_empty_indexed_result_does_not_activate_gate(tmp_path: Path) -> None:
    run, _ = _indexed_run(
        tmp_path,
        [_tool("indexed_code_search", query="missing"), _final("insufficient evidence")],
        _Retriever({}),
    )

    assert run.status is AgentRunStatus.COMPLETED
    assert run.final_answer == "insufficient evidence"


def test_indexed_tool_failure_does_not_activate_gate(tmp_path: Path) -> None:
    run, _ = _indexed_run(
        tmp_path,
        [_tool("indexed_code_search", query="jobs"), _final("used fallback evidence")],
        _FailingRetriever(),
    )

    assert run.steps[0].observation is not None
    assert run.steps[0].observation.success is False
    assert run.status is AgentRunStatus.COMPLETED


def test_second_nonempty_search_requires_new_matching_read(tmp_path: Path) -> None:
    _write(tmp_path, "src/a.py")
    _write(tmp_path, "src/b.py")
    retriever = _Retriever(
        {"first": [_result("src/a.py")], "second": [_result("src/b.py")]}
    )

    run, _ = _indexed_run(
        tmp_path,
        [
            _tool("indexed_code_search", query="first"),
            _tool("read_file", path="src/a.py"),
            _tool("indexed_code_search", query="second"),
            _final("only first verified"),
            _tool("read_file", path="src/b.py"),
            _final("both searches handled"),
        ],
        retriever,
    )

    assert run.steps[3].workflow_feedback is not None
    assert run.status is AgentRunStatus.COMPLETED
    assert run.final_answer == "both searches handled"


def test_filesystem_only_agent_behavior_is_unchanged(tmp_path: Path) -> None:
    llm = _ScriptedLLM([_final("immediate filesystem answer")])
    registry = create_default_tool_registry(ToolContext(repository_root=tmp_path))

    run = run_read_only_agent("Answer directly", llm, registry)

    assert run.status is AgentRunStatus.COMPLETED
    assert run.final_answer == "immediate filesystem answer"
    assert run.tool_calls == 0


def test_indexed_grounding_policy_is_deterministic(tmp_path: Path) -> None:
    def execute():
        return _indexed_run(
            tmp_path,
            [_tool("indexed_code_search", query="jobs"), _final("stale claim")],
            _Retriever({"jobs": [_result("src/jobs/store.py")]}),
        )[0]

    first = execute()
    second = execute()

    assert first == second
