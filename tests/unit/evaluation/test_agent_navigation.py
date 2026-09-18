"""Tests for the deterministic scripted read-only navigation evaluation."""

from pathlib import Path

import pytest

from repomind.agent import AgentRunStatus
from repomind.evaluation import (
    AgentNavigationBenchmarkCase,
    AgentNavigationBenchmarkSuite,
    evaluate_agent_navigation,
    format_agent_navigation_comparison,
)
from repomind.ingestion import CodeChunk
from repomind.retrieval import SemanticSearchResult


def _tool(name: str, **arguments: object) -> dict:
    return {"action": "tool", "tool_name": name, "tool_arguments": arguments}


def _final(answer: str) -> dict:
    return {"action": "final", "final_answer": answer}


def _fixture(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def value():\n    return 1\n", encoding="utf-8"
    )
    return tmp_path


class _FakeRetriever:
    def __init__(self, chunk: CodeChunk | None) -> None:
        self.chunk = chunk
        self.calls: list[str] = []

    def __call__(self, query: str, *, top_k: int):
        self.calls.append(query)
        if self.chunk is None:
            return []
        return [SemanticSearchResult(chunk=self.chunk, score=0.9, rank=1)]


def _chunk() -> CodeChunk:
    return CodeChunk(
        relative_path="src/app.py",
        language="python",
        start_line=1,
        end_line=2,
        content="",
        chunk_index=0,
    )


def test_filesystem_mode_runs_the_real_agent_loop_and_grounds_the_answer(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    suite = AgentNavigationBenchmarkSuite(
        cases=(
            AgentNavigationBenchmarkCase(
                id="basic",
                category="literal_friendly",
                task="What does value() return?",
                decisions=(
                    _tool("search_code", query="def value"),
                    _tool("read_file", path="src/app.py"),
                    _final("value() returns 1."),
                ),
                expected_facts=("returns 1",),
            ),
        )
    )

    report = evaluate_agent_navigation(suite, workspace, retrieval_mode="filesystem")

    assert report.case_count == 1
    result = report.case_results[0]
    assert result.status is AgentRunStatus.COMPLETED
    assert result.task_success is True
    assert result.final_answer_grounded is True
    assert result.read_file_calls == 1
    assert result.filesystem_search_calls == 1
    assert result.indexed_search_calls == 0
    assert result.verified_retrieval_followup is True


def test_indexed_mode_requires_a_retriever(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    suite = AgentNavigationBenchmarkSuite(
        cases=(
            AgentNavigationBenchmarkCase(
                id="basic",
                category="exact_symbol",
                task="Where is value?",
                decisions=(_final("done"),),
                expected_facts=("done",),
            ),
        )
    )

    with pytest.raises(ValueError, match="indexed_retriever"):
        evaluate_agent_navigation(suite, workspace, retrieval_mode="indexed")


def test_indexed_mode_records_verified_followup_when_read_file_confirms_location(
    tmp_path: Path,
) -> None:
    workspace = _fixture(tmp_path)
    retriever = _FakeRetriever(_chunk())
    suite = AgentNavigationBenchmarkSuite(
        cases=(
            AgentNavigationBenchmarkCase(
                id="verified",
                category="exact_symbol",
                task="Where is value implemented?",
                decisions=(
                    _tool("indexed_code_search", query="value"),
                    _tool("read_file", path="src/app.py"),
                    _final("value() returns 1, defined in src/app.py."),
                ),
                expected_facts=("returns 1",),
            ),
        )
    )

    report = evaluate_agent_navigation(
        suite, workspace, retrieval_mode="indexed", indexed_retriever=retriever
    )

    result = report.case_results[0]
    assert result.indexed_search_calls == 1
    assert result.verified_retrieval_followup is True
    assert result.task_success is True
    assert retriever.calls == ["value"]


def test_unverified_stale_answer_fails_grounding_and_followup(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    retriever = _FakeRetriever(_chunk())
    suite = AgentNavigationBenchmarkSuite(
        cases=(
            AgentNavigationBenchmarkCase(
                id="lazy",
                category="stale_index",
                task="What does value() return?",
                decisions=(
                    _tool("indexed_code_search", query="value"),
                    # Deliberately never reads the file before answering.
                    _final("value() returns 999."),
                ),
                expected_facts=("returns 1",),
                forbidden_facts=("returns 999",),
            ),
        )
    )

    report = evaluate_agent_navigation(
        suite, workspace, retrieval_mode="indexed", indexed_retriever=retriever
    )

    result = report.case_results[0]
    assert result.status is AgentRunStatus.MAX_ITERATIONS
    assert result.final_answer == ""
    assert result.verified_retrieval_followup is False
    assert result.final_answer_grounded is False
    assert result.task_success is False


def test_no_indexed_search_makes_verified_followup_vacuously_true(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    retriever = _FakeRetriever(_chunk())
    suite = AgentNavigationBenchmarkSuite(
        cases=(
            AgentNavigationBenchmarkCase(
                id="never-searched",
                category="literal_friendly",
                task="What does value() return?",
                decisions=(
                    _tool("read_file", path="src/app.py"),
                    _final("value() returns 1."),
                ),
                expected_facts=("returns 1",),
            ),
        )
    )

    report = evaluate_agent_navigation(
        suite, workspace, retrieval_mode="indexed", indexed_retriever=retriever
    )

    assert report.case_results[0].verified_retrieval_followup is True


def test_iteration_budget_exactly_matches_the_scripted_decision_count(tmp_path: Path) -> None:
    # The harness sizes AgentConfig.max_iterations to len(case.decisions), so
    # a fully scripted case always completes rather than hitting the
    # max-iterations limit - this is a property of the harness, not a claim
    # that a real agent always finishes in exactly this many steps.
    workspace = _fixture(tmp_path)
    suite = AgentNavigationBenchmarkSuite(
        cases=(
            AgentNavigationBenchmarkCase(
                id="incomplete",
                category="literal_friendly",
                task="What does value() return?",
                decisions=(
                    _tool("read_file", path="src/app.py"),
                    _final("value() returns 1."),
                ),
                expected_facts=("returns 1",),
            ),
        )
    )

    report = evaluate_agent_navigation(suite, workspace, retrieval_mode="filesystem")

    assert report.case_results[0].status is AgentRunStatus.COMPLETED


def test_rejects_invalid_retrieval_mode(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    suite = AgentNavigationBenchmarkSuite(
        cases=(
            AgentNavigationBenchmarkCase(
                id="basic",
                category="literal_friendly",
                task="task",
                decisions=(_final("done"),),
                expected_facts=("done",),
            ),
        )
    )

    with pytest.raises(ValueError, match="retrieval_mode"):
        evaluate_agent_navigation(suite, workspace, retrieval_mode="graph")


def test_case_must_end_with_a_final_decision() -> None:
    with pytest.raises(ValueError, match="final answer"):
        AgentNavigationBenchmarkCase(
            id="bad",
            category="literal_friendly",
            task="task",
            decisions=(_tool("read_file", path="src/app.py"),),
            expected_facts=("fact",),
        )


def test_format_agent_navigation_comparison_requires_reports() -> None:
    with pytest.raises(ValueError, match="at least one"):
        format_agent_navigation_comparison([])
