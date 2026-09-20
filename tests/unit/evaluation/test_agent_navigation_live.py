"""Tests for live-model agent navigation evaluation, using fake LLMs only.

No test in this module contacts OpenAI. Live-mode decisions come from small
fake ``StructuredAgentLLM`` implementations that stand in for a real model,
proving the plumbing (decision source, iteration budget, prompt exposure,
grading, registry selection) is correct independent of any actual model
quality.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from repomind.agent import AgentRunStatus
from repomind.evaluation import (
    AgentNavigationBenchmarkCase,
    AgentNavigationBenchmarkSuite,
    EvaluationMode,
    evaluate_agent_navigation,
)
from repomind.ingestion import CodeChunk
from repomind.retrieval import SemanticSearchResult


def _tool(name: str, **arguments: object) -> dict:
    # The provider-facing wire format: a real model serializes tool arguments
    # into one JSON object string, because a strict response schema cannot
    # express an open dict. The loop decodes it back into the internal model.
    return {
        "action": "tool",
        "tool_name": name,
        "tool_arguments_json": json.dumps(arguments),
    }


def _scripted_tool(name: str, **arguments: object) -> dict:
    # Scripted benchmark decisions are internal AgentDecision values, not
    # provider output, so they keep the ordinary tool_arguments mapping.
    return {"action": "tool", "tool_name": name, "tool_arguments": arguments}


def _final(answer: str) -> dict:
    # Valid in both shapes: a final action carries no tool arguments.
    return {"action": "final", "final_answer": answer}


def _fixture(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def value():\n    return 1\n", encoding="utf-8"
    )
    return tmp_path


class _FakeLiveLLM:
    """Stands in for a real model: returns scripted responses but records every
    prompt it was shown, so tests can assert what was (and was not) exposed.
    """

    def __init__(self, responses: Sequence[dict]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []
        self.call_count = 0

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        self.call_count += 1
        if self.call_count > len(self._responses):
            # Keep returning the last response so max-iteration tests can
            # observe MAX_ITERATIONS rather than an unhelpful crash.
            return response_model.model_validate(self._responses[-1])
        return response_model.model_validate(self._responses[self.call_count - 1])


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


def _suite(case: AgentNavigationBenchmarkCase) -> AgentNavigationBenchmarkSuite:
    return AgentNavigationBenchmarkSuite(cases=(case,))


def test_live_mode_requires_an_llm_provider(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="literal_friendly",
        task="What does value() return?",
        decisions=(_final("scripted-answer"),),
        expected_facts=("returns 1",),
    )

    with pytest.raises(ValueError, match="llm_provider"):
        evaluate_agent_navigation(
            _suite(case), workspace, retrieval_mode="filesystem", mode=EvaluationMode.LIVE_MODEL
        )


def test_live_mode_uses_model_generated_decisions_not_scripted_decisions(
    tmp_path: Path,
) -> None:
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="literal_friendly",
        task="What does value() return?",
        # A single, deliberately different scripted decision - live mode must
        # never consult this.
        decisions=(_final("scripted-answer-should-not-be-used"),),
        expected_facts=("model chose this",),
    )
    llm = _FakeLiveLLM(
        [
            _tool("read_file", path="src/app.py"),
            _final("value() returns 1; model chose this answer."),
        ]
    )

    report = evaluate_agent_navigation(
        _suite(case),
        workspace,
        retrieval_mode="filesystem",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
    )

    result = report.case_results[0]
    assert result.final_answer == "value() returns 1; model chose this answer."
    assert "scripted-answer-should-not-be-used" not in result.final_answer
    assert result.task_success is True


def test_live_mode_prompts_never_expose_evaluator_oracle_fields(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="literal_friendly",
        task="What does value() return?",
        decisions=(_final("irrelevant"),),
        expected_facts=("super-secret-expected-fact-123",),
        forbidden_facts=("super-secret-forbidden-fact-456",),
    )
    llm = _FakeLiveLLM(
        [
            _tool("read_file", path="src/app.py"),
            _final("value() returns 1."),
        ]
    )

    evaluate_agent_navigation(
        _suite(case),
        workspace,
        retrieval_mode="filesystem",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
    )

    all_text = "\n".join([*llm.prompts, *(sp or "" for sp in llm.system_prompts)])
    assert "super-secret-expected-fact-123" not in all_text
    assert "super-secret-forbidden-fact-456" not in all_text
    assert "expected_facts" not in all_text
    assert "forbidden_facts" not in all_text
    # Only the natural-language task itself should appear.
    assert "What does value() return?" in llm.prompts[0]


def test_live_mode_grading_still_uses_expected_and_forbidden_facts(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="literal_friendly",
        task="What does value() return?",
        decisions=(_final("irrelevant"),),
        expected_facts=("returns 1",),
        forbidden_facts=("returns 999",),
    )
    bad_llm = _FakeLiveLLM([_final("value() returns 999, definitely.")])

    report = evaluate_agent_navigation(
        _suite(case),
        workspace,
        retrieval_mode="filesystem",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=bad_llm,
    )

    result = report.case_results[0]
    assert result.final_answer_grounded is False
    assert result.task_success is False


def test_live_max_iterations_independent_of_scripted_decision_count(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    # Scripted decisions has length 1, but the fake live model needs 5 tool
    # calls before finalizing - proving max_iterations is NOT derived from
    # len(case.decisions).
    case = AgentNavigationBenchmarkCase(
        id="needs-many-steps",
        category="literal_friendly",
        task="What does value() return?",
        decisions=(_final("scripted-single-step"),),
        expected_facts=("returns 1",),
    )
    llm = _FakeLiveLLM(
        [
            _tool("list_directory", path="."),
            _tool("list_directory", path="src"),
            _tool("search_code", query="value"),
            _tool("read_file", path="src/app.py"),
            _final("value() returns 1."),
        ]
    )

    report = evaluate_agent_navigation(
        _suite(case),
        workspace,
        retrieval_mode="filesystem",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
        live_max_iterations=8,
    )

    result = report.case_results[0]
    assert result.status is AgentRunStatus.COMPLETED
    assert result.task_success is True
    assert llm.call_count == 5


def test_live_max_iterations_bounds_the_run_when_too_small(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="needs-many-steps",
        category="literal_friendly",
        task="What does value() return?",
        decisions=(_final("scripted-single-step"),),
        expected_facts=("returns 1",),
    )
    llm = _FakeLiveLLM(
        [
            _tool("list_directory", path="."),
            _tool("list_directory", path="src"),
            _tool("search_code", query="value"),
            _tool("read_file", path="src/app.py"),
            _final("value() returns 1."),
        ]
    )

    report = evaluate_agent_navigation(
        _suite(case),
        workspace,
        retrieval_mode="filesystem",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
        live_max_iterations=2,
    )

    result = report.case_results[0]
    assert result.status is AgentRunStatus.MAX_ITERATIONS
    assert result.task_success is False
    assert llm.call_count == 2


def test_live_max_iterations_must_be_positive(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="literal_friendly",
        task="task",
        decisions=(_final("done"),),
        expected_facts=("done",),
    )
    llm = _FakeLiveLLM([_final("done")])

    with pytest.raises(ValueError, match="live_max_iterations"):
        evaluate_agent_navigation(
            _suite(case),
            workspace,
            retrieval_mode="filesystem",
            mode=EvaluationMode.LIVE_MODEL,
            llm_provider=llm,
            live_max_iterations=0,
        )


def test_live_filesystem_mode_uses_the_real_filesystem_registry(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="literal_friendly",
        task="What does value() return?",
        decisions=(_final("irrelevant"),),
        expected_facts=("returns 1",),
    )
    # The model tries the indexed-only tool first; a real (not stubbed)
    # filesystem-mode registry must reject it as unknown before the model
    # falls back to real filesystem tools.
    llm = _FakeLiveLLM(
        [
            _tool("indexed_code_search", query="value"),
            _tool("search_code", query="value"),
            _tool("read_file", path="src/app.py"),
            _final("value() returns 1."),
        ]
    )

    report = evaluate_agent_navigation(
        _suite(case),
        workspace,
        retrieval_mode="filesystem",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
    )

    result = report.case_results[0]
    # The rejected indexed_code_search decision is still counted (it asked),
    # but the prompt shown after it must carry the registry's real rejection.
    assert result.indexed_search_calls == 1
    assert "Unknown tool: indexed_code_search" in llm.prompts[1]
    assert result.filesystem_search_calls == 1
    assert result.read_file_calls == 1
    assert result.task_success is True


def test_live_indexed_mode_uses_the_real_indexed_registry(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    retriever = _FakeRetriever(_chunk())
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="exact_symbol",
        task="Where is value implemented?",
        decisions=(_final("irrelevant"),),
        expected_facts=("returns 1",),
    )
    llm = _FakeLiveLLM(
        [
            _tool("indexed_code_search", query="value"),
            _tool("read_file", path="src/app.py"),
            _final("value() returns 1, defined in src/app.py."),
        ]
    )

    report = evaluate_agent_navigation(
        _suite(case),
        workspace,
        retrieval_mode="indexed",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
        indexed_retriever=retriever,
    )

    result = report.case_results[0]
    assert retriever.calls == ["value"]
    assert result.indexed_search_calls == 1
    assert result.verified_retrieval_followup is True
    assert result.task_success is True


def test_live_indexed_followup_verification_measured_when_unverified(tmp_path: Path) -> None:
    workspace = _fixture(tmp_path)
    retriever = _FakeRetriever(_chunk())
    case = AgentNavigationBenchmarkCase(
        id="lazy",
        category="stale_index",
        task="What does value() return?",
        decisions=(_final("irrelevant"),),
        expected_facts=("returns 1",),
        forbidden_facts=("returns 999",),
    )
    # The model never reads the file before finalizing on every attempt;
    # the indexed grounding policy should keep blocking it until the
    # iteration budget is exhausted.
    llm = _FakeLiveLLM(
        [
            _tool("indexed_code_search", query="value"),
            _final("value() returns 999."),
        ]
    )

    report = evaluate_agent_navigation(
        _suite(case),
        workspace,
        retrieval_mode="indexed",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
        indexed_retriever=retriever,
        live_max_iterations=3,
    )

    result = report.case_results[0]
    assert result.verified_retrieval_followup is False
    assert result.status is AgentRunStatus.MAX_ITERATIONS
    assert result.task_success is False


def test_offline_scripted_mode_is_unaffected_by_live_parameters(tmp_path: Path) -> None:
    # Passing live-only defaults through in scripted mode must not change
    # scripted behavior: it still uses case.decisions and ignores
    # live_max_iterations entirely.
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="literal_friendly",
        task="What does value() return?",
        decisions=(
            _scripted_tool("read_file", path="src/app.py"),
            _final("value() returns 1."),
        ),
        expected_facts=("returns 1",),
    )

    report = evaluate_agent_navigation(_suite(case), workspace, retrieval_mode="filesystem")

    assert report.mode is EvaluationMode.OFFLINE_SCRIPTED
    assert report.case_results[0].status is AgentRunStatus.COMPLETED


def test_live_indexed_case_grades_correctly_with_an_arbitrary_query_wording(
    tmp_path: Path,
) -> None:
    # Uses the REAL canonical repo-agent-eval-v1 fixture repository and case
    # (not a local minimal fixture), grounding this in the actual dataset
    # both benchmark entry points share.
    from repomind.evaluation.agent_navigation_fixtures import (
        CaseScopedIndexedRetriever,
        benchmark_cases,
        write_fixture_repository,
    )

    write_fixture_repository(tmp_path)
    case = next(c for c in benchmark_cases(indexed=True) if c.id == "exact-symbol")
    retriever = CaseScopedIndexedRetriever.for_case(case.id)

    # Deliberately NOT the benchmark's canonical scripted query
    # ("JobStore.claim") - a real model may phrase this however it likes.
    arbitrary_query = "expired worker lease recovery mechanism"
    llm = _FakeLiveLLM(
        [
            _tool("indexed_code_search", query=arbitrary_query),
            _tool("read_file", path="src/jobs/store.py", start_line=5, end_line=7),
            _final(
                "JobStore.claim atomically claims one queued job for this worker "
                "using SKIP LOCKED."
            ),
        ]
    )

    report = evaluate_agent_navigation(
        _suite(case),
        tmp_path,
        retrieval_mode="indexed",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
        indexed_retriever=retriever,
    )

    result = report.case_results[0]
    assert result.task_success is True
    assert result.indexed_search_calls == 1
    assert result.verified_retrieval_followup is True
    # The actual query the model chose is still recorded for inspectability,
    # even though it was not used for lookup.
    assert retriever.calls == [arbitrary_query]


def test_live_index_miss_case_still_returns_zero_results_for_any_query(tmp_path: Path) -> None:
    from repomind.evaluation.agent_navigation_fixtures import (
        CaseScopedIndexedRetriever,
        benchmark_cases,
        write_fixture_repository,
    )

    write_fixture_repository(tmp_path)
    case = next(
        c for c in benchmark_cases(indexed=True) if c.id == "index-miss-filesystem-fallback"
    )
    retriever = CaseScopedIndexedRetriever.for_case(case.id)

    llm = _FakeLiveLLM(
        [
            _tool("indexed_code_search", query="anything the model wants to type here"),
            _tool("search_code", query="RETRY_BACKOFF_MULTIPLIER"),
            _tool("read_file", path="src/config/retry.py", start_line=3, end_line=3),
            _final("RETRY_BACKOFF_MULTIPLIER = 2.0 is defined in src/config/retry.py."),
        ]
    )

    report = evaluate_agent_navigation(
        _suite(case),
        tmp_path,
        retrieval_mode="indexed",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
        indexed_retriever=retriever,
    )

    result = report.case_results[0]
    assert result.indexed_search_calls == 1
    assert result.filesystem_search_calls == 1
    assert result.task_success is True
