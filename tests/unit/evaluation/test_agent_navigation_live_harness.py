"""Tests for live-run authorization gating and the JSON harness result schema.

No test in this module contacts OpenAI; the harness-composition test uses a
fake LLM exactly like ``test_agent_navigation_live.py``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from repomind.agent import AgentRunStatus
from repomind.config import Settings
from repomind.evaluation import (
    EvaluationMode,
    LiveEvaluationAuthorizationError,
    build_live_navigation_harness_result,
    evaluate_agent_navigation,
    merge_agent_navigation_reports,
    require_live_authorization,
)
from repomind.evaluation.agent_navigation_fixtures import (
    VERSION,
    CaseScopedIndexedRetriever,
    benchmark_cases,
    write_fixture_repository,
)
from repomind.evaluation.models import (
    AgentNavigationBenchmarkCase,
    AgentNavigationBenchmarkSuite,
    AgentNavigationCaseResult,
    AgentNavigationEvaluationReport,
)
from repomind.llm.models import TokenUsage
from repomind.observability import InMemoryTraceRecorder, RunTrace
from repomind.observability.models import RunStatus, RunType


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"openai_api_key": "sk-live-test"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_require_live_authorization_passes_with_key_and_confirmation() -> None:
    require_live_authorization(_settings(), confirm_live=True)


def test_require_live_authorization_fails_without_api_key() -> None:
    with pytest.raises(LiveEvaluationAuthorizationError, match="OPENAI_API_KEY"):
        require_live_authorization(_settings(openai_api_key=None), confirm_live=True)


def test_require_live_authorization_fails_without_confirm_flag() -> None:
    with pytest.raises(LiveEvaluationAuthorizationError, match="confirm-live"):
        require_live_authorization(_settings(), confirm_live=False)


def test_require_live_authorization_fails_with_neither() -> None:
    with pytest.raises(LiveEvaluationAuthorizationError) as excinfo:
        require_live_authorization(_settings(openai_api_key=None), confirm_live=False)
    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert "confirm-live" in message


def test_require_live_authorization_treats_blank_key_as_missing() -> None:
    with pytest.raises(LiveEvaluationAuthorizationError, match="OPENAI_API_KEY"):
        require_live_authorization(_settings(openai_api_key=""), confirm_live=True)


@pytest.mark.parametrize("blank_key", ["   ", "\t", "\n", " \t\n "])
def test_require_live_authorization_treats_whitespace_only_key_as_missing(
    blank_key: str,
) -> None:
    # A whitespace-only key can never authenticate; accepting it would push
    # the failure past the safety gate instead of stopping at it.
    with pytest.raises(LiveEvaluationAuthorizationError, match="OPENAI_API_KEY"):
        require_live_authorization(_settings(openai_api_key=blank_key), confirm_live=True)


def _run_trace(
    *,
    llm_calls: int,
    tool_calls: int,
    usage: TokenUsage | None,
    usage_reported_calls: int | None = None,
    duration_ms: float = 42.5,
) -> RunTrace:
    started = datetime.now(UTC)
    if usage_reported_calls is None:
        # Default to fully-covered usage when a usage total is supplied, which
        # is what a real OpenAILLMClient run produces.
        usage_reported_calls = llm_calls if usage is not None else 0
    return RunTrace(
        run_id=uuid4(),
        run_type=RunType.EVALUATION,
        status=RunStatus.COMPLETED,
        started_at=started,
        ended_at=started,
        duration_ms=duration_ms,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        token_usage=usage,
        usage_reported_calls=usage_reported_calls,
    )


class _FinalOnlyLLM:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        return response_model.model_validate(
            {"action": "final", "final_answer": self._answer}
        )


def _fixture(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return tmp_path


def _live_report(tmp_path: Path):
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="literal_friendly",
        task="What is VALUE?",
        decisions=({"action": "final", "final_answer": "irrelevant"},),
        expected_facts=("1",),
    )
    suite = AgentNavigationBenchmarkSuite(version=VERSION, cases=(case,))
    recorder = InMemoryTraceRecorder()
    report = evaluate_agent_navigation(
        suite,
        workspace,
        retrieval_mode="filesystem",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=_FinalOnlyLLM("VALUE is 1."),
        recorder=recorder,
    )
    run_id = report.case_results[0].trace_run_id
    return report, recorder.traces[run_id]


def test_build_live_navigation_harness_result_rejects_empty_runs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_live_navigation_harness_result(
            benchmark_version="repo-agent-eval-v1",
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[],
        )


def test_build_live_navigation_harness_result_rejects_non_live_mode_reports(
    tmp_path: Path,
) -> None:
    workspace = _fixture(tmp_path)
    case = AgentNavigationBenchmarkCase(
        id="basic",
        category="literal_friendly",
        task="task",
        decisions=({"action": "final", "final_answer": "done"},),
        expected_facts=("done",),
    )
    scripted_report = evaluate_agent_navigation(
        AgentNavigationBenchmarkSuite(cases=(case,)), workspace, retrieval_mode="filesystem"
    )

    with pytest.raises(ValueError, match="live_model"):
        build_live_navigation_harness_result(
            benchmark_version=scripted_report.benchmark_version,
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[(scripted_report, [_run_trace(llm_calls=1, tool_calls=0, usage=None)], {})],
        )


def test_build_live_navigation_harness_result_maps_trace_metrics() -> None:
    # Build a minimal live-mode report by hand to isolate the trace-metric
    # mapping logic from a full agent run.
    run_trace = _run_trace(
        llm_calls=3,
        tool_calls=2,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140),
    )

    case_result = AgentNavigationCaseResult(
        case_id="basic",
        category="literal_friendly",
        retrieval_mode="filesystem",
        trace_run_id=run_trace.run_id,
        status=AgentRunStatus.COMPLETED,
        task_success=True,
        final_answer_grounded=True,
        tool_calls=2,
        indexed_search_calls=0,
        read_file_calls=1,
        filesystem_search_calls=1,
        verified_retrieval_followup=True,
        final_answer="VALUE is 1.",
    )
    live_report = AgentNavigationEvaluationReport(
        benchmark_version="repo-agent-eval-v1",
        mode=EvaluationMode.LIVE_MODEL,
        retrieval_mode="filesystem",
        case_results=(case_result,),
        case_count=1,
        task_success_rate=1.0,
        mean_tool_calls=2.0,
        mean_indexed_search_calls=0.0,
        mean_read_file_calls=1.0,
        mean_filesystem_search_calls=1.0,
        verified_retrieval_followup_rate=1.0,
    )

    result = build_live_navigation_harness_result(
        benchmark_version="repo-agent-eval-v1",
        model="fake-model",
        git_commit_sha="abc123",
        generated_at=datetime.now(UTC),
        max_iterations=8,
        runs=[(live_report, [run_trace], {})],
    )

    assert result.case_ids == ("basic",)
    metrics = result.runs[0].metrics
    assert metrics.llm_calls == 3
    assert metrics.tool_calls == 2
    assert metrics.prompt_tokens == 100
    assert metrics.completion_tokens == 40
    assert metrics.total_tokens == 140
    assert metrics.duration_ms == 42.5


def test_build_live_navigation_harness_result_rejects_a_missing_trace(tmp_path: Path) -> None:
    # A dropped trace would silently under-report calls, duration and tokens
    # in a committed artifact, so it must fail loudly instead.
    report, run_trace = _live_report(tmp_path)

    with pytest.raises(ValueError, match="missing"):
        build_live_navigation_harness_result(
            benchmark_version=VERSION,
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[(report, [run_trace, None], {})],
        )


def test_build_live_navigation_harness_result_rejects_an_empty_trace_list(
    tmp_path: Path,
) -> None:
    report, _ = _live_report(tmp_path)

    with pytest.raises(ValueError, match="supplied no traces"):
        build_live_navigation_harness_result(
            benchmark_version=VERSION,
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[(report, [], {})],
        )


def test_token_totals_require_complete_usage_coverage_across_every_trace() -> None:
    # Two traces, 3 + 2 LLM calls. The second reports usage for only one of
    # its two calls, so publishing 140 + 30 would read as the run's complete
    # token cost when it is not.
    complete = _run_trace(
        llm_calls=3,
        tool_calls=2,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140),
    )
    partial = _run_trace(
        llm_calls=2,
        tool_calls=1,
        usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        usage_reported_calls=1,
    )
    merged = merge_agent_navigation_reports(
        [
            _bare_live_report("case-a", trace_run_id=complete.run_id),
            _bare_live_report("case-b", trace_run_id=partial.run_id),
        ]
    )

    result = build_live_navigation_harness_result(
        benchmark_version="repo-agent-eval-v1",
        model="fake-model",
        git_commit_sha=None,
        generated_at=datetime.now(UTC),
        max_iterations=8,
        runs=[(merged, [complete, partial], {})],
    )

    metrics = result.runs[0].metrics
    # Call/tool/duration counters still aggregate across BOTH traces...
    assert metrics.llm_calls == 5
    assert metrics.tool_calls == 3
    assert metrics.duration_ms == complete.duration_ms + partial.duration_ms
    # ...but incomplete token coverage publishes nothing rather than a
    # partial sum presented as a total.
    assert metrics.prompt_tokens is None
    assert metrics.completion_tokens is None
    assert metrics.total_tokens is None


def test_token_totals_are_summed_when_every_trace_has_complete_coverage() -> None:
    first = _run_trace(
        llm_calls=3,
        tool_calls=2,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140),
    )
    second = _run_trace(
        llm_calls=2,
        tool_calls=1,
        usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )
    merged = merge_agent_navigation_reports(
        [
            _bare_live_report("case-a", trace_run_id=first.run_id),
            _bare_live_report("case-b", trace_run_id=second.run_id),
        ]
    )

    result = build_live_navigation_harness_result(
        benchmark_version="repo-agent-eval-v1",
        model="fake-model",
        git_commit_sha=None,
        generated_at=datetime.now(UTC),
        max_iterations=8,
        runs=[(merged, [first, second], {})],
    )

    metrics = result.runs[0].metrics
    assert metrics.prompt_tokens == 120
    assert metrics.completion_tokens == 50
    assert metrics.total_tokens == 170


def test_token_totals_are_none_when_no_llm_calls_are_represented() -> None:
    # Documented semantic: zero LLM calls means nothing was measured, which
    # is reported as None rather than a zero that would read as a measurement.
    empty = _run_trace(llm_calls=0, tool_calls=0, usage=None)

    result = build_live_navigation_harness_result(
        benchmark_version="repo-agent-eval-v1",
        model="fake-model",
        git_commit_sha=None,
        generated_at=datetime.now(UTC),
        max_iterations=8,
        runs=[(_bare_live_report("case-a", trace_run_id=empty.run_id), [empty], {})],
    )

    metrics = result.runs[0].metrics
    assert metrics.llm_calls == 0
    assert metrics.prompt_tokens is None
    assert metrics.completion_tokens is None
    assert metrics.total_tokens is None


def test_build_live_navigation_harness_result_rejects_mismatched_benchmark_version(
    tmp_path: Path,
) -> None:
    report, run_trace = _live_report(tmp_path)

    with pytest.raises(ValueError, match="benchmark_version"):
        build_live_navigation_harness_result(
            benchmark_version="repo-agent-eval-v2",
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[(report, [run_trace], {})],
        )


def test_build_live_navigation_harness_result_rejects_unknown_query_map_cases() -> None:
    trace = _run_trace(llm_calls=1, tool_calls=1, usage=None)
    report = _bare_live_report("case-a", trace_run_id=trace.run_id)

    with pytest.raises(ValueError, match="wrong-case"):
        build_live_navigation_harness_result(
            benchmark_version="repo-agent-eval-v1",
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[
                (
                    report,
                    [trace],
                    {"case-a": ["a query"], "wrong-case": ["orphaned query"]},
                )
            ],
        )


def test_build_live_navigation_harness_result_rejects_duplicate_retrieval_modes() -> None:
    first_trace = _run_trace(llm_calls=1, tool_calls=1, usage=None)
    second_trace = _run_trace(llm_calls=1, tool_calls=1, usage=None)
    first = _bare_live_report("case-a", trace_run_id=first_trace.run_id)
    second = _bare_live_report("case-b", trace_run_id=second_trace.run_id)
    assert first.retrieval_mode == second.retrieval_mode == "indexed"

    with pytest.raises(ValueError, match="duplicate run for retrieval_mode"):
        build_live_navigation_harness_result(
            benchmark_version="repo-agent-eval-v1",
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[
                (first, [first_trace], {}),
                (second, [second_trace], {}),
            ],
        )


def test_build_live_navigation_harness_result_rejects_an_omitted_expected_trace() -> None:
    # Two cases evaluated separately expect traces A and B; supplying only A
    # has no None to catch, so only run_id provenance detects it.
    trace_a = _run_trace(llm_calls=3, tool_calls=2, usage=None)
    trace_b = _run_trace(llm_calls=4, tool_calls=3, usage=None)
    merged = merge_agent_navigation_reports(
        [
            _bare_live_report("case-a", trace_run_id=trace_a.run_id),
            _bare_live_report("case-b", trace_run_id=trace_b.run_id),
        ]
    )

    with pytest.raises(ValueError, match="trace set does not match its report"):
        build_live_navigation_harness_result(
            benchmark_version="repo-agent-eval-v1",
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[(merged, [trace_a], {})],
        )


def test_build_live_navigation_harness_result_rejects_an_unrelated_trace() -> None:
    trace_a = _run_trace(llm_calls=1, tool_calls=1, usage=None)
    unrelated = _run_trace(llm_calls=9, tool_calls=9, usage=None)
    report = _bare_live_report("case-a", trace_run_id=trace_a.run_id)

    with pytest.raises(ValueError, match="unexpected"):
        build_live_navigation_harness_result(
            benchmark_version="repo-agent-eval-v1",
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[(report, [trace_a, unrelated], {})],
        )


def test_build_live_navigation_harness_result_rejects_a_duplicated_trace() -> None:
    # Set comparison alone would accept [A, A]; summing it would double-count.
    trace_a = _run_trace(llm_calls=3, tool_calls=2, usage=None)
    report = _bare_live_report("case-a", trace_run_id=trace_a.run_id)

    with pytest.raises(ValueError, match="same trace more than once"):
        build_live_navigation_harness_result(
            benchmark_version="repo-agent-eval-v1",
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[(report, [trace_a, trace_a], {})],
        )


def test_build_live_navigation_harness_result_rejects_a_case_without_trace_provenance() -> None:
    trace = _run_trace(llm_calls=1, tool_calls=1, usage=None)
    report = _bare_live_report("case-a", trace_run_id=trace.run_id)
    orphan_case = report.case_results[0].model_copy(update={"trace_run_id": None})
    orphan_report = report.model_copy(update={"case_results": (orphan_case,)})

    with pytest.raises(ValueError, match="no trace_run_id"):
        build_live_navigation_harness_result(
            benchmark_version="repo-agent-eval-v1",
            model="fake-model",
            git_commit_sha=None,
            generated_at=datetime.now(UTC),
            max_iterations=8,
            runs=[(orphan_report, [trace], {})],
        )


def test_per_case_indexed_traces_are_accepted_when_every_expected_trace_is_supplied() -> None:
    trace_a = _run_trace(llm_calls=3, tool_calls=2, usage=None)
    trace_b = _run_trace(llm_calls=4, tool_calls=3, usage=None)
    merged = merge_agent_navigation_reports(
        [
            _bare_live_report("case-a", trace_run_id=trace_a.run_id),
            _bare_live_report("case-b", trace_run_id=trace_b.run_id),
        ]
    )

    result = build_live_navigation_harness_result(
        benchmark_version="repo-agent-eval-v1",
        model="fake-model",
        git_commit_sha=None,
        generated_at=datetime.now(UTC),
        max_iterations=8,
        runs=[(merged, [trace_a, trace_b], {})],
    )

    assert result.runs[0].metrics.llm_calls == 7
    assert result.runs[0].metrics.tool_calls == 5


def test_filesystem_style_shared_trace_is_accepted_from_one_supplied_trace(
    tmp_path: Path,
) -> None:
    # Filesystem mode grades several cases in ONE evaluate_agent_navigation
    # call, so its case results legitimately share a single trace_run_id and
    # exactly one trace is supplied for all of them.
    workspace = _fixture(tmp_path)
    cases = tuple(
        AgentNavigationBenchmarkCase(
            id=case_id,
            category="literal_friendly",
            task="What is VALUE?",
            decisions=({"action": "final", "final_answer": "irrelevant"},),
            expected_facts=("1",),
        )
        for case_id in ("case-a", "case-b", "case-c")
    )
    recorder = InMemoryTraceRecorder()
    report = evaluate_agent_navigation(
        AgentNavigationBenchmarkSuite(version=VERSION, cases=cases),
        workspace,
        retrieval_mode="filesystem",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=_FinalOnlyLLM("VALUE is 1."),
        recorder=recorder,
    )
    shared_ids = {result.trace_run_id for result in report.case_results}
    assert len(report.case_results) == 3
    assert len(shared_ids) == 1
    run_trace = recorder.traces[report.case_results[0].trace_run_id]

    result = build_live_navigation_harness_result(
        benchmark_version=VERSION,
        model="fake-model",
        git_commit_sha=None,
        generated_at=datetime.now(UTC),
        max_iterations=8,
        runs=[(report, [run_trace], {})],
    )

    assert result.runs[0].report.case_count == 3
    assert result.runs[0].metrics.llm_calls == 3


def test_live_harness_result_from_real_run_composes_end_to_end(tmp_path: Path) -> None:
    report, run_trace = _live_report(tmp_path)
    result = build_live_navigation_harness_result(
        benchmark_version=VERSION,
        model="fake-model",
        git_commit_sha=None,
        generated_at=datetime.now(UTC),
        max_iterations=8,
        runs=[(report, [run_trace], {})],
    )
    assert result.runs[0].report.case_results[0].final_answer == "VALUE is 1."
    # A fake LLM never emits model.usage, so llm_calls is still real...
    assert result.runs[0].metrics.llm_calls == 1
    # ...but token usage is legitimately absent.
    assert result.runs[0].metrics.total_tokens is None
    # A filesystem-mode case never calls indexed_code_search.
    assert result.runs[0].indexed_search_queries_by_case == {"basic": ()}


def test_json_serialization_is_deterministic_and_contains_no_secrets(tmp_path: Path) -> None:
    report, run_trace = _live_report(tmp_path)
    generated_at = datetime(2026, 1, 1, tzinfo=UTC)
    result = build_live_navigation_harness_result(
        benchmark_version="repo-agent-eval-v1",
        model="fake-model",
        git_commit_sha="deadbeef",
        generated_at=generated_at,
        max_iterations=8,
        runs=[(report, [run_trace], {})],
    )

    first = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
    second = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
    assert first == second

    forbidden_substrings = ("sk-live-test", "OPENAI_API_KEY", "api_key", "openai_api_key")
    for forbidden in forbidden_substrings:
        assert forbidden not in first

    payload = json.loads(first)
    assert payload["harness_schema_version"] == "agent-live-eval-v1"
    assert payload["benchmark_version"] == "repo-agent-eval-v1"
    assert payload["model"] == "fake-model"
    assert payload["git_commit_sha"] == "deadbeef"
    assert payload["max_iterations"] == 8
    assert payload["case_ids"] == ["basic"]
    assert len(payload["runs"]) == 1


def _bare_live_report(
    case_id: str, *, trace_run_id: UUID | None = None
) -> AgentNavigationEvaluationReport:
    case_result = AgentNavigationCaseResult(
        case_id=case_id,
        category="exact_symbol",
        retrieval_mode="indexed",
        trace_run_id=trace_run_id if trace_run_id is not None else uuid4(),
        status=AgentRunStatus.COMPLETED,
        task_success=True,
        final_answer_grounded=True,
        tool_calls=1,
        indexed_search_calls=1,
        read_file_calls=0,
        filesystem_search_calls=0,
        verified_retrieval_followup=True,
        final_answer="done",
    )
    return AgentNavigationEvaluationReport(
        benchmark_version="repo-agent-eval-v1",
        mode=EvaluationMode.LIVE_MODEL,
        retrieval_mode="indexed",
        case_results=(case_result,),
        case_count=1,
        task_success_rate=1.0,
        mean_tool_calls=1.0,
        mean_indexed_search_calls=1.0,
        mean_read_file_calls=0.0,
        mean_filesystem_search_calls=0.0,
        verified_retrieval_followup_rate=1.0,
    )


def test_indexed_search_queries_are_persisted_per_case_with_order_preserved() -> None:
    trace_a = _run_trace(llm_calls=2, tool_calls=2, usage=None)
    trace_b = _run_trace(llm_calls=1, tool_calls=0, usage=None)
    merged = merge_agent_navigation_reports(
        [
            _bare_live_report("searched-twice", trace_run_id=trace_a.run_id),
            _bare_live_report("never-searched", trace_run_id=trace_b.run_id),
        ]
    )

    result = build_live_navigation_harness_result(
        benchmark_version="repo-agent-eval-v1",
        model="fake-model",
        git_commit_sha=None,
        generated_at=datetime.now(UTC),
        max_iterations=8,
        runs=[
            (
                merged,
                [trace_a, trace_b],
                {"searched-twice": ["first query", "second, different query"]},
            )
        ],
    )

    queries = result.runs[0].indexed_search_queries_by_case
    # Order preserved, exactly as issued.
    assert queries["searched-twice"] == ("first query", "second, different query")
    # A case that never called indexed_code_search still gets an entry -
    # an empty tuple, not a missing key.
    assert queries["never-searched"] == ()


def test_indexed_search_queries_persist_for_index_miss_case_despite_zero_results() -> None:
    # The retriever always records the attempted query, even when it
    # returns zero results (the intentional index-miss case).
    retriever = CaseScopedIndexedRetriever.for_case("index-miss-filesystem-fallback")
    assert retriever(query="an attempted query with no fixture hits", top_k=5) == []

    trace = _run_trace(llm_calls=1, tool_calls=1, usage=None)
    report = _bare_live_report("index-miss-filesystem-fallback", trace_run_id=trace.run_id)
    result = build_live_navigation_harness_result(
        benchmark_version="repo-agent-eval-v1",
        model="fake-model",
        git_commit_sha=None,
        generated_at=datetime.now(UTC),
        max_iterations=8,
        runs=[
            (
                report,
                [trace],
                {"index-miss-filesystem-fallback": list(retriever.calls)},
            )
        ],
    )

    assert result.runs[0].indexed_search_queries_by_case == {
        "index-miss-filesystem-fallback": ("an attempted query with no fixture hits",)
    }


def _tool(name: str, **arguments: object) -> dict:
    return {"action": "tool", "tool_name": name, "tool_arguments": arguments}


def _final(answer: str) -> dict:
    return {"action": "final", "final_answer": answer}


def test_multi_case_indexed_metrics_aggregate_the_sum_of_every_case_trace(
    tmp_path: Path,
) -> None:
    # A deterministic regression test with two real indexed benchmark
    # cases, each evaluated (and traced) separately - exactly how live
    # indexed mode runs - proving the combined metrics equal the sum of
    # BOTH case traces, not just one of them.
    write_fixture_repository(tmp_path)
    cases = {case.id: case for case in benchmark_cases(indexed=True)}
    case_a = cases["exact-symbol"]
    case_b = cases["literal-error-string"]

    class _FakeLLM:
        def __init__(self, responses: list[dict]) -> None:
            self._responses = list(responses)
            self.call_count = 0

        def generate_structured(self, prompt, response_model, *, system_prompt=None, temperature=None):
            response = self._responses[self.call_count]
            self.call_count += 1
            return response_model.model_validate(response)

    llm_a = _FakeLLM(
        [
            _tool("indexed_code_search", query="claim job atomically"),
            _tool("read_file", path="src/jobs/store.py", start_line=5, end_line=7),
            _final(
                "JobStore.claim atomically claims one queued job for this worker "
                "using SKIP LOCKED."
            ),
        ]
    )
    llm_b = _FakeLLM(
        [
            _tool("indexed_code_search", query="rate limit error string"),
            _tool("read_file", path="src/errors.py", start_line=3, end_line=3),
            _tool("read_file", path="src/errors.py", start_line=3, end_line=3),
            _final("The rate-limit error code is ERR_RATE_LIMIT_EXCEEDED_42."),
        ]
    )

    reports = []
    traces = []
    queries_by_case: dict[str, list[str]] = {}
    for case, llm in ((case_a, llm_a), (case_b, llm_b)):
        retriever = CaseScopedIndexedRetriever.for_case(case.id)
        recorder = InMemoryTraceRecorder()
        report = evaluate_agent_navigation(
            AgentNavigationBenchmarkSuite(version=VERSION, cases=(case,)),
            tmp_path,
            retrieval_mode="indexed",
            mode=EvaluationMode.LIVE_MODEL,
            llm_provider=llm,
            indexed_retriever=retriever,
            live_max_iterations=6,
            recorder=recorder,
        )
        run_id = report.case_results[0].trace_run_id
        trace = recorder.traces[run_id]
        reports.append(report)
        traces.append(trace)
        queries_by_case[case.id] = list(retriever.calls)

    # Sanity: the two per-case traces really do differ (3 vs. 4 LLM calls),
    # so a bug that used only one trace would be caught by the assertions
    # below rather than passing by coincidence.
    assert {t.llm_calls for t in traces} == {3, 4}

    merged_report = merge_agent_navigation_reports(reports)
    result = build_live_navigation_harness_result(
        benchmark_version="repo-agent-eval-v1",
        model="fake-model",
        git_commit_sha=None,
        generated_at=datetime.now(UTC),
        max_iterations=6,
        runs=[(merged_report, traces, queries_by_case)],
    )

    metrics = result.runs[0].metrics
    assert metrics.llm_calls == sum(t.llm_calls for t in traces)
    assert metrics.tool_calls == sum(t.tool_calls for t in traces)
    assert metrics.duration_ms == sum(t.duration_ms for t in traces)
    # Both cases succeeded, so the merged report itself must also cover both.
    assert merged_report.case_count == 2
    assert {r.case_id for r in merged_report.case_results} == {case_a.id, case_b.id}
    assert all(r.task_success for r in merged_report.case_results)
    assert result.runs[0].indexed_search_queries_by_case == {
        case_a.id: ("claim job atomically",),
        case_b.id: ("rate limit error string",),
    }


def test_merge_agent_navigation_reports_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one report"):
        merge_agent_navigation_reports([])


def test_merge_agent_navigation_reports_rejects_mismatched_benchmark_versions() -> None:
    other = _bare_live_report("case-b").model_copy(update={"benchmark_version": "repo-eval-v1"})

    with pytest.raises(ValueError, match="benchmark_version"):
        merge_agent_navigation_reports([_bare_live_report("case-a"), other])


def test_merge_agent_navigation_reports_rejects_mismatched_modes() -> None:
    scripted = _bare_live_report("case-b").model_copy(
        update={"mode": EvaluationMode.OFFLINE_SCRIPTED}
    )

    with pytest.raises(ValueError, match="mode"):
        merge_agent_navigation_reports([_bare_live_report("case-a"), scripted])


def test_merge_agent_navigation_reports_rejects_mismatched_retrieval_modes() -> None:
    filesystem_case = _bare_live_report("case-b").case_results[0].model_copy(
        update={"retrieval_mode": "filesystem"}
    )
    filesystem_report = _bare_live_report("case-b").model_copy(
        update={"retrieval_mode": "filesystem", "case_results": (filesystem_case,)}
    )

    with pytest.raises(ValueError, match="retrieval_mode"):
        merge_agent_navigation_reports([_bare_live_report("case-a"), filesystem_report])


def test_merge_agent_navigation_reports_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValueError, match="duplicate case IDs"):
        merge_agent_navigation_reports(
            [_bare_live_report("same-case"), _bare_live_report("same-case")]
        )


def test_merge_agent_navigation_reports_recomputes_aggregates_over_all_cases() -> None:
    # The merged report must average over BOTH cases, not carry the first
    # report's per-case figures forward.
    failing_case = _bare_live_report("case-b").case_results[0].model_copy(
        update={"task_success": False, "final_answer_grounded": False, "tool_calls": 3}
    )
    failing_report = _bare_live_report("case-b").model_copy(
        update={
            "case_results": (failing_case,),
            "task_success_rate": 0.0,
            "mean_tool_calls": 3.0,
        }
    )

    merged = merge_agent_navigation_reports([_bare_live_report("case-a"), failing_report])

    assert merged.case_count == 2
    assert merged.task_success_rate == 0.5
    assert merged.mean_tool_calls == 2.0
