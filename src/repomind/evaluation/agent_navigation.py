"""Deterministic scripted evaluation of read-only investigation navigation.

This measures agent orchestration, tool-selection wiring, and safety
plumbing against a synthetic fixture repository, driving the real
handwritten agent loop and real tool registries with a fully scripted fake
LLM. It answers whether indexed navigation composes correctly and safely
with the existing agent - it does not, and cannot, prove that a live model
will choose tools well.
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from repomind.agent import (
    AgentConfig,
    AgentDecision,
    AgentRunStatus,
    AgentStep,
    StructuredAgentLLM,
    run_indexed_read_only_agent,
    run_read_only_agent,
)
from repomind.evaluation.models import (
    AgentNavigationBenchmarkCase,
    AgentNavigationBenchmarkSuite,
    AgentNavigationCaseResult,
    AgentNavigationEvaluationReport,
    EvaluationMode,
)
from repomind.observability import TraceContext, TraceRecorder
from repomind.observability.instrumentation import traced_run
from repomind.tools import (
    IndexedRetriever,
    ToolContext,
    create_default_tool_registry,
    create_investigation_tool_registry,
)

_FILESYSTEM_SEARCH_TOOLS = frozenset({"search_code", "find_symbol", "list_directory"})


class _ScriptedAgentLLM:
    """Return pre-built decisions in order; never sees case grading criteria."""

    def __init__(self, decisions: Sequence[AgentDecision]) -> None:
        self._decisions = list(decisions)

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        if not self._decisions:
            raise AssertionError("scripted agent navigation case ran out of decisions")
        return self._decisions.pop(0)


def _tool_call_counts(steps: Sequence[AgentStep]) -> Counter[str]:
    return Counter(
        step.decision.tool_name for step in steps if step.decision.action == "tool"
    )


def _verified_retrieval_followup(steps: Sequence[AgentStep]) -> bool:
    """Was the latest relevant indexed result set verified with ``read_file``?

    Vacuously true when no indexed search occurred in this run: there is
    nothing indexed evidence claimed that needs verification.
    """

    pending: set[str] = set()
    for step in steps:
        observation = step.observation
        if observation is None or not observation.success or observation.output is None:
            continue
        if observation.tool_name == "indexed_code_search":
            surfaced = {
                location["relative_path"]
                for location in observation.output.get("locations", [])
            }
            if surfaced:
                pending = surfaced
        elif observation.tool_name == "read_file":
            path = observation.output.get("path")
            if path in pending:
                pending.clear()
    return not pending


def _final_answer_grounded(
    final_answer: str | None,
    expected_facts: Sequence[str],
    forbidden_facts: Sequence[str],
) -> bool:
    if final_answer is None:
        return False
    lowered = final_answer.casefold()
    has_expected = all(fact.casefold() in lowered for fact in expected_facts)
    has_forbidden = any(fact.casefold() in lowered for fact in forbidden_facts)
    return has_expected and not has_forbidden


def _run_case(
    case: AgentNavigationBenchmarkCase,
    workspace: Path,
    *,
    retrieval_mode: str,
    indexed_retriever: IndexedRetriever | None,
    trace: TraceContext,
    mode: EvaluationMode,
    llm_provider: StructuredAgentLLM | None,
    live_max_iterations: int,
) -> AgentNavigationCaseResult:
    if mode is EvaluationMode.LIVE_MODEL:
        # Live mode never sees case.decisions - a real model chooses its own
        # actions, and the run is bounded by an explicit live iteration
        # budget instead of the scripted decision count.
        if llm_provider is None:
            raise ValueError("live_model mode requires an llm_provider")
        llm = llm_provider
        max_iterations = live_max_iterations
    else:
        llm = _ScriptedAgentLLM(case.decisions)
        max_iterations = len(case.decisions)
    context = ToolContext(repository_root=workspace)
    if retrieval_mode == "indexed":
        if indexed_retriever is None:
            raise ValueError("retrieval_mode='indexed' requires an indexed_retriever")
        registry = create_investigation_tool_registry(context, indexed_retriever)
        agent_runner = run_indexed_read_only_agent
    else:
        registry = create_default_tool_registry(context)
        agent_runner = run_read_only_agent

    run = agent_runner(
        case.task,
        llm,
        registry,
        config=AgentConfig(max_iterations=max_iterations),
        trace=trace,
    )
    counts = _tool_call_counts(run.steps)
    grounded = _final_answer_grounded(run.final_answer, case.expected_facts, case.forbidden_facts)
    return AgentNavigationCaseResult(
        case_id=case.id,
        category=case.category,
        retrieval_mode=retrieval_mode,
        trace_run_id=trace.run_id,
        status=run.status,
        task_success=run.status is AgentRunStatus.COMPLETED and grounded,
        final_answer_grounded=grounded,
        tool_calls=run.tool_calls,
        indexed_search_calls=counts.get("indexed_code_search", 0),
        read_file_calls=counts.get("read_file", 0),
        filesystem_search_calls=sum(counts.get(name, 0) for name in _FILESYSTEM_SEARCH_TOOLS),
        verified_retrieval_followup=_verified_retrieval_followup(run.steps),
        final_answer=run.final_answer or "",
    )


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values)


def _aggregate_report(
    *,
    benchmark_version: str,
    mode: EvaluationMode,
    retrieval_mode: str,
    case_results: tuple[AgentNavigationCaseResult, ...],
) -> AgentNavigationEvaluationReport:
    count = len(case_results)
    return AgentNavigationEvaluationReport(
        benchmark_version=benchmark_version,
        mode=mode,
        retrieval_mode=retrieval_mode,
        case_results=case_results,
        case_count=count,
        task_success_rate=sum(result.task_success for result in case_results) / count,
        mean_tool_calls=_mean([result.tool_calls for result in case_results]),
        mean_indexed_search_calls=_mean(
            [result.indexed_search_calls for result in case_results]
        ),
        mean_read_file_calls=_mean([result.read_file_calls for result in case_results]),
        mean_filesystem_search_calls=_mean(
            [result.filesystem_search_calls for result in case_results]
        ),
        verified_retrieval_followup_rate=sum(
            result.verified_retrieval_followup for result in case_results
        )
        / count,
    )


def merge_agent_navigation_reports(
    reports: Sequence[AgentNavigationEvaluationReport],
) -> AgentNavigationEvaluationReport:
    """Combine several same-mode, same-retrieval-mode reports into one.

    Used when cases were graded through separate ``evaluate_agent_navigation``
    calls - for example, live indexed mode evaluates one case per call so
    each case can receive its own case-scoped deterministic retriever - but
    should still be reported as a single aggregate result for that retrieval
    mode, using the exact same aggregation formulas as a single-call
    evaluation over all of those cases together.
    """

    if not reports:
        raise ValueError("at least one report is required")
    first = reports[0]
    if any(
        report.mode != first.mode
        or report.retrieval_mode != first.retrieval_mode
        or report.benchmark_version != first.benchmark_version
        for report in reports
    ):
        raise ValueError(
            "all reports must share the same benchmark_version, mode, and retrieval_mode"
        )

    case_results = tuple(result for report in reports for result in report.case_results)
    if len({result.case_id for result in case_results}) != len(case_results):
        raise ValueError("merged reports must not contain duplicate case IDs")
    return _aggregate_report(
        benchmark_version=first.benchmark_version,
        mode=first.mode,
        retrieval_mode=first.retrieval_mode,
        case_results=case_results,
    )


# Live runs are bounded independently of any scripted decision count; see
# repomind.evaluation.agent_navigation_live for the confirmation-gated harness
# that supplies a real StructuredAgentLLM here.
DEFAULT_LIVE_MAX_ITERATIONS = 8


@traced_run("evaluation")
def evaluate_agent_navigation(
    suite: AgentNavigationBenchmarkSuite,
    workspace: Path,
    *,
    retrieval_mode: str,
    indexed_retriever: IndexedRetriever | None = None,
    mode: EvaluationMode = EvaluationMode.OFFLINE_SCRIPTED,
    llm_provider: StructuredAgentLLM | None = None,
    live_max_iterations: int = DEFAULT_LIVE_MAX_ITERATIONS,
    recorder: TraceRecorder | None = None,
    trace: TraceContext | None = None,
) -> AgentNavigationEvaluationReport:
    """Run every case through one real agent-loop/registry composition.

    ``workspace`` is a directory of real files the fixture cases read from;
    ``indexed_retriever`` (required when ``retrieval_mode == "indexed"``) is
    a deterministic fake standing in for persisted retrieval.

    In ``mode=EvaluationMode.LIVE_MODEL``, ``llm_provider`` (typically a real
    ``OpenAILLMClient``) makes every decision instead of the case's scripted
    ``decisions``, and each run is bounded by ``live_max_iterations`` rather
    than ``len(case.decisions)``.
    """

    trace = trace if trace is not None else TraceContext()
    resolved_mode = EvaluationMode(mode)
    if resolved_mode is EvaluationMode.OFFLINE_FIXTURE:
        raise ValueError(
            "agent navigation evaluation mode must be offline_scripted or live_model"
        )
    if retrieval_mode not in {"filesystem", "indexed"}:
        raise ValueError("retrieval_mode must be 'filesystem' or 'indexed'")
    if resolved_mode is EvaluationMode.LIVE_MODEL and llm_provider is None:
        raise ValueError("mode='live_model' requires an llm_provider")
    if live_max_iterations <= 0:
        raise ValueError("live_max_iterations must be positive")

    results: list[AgentNavigationCaseResult] = []
    for case in suite.cases:
        with trace.operation(
            "evaluation.case",
            case_id=case.id,
            suite_version=suite.version,
            mode=resolved_mode,
        ) as metadata:
            result = _run_case(
                case,
                workspace,
                retrieval_mode=retrieval_mode,
                indexed_retriever=indexed_retriever,
                trace=trace,
                mode=resolved_mode,
                llm_provider=llm_provider,
                live_max_iterations=live_max_iterations,
            )
            results.append(result)
            metadata.update(case_id=case.id, task_success=result.task_success)

    return _aggregate_report(
        benchmark_version=suite.version,
        mode=resolved_mode,
        retrieval_mode=retrieval_mode,
        case_results=tuple(results),
    )
