"""Bounded, reproducible live-model run of the repo-agent-eval-v1 tasks.

The scripted ``benchmarks/agent_navigation_eval.py`` benchmark validates
agent orchestration, tool contracts, and safety plumbing with a fully
scripted fake LLM; it intentionally never proves that a real model chooses
tools well. This harness closes that gap: it runs the exact same canonical
tasks, fixture repository, expected/forbidden facts, and deterministic
indexed-retriever fixture chunks (from
``repomind.evaluation.agent_navigation_fixtures``, the single source of
truth also used by the offline benchmark) through the real read-only
investigation agent (``run_read_only_agent`` / ``run_indexed_read_only_agent``)
and real tool registries, but with a real ``OpenAILLMClient`` choosing every
action instead of a scripted decision queue.

In indexed mode, once the model calls ``indexed_code_search`` for a given
benchmark case, it receives that case's fixed fixture result set regardless
of the exact query text it chose - a real model may reasonably phrase
``indexed_code_search("expired worker lease recovery")`` instead of the
scripted benchmark's canonical query, and that must not be graded as a
retrieval miss. The intentional ``index-miss-filesystem-fallback`` case
still always returns zero indexed results. Each case is evaluated through
its own ``evaluate_agent_navigation`` call (so it gets its own case-scoped
retriever) and the per-case reports/traces are merged back into one
aggregate result for the "indexed" retrieval mode - the JSON result records
the actual query the model chose for every case, and the mode's aggregate
call/token/duration metrics are summed across every contributing case
trace, never taken from just one of them.

Grading stays fully deterministic (expected/forbidden substring checks, tool-
call counts, indexed-result verification) - no LLM is used as a judge.

This measures real-model tool-selection/navigation behavior under
*controlled* retrieval results; it is not a retrieval-ranking benchmark
(RepoMind's separate retrieval benchmarks cover that).

COSTS MONEY. Never run automatically. A live request requires both a
configured OPENAI_API_KEY and the explicit --confirm-live flag; this script
never runs in CI, and importing it (or calling --help) never makes a
network request.

Example:
    uv run python -m benchmarks.agent_live_eval --retrieval-mode both \\
        --confirm-live --output results/agent_live_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from repomind.config import get_settings
from repomind.evaluation import (
    DEFAULT_LIVE_MAX_ITERATIONS,
    AgentNavigationBenchmarkSuite,
    EvaluationMode,
    LiveEvaluationAuthorizationError,
    build_live_navigation_harness_result,
    evaluate_agent_navigation,
    git_commit_sha,
    merge_agent_navigation_reports,
    require_live_authorization,
)
from repomind.evaluation.agent_navigation_fixtures import (
    VERSION,
    CaseScopedIndexedRetriever,
    benchmark_cases,
    write_fixture_repository,
)
from repomind.observability import InMemoryTraceRecorder, RunTrace

if TYPE_CHECKING:
    from repomind.evaluation.agent_navigation_live import LiveAgentNavigationHarnessResult
    from repomind.evaluation.models import (
        AgentNavigationBenchmarkCase,
        AgentNavigationEvaluationReport,
    )
    from repomind.llm import OpenAILLMClient

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# One retrieval mode's aggregate result: the (possibly merged) report
# covering every selected case in that mode, every trace that contributed
# to it (one per case in indexed mode; one total in filesystem mode), and a
# case-id -> indexed_code_search-queries-issued mapping (empty for
# filesystem mode, which never has that tool available).
_ModeRun = tuple[
    "AgentNavigationEvaluationReport", list["RunTrace | None"], dict[str, list[str]]
]


def _select_cases(
    cases: tuple[AgentNavigationBenchmarkCase, ...],
    *,
    retrieval_mode: str,
    case_ids: list[str] | None,
    limit: int | None,
) -> list[AgentNavigationBenchmarkCase]:
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if case.id in wanted]
        missing = wanted - {case.id for case in selected}
        if missing:
            raise SystemExit(
                f"Unknown case id(s) for retrieval_mode={retrieval_mode!r}: "
                f"{', '.join(sorted(missing))}"
            )
    else:
        selected = list(cases)
    if limit is not None:
        selected = selected[:limit]
    return selected


def _run_filesystem_mode(
    *,
    llm: OpenAILLMClient,
    workspace: Path,
    case_ids: list[str] | None,
    limit: int | None,
    max_iterations: int,
) -> _ModeRun:
    cases = _select_cases(
        benchmark_cases(indexed=False), retrieval_mode="filesystem", case_ids=case_ids, limit=limit
    )
    if not cases:
        raise SystemExit("No cases selected for retrieval_mode='filesystem'")

    suite = AgentNavigationBenchmarkSuite(version=VERSION, cases=tuple(cases))
    recorder = InMemoryTraceRecorder()
    report = evaluate_agent_navigation(
        suite,
        workspace,
        retrieval_mode="filesystem",
        mode=EvaluationMode.LIVE_MODEL,
        llm_provider=llm,
        live_max_iterations=max_iterations,
        recorder=recorder,
    )
    run_id = report.case_results[0].trace_run_id if report.case_results else None
    run_trace = recorder.traces.get(run_id) if run_id is not None else None
    return report, [run_trace], {}


def _run_indexed_mode(
    *,
    llm: OpenAILLMClient,
    workspace: Path,
    case_ids: list[str] | None,
    limit: int | None,
    max_iterations: int,
) -> _ModeRun:
    cases = _select_cases(
        benchmark_cases(indexed=True), retrieval_mode="indexed", case_ids=case_ids, limit=limit
    )
    if not cases:
        raise SystemExit("No cases selected for retrieval_mode='indexed'")

    # Each case gets its own CaseScopedIndexedRetriever, bound to that case's
    # fixed fixture chunks regardless of the query text the model chooses -
    # so cases run one at a time (each its own evaluate_agent_navigation
    # call/trace) rather than batched into one shared suite, then the
    # per-case reports and traces are merged back into one aggregate result
    # for the "indexed" retrieval mode below.
    reports: list[AgentNavigationEvaluationReport] = []
    traces: list[RunTrace | None] = []
    queries_by_case: dict[str, list[str]] = {}
    for case in cases:
        retriever = CaseScopedIndexedRetriever.for_case(case.id)
        suite = AgentNavigationBenchmarkSuite(version=VERSION, cases=(case,))
        recorder = InMemoryTraceRecorder()
        report = evaluate_agent_navigation(
            suite,
            workspace,
            retrieval_mode="indexed",
            mode=EvaluationMode.LIVE_MODEL,
            llm_provider=llm,
            live_max_iterations=max_iterations,
            recorder=recorder,
            indexed_retriever=retriever,
        )
        run_id = report.case_results[0].trace_run_id
        run_trace = recorder.traces.get(run_id) if run_id is not None else None
        reports.append(report)
        traces.append(run_trace)
        # Recorded regardless of how many results were returned, so the
        # intentional index-miss case still persists its attempted query.
        queries_by_case[case.id] = list(retriever.calls)

    return merge_agent_navigation_reports(reports), traces, queries_by_case


def _print_console_summary(result: LiveAgentNavigationHarnessResult) -> None:
    print(f"{result.benchmark_version.upper()} LIVE-MODEL AGENT NAVIGATION ({result.model})")
    print(
        "(real OpenAI tool-selection under controlled, case-scoped retrieval; "
        "grading is deterministic, not LLM-judged; not a retrieval-ranking "
        "benchmark)\n"
    )
    for run in result.runs:
        report = run.report
        metrics = run.metrics
        print(f"--- retrieval_mode={report.retrieval_mode} ---")
        print(
            f"task_success_rate={report.task_success_rate:.3f} "
            f"mean_tool_calls={report.mean_tool_calls:.3f} "
            f"mean_indexed_search_calls={report.mean_indexed_search_calls:.3f} "
            f"mean_read_file_calls={report.mean_read_file_calls:.3f} "
            f"verified_retrieval_followup_rate={report.verified_retrieval_followup_rate:.3f}"
        )
        print(
            f"llm_calls={metrics.llm_calls} tool_calls={metrics.tool_calls} "
            f"duration_ms={metrics.duration_ms}"
        )
        if metrics.total_tokens is not None:
            print(
                f"tokens: prompt={metrics.prompt_tokens} "
                f"completion={metrics.completion_tokens} total={metrics.total_tokens}"
            )
        for case_result in report.case_results:
            print(
                f"  {case_result.case_id} [{case_result.category}]: "
                f"status={case_result.status}, task_success={case_result.task_success}, "
                f"final_answer_grounded={case_result.final_answer_grounded}, "
                f"tool_calls={case_result.tool_calls}, "
                f"verified_retrieval_followup={case_result.verified_retrieval_followup}"
            )
            queries = run.indexed_search_queries_by_case.get(case_result.case_id)
            if queries:
                print(f"    indexed_code_search queries issued: {list(queries)}")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.agent_live_eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("filesystem", "indexed", "both"),
        default="filesystem",
        help="Which tool-registry composition(s) to run (default: filesystem).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_LIVE_MAX_ITERATIONS,
        help=f"Live agent iteration budget per case (default: {DEFAULT_LIVE_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=None,
        metavar="CASE_ID",
        help="Restrict to this case id; may be passed more than once.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most this many cases per retrieval mode (after --case filtering).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the JSON harness result to this path.",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help=(
            "Required to actually call OpenAI. Without this flag (or without "
            "OPENAI_API_KEY configured) the harness refuses to run and makes "
            "no network request."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    settings = get_settings()
    try:
        require_live_authorization(settings, confirm_live=args.confirm_live)
    except LiveEvaluationAuthorizationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Only imported/constructed once authorization has passed.
    from repomind.llm import OpenAILLMClient

    llm = OpenAILLMClient(settings)
    sha = git_commit_sha(_REPOSITORY_ROOT)
    modes = ["filesystem", "indexed"] if args.retrieval_mode == "both" else [args.retrieval_mode]

    with TemporaryDirectory(prefix="repomind-agent-live-eval-") as temporary:
        workspace = Path(temporary)
        write_fixture_repository(workspace)

        mode_runs: list[_ModeRun] = [
            (_run_filesystem_mode if mode == "filesystem" else _run_indexed_mode)(
                llm=llm,
                workspace=workspace,
                case_ids=args.case_ids,
                limit=args.limit,
                max_iterations=args.max_iterations,
            )
            for mode in modes
        ]

    result = build_live_navigation_harness_result(
        benchmark_version=VERSION,
        model=llm.model,
        git_commit_sha=sha,
        generated_at=datetime.now(UTC),
        max_iterations=args.max_iterations,
        runs=mode_runs,
    )

    _print_console_summary(result)

    if args.output is not None:
        payload = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote JSON results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
