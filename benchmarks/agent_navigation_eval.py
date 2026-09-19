"""Run the deterministic offline ``repo-agent-eval-v1`` navigation benchmark.

Milestone 25 adds an optional indexed navigation tool to the read-only
investigation agent. This benchmark answers a narrow question: does indexed
navigation help the *orchestration* locate relevant code more efficiently,
or solve cases literal/declaration search handles poorly, without breaking
safety guarantees?

This is a fully scripted, deterministic comparison: the real handwritten
agent loop and real tool registries run against a synthetic fixture
repository with a fake, deterministic indexed retriever standing in for
persisted retrieval. No live model call proves a real LLM would choose
tools this well; this benchmark validates orchestration, tool contracts,
and safety plumbing only.

The fixture repository, task list, and deterministic indexed-retriever data
live in ``repomind.evaluation.agent_navigation_fixtures`` - the single
source of truth also used by the live-model harness in
``benchmarks/agent_live_eval.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from repomind.evaluation import (
    AgentNavigationBenchmarkSuite,
    evaluate_agent_navigation,
    format_agent_navigation_comparison,
)
from repomind.evaluation.agent_navigation_fixtures import (
    RETRIEVER_RESPONSES,
    VERSION,
    FixtureIndexedRetriever,
    benchmark_cases,
    stale_index_safety_suites,
    write_fixture_repository,
)


def main() -> None:
    """Print offline, deterministic agent-navigation evidence. No live model calls."""

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.agent_navigation_eval",
        description=__doc__,
    )
    parser.parse_args()

    with TemporaryDirectory(prefix="repomind-agent-nav-eval-") as temporary:
        workspace = Path(temporary)
        write_fixture_repository(workspace)

        filesystem_suite = AgentNavigationBenchmarkSuite(
            version=VERSION, cases=benchmark_cases(indexed=False)
        )
        indexed_suite = AgentNavigationBenchmarkSuite(version=VERSION, cases=benchmark_cases(indexed=True))
        retriever = FixtureIndexedRetriever(RETRIEVER_RESPONSES)

        filesystem_report = evaluate_agent_navigation(
            filesystem_suite, workspace, retrieval_mode="filesystem"
        )
        indexed_report = evaluate_agent_navigation(
            indexed_suite, workspace, retrieval_mode="indexed", indexed_retriever=retriever
        )

        print(f"{VERSION.upper()} AGENT NAVIGATION COMPARISON")
        print(
            "(scripted orchestration only - this validates tool composition and "
            "safety, not live-model tool-selection quality)\n"
        )
        print(format_agent_navigation_comparison([filesystem_report, indexed_report]))

        verified_suite, lazy_suite = stale_index_safety_suites()
        verified_report = evaluate_agent_navigation(
            verified_suite, workspace, retrieval_mode="indexed", indexed_retriever=retriever
        )
        lazy_report = evaluate_agent_navigation(
            lazy_suite, workspace, retrieval_mode="indexed", indexed_retriever=retriever
        )
        print("\nSTALE-INDEX SAFETY DEMONSTRATION (excluded from the comparison above)")
        print(
            f"  verified (reads current file before answering): "
            f"task_success={verified_report.case_results[0].task_success}"
        )
        print(
            f"  lazy (answers from the indexed location without reading it): "
            f"status={lazy_report.case_results[0].status}, "
            f"task_success={lazy_report.case_results[0].task_success} "
            "(runtime finalization blocked - current source was not read)"
        )


if __name__ == "__main__":
    main()
