"""Deterministic, dependency-free text reporting for evaluation results."""

from collections.abc import Sequence
from typing import Protocol

from repomind.evaluation.models import (
    CodingEvaluationReport,
    RAGEvaluationReport,
    RetrievalEvaluationReport,
)


class _ComparableReport(Protocol):
    benchmark_version: str
    mode: object


def _validate_comparable(reports: Sequence[_ComparableReport]) -> None:
    versions_and_modes = {
        (report.benchmark_version, report.mode) for report in reports
    }
    if len(versions_and_modes) != 1:
        raise ValueError("comparison reports must use the same benchmark version and mode")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    header = "  ".join(value.ljust(width) for value, width in zip(headers, widths, strict=True))
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def format_retrieval_comparison(
    reports: Sequence[RetrievalEvaluationReport],
) -> str:
    """Format aggregate ranking metrics and inspectable per-case ranks."""

    if not reports:
        raise ValueError("at least one retrieval report is required")
    _validate_comparable(reports)
    k_values = {report.k for report in reports}
    if len(k_values) != 1:
        raise ValueError("retrieval comparison reports must use the same k")
    k = reports[0].k
    rows = [
        (
            report.strategy,
            f"{report.mean_recall_at_k:.3f}",
            f"{report.mrr:.3f}",
            f"{report.mean_ndcg_at_k:.3f}",
        )
        for report in reports
    ]
    details = [
        (
            f"{report.strategy}/{result.case_id}: first_relevant_rank="
            f"{result.first_relevant_rank}, retrieved={result.retrieved_chunk_ids}"
        )
        for report in reports
        for result in report.case_results
    ]
    return (
        _table(
            ("Strategy", f"Recall@{k}", "MRR", f"nDCG@{k}"),
            rows,
        )
        + "\n\nCase details\n"
        + "\n".join(details)
    )


def format_rag_comparison(reports: Sequence[RAGEvaluationReport]) -> str:
    """Format separate retrieval, context, citation, and answer measurements."""

    if not reports:
        raise ValueError("at least one RAG report is required")
    _validate_comparable(reports)
    rows = [
        (
            report.strategy,
            f"{report.mean_retrieval_recall:.3f}",
            f"{report.mean_context_recall:.3f}",
            f"{report.mean_citation_recall:.3f}",
            f"{report.answer_pass_rate:.3f}",
        )
        for report in reports
    ]
    details = [
        (
            f"{report.strategy}/{result.case_id}: answer_passed="
            f"{result.answer_passed}, missing_facts={result.missing_required_facts}, "
            f"context={result.context_chunk_ids}, citations={result.cited_chunk_ids}"
        )
        for report in reports
        for result in report.case_results
    ]
    return (
        _table(
            (
                "Strategy",
                "Retrieval Recall",
                "Context Recall",
                "Citation Recall",
                "Answer Passed",
            ),
            rows,
        )
        + "\n\nCase details\n"
        + "\n".join(details)
    )


def format_coding_report(report: CodingEvaluationReport) -> str:
    """Format workflow, hidden correctness, recovery, and resource metrics."""

    recovery = "n/a" if report.recovery_rate is None else f"{report.recovery_rate:.3f}"
    summary = "\n".join(
        (
            f"Benchmark: {report.benchmark_version} ({report.mode})",
            f"Cases: {report.case_count}",
            f"Workflow completion: {report.workflow_completion_rate:.3f}",
            f"True task success: {report.task_success_rate:.3f}",
            f"False-positive completion: {report.false_positive_completion_rate:.3f}",
            f"Verification pass: {report.verification_pass_rate:.3f}",
            f"Recovery rate: {recovery}",
            f"Mean LLM calls: {report.mean_llm_calls:.3f}",
            f"Mean tool calls: {report.mean_tool_calls:.3f}",
            f"Mean successful mutations: {report.mean_successful_mutations:.3f}",
            f"Mean agent iterations: {report.mean_agent_iterations:.3f}",
            f"Mean completion attempts: {report.mean_completion_attempts:.3f}",
        )
    )
    details = [
        (
            f"{result.case_id}: status={result.workflow_status}, "
            f"oracle_passed={result.oracle_passed}, task_success={result.task_success}, "
            f"false_positive={result.false_positive_completion}, "
            "changed_files="
            f"{tuple(path.as_posix() for path in result.changed_files)}, "
            f"failures={result.oracle_failures}"
        )
        for result in report.case_results
    ]
    return summary + "\n\nCase details\n" + "\n".join(details)
