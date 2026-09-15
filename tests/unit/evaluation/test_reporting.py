"""Tests for deterministic dependency-free evaluation reports."""

from pathlib import Path

import pytest

from repomind.coding import CodingTaskStatus
from repomind.evaluation import (
    CodingBenchmarkCaseResult,
    CodingEvaluationReport,
    CodingOracleResult,
    EvaluationMode,
    OracleCheckResult,
    RAGCaseResult,
    RAGEvaluationReport,
    RetrievalCaseResult,
    RetrievalEvaluationReport,
    format_coding_report,
    format_rag_comparison,
    format_retrieval_comparison,
)

CHUNK_ID = ("src/example.py", 0, 1, 1)


def _retrieval_report(strategy: str, score: float) -> RetrievalEvaluationReport:
    case = RetrievalCaseResult(
        case_id="case",
        strategy=strategy,
        retrieved_chunk_ids=(CHUNK_ID,),
        relevant_chunk_ids=(CHUNK_ID,),
        recall_at_k=score,
        reciprocal_rank=score,
        ndcg_at_k=score,
        first_relevant_rank=1 if score else None,
    )
    return RetrievalEvaluationReport(
        benchmark_version="repo-eval-v1",
        mode=EvaluationMode.OFFLINE_FIXTURE,
        strategy=strategy,
        k=5,
        case_results=(case,),
        case_count=1,
        mean_recall_at_k=score,
        mrr=score,
        mean_ndcg_at_k=score,
    )


def test_retrieval_comparison_is_deterministic_and_has_no_winner_claim() -> None:
    reports = [_retrieval_report("semantic", 0.5), _retrieval_report("hybrid", 1.0)]

    first = format_retrieval_comparison(reports)
    second = format_retrieval_comparison(reports)

    assert first == second
    assert "Strategy" in first
    assert "Recall@5" in first
    assert "MRR" in first
    assert "nDCG@5" in first
    assert "first_relevant_rank" in first
    assert "winner" not in first.casefold()


def test_rag_comparison_keeps_pipeline_stages_separate() -> None:
    case = RAGCaseResult(
        case_id="case",
        strategy="hybrid",
        answer="Supported answer",
        insufficient_evidence=False,
        retrieved_chunk_ids=(CHUNK_ID,),
        context_chunk_ids=(CHUNK_ID,),
        cited_chunk_ids=(CHUNK_ID,),
        relevant_chunk_ids=(CHUNK_ID,),
        retrieval_recall=1.0,
        context_recall=0.5,
        citation_recall=0.5,
        required_facts_passed=False,
        missing_required_facts=("required fact",),
        answer_passed=False,
    )
    report = RAGEvaluationReport(
        benchmark_version="repo-eval-v1",
        mode=EvaluationMode.OFFLINE_FIXTURE,
        strategy="hybrid",
        case_results=(case,),
        case_count=1,
        mean_retrieval_recall=1.0,
        mean_context_recall=0.5,
        mean_citation_recall=0.5,
        answer_pass_rate=0.0,
    )

    rendered = format_rag_comparison([report])

    assert "Retrieval Recall" in rendered
    assert "Context Recall" in rendered
    assert "Citation Recall" in rendered
    assert "missing_facts=('required fact',)" in rendered


def test_coding_report_distinguishes_completion_and_true_success() -> None:
    result = CodingBenchmarkCaseResult(
        case_id="false-positive",
        workflow_status=CodingTaskStatus.COMPLETED,
        oracle=CodingOracleResult(
            passed=False,
            checks=(
                OracleCheckResult(
                    check="file_contains",
                    path=Path("app.py"),
                    passed=False,
                    message="hidden check failed",
                ),
            ),
        ),
        oracle_passed=False,
        task_success=False,
        false_positive_completion=True,
        final_verification_passed=True,
        recovery_observed=False,
        llm_calls=1,
        tool_calls=0,
        successful_mutations=0,
        agent_iterations=1,
        completion_attempts=1,
        changed_files=(Path("app.py"),),
        oracle_failures=("hidden check failed",),
        workflow_blockers=(),
    )
    report = CodingEvaluationReport(
        benchmark_version="repo-eval-v1",
        mode=EvaluationMode.OFFLINE_SCRIPTED,
        case_results=(result,),
        case_count=1,
        workflow_completion_rate=1.0,
        task_success_rate=0.0,
        false_positive_completion_rate=1.0,
        verification_pass_rate=1.0,
        recovery_rate=None,
        mean_llm_calls=1.0,
        mean_tool_calls=0.0,
        mean_successful_mutations=0.0,
        mean_agent_iterations=1.0,
        mean_completion_attempts=1.0,
    )

    rendered = format_coding_report(report)

    assert "Workflow completion: 1.000" in rendered
    assert "True task success: 0.000" in rendered
    assert "False-positive completion: 1.000" in rendered
    assert "Recovery rate: n/a" in rendered
    assert "hidden check failed" in rendered


def test_comparison_formatters_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        format_retrieval_comparison([])
    with pytest.raises(ValueError, match="at least one"):
        format_rag_comparison([])


def test_comparison_does_not_mix_fixture_and_live_reports() -> None:
    fixture = _retrieval_report("semantic", 1.0)
    live = fixture.model_copy(update={"mode": EvaluationMode.LIVE_MODEL})

    with pytest.raises(ValueError, match="same benchmark version and mode"):
        format_retrieval_comparison([fixture, live])
