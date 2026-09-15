"""Strategy-agnostic evaluation of ranked repository chunks."""

from collections.abc import Sequence
from typing import Protocol

from repomind.evaluation.metrics import (
    first_relevant_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from repomind.evaluation.models import (
    EvaluationMode,
    RetrievalBenchmarkSuite,
    RetrievalCaseResult,
    RetrievalEvaluationReport,
)
from repomind.retrieval import RankedChunk, chunk_identity


class RetrievalStrategy(Protocol):
    """One configured retrieval path evaluated without exposing its internals."""

    def __call__(self, query: str, *, top_k: int) -> Sequence[RankedChunk]:
        """Return ranked chunks for the exact benchmark query."""


def _strategy_name(strategy: str) -> str:
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must not be blank")
    return strategy


def evaluate_retrieval(
    suite: RetrievalBenchmarkSuite,
    strategy: str,
    retriever: RetrievalStrategy,
    *,
    k: int,
    mode: EvaluationMode = EvaluationMode.OFFLINE_FIXTURE,
) -> RetrievalEvaluationReport:
    """Evaluate one configured retriever sequentially over a versioned suite."""

    resolved_mode = EvaluationMode(mode)
    if resolved_mode is EvaluationMode.OFFLINE_SCRIPTED:
        raise ValueError("retrieval evaluation mode must be offline_fixture or live_model")
    name = _strategy_name(strategy)
    case_results: list[RetrievalCaseResult] = []
    for case in suite.cases:
        ranked = retriever(case.query, top_k=k)
        retrieved = tuple(chunk_identity(result.chunk) for result in ranked[:k])
        relevant = case.relevant_chunks
        case_results.append(
            RetrievalCaseResult(
                case_id=case.id,
                strategy=name,
                retrieved_chunk_ids=retrieved,
                relevant_chunk_ids=relevant,
                recall_at_k=recall_at_k(relevant, retrieved, k=k),
                reciprocal_rank=reciprocal_rank(relevant, retrieved),
                ndcg_at_k=ndcg_at_k(relevant, retrieved, k=k),
                first_relevant_rank=first_relevant_rank(relevant, retrieved),
            )
        )

    reciprocal_ranks = [result.reciprocal_rank for result in case_results]
    count = len(case_results)
    return RetrievalEvaluationReport(
        benchmark_version=suite.version,
        mode=resolved_mode,
        strategy=name,
        k=k,
        case_results=tuple(case_results),
        case_count=count,
        mean_recall_at_k=sum(result.recall_at_k for result in case_results) / count,
        mrr=mean_reciprocal_rank(reciprocal_ranks),
        mean_ndcg_at_k=sum(result.ndcg_at_k for result in case_results) / count,
    )
