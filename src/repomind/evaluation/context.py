"""Deterministic evaluation of context assembly, separate from retrieval ranking.

This module answers a narrower question than retrieval evaluation: given an
already-ranked seed list, did context assembly deliver the gold evidence
inside the final packed, budget-bounded context? It never recomputes or
reports retrieval-ranking metrics (recall/MRR/nDCG); those stay in
:mod:`repomind.evaluation.retrieval`.
"""

from collections.abc import Sequence
from typing import Protocol

from repomind.evaluation.models import (
    ContextAssemblyBenchmarkSuite,
    ContextAssemblyCaseResult,
    ContextAssemblyEvaluationReport,
    EvaluationMode,
)
from repomind.rag import ContextAssemblyConfig, ContextStrategy, NeighborLoader, assemble_context
from repomind.retrieval import ChunkIdentity, RankedChunk, chunk_identity


class ContextRetrievalStrategy(Protocol):
    """One configured seed retriever evaluated without exposing its internals."""

    def __call__(self, query: str, *, top_k: int) -> Sequence[RankedChunk]:
        """Return ranked seed chunks for the exact benchmark query."""


def evaluate_context_assembly(
    suite: ContextAssemblyBenchmarkSuite,
    retriever: ContextRetrievalStrategy,
    *,
    top_k: int,
    config: ContextAssemblyConfig,
    neighbor_loader: NeighborLoader | None = None,
    mode: EvaluationMode = EvaluationMode.OFFLINE_FIXTURE,
) -> ContextAssemblyEvaluationReport:
    """Evaluate one configured context-assembly strategy over a versioned suite."""

    resolved_mode = EvaluationMode(mode)
    if resolved_mode is EvaluationMode.OFFLINE_SCRIPTED:
        raise ValueError("context assembly evaluation mode must be offline_fixture or live_model")
    case_results: list[ContextAssemblyCaseResult] = []
    for case in suite.cases:
        seeds = retriever(case.query, top_k=top_k)
        result = assemble_context(seeds, neighbor_loader, config)
        packed_ids: tuple[ChunkIdentity, ...] = tuple(
            chunk_identity(packed.chunk) for packed in result.chunks
        )
        relevant = set(case.relevant_chunks)
        packed_set = set(packed_ids)
        found = relevant & packed_set
        coverage = len(found) / len(relevant)
        precision = len(found) / len(packed_set) if packed_set else 0.0
        utilization = result.estimated_tokens / result.budget_tokens
        case_results.append(
            ContextAssemblyCaseResult(
                case_id=case.id,
                mode=resolved_mode,
                strategy=result.strategy,
                packed_chunk_ids=packed_ids,
                relevant_chunk_ids=case.relevant_chunks,
                seed_count=result.seed_count,
                expanded_candidate_count=result.expanded_candidate_count,
                deduplicated_count=result.deduplicated_count,
                dropped_for_budget_count=result.dropped_for_budget_count,
                packed_count=result.packed_count,
                estimated_tokens=result.estimated_tokens,
                budget_tokens=result.budget_tokens,
                gold_evidence_coverage=coverage,
                context_precision=precision,
                budget_utilization=utilization,
            )
        )

    count = len(case_results)
    return ContextAssemblyEvaluationReport(
        benchmark_version=suite.version,
        mode=resolved_mode,
        strategy=ContextStrategy(config.strategy),
        case_results=tuple(case_results),
        case_count=count,
        mean_gold_evidence_coverage=sum(r.gold_evidence_coverage for r in case_results) / count,
        mean_context_precision=sum(r.context_precision for r in case_results) / count,
        mean_budget_utilization=sum(r.budget_utilization for r in case_results) / count,
        mean_packed_count=sum(r.packed_count for r in case_results) / count,
        mean_deduplicated_count=sum(r.deduplicated_count for r in case_results) / count,
    )
