"""Deterministic end-to-end evaluation of configurable repository RAG."""

from collections.abc import Sequence

from repomind.evaluation.metrics import recall_at_k
from repomind.evaluation.models import (
    EvaluationMode,
    RAGBenchmarkSuite,
    RAGCaseResult,
    RAGEvaluationReport,
)
from repomind.observability import TraceContext, TraceRecorder
from repomind.observability.instrumentation import traced_run
from repomind.rag import (
    NeighborLoader,
    RAGConfig,
    Retriever,
    StructuredLLMProvider,
    answer_repository_question_with_retriever,
    build_rag_context,
)
from repomind.rag.models import BuiltRepositoryContext, RepositoryAnswer
from repomind.retrieval import ChunkIdentity, RankedChunk, chunk_identity


class _RecordingRetriever:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self.results: tuple[RankedChunk, ...] = ()

    def __call__(self, query: str, *, top_k: int) -> Sequence[RankedChunk]:
        self.results = tuple(self._retriever(query, top_k=top_k))
        return self.results


def _citation_identities(
    answer: RepositoryAnswer,
    context: BuiltRepositoryContext,
) -> tuple[ChunkIdentity, ...]:
    identities: list[ChunkIdentity] = []
    for citation in answer.citations:
        matching = next(
            (
                source.chunk
                for source in context.sources
                if source.chunk.relative_path == citation.relative_path
                and source.chunk.start_line == citation.start_line
                and source.chunk.end_line == citation.end_line
            ),
            None,
        )
        if matching is not None:
            identities.append(chunk_identity(matching))
    return tuple(identities)


def _identity_recall(
    relevant: tuple[ChunkIdentity, ...],
    actual: tuple[ChunkIdentity, ...],
) -> float:
    return recall_at_k(relevant, actual, k=max(1, len(actual)))


@traced_run("evaluation")
def evaluate_rag(
    suite: RAGBenchmarkSuite,
    strategy: str,
    retriever: Retriever,
    llm_client: StructuredLLMProvider,
    *,
    config: RAGConfig | None = None,
    mode: EvaluationMode = EvaluationMode.OFFLINE_FIXTURE,
    recorder: TraceRecorder | None = None,
    trace: TraceContext | None = None,
    neighbor_loader: NeighborLoader | None = None,
) -> RAGEvaluationReport:
    """Evaluate retrieval, final context, and answer oracles as separate stages."""

    trace = trace if trace is not None else TraceContext()
    resolved_mode = EvaluationMode(mode)
    if resolved_mode is EvaluationMode.OFFLINE_SCRIPTED:
        raise ValueError("RAG evaluation mode must be offline_fixture or live_model")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must not be blank")
    rag_config = config or RAGConfig()
    case_results: list[RAGCaseResult] = []
    for case in suite.cases:
        with trace.operation(
            "evaluation.case",
            case_id=case.id,
            suite_version=suite.version,
            mode=resolved_mode,
            strategy=strategy,
        ) as metadata:
            recording_retriever = _RecordingRetriever(retriever)
            answer = answer_repository_question_with_retriever(
                case.question,
                recording_retriever,
                llm_client,
                config=rag_config,
                trace=trace,
                strategy=strategy,
                neighbor_loader=neighbor_loader,
            )
            context = build_rag_context(
                recording_retriever.results,
                rag_config,
                neighbor_loader=neighbor_loader,
            )
            retrieved = tuple(
                chunk_identity(result.chunk) for result in recording_retriever.results
            )
            context_ids = tuple(chunk_identity(source.chunk) for source in context.sources)
            cited = _citation_identities(answer, context)
            missing_facts = tuple(
                fact
                for fact in case.expected_answer_facts
                if fact.casefold() not in answer.answer.casefold()
            )
            citation_recall = _identity_recall(case.relevant_chunks, cited)
            insufficient_matches = (
                answer.insufficient_evidence is case.expected_insufficient_evidence
            )
            facts_passed = not missing_facts
            case_results.append(
                RAGCaseResult(
                    case_id=case.id,
                    trace_run_id=trace.run_id,
                    strategy=strategy,
                    answer=answer.answer,
                    insufficient_evidence=answer.insufficient_evidence,
                    retrieved_chunk_ids=retrieved,
                    context_chunk_ids=context_ids,
                    cited_chunk_ids=cited,
                    relevant_chunk_ids=case.relevant_chunks,
                    retrieval_recall=_identity_recall(case.relevant_chunks, retrieved),
                    context_recall=_identity_recall(case.relevant_chunks, context_ids),
                    citation_recall=citation_recall,
                    required_facts_passed=facts_passed,
                    missing_required_facts=missing_facts,
                    answer_passed=facts_passed and citation_recall == 1.0 and insufficient_matches,
                )
            )
            metadata.update(
                {
                    key: value
                    for key, value in case_results[-1].model_dump(mode="json").items()
                    if key
                    in {"retrieval_recall", "context_recall", "citation_recall", "answer_passed"}
                }
            )

    count = len(case_results)
    return RAGEvaluationReport(
        benchmark_version=suite.version,
        mode=resolved_mode,
        strategy=strategy,
        case_results=tuple(case_results),
        case_count=count,
        mean_retrieval_recall=sum(result.retrieval_recall for result in case_results) / count,
        mean_context_recall=sum(result.context_recall for result in case_results) / count,
        mean_citation_recall=sum(result.citation_recall for result in case_results) / count,
        answer_pass_rate=sum(result.answer_passed for result in case_results) / count,
    )
