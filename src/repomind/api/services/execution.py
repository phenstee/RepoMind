"""Compose existing RAG and agent workflows; never reproduce their algorithms/gates."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from repomind.agent import (
    AgentConfig,
    EditingAgentConfig,
    extract_observed_source_evidence,
    run_indexed_read_only_agent,
    run_read_only_agent,
)
from repomind.api.errors import APIError, public_error
from repomind.api.models import (
    AgentRequest,
    AgentResponse,
    CitationResponse,
    CodingRequest,
    CodingResponse,
    IndexResponse,
    ObservedLocationResponse,
    RAGRequest,
    RAGResponse,
    VerificationSummary,
)
from repomind.api.privacy import public_metadata, public_review_metadata, public_text
from repomind.api.services.repositories import ChunkEmbedder, RepositoryService
from repomind.api.services.runs import TraceStore
from repomind.coding import (
    CodingPlan,
    CodingReview,
    CodingTask,
    CodingTaskStatus,
    VerificationPolicy,
    run_coding_task,
)
from repomind.jobs.control import (
    CooperativeCancellation,
    JobCancellationRequested,
    NoCancellation,
)
from repomind.observability import InMemoryTraceRecorder, RunType, TraceContext
from repomind.rag import (
    ContextStrategy,
    RAGConfig,
    StructuredLLMProvider,
    answer_repository_question_with_retriever,
)
from repomind.retrieval import EmbeddingVector, LLMReranker, RankedChunk
from repomind.tools import (
    ToolContext,
    create_default_tool_registry,
    create_editing_tool_registry,
    create_investigation_tool_registry,
)


class EmbeddingProvider(ChunkEmbedder, Protocol):
    def embed_text(self, text: str) -> EmbeddingVector: ...


LLMFactory = Callable[[TraceContext], StructuredLLMProvider]
EmbeddingFactory = Callable[[TraceContext], EmbeddingProvider]


class ExecutionService:
    def __init__(
        self,
        repositories: RepositoryService,
        trace_store: TraceStore,
        llm_factory: LLMFactory,
        embedding_factory: EmbeddingFactory,
    ):
        self.repositories = repositories
        self.trace_store = trace_store
        self.llm_factory = llm_factory
        self.embedding_factory = embedding_factory

    @contextmanager
    def tracing(
        self,
        enabled: bool,
        kind: RunType,
        *,
        trace: TraceContext | None = None,
    ) -> Iterator[TraceContext]:
        if trace is None:
            recorder = (
                InMemoryTraceRecorder(sink=self.trace_store.persist_run_trace) if enabled else None
            )
            trace = TraceContext(recorder, kind)
        try:
            yield trace
        except JobCancellationRequested:
            trace.finish("cancelled")
            raise
        except Exception as exc:
            trace.finish("error", error=exc)
            error = public_error(exc)
            error.trace_run_id = trace.run_id
            raise error from exc
        else:
            trace.finish()

    def index(
        self,
        repository_id: int,
        *,
        trace: TraceContext | None = None,
        cancellation: CooperativeCancellation | None = None,
    ) -> IndexResponse:
        cancellation = cancellation or NoCancellation()
        if trace is None:
            return self.repositories.index(
                repository_id,
                self.embedding_factory(TraceContext()),
                cancellation=cancellation,
            )
        with self.tracing(True, RunType.INDEX, trace=trace):
            return self.repositories.index(
                repository_id,
                self.embedding_factory(trace),
                trace=trace,
                cancellation=cancellation,
            )

    def rag(
        self,
        repository_id: int,
        request: RAGRequest,
        *,
        trace: TraceContext | None = None,
        cancellation: CooperativeCancellation | None = None,
    ) -> RAGResponse:
        cancellation = cancellation or NoCancellation()
        with self.tracing(request.trace, RunType.RAG, trace=trace) as run_trace:
            cancellation.checkpoint()
            _, root = self.repositories.locate(repository_id)
            with self.repositories.workspace.operation(root):
                llm = self.llm_factory(run_trace)
                embedder = self.embedding_factory(run_trace)

                def retrieve(query: str, *, top_k: int) -> list[RankedChunk]:
                    rerank = request.strategy == "hybrid_rerank"
                    include_symbols = request.strategy == "hybrid_symbol"
                    candidates = self.repositories.store.search(
                        repository_id,
                        query,
                        embedder.embed_text(query),
                        hybrid=request.strategy != "semantic",
                        top_k=max(20, top_k) if rerank else top_k,
                        include_symbols=include_symbols,
                    )
                    if include_symbols:
                        run_trace.emit(
                            "symbol.matched",
                            symbol_candidate_count=sum(
                                1 for c in candidates if getattr(c, "symbol_rank", None) is not None
                            ),
                            symbol_match_detected=any(
                                getattr(c, "symbol_rank", None) is not None for c in candidates
                            ),
                            fused_candidate_count=len(candidates),
                        )
                    if rerank:
                        return LLMReranker(llm, trace=run_trace).rerank(
                            query, candidates, top_k=top_k
                        )
                    return list(candidates)

                context_strategy = ContextStrategy(request.context_strategy)
                neighbor_loader = (
                    (lambda keys: self.repositories.store.load_neighbors(repository_id, keys))
                    if context_strategy is ContextStrategy.EXPANDED
                    else None
                )
                answer = answer_repository_question_with_retriever(
                    request.question,
                    retrieve,
                    llm,
                    config=RAGConfig(top_k=request.top_k, context_strategy=context_strategy),
                    trace=run_trace,
                    strategy="hybrid+rerank"
                    if request.strategy == "hybrid_rerank"
                    else request.strategy,
                    cancellation=cancellation,
                    neighbor_loader=neighbor_loader,
                )
                return RAGResponse(
                    answer=public_text(answer.answer),
                    insufficient_evidence=answer.insufficient_evidence,
                    citations=[
                        CitationResponse(
                            relative_path=c.relative_path.as_posix(),
                            start_line=c.start_line,
                            end_line=c.end_line,
                        )
                        for c in answer.citations
                    ],
                    trace_run_id=run_trace.run_id,
                )

    def agent(
        self,
        repository_id: int,
        request: AgentRequest,
        *,
        trace: TraceContext | None = None,
        cancellation: CooperativeCancellation | None = None,
    ) -> AgentResponse:
        cancellation = cancellation or NoCancellation()
        with self.tracing(request.trace, RunType.READ_ONLY_AGENT, trace=trace) as run_trace:
            cancellation.checkpoint()
            _, root = self.repositories.locate(repository_id)
            with self.repositories.workspace.operation(root):
                context = ToolContext(repository_root=root)
                if request.retrieval_mode == "indexed":
                    embedder = self.embedding_factory(run_trace)

                    def indexed_retriever(query: str, *, top_k: int) -> list[RankedChunk]:
                        # M24's strongest fused strategy; navigation only, so
                        # this never sees embeddings/SQL beyond one lazy call.
                        return list(
                            self.repositories.store.search(
                                repository_id,
                                query,
                                embedder.embed_text(query),
                                hybrid=True,
                                top_k=top_k,
                                include_symbols=True,
                            )
                        )

                    registry = create_investigation_tool_registry(context, indexed_retriever)
                    agent_runner = run_indexed_read_only_agent
                else:
                    registry = create_default_tool_registry(context)
                    agent_runner = run_read_only_agent
                result = agent_runner(
                    request.query,
                    self.llm_factory(run_trace),
                    registry,
                    config=AgentConfig(max_iterations=request.max_iterations),
                    trace=run_trace,
                    cancellation=cancellation,
                )
                run_trace.finish(result.status.value)
                evidence = extract_observed_source_evidence(result.steps)
                return AgentResponse(
                    status=result.status.value,
                    final_answer=public_text(result.final_answer),
                    iterations=result.iterations,
                    llm_calls=result.llm_calls,
                    tool_execution_attempts=result.tool_calls,
                    evidence=[
                        ObservedLocationResponse(
                            relative_path=location.relative_path.as_posix(),
                            start_line=location.start_line,
                            end_line=location.end_line,
                            observed_via=location.observed_via,
                        )
                        for location in evidence.locations
                    ],
                    evidence_truncated=evidence.truncated,
                    trace_run_id=run_trace.run_id,
                )

    def coding(
        self,
        repository_id: int,
        request: CodingRequest,
        *,
        trace: TraceContext | None = None,
        cancellation: CooperativeCancellation | None = None,
    ) -> CodingResponse:
        cancellation = cancellation or NoCancellation()
        with self.tracing(request.trace, RunType.CODING_TASK, trace=trace) as run_trace:
            cancellation.checkpoint()
            _, root = self.repositories.locate(repository_id)
            with (
                self.repositories.workspace.operation(root),
                self.repositories.execution_lock.hold(repository_id),
            ):
                registry = create_editing_tool_registry(ToolContext(repository_root=root))
                result = run_coding_task(
                    CodingTask(
                        objective=request.objective,
                        acceptance_criteria=tuple(request.acceptance_criteria),
                    ),
                    self.llm_factory(run_trace),
                    registry,
                    verification_policy=VerificationPolicy(
                        test_paths=tuple(Path(p) for p in request.verification.test_paths),
                        ruff_paths=tuple(Path(p) for p in request.verification.ruff_paths),
                    ),
                    agent_config=EditingAgentConfig(max_iterations=request.max_iterations),
                    trace=run_trace,
                    cancellation=cancellation,
                )
                run_trace.finish(result.status.value)
                if result.status == CodingTaskStatus.PRECONDITION_FAILED:
                    raise APIError(
                        409, "coding_precondition_failed", "Coding preflight requirements failed."
                    )
                if result.status == CodingTaskStatus.ERROR:
                    raise APIError(500, "coding_failed", "Coding workflow could not be completed.")
                verification = result.verification

                def summary(kind: str) -> VerificationSummary:
                    evidence = getattr(verification, f"{kind}_result")
                    return VerificationSummary(
                        passed=getattr(verification, f"{kind}_passed"),
                        workspace_revision=getattr(verification, f"{kind}_revision"),
                        exit_code=evidence.exit_code if evidence else None,
                        timed_out=evidence.timed_out if evidence else None,
                        execution_failed=getattr(verification, f"{kind}_execution_error")
                        is not None,
                    )

                return CodingResponse(
                    status=result.status,
                    final_answer=public_text(result.final_answer),
                    tests=summary("tests"),
                    ruff=summary("ruff"),
                    changed_files=[p.as_posix() for p in result.changed_files],
                    completion_attempts=result.completion_attempts,
                    workspace_revision=result.workspace_revision,
                    plan=(
                        CodingPlan.model_validate(
                            public_metadata(result.plan.model_dump(mode="json"))
                        )
                        if result.plan is not None
                        else None
                    ),
                    review=(
                        CodingReview.model_validate(
                            public_review_metadata(
                                result.review.model_dump(mode="json"),
                                tuple(
                                    diff.content
                                    for diff in (
                                        result.final_review.unstaged_diff
                                        if result.final_review is not None
                                        else None,
                                        result.final_review.staged_diff
                                        if result.final_review is not None
                                        else None,
                                    )
                                    if diff is not None
                                ),
                            )
                        )
                        if result.review is not None
                        else None
                    ),
                    review_attempts=result.review_attempts,
                    review_blocks=result.review_blocks,
                    trace_run_id=run_trace.run_id,
                )
