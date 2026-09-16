"""Compose existing RAG and agent workflows; never reproduce their algorithms/gates."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from repomind.agent import AgentConfig, EditingAgentConfig, run_read_only_agent
from repomind.api.errors import APIError, public_error
from repomind.api.models import (
    AgentRequest,
    AgentResponse,
    CitationResponse,
    CodingRequest,
    CodingResponse,
    IndexResponse,
    RAGRequest,
    RAGResponse,
    VerificationSummary,
)
from repomind.api.privacy import public_text
from repomind.api.services.repositories import ChunkEmbedder, RepositoryService
from repomind.api.services.runs import TraceStore
from repomind.coding import CodingTask, CodingTaskStatus, VerificationPolicy, run_coding_task
from repomind.observability import InMemoryTraceRecorder, RunType, TraceContext
from repomind.rag import RAGConfig, StructuredLLMProvider, answer_repository_question_with_retriever
from repomind.retrieval import EmbeddingVector, LLMReranker, RankedChunk
from repomind.tools import ToolContext, create_default_tool_registry, create_editing_tool_registry


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
    def tracing(self, enabled: bool, kind: RunType) -> Iterator[TraceContext]:
        recorder = (
            InMemoryTraceRecorder(sink=self.trace_store.persist_run_trace) if enabled else None
        )
        trace = TraceContext(recorder, kind)
        try:
            yield trace
        except Exception as exc:
            trace.finish("error", error=exc)
            error = public_error(exc)
            error.trace_run_id = trace.run_id
            raise error from exc
        else:
            trace.finish()

    def index(self, repository_id: int) -> IndexResponse:
        return self.repositories.index(repository_id, self.embedding_factory(TraceContext()))

    def rag(self, repository_id: int, request: RAGRequest) -> RAGResponse:
        with self.tracing(request.trace, RunType.RAG) as trace:
            _, root = self.repositories.locate(repository_id)
            with self.repositories.workspace.operation(root):
                llm = self.llm_factory(trace)
                embedder = self.embedding_factory(trace)

                def retrieve(query: str, *, top_k: int) -> list[RankedChunk]:
                    rerank = request.strategy == "hybrid_rerank"
                    candidates = self.repositories.store.search(
                        repository_id,
                        query,
                        embedder.embed_text(query),
                        hybrid=request.strategy != "semantic",
                        top_k=max(20, top_k) if rerank else top_k,
                    )
                    if rerank:
                        return LLMReranker(llm, trace=trace).rerank(query, candidates, top_k=top_k)
                    return list(candidates)

                answer = answer_repository_question_with_retriever(
                    request.question,
                    retrieve,
                    llm,
                    config=RAGConfig(top_k=request.top_k),
                    trace=trace,
                    strategy="hybrid+rerank"
                    if request.strategy == "hybrid_rerank"
                    else request.strategy,
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
                    trace_run_id=trace.run_id,
                )

    def agent(self, repository_id: int, request: AgentRequest) -> AgentResponse:
        with self.tracing(request.trace, RunType.READ_ONLY_AGENT) as trace:
            _, root = self.repositories.locate(repository_id)
            with self.repositories.workspace.operation(root):
                registry = create_default_tool_registry(ToolContext(repository_root=root))
                result = run_read_only_agent(
                    request.query,
                    self.llm_factory(trace),
                    registry,
                    config=AgentConfig(max_iterations=request.max_iterations),
                    trace=trace,
                )
                trace.finish(result.status.value)
                return AgentResponse(
                    status=result.status.value,
                    final_answer=public_text(result.final_answer),
                    iterations=result.iterations,
                    llm_calls=result.llm_calls,
                    tool_execution_attempts=result.tool_calls,
                    trace_run_id=trace.run_id,
                )

    def coding(self, repository_id: int, request: CodingRequest) -> CodingResponse:
        with self.tracing(request.trace, RunType.CODING_TASK) as trace:
            _, root = self.repositories.locate(repository_id)
            with self.repositories.workspace.operation(root):
                registry = create_editing_tool_registry(ToolContext(repository_root=root))
                result = run_coding_task(
                    CodingTask(
                        objective=request.objective,
                        acceptance_criteria=tuple(request.acceptance_criteria),
                    ),
                    self.llm_factory(trace),
                    registry,
                    verification_policy=VerificationPolicy(
                        test_paths=tuple(Path(p) for p in request.verification.test_paths),
                        ruff_paths=tuple(Path(p) for p in request.verification.ruff_paths),
                    ),
                    agent_config=EditingAgentConfig(max_iterations=request.max_iterations),
                    trace=trace,
                )
                trace.finish(result.status.value)
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
                    trace_run_id=trace.run_id,
                )
