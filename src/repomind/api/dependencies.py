"""App-scoped, lazily built services with injectable persistence and providers."""

from dataclasses import dataclass
from threading import Lock

from fastapi import Request

from repomind.api.services.execution import EmbeddingFactory, ExecutionService, LLMFactory
from repomind.api.services.repositories import RepositoryService, WorkspacePolicy
from repomind.api.services.runs import RunService, TraceStore
from repomind.api.services.streaming import StreamingService
from repomind.api.store import PostgresRepositoryStore, RepositoryStore
from repomind.config import Settings, get_settings
from repomind.db import create_database_engine, create_session_factory
from repomind.llm import OpenAILLMClient
from repomind.observability import PostgresTraceStore
from repomind.retrieval import OpenAIEmbeddingClient


@dataclass(frozen=True)
class Services:
    repositories: RepositoryService
    execution: ExecutionService
    runs: RunService
    streaming: StreamingService


class ServiceContainer:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        services: Services | None = None,
        repository_store: RepositoryStore | None = None,
        trace_store: TraceStore | None = None,
        llm_factory: LLMFactory | None = None,
        embedding_factory: EmbeddingFactory | None = None,
    ):
        self.settings = settings
        self.services = services
        self.repository_store = repository_store
        self.trace_store = trace_store
        self.llm_factory = llm_factory
        self.embedding_factory = embedding_factory
        self.engine = None
        self._lock = Lock()

    def get(self) -> Services:
        with self._lock:
            if self.services is None:
                settings = self.settings if self.settings is not None else get_settings()
                if self.repository_store is None or self.trace_store is None:
                    self.engine = create_database_engine(settings.database_url)
                    factory = create_session_factory(self.engine)
                store = (
                    self.repository_store
                    if self.repository_store is not None
                    else PostgresRepositoryStore(factory)
                )
                traces = (
                    self.trace_store
                    if self.trace_store is not None
                    else PostgresTraceStore(factory)
                )
                repositories = RepositoryService(
                    store, WorkspacePolicy(settings.repomind_workspace_root)
                )
                execution = ExecutionService(
                    repositories,
                    traces,
                    self.llm_factory or (lambda trace: OpenAILLMClient(settings, trace=trace)),
                    self.embedding_factory
                    or (lambda trace: OpenAIEmbeddingClient(settings, trace=trace)),
                )
                self.services = Services(
                    repositories,
                    execution,
                    RunService(traces),
                    StreamingService(execution),
                )
            return self.services

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()


def get_services(request: Request) -> Services:
    return request.app.state.container.get()
