"""Lazy local-development ASGI application. No model or database calls at import."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from repomind.api.dependencies import ServiceContainer, Services
from repomind.api.errors import install_error_handlers
from repomind.api.routes import router
from repomind.api.services.execution import EmbeddingFactory, LLMFactory
from repomind.api.services.runs import TraceStore
from repomind.api.store import RepositoryStore
from repomind.config import Settings


def create_app(
    *,
    settings: Settings | None = None,
    services: Services | None = None,
    repository_store: RepositoryStore | None = None,
    trace_store: TraceStore | None = None,
    llm_factory: LLMFactory | None = None,
    embedding_factory: EmbeddingFactory | None = None,
) -> FastAPI:
    container = ServiceContainer(
        settings=settings,
        services=services,
        repository_store=repository_store,
        trace_store=trace_store,
        llm_factory=llm_factory,
        embedding_factory=embedding_factory,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await run_in_threadpool(container.close)

    application = FastAPI(
        title="RepoMind",
        version="0.1.0",
        lifespan=lifespan,
        description="Trusted local development only. No authentication or authorization. "
        "Do not expose directly to the public Internet.",
    )
    application.state.container = container
    install_error_handlers(application)
    application.include_router(router)
    return application


app = create_app()
