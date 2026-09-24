"""Workspace confinement and bounded, synchronous indexing composition."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Protocol

from repomind.api.errors import APIError
from repomind.api.models import (
    IndexResponse,
    RegisterRepositoryRequest,
    RepositoryFilesResponse,
    RepositoryListItemResponse,
    RepositoryListResponse,
    RepositoryResponse,
)
from repomind.api.store import RepositoryBinding, RepositoryStore
from repomind.db import content_sha256
from repomind.indexing import (
    FileClassification,
    IndexSummary,
    IndexUpdate,
    classify_files,
    index_fingerprint,
)
from repomind.ingestion import (
    ChunkingConfig,
    CodeChunk,
    RepositorySnapshot,
    chunk_source_file,
    find_source_files,
    ingest_repository,
    resolve_repository_path,
)
from repomind.jobs.control import CooperativeCancellation, NoCancellation
from repomind.jobs.locks import NullRepositoryExecutionLock, RepositoryExecutionLock
from repomind.observability import TraceContext
from repomind.retrieval import EmbeddedChunk, EmbeddingTextStrategy


class ChunkEmbedder(Protocol):
    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> list[EmbeddedChunk]: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def embedding_text_strategy(self) -> EmbeddingTextStrategy: ...


class WorkspacePolicy:
    """Use the domain resolver; never turn a client path into a trusted root directly."""

    def __init__(self, root: Path | None):
        self.root = root
        self._canonical: Path | None = None
        self._mutex = Lock()
        self._active: set[Path] = set()

    def resolve(self, relative: str) -> Path:
        try:
            if self.root is None or not self.root.is_absolute():
                raise APIError(
                    503, "workspace_unavailable", "An absolute workspace root is required."
                )
            if self.root.is_symlink() or self.root.is_junction():
                raise APIError(503, "workspace_unavailable", "Workspace root is unavailable.")
            canonical = self.root.resolve(strict=True)
            if not canonical.is_dir() or canonical == Path(canonical.anchor):
                raise APIError(503, "workspace_unavailable", "Workspace root is unavailable.")
            with self._mutex:
                if self._canonical is not None and canonical != self._canonical:
                    raise APIError(503, "workspace_unavailable", "Workspace root changed.")
                self._canonical = canonical
        except (OSError, ValueError) as exc:
            raise APIError(503, "workspace_unavailable", "Workspace root is unavailable.") from exc
        try:
            # Reject Windows stream/alias spelling as well as the domain's traversal checks.
            if (
                "\x00" in relative
                or ":" in relative
                or any(part.endswith((" ", ".")) for part in relative.replace("\\", "/").split("/"))
            ):
                raise ValueError("Unsafe repository path")
            resolved = resolve_repository_path(canonical, Path(relative))
            if not resolved.is_dir():
                raise APIError(
                    404, "repository_directory_not_found", "Repository directory not found."
                )
            return resolved
        except (ValueError, OSError) as exc:
            raise APIError(
                400, "invalid_repository_path", "Repository path is not allowed."
            ) from exc

    @contextmanager
    def operation(self, path: Path) -> Iterator[None]:
        """Prevent overlapping API operations, including nested workspace aliases, in one process."""
        with self._mutex:
            if any(
                path.is_relative_to(active) or active.is_relative_to(path)
                for active in self._active
            ):
                raise APIError(409, "repository_busy", "Another repository operation is running.")
            self._active.add(path)
        try:
            yield
        finally:
            with self._mutex:
                self._active.remove(path)


class RepositoryService:
    MAX_FILES = 10_000
    MAX_SOURCE_BYTES = 50 * 1024 * 1024
    MAX_CHUNKS = 10_000

    def __init__(
        self,
        store: RepositoryStore,
        workspace: WorkspacePolicy,
        execution_lock: RepositoryExecutionLock | None = None,
        chunking_config: ChunkingConfig | None = None,
    ):
        self.store = store
        self.workspace = workspace
        self.execution_lock = execution_lock or NullRepositoryExecutionLock()
        self.chunking_config = chunking_config or ChunkingConfig()

    @staticmethod
    def response(binding: RepositoryBinding) -> RepositoryResponse:
        return RepositoryResponse(id=binding.id, name=binding.name, created_at=binding.created_at)

    def register(self, request: RegisterRepositoryRequest) -> RepositoryResponse:
        root = self.workspace.resolve(request.path)
        relative = root.relative_to(self.workspace.root.resolve()).as_posix()
        with self.workspace.operation(root):
            return self.response(self.store.register(request.name, relative))

    def locate(self, repository_id: int) -> tuple[RepositoryBinding, Path]:
        binding = self.store.get(repository_id)
        if binding.workspace_relative_path is None:
            raise APIError(409, "repository_unbound", "Repository has no API workspace binding.")
        return binding, self.workspace.resolve(binding.workspace_relative_path)

    def get(self, repository_id: int) -> RepositoryResponse:
        binding, _ = self.locate(repository_id)
        return self.response(binding)

    def list(self, limit: int) -> RepositoryListResponse:
        return RepositoryListResponse(
            repositories=[
                RepositoryListItemResponse(
                    id=binding.id,
                    name=binding.name,
                    created_at=binding.created_at,
                    workspace_relative_path=binding.workspace_relative_path,
                )
                for binding in self.store.list_repositories(limit)
            ]
        )

    def files(self, repository_id: int, limit: int, offset: int) -> RepositoryFilesResponse:
        self.locate(repository_id)
        return RepositoryFilesResponse(
            repository_id=repository_id,
            files=self.store.files(repository_id, limit, offset),
            limit=limit,
            offset=offset,
        )

    def index(
        self,
        repository_id: int,
        embedder: ChunkEmbedder,
        *,
        trace: TraceContext | None = None,
        cancellation: CooperativeCancellation | None = None,
    ) -> IndexResponse:
        cancellation = cancellation or NoCancellation()
        cancellation.checkpoint()
        binding, root = self.locate(repository_id)
        with self.workspace.operation(root), self.execution_lock.hold(repository_id):
            cancellation.checkpoint()
            if trace is not None:
                trace.emit("index.started", repository_id=repository_id)
            sources = find_source_files(root)
            if (
                len(sources) > self.MAX_FILES
                or sum(p.stat().st_size for p in sources) > self.MAX_SOURCE_BYTES
            ):
                raise APIError(
                    413, "repository_too_large", "Repository exceeds synchronous index limits."
                )
            snapshot = ingest_repository(root).model_copy(update={"name": binding.name})
            cancellation.checkpoint()
            if (
                snapshot.file_count > self.MAX_FILES
                or snapshot.total_size_bytes > self.MAX_SOURCE_BYTES
            ):
                raise APIError(
                    413, "repository_too_large", "Repository exceeds synchronous index limits."
                )
            if trace is not None:
                trace.emit(
                    "ingestion.completed",
                    repository_id=repository_id,
                    file_count=snapshot.file_count,
                    total_size_bytes=snapshot.total_size_bytes,
                )
            fingerprint = index_fingerprint(
                chunking=self.chunking_config,
                embedding_model=embedder.embedding_model,
                embedding_text_strategy=embedder.embedding_text_strategy,
            )
            manifest = self.store.index_manifest(repository_id)
            plan = classify_files(
                current={
                    source.relative_path.as_posix(): content_sha256(source.content)
                    for source in snapshot.files
                },
                manifest=manifest,
                compatible=manifest.is_compatible_with(fingerprint),
            )
            # Only changed/new files are chunked; unchanged files keep the
            # chunks and vectors already persisted for them.
            chunks = self._chunk_selected(snapshot, plan.reindexed)
            cancellation.checkpoint()
            reused_chunks = sum(
                manifest.files[path].chunk_count for path in plan.unchanged
            )
            if reused_chunks + len(chunks) > self.MAX_CHUNKS:
                raise APIError(
                    413, "repository_too_large", "Repository exceeds synchronous index limits."
                )
            if trace is not None:
                trace.emit(
                    "chunking.completed",
                    repository_id=repository_id,
                    chunk_count=reused_chunks + len(chunks),
                    chunking_strategy=self.chunking_config.strategy.value,
                )
                trace.emit(
                    "embedding.started", repository_id=repository_id, chunk_count=len(chunks)
                )
            embedded = embedder.embed_chunks(chunks) if chunks else []
            self._verify_embedder_output(chunks, embedded, embedder.embedding_model)
            cancellation.checkpoint()
            if trace is not None:
                trace.emit(
                    "embedding.completed",
                    repository_id=repository_id,
                    chunk_count=len(embedded),
                    embedding_model=embedder.embedding_model if embedded else None,
                )
            self.workspace.resolve(binding.workspace_relative_path)
            cancellation.checkpoint()
            summary = self.store.apply_index_update(
                repository_id,
                IndexUpdate(
                    fingerprint=fingerprint,
                    snapshot=snapshot,
                    upserted=plan.reindexed,
                    deleted=plan.deleted,
                    chunks=embedded,
                    full_rebuild=plan.full_rebuild,
                ),
            )
            if trace is not None:
                self._emit_index_completed(trace, repository_id, plan, summary, len(embedded))
            cancellation.checkpoint()
            return IndexResponse(
                repository_id=repository_id,
                files_indexed=summary.file_count,
                chunks_indexed=summary.chunk_count,
                embedding_model=summary.embedding_model,
            )

    @staticmethod
    def _verify_embedder_output(
        requested: Sequence[CodeChunk],
        embedded: Sequence[EmbeddedChunk],
        embedding_model: str,
    ) -> None:
        """Fail closed unless the provider returned exactly what we asked for.

        The caller knows the precise chunk sequence it submitted, so a missing,
        reordered, substituted, or foreign result is detectable here - before
        anything is persisted under a fingerprint that claims this model. The
        messages name only chunk identity, never source text.
        """

        if len(embedded) != len(requested):
            raise APIError(
                502,
                "embedding_contract_violation",
                "The embedding provider returned an unexpected number of results.",
            )
        for chunk, result in zip(requested, embedded, strict=True):
            if result.chunk is not chunk and result.chunk != chunk:
                raise APIError(
                    502,
                    "embedding_contract_violation",
                    "The embedding provider returned results for unexpected chunks.",
                )
            if result.embedding.model != embedding_model:
                # The fingerprint authorizing future reuse records
                # embedder.embedding_model; storing a vector from a different
                # model would make that claim false.
                raise APIError(
                    502,
                    "embedding_contract_violation",
                    "The embedding provider returned an unexpected model identity.",
                )

    def _chunk_selected(
        self, snapshot: RepositorySnapshot, paths: Sequence[str]
    ) -> Sequence[CodeChunk]:
        """Chunk only the selected files with the same per-file chunker.

        ``chunk_repository`` is simply this loop over every file, so chunking a
        subset produces byte-identical chunks for those files: ``chunk_index``
        already restarts per file and never depends on other files.
        """

        if not paths:
            return []
        selected = set(paths)
        chunks: list[CodeChunk] = []
        for source in snapshot.files:
            if source.relative_path.as_posix() in selected:
                chunks.extend(chunk_source_file(source, self.chunking_config))
        return chunks

    @staticmethod
    def _emit_index_completed(
        trace: TraceContext,
        repository_id: int,
        plan: FileClassification,
        summary: IndexSummary,
        embedded_chunks: int,
    ) -> None:
        """Report aggregate index work only: never paths, source text, or vectors."""

        trace.emit(
            "persistence.completed",
            repository_id=repository_id,
            file_count=summary.file_count,
            chunk_count=summary.chunk_count,
            full_rebuild=plan.full_rebuild,
            files_added=len(plan.added),
            files_changed=len(plan.changed),
            files_unchanged=len(plan.unchanged),
            files_deleted=len(plan.deleted),
            chunks_embedded=embedded_chunks,
        )
