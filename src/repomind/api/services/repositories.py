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
    RepositoryResponse,
)
from repomind.api.store import RepositoryBinding, RepositoryStore
from repomind.ingestion import (
    CodeChunk,
    chunk_repository,
    find_source_files,
    ingest_repository,
    resolve_repository_path,
)
from repomind.retrieval import EmbeddedChunk


class ChunkEmbedder(Protocol):
    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> list[EmbeddedChunk]: ...


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

    def __init__(self, store: RepositoryStore, workspace: WorkspacePolicy):
        self.store = store
        self.workspace = workspace

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

    def files(self, repository_id: int, limit: int, offset: int) -> RepositoryFilesResponse:
        self.locate(repository_id)
        return RepositoryFilesResponse(
            repository_id=repository_id,
            files=self.store.files(repository_id, limit, offset),
            limit=limit,
            offset=offset,
        )

    def index(self, repository_id: int, embedder: ChunkEmbedder) -> IndexResponse:
        binding, root = self.locate(repository_id)
        with self.workspace.operation(root):
            sources = find_source_files(root)
            if (
                len(sources) > self.MAX_FILES
                or sum(p.stat().st_size for p in sources) > self.MAX_SOURCE_BYTES
            ):
                raise APIError(
                    413, "repository_too_large", "Repository exceeds synchronous index limits."
                )
            snapshot = ingest_repository(root).model_copy(update={"name": binding.name})
            if (
                snapshot.file_count > self.MAX_FILES
                or snapshot.total_size_bytes > self.MAX_SOURCE_BYTES
            ):
                raise APIError(
                    413, "repository_too_large", "Repository exceeds synchronous index limits."
                )
            chunks = chunk_repository(snapshot)
            if len(chunks) > self.MAX_CHUNKS:
                raise APIError(
                    413, "repository_too_large", "Repository exceeds synchronous index limits."
                )
            embedded = embedder.embed_chunks(chunks) if chunks else []
            self.workspace.resolve(binding.workspace_relative_path)
            self.store.replace_index(repository_id, snapshot, embedded)
            return IndexResponse(
                repository_id=repository_id,
                files_indexed=snapshot.file_count,
                chunks_indexed=len(embedded),
                embedding_model=embedded[0].embedding.model if embedded else None,
            )
