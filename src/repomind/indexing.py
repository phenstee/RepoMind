"""Typed contracts for configuration-compatible incremental repository indexing.

RepoMind reuses an existing persisted index only when it can *prove* the stored
index was produced by the same indexing configuration. Source equality alone is
not sufficient: two identical files are not interchangeable if one index was
built with different chunk boundaries or a different embedding representation.

:func:`index_fingerprint` therefore derives a deterministic digest from every
setting that changes what a stored chunk or vector *means*. Operational
settings that do not change stored semantics (embedding batch size, for
example) are deliberately excluded so routine tuning never forces a rebuild.

When the fingerprint does not match - including a legacy index that never
persisted one - the caller must fall back to a full rebuild. Safe rebuild is
always preferable to reusing possibly incompatible vectors.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256

from repomind.ingestion import ChunkingConfig, RepositorySnapshot
from repomind.retrieval import EmbeddedChunk, EmbeddingTextStrategy

# Bumping this invalidates every persisted fingerprint, which forces one safe
# rebuild everywhere. Bump it whenever the digest inputs or encoding change.
INDEX_FINGERPRINT_VERSION = 1


def content_digest(content: str) -> str:
    """Return a deterministic SHA-256 digest of exact UTF-8 source text.

    ``db.content_sha256`` delegates here so classification, validation, and
    persistence can never disagree about what "unchanged" means.
    """

    return sha256(content.encode("utf-8")).hexdigest()


def index_fingerprint(
    *,
    chunking: ChunkingConfig,
    embedding_model: str,
    embedding_text_strategy: EmbeddingTextStrategy,
) -> str:
    """Digest every setting that changes the meaning of a stored chunk/vector.

    The payload is canonically serialized (sorted keys, no insignificant
    whitespace) and hashed with SHA-256 so the value is stable across
    processes. Python's randomized ``hash()`` is deliberately not used.
    """

    payload = {
        "version": INDEX_FINGERPRINT_VERSION,
        "chunking_strategy": chunking.strategy.value,
        "max_lines_per_chunk": chunking.max_lines_per_chunk,
        "overlap_lines": chunking.overlap_lines,
        "max_chars_per_chunk": chunking.max_chars_per_chunk,
        "embedding_model": embedding_model,
        "embedding_text_strategy": EmbeddingTextStrategy(embedding_text_strategy).value,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IndexedFileState:
    """What the persisted index already holds for one repository-relative path."""

    content_hash: str
    chunk_count: int


@dataclass(frozen=True)
class IndexManifest:
    """Cheap projection of the persisted index: no source text, no vectors.

    ``fingerprint`` is ``None`` for a repository indexed before fingerprints
    existed, which is treated as incompatible rather than assumed reusable.
    """

    fingerprint: str | None = None
    files: Mapping[str, IndexedFileState] = field(default_factory=dict)

    def is_compatible_with(self, fingerprint: str) -> bool:
        return self.fingerprint is not None and self.fingerprint == fingerprint


@dataclass(frozen=True)
class FileClassification:
    """Deterministic file-granularity plan for one indexing run."""

    unchanged: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    full_rebuild: bool = False

    @property
    def reindexed(self) -> tuple[str, ...]:
        """Paths whose chunks and vectors must be produced again."""

        return tuple(sorted((*self.changed, *self.added)))


def classify_files(
    *,
    current: Mapping[str, str],
    manifest: IndexManifest,
    compatible: bool,
) -> FileClassification:
    """Classify current source paths against the persisted manifest.

    ``current`` maps repository-relative path to the current content hash. An
    incompatible (or legacy) manifest degrades to a full rebuild in which every
    current file is treated as added and nothing is reused.
    """

    if not compatible:
        return FileClassification(added=tuple(sorted(current)), full_rebuild=True)

    unchanged: list[str] = []
    changed: list[str] = []
    added: list[str] = []
    for path in sorted(current):
        stored = manifest.files.get(path)
        if stored is None:
            added.append(path)
        elif stored.content_hash == current[path]:
            unchanged.append(path)
        else:
            changed.append(path)
    deleted = tuple(sorted(set(manifest.files) - set(current)))
    return FileClassification(
        unchanged=tuple(unchanged),
        changed=tuple(changed),
        added=tuple(added),
        deleted=deleted,
    )


@dataclass(frozen=True)
class IndexUpdate:
    """One atomic index delta: everything the store needs, already prepared.

    ``chunks`` carries embedded chunks only for ``upserted`` paths. Unchanged
    files are absent from both collections precisely because their persisted
    rows and vectors are being reused untouched.
    """

    fingerprint: str
    snapshot: RepositorySnapshot
    upserted: tuple[str, ...]
    deleted: tuple[str, ...]
    chunks: Sequence[EmbeddedChunk]
    full_rebuild: bool


@dataclass(frozen=True)
class IndexSummary:
    """Totals for the complete resulting index, not for this run's delta."""

    file_count: int
    chunk_count: int
    embedding_model: str | None


class IndexUpdateError(ValueError):
    """Raised when a delta cannot be applied without corrupting the index."""


def validate_index_update(
    update: IndexUpdate,
    *,
    manifest: IndexManifest,
) -> None:
    """Reject any delta that would leave a wrong index stamped as trustworthy.

    ``apply_index_update`` writes a fingerprint that authorizes *future* reuse,
    so an internally inconsistent delta is worse than a failed index: it would
    certify stale or missing rows as current. This runs before any mutation and
    checks the delta against both the complete snapshot and the complete
    persisted manifest, so a rejected update leaves the previous index and its
    fingerprint exactly as they were.

    The manifest is taken whole, not just its files: matching content hashes
    prove a *file* is unchanged, but only the persisted fingerprint proves the
    rows and vectors behind it were built under the configuration this update
    claims.
    """

    persisted = manifest.files
    snapshot_hashes = {
        source.relative_path.as_posix(): content_digest(source.content)
        for source in update.snapshot.files
    }
    upserted, deleted = set(update.upserted), set(update.deleted)
    if len(upserted) != len(update.upserted):
        raise IndexUpdateError("upserted paths contain duplicates")
    if len(deleted) != len(update.deleted):
        raise IndexUpdateError("deleted paths contain duplicates")
    if overlap := sorted(upserted & deleted):
        raise IndexUpdateError(f"path {overlap[0]!r} is both upserted and deleted")
    if unknown := sorted(upserted - set(snapshot_hashes)):
        raise IndexUpdateError(
            f"upserted path {unknown[0]!r} is absent from the snapshot"
        )

    if update.full_rebuild:
        # A rebuild drops every prior row, so the upsert list alone must
        # reconstruct the entire current snapshot. Incompatible or missing
        # prior provenance is expected here and is exactly what a rebuild is
        # for, so the persisted fingerprint is deliberately not consulted.
        if missing := sorted(set(snapshot_hashes) - upserted):
            raise IndexUpdateError(
                f"full rebuild omits snapshot path {missing[0]!r} from upserted"
            )
    else:
        # An incremental update keeps rows it did not rewrite, so it may only
        # proceed when the stored index was produced under the very
        # configuration this update is about to stamp. A missing fingerprint
        # (legacy data, or provenance revoked by a legacy mutation) can never
        # authorize reuse.
        if not manifest.is_compatible_with(update.fingerprint):
            raise IndexUpdateError(
                "incremental update requires a persisted fingerprint matching "
                "the update; rebuild instead"
            )
        expected_deletions = set(persisted) - set(snapshot_hashes)
        if surplus := sorted(deleted - expected_deletions):
            raise IndexUpdateError(
                f"deleted path {surplus[0]!r} is still present in the snapshot"
            )
        if omitted := sorted(expected_deletions - deleted):
            raise IndexUpdateError(
                f"path {omitted[0]!r} left the snapshot but is not deleted"
            )
        for path in sorted(set(snapshot_hashes) - upserted):
            stored = persisted.get(path)
            if stored is None:
                raise IndexUpdateError(
                    f"new path {path!r} is not upserted and is not already indexed"
                )
            if stored.content_hash != snapshot_hashes[path]:
                raise IndexUpdateError(
                    f"changed path {path!r} would be reused without re-indexing"
                )

    for chunk in update.chunks:
        path = chunk.chunk.relative_path.as_posix()
        if path not in upserted:
            raise IndexUpdateError(
                f"embedded chunk targets {path!r}, which is not being upserted"
            )
