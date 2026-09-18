"""Transactional persistence plus exact and HNSW pgvector retrieval."""

from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256

from pgvector.sqlalchemy import Vector
from sqlalchemy import case, cast, distinct, false, or_, select, text, tuple_
from sqlalchemy.orm import Session

from repomind.db.models import (
    HNSW_EMBEDDING_DIMENSIONS,
    CodeChunkRecord,
    RepositoryFileRecord,
    RepositoryRecord,
)
from repomind.ingestion import CodeChunk, RepositorySnapshot
from repomind.retrieval import (
    EmbeddedChunk,
    EmbeddingVector,
    IdentifierCandidate,
    IdentifierConfidence,
    SemanticSearchError,
    SemanticSearchMode,
    SemanticSearchResult,
    SimilarityError,
    SymbolMatchTier,
    SymbolSearchError,
    SymbolSearchResult,
    cosine_similarity,
)
from repomind.retrieval.symbols import MAX_SYMBOL_CANDIDATE_LIMIT


class PersistenceError(ValueError):
    """Raised when persistent repository data violates a stable domain contract."""


class RepositoryNotFoundError(PersistenceError):
    """Raised when a persistence operation targets an unknown repository."""


def content_sha256(content: str) -> str:
    """Return a deterministic SHA-256 digest of exact UTF-8 source text."""

    return sha256(content.encode("utf-8")).hexdigest()


def _repository_or_raise(session: Session, repository_id: int) -> RepositoryRecord:
    repository = session.get(RepositoryRecord, repository_id)
    if repository is None:
        raise RepositoryNotFoundError(f"Repository {repository_id} does not exist")
    return repository


def persist_repository_snapshot(
    session: Session,
    snapshot: RepositorySnapshot,
) -> RepositoryRecord:
    """Replace a named repository's stored file snapshot without committing.

    Replacing files also removes prior chunks through delete cascades. The
    snapshot's absolute root is intentionally not persisted.
    """

    repository = session.scalar(
        select(RepositoryRecord).where(RepositoryRecord.name == snapshot.name)
    )
    if repository is None:
        repository = RepositoryRecord(name=snapshot.name)
        session.add(repository)
        session.flush()
    else:
        repository.files.clear()
        session.flush()
        repository.updated_at = datetime.now(UTC)

    repository.files.extend(
        RepositoryFileRecord(
            relative_path=source_file.relative_path.as_posix(),
            language=source_file.language,
            size_bytes=source_file.size_bytes,
            line_count=source_file.line_count,
            content=source_file.content,
            content_hash=content_sha256(source_file.content),
        )
        for source_file in snapshot.files
    )
    session.flush()
    return repository


def _files_by_path(
    session: Session,
    repository_id: int,
) -> dict[str, RepositoryFileRecord]:
    _repository_or_raise(session, repository_id)
    files = session.scalars(
        select(RepositoryFileRecord).where(
            RepositoryFileRecord.repository_id == repository_id
        )
    ).all()
    return {repository_file.relative_path: repository_file for repository_file in files}


def _validate_chunk_inputs(
    files_by_path: dict[str, RepositoryFileRecord],
    chunks: Sequence[CodeChunk],
) -> None:
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        path = chunk.relative_path.as_posix()
        if path not in files_by_path:
            raise PersistenceError(
                f"Chunk references {path!r}, which is not stored for this repository"
            )
        identity = (path, chunk.chunk_index)
        if identity in seen:
            raise PersistenceError(
                f"Duplicate chunk index {chunk.chunk_index} for repository file {path!r}"
            )
        seen.add(identity)


def _replace_chunks(
    session: Session,
    repository_id: int,
    chunks: Sequence[CodeChunk],
    embeddings: dict[tuple[str, int], EmbeddingVector],
) -> list[CodeChunkRecord]:
    files_by_path = _files_by_path(session, repository_id)
    _validate_chunk_inputs(files_by_path, chunks)

    for repository_file in files_by_path.values():
        repository_file.chunks.clear()
    session.flush()

    records: list[CodeChunkRecord] = []
    for chunk in chunks:
        path = chunk.relative_path.as_posix()
        embedding = embeddings.get((path, chunk.chunk_index))
        record = CodeChunkRecord(
            chunk_index=chunk.chunk_index,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content=chunk.content,
            content_hash=content_sha256(chunk.content),
            chunking_strategy=chunk.chunking_strategy.value,
            chunk_kind=chunk.chunk_kind.value,
            symbol_name=chunk.symbol_name,
            qualified_symbol_name=chunk.qualified_symbol_name,
            parent_symbol=chunk.parent_symbol,
            fragment_index=chunk.fragment_index,
            fragment_count=chunk.fragment_count,
            embedding=list(embedding.values) if embedding is not None else None,
            embedding_model=embedding.model if embedding is not None else None,
            embedding_dimensions=embedding.dimensions if embedding is not None else None,
        )
        files_by_path[path].chunks.append(record)
        records.append(record)

    session.flush()
    return records


def persist_chunks(
    session: Session,
    repository_id: int,
    chunks: Sequence[CodeChunk],
) -> list[CodeChunkRecord]:
    """Replace all chunks for a repository without generating embeddings or committing."""

    return _replace_chunks(session, repository_id, chunks, {})


def persist_embedded_chunks(
    session: Session,
    repository_id: int,
    embedded_chunks: Sequence[EmbeddedChunk],
) -> list[CodeChunkRecord]:
    """Replace all chunks and pgvector embeddings for a repository without committing."""

    model_dimensions: dict[str, int] = {}
    embeddings: dict[tuple[str, int], EmbeddingVector] = {}
    chunks: list[CodeChunk] = []
    for embedded_chunk in embedded_chunks:
        chunk = embedded_chunk.chunk
        embedding = embedded_chunk.embedding
        expected_dimensions = model_dimensions.setdefault(
            embedding.model,
            embedding.dimensions,
        )
        if embedding.dimensions != expected_dimensions:
            raise PersistenceError(
                f"Embedding model {embedding.model!r} has inconsistent dimensions: "
                f"{expected_dimensions} and {embedding.dimensions}"
            )
        try:
            cosine_similarity(embedding.values, embedding.values)
        except SimilarityError as exc:
            raise PersistenceError(
                f"Chunk {chunk.relative_path.as_posix()!r} has an unusable embedding"
            ) from exc

        identity = (chunk.relative_path.as_posix(), chunk.chunk_index)
        if identity in embeddings:
            raise PersistenceError(
                f"Duplicate embedded chunk index {chunk.chunk_index} for "
                f"repository file {identity[0]!r}"
            )
        chunks.append(chunk)
        embeddings[identity] = embedding

    return _replace_chunks(session, repository_id, chunks, embeddings)


def _code_chunk_from_record(
    record: CodeChunkRecord,
    relative_path: str,
    language: str | None,
) -> CodeChunk:
    return CodeChunk(
        relative_path=relative_path,
        language=language,
        start_line=record.start_line,
        end_line=record.end_line,
        content=record.content,
        chunk_index=record.chunk_index,
        chunking_strategy=record.chunking_strategy or "line_v1",
        chunk_kind=record.chunk_kind or "line",
        symbol_name=record.symbol_name,
        qualified_symbol_name=record.qualified_symbol_name,
        parent_symbol=record.parent_symbol,
        fragment_index=record.fragment_index,
        fragment_count=record.fragment_count,
    )


def _embedded_chunk_from_record(
    record: CodeChunkRecord,
    relative_path: str,
    language: str | None,
) -> EmbeddedChunk:
    if (
        record.embedding is None
        or record.embedding_model is None
        or record.embedding_dimensions is None
    ):
        raise PersistenceError(f"Chunk record {record.id!r} does not have an embedding")
    values = tuple(float(value) for value in record.embedding)
    if len(values) != record.embedding_dimensions:
        raise PersistenceError(
            f"Chunk record {record.id!r} has inconsistent embedding dimensions"
        )
    return EmbeddedChunk(
        chunk=_code_chunk_from_record(record, relative_path, language),
        embedding=EmbeddingVector(values=values, model=record.embedding_model),
    )


def load_embedded_chunks(
    session: Session,
    repository_id: int,
) -> list[EmbeddedChunk]:
    """Reconstruct embedded domain chunks in deterministic source order."""

    _repository_or_raise(session, repository_id)
    rows = session.execute(
        select(
            CodeChunkRecord,
            RepositoryFileRecord.relative_path,
            RepositoryFileRecord.language,
        )
        .join(RepositoryFileRecord)
        .where(
            RepositoryFileRecord.repository_id == repository_id,
            CodeChunkRecord.embedding.is_not(None),
        )
        .order_by(
            RepositoryFileRecord.relative_path,
            CodeChunkRecord.start_line,
            CodeChunkRecord.chunk_index,
            CodeChunkRecord.id,
        )
    ).all()
    return [
        _embedded_chunk_from_record(record, relative_path, language)
        for record, relative_path, language in rows
    ]


def load_chunks(
    session: Session,
    repository_id: int,
) -> list[CodeChunk]:
    """Reconstruct every persisted chunk without loading embedding vectors."""

    _repository_or_raise(session, repository_id)
    rows = session.execute(
        select(
            RepositoryFileRecord.relative_path,
            RepositoryFileRecord.language,
            CodeChunkRecord.chunk_index,
            CodeChunkRecord.start_line,
            CodeChunkRecord.end_line,
            CodeChunkRecord.content,
            CodeChunkRecord.chunking_strategy,
            CodeChunkRecord.chunk_kind,
            CodeChunkRecord.symbol_name,
            CodeChunkRecord.qualified_symbol_name,
            CodeChunkRecord.parent_symbol,
            CodeChunkRecord.fragment_index,
            CodeChunkRecord.fragment_count,
        )
        .join(RepositoryFileRecord)
        .where(RepositoryFileRecord.repository_id == repository_id)
        .order_by(
            RepositoryFileRecord.relative_path,
            CodeChunkRecord.start_line,
            CodeChunkRecord.chunk_index,
            CodeChunkRecord.id,
        )
    ).all()
    return [
        CodeChunk(
            relative_path=relative_path,
            language=language,
            start_line=start_line,
            end_line=end_line,
            content=content,
            chunk_index=chunk_index,
            chunking_strategy=chunking_strategy,
            chunk_kind=chunk_kind,
            symbol_name=symbol_name,
            qualified_symbol_name=qualified_symbol_name,
            parent_symbol=parent_symbol,
            fragment_index=fragment_index,
            fragment_count=fragment_count,
        )
        for (
            relative_path,
            language,
            chunk_index,
            start_line,
            end_line,
            content,
            chunking_strategy,
            chunk_kind,
            symbol_name,
            qualified_symbol_name,
            parent_symbol,
            fragment_index,
            fragment_count,
        ) in rows
    ]


def load_neighbor_chunks(
    session: Session,
    repository_id: int,
    keys: Sequence[tuple[str, int]],
) -> list[CodeChunk]:
    """Batch-load specific (relative_path, chunk_index) chunks for one repository.

    Uses a single row-value ``IN`` query regardless of how many keys are
    requested, avoiding one query per neighbor.
    """

    _repository_or_raise(session, repository_id)
    unique_keys = sorted(set(keys))
    if not unique_keys:
        return []

    rows = session.execute(
        select(
            RepositoryFileRecord.relative_path,
            RepositoryFileRecord.language,
            CodeChunkRecord.chunk_index,
            CodeChunkRecord.start_line,
            CodeChunkRecord.end_line,
            CodeChunkRecord.content,
            CodeChunkRecord.chunking_strategy,
            CodeChunkRecord.chunk_kind,
            CodeChunkRecord.symbol_name,
            CodeChunkRecord.qualified_symbol_name,
            CodeChunkRecord.parent_symbol,
            CodeChunkRecord.fragment_index,
            CodeChunkRecord.fragment_count,
        )
        .join(RepositoryFileRecord)
        .where(
            RepositoryFileRecord.repository_id == repository_id,
            tuple_(RepositoryFileRecord.relative_path, CodeChunkRecord.chunk_index).in_(
                unique_keys
            ),
        )
        .order_by(
            RepositoryFileRecord.relative_path,
            CodeChunkRecord.chunk_index,
        )
    ).all()
    return [
        CodeChunk(
            relative_path=relative_path,
            language=language,
            start_line=start_line,
            end_line=end_line,
            content=content,
            chunk_index=chunk_index,
            chunking_strategy=chunking_strategy,
            chunk_kind=chunk_kind,
            symbol_name=symbol_name,
            qualified_symbol_name=qualified_symbol_name,
            parent_symbol=parent_symbol,
            fragment_index=fragment_index,
            fragment_count=fragment_count,
        )
        for (
            relative_path,
            language,
            chunk_index,
            start_line,
            end_line,
            content,
            chunking_strategy,
            chunk_kind,
            symbol_name,
            qualified_symbol_name,
            parent_symbol,
            fragment_index,
            fragment_count,
        ) in rows
    ]


_SYMBOL_FRAGMENT_OVERFETCH = 4


def find_symbol_candidates(
    session: Session,
    repository_id: int,
    candidates: Sequence[IdentifierCandidate],
    *,
    limit: int,
) -> list[SymbolSearchResult]:
    """Match persisted structural symbol metadata for one repository.

    One bounded, tier-ordered SQL query, never a full-repository chunk scan:
    the tier is computed in SQL with ``CASE`` and only the matching rows are
    fetched, capped by ``LIMIT``. Multiple fragments of one symbol in one
    file collapse to their first fragment in Python (over a result set
    already bounded by the SQL ``LIMIT``, not the repository size) so one
    oversized function cannot consume the whole candidate budget.
    """

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
        or limit > MAX_SYMBOL_CANDIDATE_LIMIT
    ):
        raise SymbolSearchError(
            f"limit must be a positive integer no greater than {MAX_SYMBOL_CANDIDATE_LIMIT}"
        )
    _repository_or_raise(session, repository_id)

    qualified_texts = sorted({c.text for c in candidates if c.qualified})
    strong_texts = sorted(
        {c.text for c in candidates if not c.qualified and c.confidence is IdentifierConfidence.STRONG}
    )
    weak_texts = sorted(
        {c.text for c in candidates if not c.qualified and c.confidence is IdentifierConfidence.WEAK}
    )
    if not qualified_texts and not strong_texts and not weak_texts:
        return []

    tier_expression = case(
        (CodeChunkRecord.qualified_symbol_name.in_(qualified_texts), 0),
        (CodeChunkRecord.symbol_name.in_(strong_texts), 1),
        (CodeChunkRecord.symbol_name.in_(weak_texts), 2),
        else_=None,
    ).label("tier")
    match_condition = or_(
        CodeChunkRecord.qualified_symbol_name.in_(qualified_texts) if qualified_texts else false(),
        CodeChunkRecord.symbol_name.in_(strong_texts) if strong_texts else false(),
        CodeChunkRecord.symbol_name.in_(weak_texts) if weak_texts else false(),
    )

    rows = session.execute(
        select(
            RepositoryFileRecord.relative_path,
            RepositoryFileRecord.language,
            CodeChunkRecord.chunk_index,
            CodeChunkRecord.start_line,
            CodeChunkRecord.end_line,
            CodeChunkRecord.content,
            CodeChunkRecord.chunking_strategy,
            CodeChunkRecord.chunk_kind,
            CodeChunkRecord.symbol_name,
            CodeChunkRecord.qualified_symbol_name,
            CodeChunkRecord.parent_symbol,
            CodeChunkRecord.fragment_index,
            CodeChunkRecord.fragment_count,
            tier_expression,
        )
        .join(RepositoryFileRecord)
        .where(RepositoryFileRecord.repository_id == repository_id, match_condition)
        .order_by(
            tier_expression,
            RepositoryFileRecord.relative_path,
            CodeChunkRecord.chunk_index,
        )
        .limit(limit * _SYMBOL_FRAGMENT_OVERFETCH)
    ).all()

    tiers = (
        SymbolMatchTier.QUALIFIED_SYMBOL,
        SymbolMatchTier.SIMPLE_SYMBOL_STRONG,
        SymbolMatchTier.SIMPLE_SYMBOL_WEAK,
    )
    seen_symbols: set[tuple[str, str]] = set()
    results: list[SymbolSearchResult] = []
    for row in rows:
        (
            relative_path,
            language,
            chunk_index,
            start_line,
            end_line,
            content,
            chunking_strategy,
            chunk_kind,
            symbol_name,
            qualified_symbol_name,
            parent_symbol,
            fragment_index,
            fragment_count,
            tier_value,
        ) = row
        if tier_value is None or len(results) >= limit:
            continue
        dedup_key = (relative_path, qualified_symbol_name or symbol_name or "")
        if dedup_key in seen_symbols:
            continue
        seen_symbols.add(dedup_key)
        results.append(
            SymbolSearchResult(
                chunk=CodeChunk(
                    relative_path=relative_path,
                    language=language,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    chunk_index=chunk_index,
                    chunking_strategy=chunking_strategy,
                    chunk_kind=chunk_kind,
                    symbol_name=symbol_name,
                    qualified_symbol_name=qualified_symbol_name,
                    parent_symbol=parent_symbol,
                    fragment_index=fragment_index,
                    fragment_count=fragment_count,
                ),
                rank=len(results) + 1,
                match_tier=tiers[tier_value],
                matched_identifier=qualified_symbol_name or symbol_name,
            )
        )
    return results


def pgvector_semantic_search(
    session: Session,
    repository_id: int,
    query_embedding: EmbeddingVector,
    *,
    top_k: int = 5,
    mode: SemanticSearchMode = SemanticSearchMode.EXACT,
) -> list[SemanticSearchResult]:
    """Run exact or HNSW cosine search within one repository and model."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise SemanticSearchError("top_k must be a positive integer")
    try:
        resolved_mode = SemanticSearchMode(mode)
    except ValueError as exc:
        raise SemanticSearchError("mode must be 'exact' or 'ann'") from exc
    _repository_or_raise(session, repository_id)
    try:
        cosine_similarity(query_embedding.values, query_embedding.values)
    except SimilarityError as exc:
        raise PersistenceError("Query embedding is unusable for cosine search") from exc

    dimensions = session.scalars(
        select(distinct(CodeChunkRecord.embedding_dimensions))
        .join(RepositoryFileRecord)
        .where(
            RepositoryFileRecord.repository_id == repository_id,
            CodeChunkRecord.embedding.is_not(None),
            CodeChunkRecord.embedding_model == query_embedding.model,
            CodeChunkRecord.embedding_dimensions.is_not(None),
        )
    ).all()
    if not dimensions:
        return []
    if len(dimensions) != 1:
        raise PersistenceError(
            f"Stored embeddings for model {query_embedding.model!r} have mixed dimensions"
        )
    stored_dimensions = dimensions[0]
    if stored_dimensions != query_embedding.dimensions:
        raise PersistenceError(
            f"Query embedding has {query_embedding.dimensions} dimensions, but stored "
            f"model {query_embedding.model!r} uses {stored_dimensions}"
        )

    if resolved_mode is SemanticSearchMode.ANN:
        if query_embedding.dimensions != HNSW_EMBEDDING_DIMENSIONS:
            raise PersistenceError(
                "ANN search requires "
                f"{HNSW_EMBEDDING_DIMENSIONS}-dimensional embeddings; use exact mode "
                f"for {query_embedding.dimensions}-dimensional vectors"
            )
        # PostgreSQL 17 + pgvector 0.8.x can continue filtered HNSW scans until
        # enough repository/model matches are found. SET LOCAL never changes the
        # database-wide setting and expires with the current transaction.
        session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        indexed_embedding = cast(
            CodeChunkRecord.embedding,
            Vector(HNSW_EMBEDDING_DIMENSIONS),
        )
        distance = indexed_embedding.cosine_distance(
            list(query_embedding.values)
        ).label("cosine_distance")
    else:
        # The uncast expression intentionally cannot match the fixed-dimension
        # expression index, preserving an exact pgvector baseline.
        distance = CodeChunkRecord.embedding.cosine_distance(
            list(query_embedding.values)
        ).label("cosine_distance")
    rows = session.execute(
        select(
            CodeChunkRecord,
            RepositoryFileRecord.relative_path,
            RepositoryFileRecord.language,
            distance,
        )
        .join(RepositoryFileRecord)
        .where(
            RepositoryFileRecord.repository_id == repository_id,
            CodeChunkRecord.embedding.is_not(None),
            CodeChunkRecord.embedding_model == query_embedding.model,
            CodeChunkRecord.embedding_dimensions == query_embedding.dimensions,
        )
        .order_by(
            distance,
            RepositoryFileRecord.relative_path,
            CodeChunkRecord.start_line,
            CodeChunkRecord.chunk_index,
            CodeChunkRecord.id,
        )
        .limit(top_k)
    ).all()

    results: list[SemanticSearchResult] = []
    for rank, (record, relative_path, language, cosine_distance) in enumerate(
        rows,
        start=1,
    ):
        score = max(-1.0, min(1.0, 1.0 - float(cosine_distance)))
        results.append(
            SemanticSearchResult(
                chunk=_code_chunk_from_record(record, relative_path, language),
                score=score,
                rank=rank,
            )
        )
    return results
