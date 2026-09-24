"""Real-PostgreSQL proof that unchanged rows and vectors survive a re-index."""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from repomind.db import (
    apply_index_update,
    persist_chunks,
    persist_embedded_chunks,
    persist_repository_snapshot,
    read_index_manifest,
)
from repomind.db.models import CodeChunkRecord, RepositoryFileRecord, RepositoryRecord
from repomind.db.repositories import pgvector_semantic_search
from repomind.indexing import (
    IndexUpdate,
    IndexUpdateError,
    classify_files,
    index_fingerprint,
)
from repomind.ingestion import ChunkingConfig, CodeChunk, RepositorySnapshot, SourceFile
from repomind.retrieval import EmbeddedChunk, EmbeddingTextStrategy, EmbeddingVector

pytestmark = pytest.mark.postgres

_FINGERPRINT = index_fingerprint(
    chunking=ChunkingConfig(),
    embedding_model="model-a",
    embedding_text_strategy=EmbeddingTextStrategy.RAW_SOURCE,
)


def _source(path: str, content: str) -> SourceFile:
    return SourceFile(
        relative_path=path,
        language="python",
        content=content,
        size_bytes=len(content.encode("utf-8")),
        line_count=len(content.splitlines()),
    )


def _snapshot(name: str, *sources: SourceFile) -> RepositorySnapshot:
    return RepositorySnapshot(
        root=Path("C:/private/local/repository"),
        name=name,
        files=list(sources),
        skipped=[],
        file_count=len(sources),
        total_size_bytes=sum(source.size_bytes for source in sources),
        languages={"python": len(sources)},
    )


def _embedded(source: SourceFile, values: tuple[float, ...]) -> EmbeddedChunk:
    chunk = CodeChunk(
        relative_path=source.relative_path,
        language="python",
        start_line=1,
        end_line=max(1, source.line_count),
        content=source.content,
        chunk_index=0,
    )
    return EmbeddedChunk(chunk=chunk, embedding=EmbeddingVector(values=values, model="model-a"))


def _repository(session: Session) -> RepositoryRecord:
    record = RepositoryRecord(name=f"incremental-{uuid4().hex}")
    session.add(record)
    session.flush()
    return record


def _row_ids(session: Session, repository_id: int, path: str) -> tuple[int, list[int]]:
    file_record = session.scalar(
        select(RepositoryFileRecord).where(
            RepositoryFileRecord.repository_id == repository_id,
            RepositoryFileRecord.relative_path == path,
        )
    )
    assert file_record is not None, f"{path} is not persisted"
    chunk_ids = sorted(chunk.id for chunk in file_record.chunks)
    return file_record.id, chunk_ids


def _paths(session: Session, repository_id: int) -> set[str]:
    return set(
        session.scalars(
            select(RepositoryFileRecord.relative_path).where(
                RepositoryFileRecord.repository_id == repository_id
            )
        )
    )


def _initial_index(session: Session, record: RepositoryRecord) -> RepositorySnapshot:
    a, b, doomed = (
        _source("a.py", "def a():\n    return 1\n"),
        _source("b.py", "def b():\n    return 2\n"),
        _source("doomed.py", "def doomed():\n    return 3\n"),
    )
    snapshot = _snapshot(record.name, a, b, doomed)
    apply_index_update(
        session,
        record.id,
        IndexUpdate(
            fingerprint=_FINGERPRINT,
            snapshot=snapshot,
            upserted=("a.py", "b.py", "doomed.py"),
            deleted=(),
            chunks=[
                _embedded(a, (1.0, 0.0, 0.0)),
                _embedded(b, (0.0, 1.0, 0.0)),
                _embedded(doomed, (0.0, 0.0, 1.0)),
            ],
            full_rebuild=True,
        ),
    )
    return snapshot


def test_unchanged_file_keeps_its_rows_while_changed_added_deleted_apply(db_session: Session):
    record = _repository(db_session)
    _initial_index(db_session, record)
    a_file_id, a_chunk_ids = _row_ids(db_session, record.id, "a.py")
    b_file_id, _ = _row_ids(db_session, record.id, "b.py")

    changed_b = _source("b.py", "def b():\n    return 222\n")
    added_c = _source("c.py", "def c():\n    return 4\n")
    unchanged_a = _source("a.py", "def a():\n    return 1\n")
    snapshot = _snapshot(record.name, unchanged_a, changed_b, added_c)

    manifest = read_index_manifest(db_session, record.id)
    assert manifest.fingerprint == _FINGERPRINT
    plan = classify_files(
        current={
            "a.py": manifest.files["a.py"].content_hash,
            "b.py": "a-different-hash",
            "c.py": "new-hash",
        },
        manifest=manifest,
        compatible=True,
    )
    assert plan.unchanged == ("a.py",)
    assert plan.changed == ("b.py",)
    assert plan.added == ("c.py",)
    assert plan.deleted == ("doomed.py",)

    summary = apply_index_update(
        db_session,
        record.id,
        IndexUpdate(
            fingerprint=_FINGERPRINT,
            snapshot=snapshot,
            upserted=plan.reindexed,
            deleted=plan.deleted,
            chunks=[_embedded(changed_b, (0.0, 1.0, 0.0)), _embedded(added_c, (0.0, 0.0, 1.0))],
            full_rebuild=False,
        ),
    )

    # The unchanged file keeps stable database identity: its row and chunk
    # rows were never deleted and recreated.
    assert _row_ids(db_session, record.id, "a.py") == (a_file_id, a_chunk_ids)
    assert _row_ids(db_session, record.id, "b.py")[0] != b_file_id
    assert _paths(db_session, record.id) == {"a.py", "b.py", "c.py"}
    assert summary.file_count == 3
    assert summary.chunk_count == 3
    assert summary.embedding_model == "model-a"


def test_reused_vector_remains_searchable_and_deleted_source_disappears(db_session: Session):
    record = _repository(db_session)
    _initial_index(db_session, record)

    unchanged_a = _source("a.py", "def a():\n    return 1\n")
    snapshot = _snapshot(record.name, unchanged_a)
    apply_index_update(
        db_session,
        record.id,
        IndexUpdate(
            fingerprint=_FINGERPRINT,
            snapshot=snapshot,
            upserted=(),
            deleted=("b.py", "doomed.py"),
            chunks=[],
            full_rebuild=False,
        ),
    )

    # a.py was never re-embedded this run, yet its persisted vector still
    # answers retrieval; the deleted files are gone from the corpus.
    results = pgvector_semantic_search(
        db_session,
        record.id,
        EmbeddingVector(values=(1.0, 0.0, 0.0), model="model-a"),
        top_k=5,
    )
    assert [result.chunk.relative_path.as_posix() for result in results] == ["a.py"]
    assert _paths(db_session, record.id) == {"a.py"}
    assert db_session.scalar(
        select(CodeChunkRecord)
        .join(RepositoryFileRecord)
        .where(
            RepositoryFileRecord.repository_id == record.id,
            RepositoryFileRecord.relative_path == "doomed.py",
        )
    ) is None


def test_manifest_reports_hashes_chunk_counts_and_legacy_null_fingerprint(db_session: Session):
    record = _repository(db_session)
    snapshot = _initial_index(db_session, record)

    manifest = read_index_manifest(db_session, record.id)

    assert set(manifest.files) == {"a.py", "b.py", "doomed.py"}
    assert all(state.chunk_count == 1 for state in manifest.files.values())
    stored = {source.relative_path.as_posix(): source.content for source in snapshot.files}
    for path, state in manifest.files.items():
        from repomind.db import content_sha256

        assert state.content_hash == content_sha256(stored[path])

    # A repository indexed before fingerprints existed is never compatible.
    record.index_fingerprint = None
    db_session.flush()
    legacy = read_index_manifest(db_session, record.id)
    assert legacy.fingerprint is None
    assert legacy.is_compatible_with(_FINGERPRINT) is False


def test_full_rebuild_discards_every_prior_row(db_session: Session):
    record = _repository(db_session)
    _initial_index(db_session, record)
    original_file_id, _ = _row_ids(db_session, record.id, "a.py")

    rebuilt_a = _source("a.py", "def a():\n    return 1\n")
    snapshot = _snapshot(record.name, rebuilt_a)
    summary = apply_index_update(
        db_session,
        record.id,
        IndexUpdate(
            fingerprint="b" * 64,
            snapshot=snapshot,
            upserted=("a.py",),
            deleted=(),
            chunks=[_embedded(rebuilt_a, (1.0, 0.0, 0.0))],
            full_rebuild=True,
        ),
    )

    # Even though only a.py was upserted, the rebuild removed b.py/doomed.py
    # because nothing built under the old configuration may survive.
    assert _paths(db_session, record.id) == {"a.py"}
    assert _row_ids(db_session, record.id, "a.py")[0] != original_file_id
    assert summary.file_count == 1 and summary.chunk_count == 1
    assert read_index_manifest(db_session, record.id).fingerprint == "b" * 64


def test_update_rejects_upserting_a_path_absent_from_the_snapshot(db_session: Session):
    record = _repository(db_session)
    _initial_index(db_session, record)

    with pytest.raises(IndexUpdateError, match="absent from the snapshot"):
        apply_index_update(
            db_session,
            record.id,
            IndexUpdate(
                fingerprint=_FINGERPRINT,
                snapshot=_snapshot(record.name, _source("a.py", "def a():\n    return 1\n")),
                upserted=("ghost.py",),
                deleted=(),
                chunks=[],
                full_rebuild=False,
            ),
        )


def _current_sources() -> tuple[SourceFile, ...]:
    return (
        _source("a.py", "def a():\n    return 1\n"),
        _source("b.py", "def b():\n    return 2\n"),
        _source("doomed.py", "def doomed():\n    return 3\n"),
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda name, sources: IndexUpdate(
                fingerprint=_FINGERPRINT,
                snapshot=_snapshot(name, _source("a.py", "def a():\n    return CHANGED\n"), *sources[1:]),
                upserted=(),
                deleted=(),
                chunks=[],
                full_rebuild=False,
            ),
            id="changed-file-omitted-from-upserted",
        ),
        pytest.param(
            lambda name, sources: IndexUpdate(
                fingerprint=_FINGERPRINT,
                snapshot=_snapshot(name, *sources[:2]),
                upserted=(),
                deleted=(),
                chunks=[],
                full_rebuild=False,
            ),
            id="deleted-file-omitted-from-deleted",
        ),
        pytest.param(
            lambda name, sources: IndexUpdate(
                fingerprint=_FINGERPRINT,
                snapshot=_snapshot(name, *sources, _source("new.py", "def new():\n    return 4\n")),
                upserted=(),
                deleted=(),
                chunks=[],
                full_rebuild=False,
            ),
            id="new-file-omitted-from-upserted",
        ),
        pytest.param(
            lambda name, sources: IndexUpdate(
                fingerprint=_FINGERPRINT,
                snapshot=_snapshot(name, *sources),
                upserted=(),
                deleted=(),
                chunks=[_embedded(sources[0], (1.0, 0.0, 0.0))],
                full_rebuild=False,
            ),
            id="embedded-chunk-for-non-upserted-path",
        ),
        pytest.param(
            lambda name, sources: IndexUpdate(
                fingerprint=_FINGERPRINT,
                snapshot=_snapshot(name, *sources),
                upserted=("a.py",),
                deleted=(),
                chunks=[_embedded(sources[0], (1.0, 0.0, 0.0))],
                full_rebuild=True,
            ),
            id="incomplete-full-rebuild-upsert-list",
        ),
    ],
)
def test_malformed_updates_are_rejected_before_any_mutation(db_session: Session, build):
    record = _repository(db_session)
    _initial_index(db_session, record)
    before_paths = _paths(db_session, record.id)
    before_rows = {path: _row_ids(db_session, record.id, path) for path in before_paths}
    before_fingerprint = read_index_manifest(db_session, record.id).fingerprint

    with pytest.raises(IndexUpdateError):
        apply_index_update(db_session, record.id, build(record.name, _current_sources()))

    # Nothing was removed, added, or restamped.
    assert _paths(db_session, record.id) == before_paths
    assert {path: _row_ids(db_session, record.id, path) for path in before_paths} == before_rows
    assert read_index_manifest(db_session, record.id).fingerprint == before_fingerprint


# --------------------------------------------------------------------------
# provenance gates incremental reuse
# --------------------------------------------------------------------------


def _unchanged_incremental(name: str, fingerprint: str) -> IndexUpdate:
    """A structurally perfect no-op delta carrying the given provenance."""

    return IndexUpdate(
        fingerprint=fingerprint,
        snapshot=_snapshot(name, *_current_sources()),
        upserted=(),
        deleted=(),
        chunks=[],
        full_rebuild=False,
    )


@pytest.mark.parametrize("prior", ["a" * 64, None], ids=["mismatched", "missing"])
def test_incremental_update_requires_matching_persisted_provenance(
    db_session: Session, prior
):
    record = _repository(db_session)
    _initial_index(db_session, record)
    if prior is None:
        record.index_fingerprint = None
    else:
        record.index_fingerprint = prior
    db_session.flush()

    before_paths = _paths(db_session, record.id)
    before_rows = {path: _row_ids(db_session, record.id, path) for path in before_paths}

    # Byte-identical sources, nothing to reindex - but the persisted rows were
    # not produced under _FINGERPRINT, so they may not be reused.
    with pytest.raises(IndexUpdateError, match="requires a persisted fingerprint"):
        apply_index_update(
            db_session, record.id, _unchanged_incremental(record.name, _FINGERPRINT)
        )

    assert _paths(db_session, record.id) == before_paths
    assert {p: _row_ids(db_session, record.id, p) for p in before_paths} == before_rows
    assert read_index_manifest(db_session, record.id).fingerprint == prior


@pytest.mark.parametrize("prior", ["a" * 64, None], ids=["mismatched", "missing"])
def test_full_rebuild_succeeds_across_incompatible_provenance(db_session: Session, prior):
    record = _repository(db_session)
    _initial_index(db_session, record)
    record.index_fingerprint = prior
    db_session.flush()
    original_file_id, _ = _row_ids(db_session, record.id, "a.py")

    sources = _current_sources()
    summary = apply_index_update(
        db_session,
        record.id,
        IndexUpdate(
            fingerprint=_FINGERPRINT,
            snapshot=_snapshot(record.name, *sources),
            upserted=tuple(s.relative_path.as_posix() for s in sources),
            deleted=(),
            chunks=[_embedded(s, (1.0, 0.0, 0.0)) for s in sources],
            full_rebuild=True,
        ),
    )

    # A rebuild is exactly how incompatible provenance is resolved.
    assert read_index_manifest(db_session, record.id).fingerprint == _FINGERPRINT
    assert summary.file_count == len(sources)
    assert _row_ids(db_session, record.id, "a.py")[0] != original_file_id


def test_matching_provenance_still_permits_incremental_reuse(db_session: Session):
    record = _repository(db_session)
    _initial_index(db_session, record)
    before_rows = _row_ids(db_session, record.id, "a.py")

    summary = apply_index_update(
        db_session, record.id, _unchanged_incremental(record.name, _FINGERPRINT)
    )

    # The provenance gate must not block a legitimate no-op reuse.
    assert _row_ids(db_session, record.id, "a.py") == before_rows
    assert summary.file_count == 3 and summary.chunk_count == 3
    assert read_index_manifest(db_session, record.id).fingerprint == _FINGERPRINT


# --------------------------------------------------------------------------
# legacy persistence invalidates reuse authorization
# --------------------------------------------------------------------------


def test_persist_repository_snapshot_clears_the_fingerprint(db_session: Session):
    record = _repository(db_session)
    _initial_index(db_session, record)
    assert read_index_manifest(db_session, record.id).fingerprint == _FINGERPRINT

    persist_repository_snapshot(
        db_session, _snapshot(record.name, _source("a.py", "def a():\n    return 1\n"))
    )

    # The snapshot was replaced without proving which configuration built it,
    # so incremental reuse is no longer authorized.
    assert read_index_manifest(db_session, record.id).fingerprint is None


def test_persist_chunks_clears_the_fingerprint(db_session: Session):
    record = _repository(db_session)
    _initial_index(db_session, record)

    persist_chunks(
        db_session,
        record.id,
        [
            CodeChunk(
                relative_path="a.py",
                language="python",
                start_line=1,
                end_line=2,
                content="def a():\n    return 1\n",
                chunk_index=0,
            )
        ],
    )

    assert read_index_manifest(db_session, record.id).fingerprint is None


def test_persist_embedded_chunks_clears_the_fingerprint(db_session: Session):
    record = _repository(db_session)
    _initial_index(db_session, record)

    persist_embedded_chunks(
        db_session,
        record.id,
        [_embedded(_source("a.py", "def a():\n    return 1\n"), (1.0, 0.0, 0.0))],
    )

    assert read_index_manifest(db_session, record.id).fingerprint is None


def test_legacy_mutation_forces_the_next_run_to_rebuild(db_session: Session):
    record = _repository(db_session)
    _initial_index(db_session, record)
    persist_repository_snapshot(
        db_session, _snapshot(record.name, _source("a.py", "def a():\n    return 1\n"))
    )

    manifest = read_index_manifest(db_session, record.id)

    # Even though a.py's content is byte-identical, the missing provenance
    # means nothing may be reused.
    plan = classify_files(
        current={"a.py": manifest.files["a.py"].content_hash},
        manifest=manifest,
        compatible=manifest.is_compatible_with(_FINGERPRINT),
    )
    assert plan.full_rebuild is True
    assert plan.unchanged == ()
    assert plan.added == ("a.py",)
