"""File-granularity incremental indexing: reuse, deltas, and safe fallback."""

import pytest

from repomind import indexing
from repomind.indexing import (
    INDEX_FINGERPRINT_VERSION,
    IndexedFileState,
    IndexManifest,
    IndexUpdate,
    IndexUpdateError,
    classify_files,
    index_fingerprint,
    validate_index_update,
)
from repomind.ingestion import ChunkingConfig, ChunkingStrategy
from repomind.retrieval import EmbeddingTextStrategy


def _index(api, repository_id: int = 1):
    response = api.client.post(f"/api/v1/repositories/{repository_id}/index")
    assert response.status_code == 200, response.text
    return response.json()


def _write(api, name: str, body: str) -> None:
    (api.repo / name).write_text(body, encoding="utf-8", newline="\n")


def _paths(api) -> set[str]:
    return set(api.store.indexes[1])


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------


def test_fingerprint_is_deterministic_across_calls():
    kwargs = {
        "chunking": ChunkingConfig(),
        "embedding_model": "text-embedding-3-small",
        "embedding_text_strategy": EmbeddingTextStrategy.RAW_SOURCE,
    }

    first = index_fingerprint(**kwargs)

    assert first == index_fingerprint(**kwargs)
    assert len(first) == 64 and first == first.lower()


@pytest.mark.parametrize(
    "override",
    [
        {"chunking": ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL, overlap_lines=0)},
        {"chunking": ChunkingConfig(max_lines_per_chunk=80)},
        {"chunking": ChunkingConfig(overlap_lines=5)},
        {"chunking": ChunkingConfig(max_chars_per_chunk=9_000)},
        {"embedding_model": "text-embedding-3-large"},
        {"embedding_text_strategy": EmbeddingTextStrategy.STRUCTURAL_CONTEXT},
    ],
)
def test_every_indexing_relevant_setting_changes_the_fingerprint(override):
    baseline = {
        "chunking": ChunkingConfig(),
        "embedding_model": "text-embedding-3-small",
        "embedding_text_strategy": EmbeddingTextStrategy.RAW_SOURCE,
    }

    assert index_fingerprint(**{**baseline, **override}) != index_fingerprint(**baseline)


def test_operational_only_settings_do_not_invalidate_compatibility():
    """Batch size changes throughput, not what a stored vector means."""

    from repomind.retrieval import EmbeddingConfig

    assert EmbeddingConfig(batch_size=8).text_strategy == EmbeddingConfig(
        batch_size=512
    ).text_strategy
    # Batch size is not an input at all, so it cannot perturb the digest.
    assert index_fingerprint(
        chunking=ChunkingConfig(),
        embedding_model="m",
        embedding_text_strategy=EmbeddingTextStrategy.RAW_SOURCE,
    ) == index_fingerprint(
        chunking=ChunkingConfig(),
        embedding_model="m",
        embedding_text_strategy=EmbeddingTextStrategy.RAW_SOURCE,
    )


def test_fingerprint_version_participates_in_the_digest(monkeypatch):
    """Bumping the version must invalidate every persisted fingerprint."""

    kwargs = {
        "chunking": ChunkingConfig(),
        "embedding_model": "text-embedding-3-small",
        "embedding_text_strategy": EmbeddingTextStrategy.RAW_SOURCE,
    }
    at_current_version = index_fingerprint(**kwargs)

    monkeypatch.setattr(
        indexing, "INDEX_FINGERPRINT_VERSION", INDEX_FINGERPRINT_VERSION + 1
    )
    at_next_version = index_fingerprint(**kwargs)

    assert at_next_version != at_current_version
    assert len(at_next_version) == 64


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def test_classification_splits_unchanged_changed_added_and_deleted():
    manifest = IndexManifest(
        fingerprint="f",
        files={
            "keep.py": IndexedFileState(content_hash="a", chunk_count=2),
            "edit.py": IndexedFileState(content_hash="b", chunk_count=1),
            "gone.py": IndexedFileState(content_hash="c", chunk_count=3),
        },
    )

    plan = classify_files(
        current={"keep.py": "a", "edit.py": "CHANGED", "new.py": "d"},
        manifest=manifest,
        compatible=True,
    )

    assert plan.unchanged == ("keep.py",)
    assert plan.changed == ("edit.py",)
    assert plan.added == ("new.py",)
    assert plan.deleted == ("gone.py",)
    assert plan.reindexed == ("edit.py", "new.py")
    assert plan.full_rebuild is False


def test_incompatible_manifest_classifies_everything_as_a_rebuild():
    manifest = IndexManifest(
        fingerprint="old", files={"a.py": IndexedFileState(content_hash="x", chunk_count=1)}
    )

    plan = classify_files(current={"a.py": "x"}, manifest=manifest, compatible=False)

    assert plan.full_rebuild is True
    assert plan.added == ("a.py",)
    assert plan.unchanged == ()


def test_legacy_manifest_without_fingerprint_is_never_compatible():
    manifest = IndexManifest(files={"a.py": IndexedFileState(content_hash="x", chunk_count=1)})

    assert manifest.fingerprint is None
    assert manifest.is_compatible_with("anything") is False


# --------------------------------------------------------------------------
# end-to-end indexing behavior
# --------------------------------------------------------------------------


def test_first_index_builds_the_complete_index(api):
    body = _index(api)

    assert body == {
        "repository_id": 1,
        "files_indexed": 1,
        "chunks_indexed": 1,
        "embedding_model": "offline-model",
    }
    assert api.embeddings.embedded_paths == ["app.py"]


def test_unchanged_reindex_embeds_nothing_and_reuses_persisted_chunks(api):
    first = _index(api)
    reused = api.store.chunks[1][0]
    api.embeddings.embedded_batches.clear()

    second = _index(api)

    assert api.embeddings.embedded_paths == []
    assert second == first
    # Identity proves the persisted chunk/vector was reused, not rebuilt.
    assert api.store.chunks[1][0] is reused


def test_changing_one_file_embeds_only_that_file(api):
    _write(api, "other.py", "def other():\n    return 2\n")
    _index(api)
    untouched = {
        path: api.store.indexes[1][path]["chunks"][0] for path in ("app.py", "other.py")
    }
    api.embeddings.embedded_batches.clear()

    _write(api, "other.py", "def other():\n    return 99\n")
    body = _index(api)

    assert api.embeddings.embedded_paths == ["other.py"]
    assert api.store.indexes[1]["app.py"]["chunks"][0] is untouched["app.py"]
    assert api.store.indexes[1]["other.py"]["chunks"][0] is not untouched["other.py"]
    assert body["files_indexed"] == 2


def test_adding_one_file_embeds_only_the_new_file(api):
    _index(api)
    original = api.store.indexes[1]["app.py"]["chunks"][0]
    api.embeddings.embedded_batches.clear()

    _write(api, "added.py", "def added():\n    return 3\n")
    body = _index(api)

    assert api.embeddings.embedded_paths == ["added.py"]
    assert api.store.indexes[1]["app.py"]["chunks"][0] is original
    assert body["files_indexed"] == 2
    assert _paths(api) == {"app.py", "added.py"}


def test_deleting_a_file_removes_it_without_embedding_survivors(api):
    _write(api, "doomed.py", "def doomed():\n    return 4\n")
    _index(api)
    survivor = api.store.indexes[1]["app.py"]["chunks"][0]
    api.embeddings.embedded_batches.clear()

    (api.repo / "doomed.py").unlink()
    body = _index(api)

    assert api.embeddings.embedded_paths == []
    assert _paths(api) == {"app.py"}
    assert api.store.indexes[1]["app.py"]["chunks"][0] is survivor
    assert body["files_indexed"] == 1


def test_mixed_add_change_delete_matches_a_full_rebuild_corpus(api):
    _write(api, "keep.py", "def keep():\n    return 1\n")
    _write(api, "edit.py", "def edit():\n    return 1\n")
    _write(api, "drop.py", "def drop():\n    return 1\n")
    _index(api)
    api.embeddings.embedded_batches.clear()

    _write(api, "edit.py", "def edit():\n    return 222\n")
    _write(api, "fresh.py", "def fresh():\n    return 1\n")
    (api.repo / "drop.py").unlink()
    incremental = _index(api)

    assert sorted(api.embeddings.embedded_paths) == ["edit.py", "fresh.py"]
    assert _paths(api) == {"app.py", "keep.py", "edit.py", "fresh.py"}

    # A forced rebuild of the same tree must yield the same logical corpus.
    api.store.fingerprints[1] = "incompatible-fingerprint"
    rebuilt = _index(api)
    assert rebuilt["files_indexed"] == incremental["files_indexed"]
    assert rebuilt["chunks_indexed"] == incremental["chunks_indexed"]
    assert _paths(api) == {"app.py", "keep.py", "edit.py", "fresh.py"}


def test_embedding_failure_while_preparing_a_delta_leaves_the_index_untouched(api, monkeypatch):
    _index(api)
    snapshot, chunks = api.store.snapshots[1], api.store.chunks[1]
    fingerprint = api.store.fingerprints[1]

    _write(api, "broken.py", "def broken():\n    return 1\n")
    monkeypatch.setattr(
        api.embeddings, "embed_chunks", lambda chunks: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert api.client.post("/api/v1/repositories/1/index").status_code == 500
    assert api.store.snapshots[1] is snapshot
    assert api.store.chunks[1] is chunks
    assert api.store.fingerprints[1] == fingerprint
    assert _paths(api) == {"app.py"}


def test_persistence_failure_rolls_back_the_whole_delta(api, monkeypatch):
    _index(api)
    before = dict(api.store.indexes[1])

    _write(api, "late.py", "def late():\n    return 1\n")
    monkeypatch.setattr(
        api.store,
        "apply_index_update",
        lambda repository_id, update: (_ for _ in ()).throw(RuntimeError("db down")),
    )

    assert api.client.post("/api/v1/repositories/1/index").status_code == 500
    assert api.store.indexes[1] == before


def test_empty_repository_stays_empty_without_embedding(api):
    (api.root / "blank").mkdir()
    repository_id = api.client.post(
        "/api/v1/repositories", json={"name": "blank", "path": "blank"}
    ).json()["id"]

    first = _index(api, repository_id)
    second = _index(api, repository_id)

    assert first == second
    assert first["files_indexed"] == first["chunks_indexed"] == 0
    assert first["embedding_model"] is None
    assert api.embeddings.embedded_paths == []


def test_non_empty_to_empty_clears_stale_data_without_embedding(api):
    _index(api)
    assert _paths(api) == {"app.py"}
    api.embeddings.embedded_batches.clear()

    (api.repo / "app.py").unlink()
    body = _index(api)

    assert api.embeddings.embedded_paths == []
    assert api.store.indexes[1] == {}
    assert body["files_indexed"] == body["chunks_indexed"] == 0
    assert body["embedding_model"] is None


# --------------------------------------------------------------------------
# compatibility fallback
# --------------------------------------------------------------------------


def test_changed_embedding_model_forces_a_full_rebuild(api):
    first = _index(api)
    original = api.store.indexes[1]["app.py"]["chunks"][0]
    assert first["embedding_model"] == "offline-model"
    api.embeddings.embedded_batches.clear()

    api.embeddings.model = "different-embedding-model"
    second = _index(api)

    # The file is genuinely re-embedded and the old vector is not reused.
    assert api.embeddings.embedded_paths == ["app.py"]
    assert api.store.indexes[1]["app.py"]["chunks"][0] is not original
    # The stored vector carries the NEW model, and the response agrees, so the
    # persisted fingerprint never claims a model the vectors contradict.
    assert api.store.chunks[1][0].embedding.model == "different-embedding-model"
    assert second["embedding_model"] == "different-embedding-model"

    # The rebuild established new provenance, so a third run reuses it.
    api.embeddings.embedded_batches.clear()
    third = _index(api)
    assert api.embeddings.embedded_paths == []
    assert third == second


def test_changed_embedding_text_strategy_forces_a_full_rebuild(api):
    _index(api)
    api.embeddings.embedded_batches.clear()

    api.embeddings.text_strategy = EmbeddingTextStrategy.STRUCTURAL_CONTEXT
    _index(api)

    assert api.embeddings.embedded_paths == ["app.py"]


def test_changed_chunking_config_forces_a_full_rebuild(api):
    _index(api)
    api.embeddings.embedded_batches.clear()

    service = api.app.state.container.get().repositories
    service.chunking_config = ChunkingConfig(
        strategy=ChunkingStrategy.STRUCTURAL, overlap_lines=0
    )
    _index(api)

    assert api.embeddings.embedded_paths == ["app.py"]


def test_legacy_index_without_fingerprint_rebuilds_once_then_reuses(api):
    _index(api)
    api.embeddings.embedded_batches.clear()
    # Simulate a repository indexed before fingerprints were persisted.
    api.store.fingerprints[1] = None

    _index(api)
    assert api.embeddings.embedded_paths == ["app.py"]

    api.embeddings.embedded_batches.clear()
    _index(api)
    assert api.embeddings.embedded_paths == []


def test_rebuild_discards_files_absent_from_the_current_snapshot(api):
    _write(api, "stale.py", "def stale():\n    return 1\n")
    _index(api)
    assert _paths(api) == {"app.py", "stale.py"}

    (api.repo / "stale.py").unlink()
    api.store.fingerprints[1] = "incompatible"
    body = _index(api)

    assert _paths(api) == {"app.py"}
    assert body["files_indexed"] == 1


# --------------------------------------------------------------------------
# limits, response contract, isolation, trace
# --------------------------------------------------------------------------


def test_max_chunks_counts_reused_chunks_not_only_new_ones(api):
    _write(api, "second.py", "def second():\n    return 1\n")
    _index(api)
    service = api.app.state.container.get().repositories
    before = dict(api.store.indexes[1])
    api.embeddings.embedded_batches.clear()

    # Two files are already indexed; adding a third must fail the total limit
    # even though only one new chunk would be embedded this run.
    service.MAX_CHUNKS = 2
    _write(api, "third.py", "def third():\n    return 1\n")

    assert api.client.post("/api/v1/repositories/1/index").status_code == 413
    assert api.embeddings.embedded_paths == []
    assert api.store.indexes[1] == before


def test_index_response_reports_totals_not_the_delta(api):
    _write(api, "extra.py", "def extra():\n    return 1\n")
    _index(api)
    api.embeddings.embedded_batches.clear()

    _write(api, "extra.py", "def extra():\n    return 7\n")
    body = _index(api)

    assert api.embeddings.embedded_paths == ["extra.py"]
    # Totals for the resulting index, not "changed this run".
    assert body["files_indexed"] == 2
    assert body["chunks_indexed"] == 2
    assert body["embedding_model"] == "offline-model"


def test_incremental_run_is_isolated_per_repository(api):
    other = api.root / "other"
    other.mkdir()
    (other / "mine.py").write_text("def mine():\n    return 1\n", encoding="utf-8")
    other_id = api.client.post(
        "/api/v1/repositories", json={"name": "other", "path": "other"}
    ).json()["id"]
    _index(api)
    _index(api, other_id)
    api.embeddings.embedded_batches.clear()

    _write(api, "app.py", "def value():\n    return 42\n")
    _index(api)

    assert api.embeddings.embedded_paths == ["app.py"]
    assert set(api.store.indexes[other_id]) == {"mine.py"}
    assert api.store.fingerprints[1] == api.store.fingerprints[other_id]


def test_trace_reports_incremental_counts_without_source_content(api):
    service = api.app.state.container.get().repositories
    _index(api)
    _write(api, "added.py", "def added():\n    return 1\n")
    _write(api, "app.py", "def value():\n    return 5\n")

    recorder = _Recorder()
    service.index(1, api.embeddings, trace=recorder)

    completed = recorder.events["persistence.completed"]
    assert completed["full_rebuild"] is False
    assert completed["files_added"] == 1
    assert completed["files_changed"] == 1
    assert completed["files_unchanged"] == 0
    assert completed["files_deleted"] == 0
    assert completed["chunks_embedded"] == 2
    assert completed["file_count"] == 2
    assert completed["chunk_count"] == 2
    payload = str(recorder.events)
    for secret in ("def value", "return 5", str(api.repo)):
        assert secret not in payload


# --------------------------------------------------------------------------
# malformed deltas must be rejected before any mutation
# --------------------------------------------------------------------------


def _snapshot_of(api, repository_id: int = 1):
    """Build the snapshot the service would produce for the current worktree."""

    from repomind.ingestion import ingest_repository

    root = api.repo if repository_id == 1 else api.root / "other"
    return ingest_repository(root).model_copy(update={"name": "sample"})


def _manifest(api, repository_id: int = 1):
    return api.store.index_manifest(repository_id)


def _malformed_update(api, **overrides):
    """An update that is structurally wrong but provenance-valid by default.

    The fingerprint defaults to the persisted one so these cases exercise the
    structural invariants rather than tripping the provenance gate first.
    """

    defaults = {
        "fingerprint": _manifest(api).fingerprint or "f" * 64,
        "snapshot": _snapshot_of(api),
        "upserted": (),
        "deleted": (),
        "chunks": [],
        "full_rebuild": False,
    }
    return IndexUpdate(**{**defaults, **overrides})


def test_changed_file_omitted_from_upserted_is_rejected(api):
    _index(api)
    _write(api, "app.py", "def value():\n    return 777\n")

    with pytest.raises(IndexUpdateError, match="would be reused without re-indexing"):
        validate_index_update(_malformed_update(api), manifest=_manifest(api))


def test_deleted_file_omitted_from_deleted_is_rejected(api):
    _write(api, "gone.py", "def gone():\n    return 1\n")
    _index(api)
    (api.repo / "gone.py").unlink()

    with pytest.raises(IndexUpdateError, match="left the snapshot but is not deleted"):
        validate_index_update(_malformed_update(api), manifest=_manifest(api))


def test_new_snapshot_file_omitted_from_upserted_is_rejected(api):
    _index(api)
    _write(api, "fresh.py", "def fresh():\n    return 1\n")

    with pytest.raises(IndexUpdateError, match="not upserted and is not already indexed"):
        validate_index_update(_malformed_update(api), manifest=_manifest(api))


def test_embedded_chunk_for_a_non_upserted_path_is_rejected(api):
    _index(api)
    reused = api.store.indexes[1]["app.py"]["chunks"][0]

    with pytest.raises(IndexUpdateError, match="not being upserted"):
        validate_index_update(
            _malformed_update(api, chunks=[reused]), manifest=_manifest(api)
        )


def test_incomplete_full_rebuild_upsert_list_is_rejected(api):
    _write(api, "second.py", "def second():\n    return 1\n")
    _index(api)

    with pytest.raises(IndexUpdateError, match="omits snapshot path"):
        validate_index_update(
            _malformed_update(api, upserted=("app.py",), full_rebuild=True),
            manifest=_manifest(api),
        )


def test_duplicate_and_contradictory_path_declarations_are_rejected(api):
    _index(api)

    with pytest.raises(IndexUpdateError, match="duplicates"):
        validate_index_update(
            _malformed_update(api, upserted=("app.py", "app.py")),
            manifest=_manifest(api),
        )
    with pytest.raises(IndexUpdateError, match="both upserted and deleted"):
        validate_index_update(
            _malformed_update(api, upserted=("app.py",), deleted=("app.py",)),
            manifest=_manifest(api),
        )
    with pytest.raises(IndexUpdateError, match="absent from the snapshot"):
        validate_index_update(
            _malformed_update(api, upserted=("ghost.py",)), manifest=_manifest(api)
        )


@pytest.mark.parametrize(
    "overrides,setup",
    [
        ({}, lambda api: _write(api, "app.py", "def value():\n    return 777\n")),
        ({}, lambda api: _write(api, "fresh.py", "def fresh():\n    return 1\n")),
        ({"upserted": ("ghost.py",)}, lambda api: None),
    ],
)
def test_rejected_update_leaves_rows_and_fingerprint_untouched(api, overrides, setup):
    _index(api)
    before_index = {path: dict(entry) for path, entry in api.store.indexes[1].items()}
    before_fingerprint = api.store.fingerprints[1]
    setup(api)

    with pytest.raises(IndexUpdateError):
        api.store.apply_index_update(1, _malformed_update(api, **overrides))

    assert {path: dict(entry) for path, entry in api.store.indexes[1].items()} == before_index
    assert api.store.fingerprints[1] == before_fingerprint


# --------------------------------------------------------------------------
# provenance gates incremental reuse
# --------------------------------------------------------------------------


def test_mismatched_fingerprint_rejects_incremental_reuse(api):
    """Matching content hashes prove a file is unchanged, not that its
    persisted vectors were built under the configuration being stamped."""

    _index(api)
    before_index = {path: dict(entry) for path, entry in api.store.indexes[1].items()}
    fingerprint_a = api.store.fingerprints[1]

    # Byte-identical tree, nothing to reindex - but a different provenance.
    update = _malformed_update(api, fingerprint="b" * 64)

    with pytest.raises(IndexUpdateError, match="requires a persisted fingerprint"):
        api.store.apply_index_update(1, update)

    assert {p: dict(e) for p, e in api.store.indexes[1].items()} == before_index
    assert api.store.fingerprints[1] == fingerprint_a


def test_missing_fingerprint_rejects_incremental_reuse(api):
    _index(api)
    before_index = {path: dict(entry) for path, entry in api.store.indexes[1].items()}
    # Simulate provenance revoked by a legacy mutation helper.
    api.store.fingerprints[1] = None

    update = _malformed_update(api, fingerprint="c" * 64)

    with pytest.raises(IndexUpdateError, match="requires a persisted fingerprint"):
        api.store.apply_index_update(1, update)

    assert {p: dict(e) for p, e in api.store.indexes[1].items()} == before_index
    assert api.store.fingerprints[1] is None


@pytest.mark.parametrize("prior", ["a" * 64, None])
def test_incompatible_fingerprint_still_allows_a_full_rebuild(api, prior):
    _index(api)
    original = api.store.indexes[1]["app.py"]["chunks"][0]
    api.store.fingerprints[1] = prior

    rebuilt = api.store.apply_index_update(
        1,
        _malformed_update(
            api,
            fingerprint="b" * 64,
            upserted=("app.py",),
            chunks=[original],
            full_rebuild=True,
        ),
    )

    # A rebuild discards the prior index, so incompatible provenance is
    # exactly the situation it exists to resolve.
    assert api.store.fingerprints[1] == "b" * 64
    assert rebuilt.file_count == 1 and rebuilt.chunk_count == 1
    assert _paths(api) == {"app.py"}


def test_service_rebuilds_rather_than_hitting_the_provenance_gate(api):
    """The service classifies an incompatible index as a rebuild, so the new
    validator rule never blocks a legitimate run."""

    _index(api)
    api.store.fingerprints[1] = "stale" + "0" * 59
    api.embeddings.embedded_batches.clear()

    body = _index(api)

    assert api.embeddings.embedded_paths == ["app.py"]
    assert body["files_indexed"] == 1
    assert api.store.fingerprints[1] not in (None, "stale" + "0" * 59)


# --------------------------------------------------------------------------
# embedder output contract
# --------------------------------------------------------------------------


def test_missing_embedding_result_does_not_mutate_the_index(api, monkeypatch):
    _index(api)
    before, fingerprint = dict(api.store.indexes[1]), api.store.fingerprints[1]
    _write(api, "late.py", "def late():\n    return 1\n")

    monkeypatch.setattr(api.embeddings, "embed_chunks", lambda chunks: [])

    assert api.client.post("/api/v1/repositories/1/index").status_code == 502
    assert api.store.indexes[1] == before
    assert api.store.fingerprints[1] == fingerprint


def test_substituted_foreign_chunk_does_not_mutate_the_index(api, monkeypatch):
    from repomind.ingestion import CodeChunk
    from repomind.retrieval import EmbeddedChunk

    _index(api)
    before, fingerprint = dict(api.store.indexes[1]), api.store.fingerprints[1]
    _write(api, "late.py", "def late():\n    return 1\n")

    foreign = CodeChunk(
        relative_path="somewhere/else.py",
        language="python",
        start_line=1,
        end_line=1,
        content="def foreign():\n    return 0\n",
        chunk_index=0,
    )
    monkeypatch.setattr(
        api.embeddings,
        "embed_chunks",
        lambda chunks: [EmbeddedChunk(chunk=foreign, embedding=api.embeddings.vector)],
    )

    response = api.client.post("/api/v1/repositories/1/index")
    assert response.status_code == 502
    assert "def foreign" not in response.text
    assert api.store.indexes[1] == before
    assert api.store.fingerprints[1] == fingerprint


def test_model_identity_mismatch_fails_closed(api, monkeypatch):
    from repomind.retrieval import EmbeddedChunk, EmbeddingVector

    _index(api)
    before, fingerprint = dict(api.store.indexes[1]), api.store.fingerprints[1]
    _write(api, "late.py", "def late():\n    return 1\n")

    # The embedder advertises one model for the fingerprint but returns
    # vectors stamped with another.
    contradictory = EmbeddingVector(values=(1.0, 0.0), model="some-other-model")
    monkeypatch.setattr(
        api.embeddings,
        "embed_chunks",
        lambda chunks: [EmbeddedChunk(chunk=c, embedding=contradictory) for c in chunks],
    )

    assert api.client.post("/api/v1/repositories/1/index").status_code == 502
    assert api.store.indexes[1] == before
    assert api.store.fingerprints[1] == fingerprint


def test_reordered_embedding_results_fail_closed(api, monkeypatch):
    _write(api, "b.py", "def b():\n    return 1\n")
    _write(api, "c.py", "def c():\n    return 1\n")
    _index(api)
    before, fingerprint = dict(api.store.indexes[1]), api.store.fingerprints[1]
    _write(api, "b.py", "def b():\n    return 22\n")
    _write(api, "c.py", "def c():\n    return 33\n")

    real = api.embeddings.embed_chunks
    monkeypatch.setattr(
        api.embeddings, "embed_chunks", lambda chunks: list(reversed(real(chunks)))
    )

    assert api.client.post("/api/v1/repositories/1/index").status_code == 502
    assert api.store.indexes[1] == before
    assert api.store.fingerprints[1] == fingerprint


class _Recorder:
    """Minimal trace sink capturing emitted metadata for assertions."""

    def __init__(self) -> None:
        self.events: dict[str, dict] = {}
        self.run_id = None

    def emit(self, event: str, **metadata) -> None:
        self.events[event] = metadata

    def finish(self, status: str) -> None:
        self.events["finished"] = {"status": status}
