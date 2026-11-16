"""Tests for chunking-independent span-gold retrieval evaluation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repomind.evaluation.spans import (
    GoldSpan,
    SpanBenchmarkCase,
    SpanBenchmarkSuite,
    SpanEvaluationError,
    SpanTarget,
    chunk_covers,
    evaluate_span_retrieval,
    resolve_suite,
    resolve_target,
)
from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    CodeChunk,
    RepositorySnapshot,
    SourceFile,
    chunk_repository,
)
from repomind.retrieval import BM25Index

SERVICE = '''"""Job service."""


class JobStore:
    @staticmethod
    def helper():
        return 1

    def claim_next(self, worker_id):
        record = self.oldest_queued()
        record.lease_owner = worker_id
        record.status = "running"
        return record


def unrelated():
    return None
'''


def _snapshot() -> RepositorySnapshot:
    source = SourceFile(
        relative_path=Path("src/store.py"),
        language="python",
        content=SERVICE,
        size_bytes=len(SERVICE),
        line_count=len(SERVICE.splitlines()),
    )
    return RepositorySnapshot(
        root=Path("/fixture"),
        name="fixture",
        files=[source],
        skipped=[],
        file_count=1,
        total_size_bytes=source.size_bytes,
        languages={"python": 1},
    )


def _chunk(start: int, end: int, path: str = "src/store.py") -> CodeChunk:
    return CodeChunk(
        relative_path=path,
        language="python",
        start_line=start,
        end_line=end,
        content="x\n" * (end - start + 1),
        chunk_index=start,
    )


def test_symbol_target_resolves_to_its_definition_including_decorators() -> None:
    sources = {"src/store.py": SERVICE}

    method = resolve_target(SpanTarget(path="src/store.py", symbol="JobStore.claim_next"), sources)
    decorated = resolve_target(SpanTarget(path="src/store.py", symbol="JobStore.helper"), sources)

    assert (method.start_line, method.end_line) == (9, 13)
    assert (decorated.start_line, decorated.end_line) == (5, 7)


def test_unknown_gold_fails_loudly_instead_of_scoring_zero() -> None:
    sources = {"src/store.py": SERVICE}

    with pytest.raises(SpanEvaluationError, match="not found"):
        resolve_target(SpanTarget(path="src/store.py", symbol="JobStore.missing"), sources)
    with pytest.raises(SpanEvaluationError, match="not in the corpus"):
        resolve_target(SpanTarget(path="src/gone.py", symbol="x"), sources)
    with pytest.raises(SpanEvaluationError, match="exceed"):
        resolve_target(SpanTarget(path="src/store.py", lines=(1, 999)), sources)


def test_target_needs_exactly_one_locator() -> None:
    with pytest.raises(ValidationError):
        SpanTarget(path="a.py")
    with pytest.raises(ValidationError):
        SpanTarget(path="a.py", symbol="f", lines=(1, 2))
    with pytest.raises(ValidationError):
        SpanTarget(path="a.py", lines=(5, 2))


def test_a_shared_boundary_line_is_not_evidence() -> None:
    span = GoldSpan(relative_path="src/store.py", start_line=10, end_line=40, label="f")

    assert not chunk_covers(_chunk(1, 10), span)
    assert not chunk_covers(_chunk(1, 13), span)
    assert chunk_covers(_chunk(1, 14), span)
    assert chunk_covers(_chunk(30, 90), span)
    assert not chunk_covers(_chunk(10, 40, path="src/other.py"), span)


def test_short_span_must_be_contained_whole() -> None:
    span = GoldSpan(relative_path="src/store.py", start_line=40, end_line=42, label="const")

    assert chunk_covers(_chunk(38, 45), span)
    assert not chunk_covers(_chunk(41, 45), span)


def test_same_suite_scores_line_and_structural_chunking_on_equal_terms() -> None:
    suite = SpanBenchmarkSuite(
        version="spans-test",
        cases=(
            SpanBenchmarkCase(
                id="claim",
                category="behavior",
                query="lease owner worker",
                targets=(SpanTarget(path="src/store.py", symbol="JobStore.claim_next"),),
            ),
        ),
    )
    snapshot = _snapshot()
    gold = resolve_suite(suite, snapshot)

    for strategy in ChunkingStrategy:
        chunks = chunk_repository(snapshot, ChunkingConfig(strategy=strategy))
        report = evaluate_span_retrieval(
            suite, gold, strategy.value, BM25Index.from_chunks(chunks).search, k=3
        )

        assert report.hit_rate == 1.0, strategy
        assert report.mrr == 1.0, strategy
        assert report.mean_target_recall == 1.0, strategy


def test_report_aggregates_and_groups_by_category() -> None:
    suite = SpanBenchmarkSuite(
        version="spans-test",
        cases=(
            SpanBenchmarkCase(
                id="hit-second",
                category="a",
                query="q1",
                targets=(SpanTarget(path="src/store.py", lines=(9, 13)),),
            ),
            SpanBenchmarkCase(
                id="miss",
                category="b",
                query="q2",
                targets=(SpanTarget(path="src/store.py", lines=(16, 17)),),
            ),
        ),
    )
    gold = resolve_suite(suite, _snapshot())

    class _Ranked:
        def __init__(self, chunk: CodeChunk) -> None:
            self.chunk = chunk

    ranked = [_Ranked(_chunk(1, 3)), _Ranked(_chunk(8, 14))]
    report = evaluate_span_retrieval(suite, gold, "fixed", lambda q, *, top_k: ranked, k=2)

    assert report.hit_rate == 0.5
    assert report.mrr == pytest.approx(0.25)
    assert report.case_results[0].first_hit_rank == 2
    assert report.case_results[1].first_hit_rank is None
    assert set(report.by_category()) == {"a", "b"}
    assert report.by_category()["a"].hit_rate == 1.0


def test_duplicate_case_ids_are_rejected() -> None:
    case = SpanBenchmarkCase(
        id="dup", category="a", query="q", targets=(SpanTarget(path="a.py", lines=(1, 1)),)
    )
    with pytest.raises(ValidationError):
        SpanBenchmarkSuite(version="v", cases=(case, case))
