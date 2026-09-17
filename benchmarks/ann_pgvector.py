"""Opt-in exact-vs-HNSW PostgreSQL benchmark with rollback-only fixtures."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from repomind.db import (
    create_database_engine,
    persist_embedded_chunks,
    persist_repository_snapshot,
    pgvector_semantic_search,
)
from repomind.db.models import HNSW_INDEX_NAME
from repomind.ingestion import CodeChunk, RepositorySnapshot, SourceFile
from repomind.retrieval import EmbeddedChunk, EmbeddingVector, SemanticSearchMode

DIMENSIONS = 1536
MODEL = "repomind-ann-benchmark-1536"


def _vector(position: float) -> tuple[float, ...]:
    values = [0.0] * DIMENSIONS
    values[0] = math.cos(position)
    values[1] = math.sin(position)
    values[2] = 0.05 * math.cos(position * 7)
    values[3] = 0.05 * math.sin(position * 7)
    return tuple(values)


def _snapshot(size: int) -> RepositorySnapshot:
    content = "".join(f"benchmark_value_{index} = {index}\n" for index in range(size))
    source = SourceFile(
        relative_path=Path("benchmark/vectors.py"),
        language="python",
        content=content,
        size_bytes=len(content.encode()),
        line_count=size,
    )
    return RepositorySnapshot(
        root=Path("C:/isolated/repomind-ann-benchmark"),
        name=f"repomind-ann-benchmark-{uuid4().hex}",
        files=[source],
        skipped=[],
        file_count=1,
        total_size_bytes=source.size_bytes,
        languages={"python": 1},
    )


def _corpus(size: int) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk=CodeChunk(
                relative_path="benchmark/vectors.py",
                language="python",
                start_line=index + 1,
                end_line=index + 1,
                content=f"benchmark_value_{index} = {index}\n",
                chunk_index=index,
            ),
            embedding=EmbeddingVector(values=_vector(index * 0.002), model=MODEL),
        )
        for index in range(size)
    ]


def _time_search(
    session: Session,
    repository_id: int,
    query: EmbeddingVector,
    mode: SemanticSearchMode,
    k: int,
) -> tuple[float, list[int]]:
    started = perf_counter()
    results = pgvector_semantic_search(
        session,
        repository_id,
        query,
        top_k=k,
        mode=mode,
    )
    elapsed_ms = (perf_counter() - started) * 1_000
    return elapsed_ms, [result.chunk.chunk_index for result in results]


def _planner_uses_hnsw(
    session: Session,
    repository_id: int,
    query: EmbeddingVector,
    k: int,
) -> bool:
    vector_literal = "[" + ",".join(str(value) for value in query.values) + "]"
    rows = session.execute(
        text(
            "EXPLAIN (COSTS OFF) SELECT c.id FROM code_chunks c "
            "JOIN repository_files f ON f.id = c.repository_file_id "
            "WHERE f.repository_id = :repository_id AND c.embedding IS NOT NULL "
            "AND c.embedding_model = :model AND c.embedding_dimensions = 1536 "
            "ORDER BY (c.embedding::vector(1536)) <=> CAST(:query AS vector(1536)) "
            "LIMIT :k"
        ),
        {
            "repository_id": repository_id,
            "model": query.model,
            "query": vector_literal,
            "k": k,
        },
    ).scalars()
    return HNSW_INDEX_NAME in "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.ann_pgvector",
        description=__doc__,
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 500, 2_000])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=5)
    args = parser.parse_args()
    if any(size <= 0 for size in args.sizes) or args.k <= 0 or args.queries <= 0:
        parser.error("sizes, k, and queries must be positive")
    database_url = os.environ.get("REPOMIND_TEST_DATABASE_URL")
    if not database_url:
        parser.error("set REPOMIND_TEST_DATABASE_URL to an isolated migrated test database")

    engine = create_database_engine(database_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    try:
        max_size = max(args.sizes)
        snapshot = _snapshot(max_size)
        repository = persist_repository_snapshot(session, snapshot)
        corpus = _corpus(max_size)
        print("Corpus  k   Exact ms  ANN ms  ANN Recall@k  Planner")
        print("------  --  --------  ------  ------------  -------")
        for size in args.sizes:
            persist_embedded_chunks(session, repository.id, corpus[:size])
            exact_times: list[float] = []
            ann_times: list[float] = []
            recalls: list[float] = []
            for query_number in range(args.queries):
                position = ((query_number + 0.37) / args.queries) * (size - 1) * 0.002
                query = EmbeddingVector(values=_vector(position), model=MODEL)
                exact_ms, exact_ids = _time_search(
                    session, repository.id, query, SemanticSearchMode.EXACT, args.k
                )
                ann_ms, ann_ids = _time_search(
                    session, repository.id, query, SemanticSearchMode.ANN, args.k
                )
                exact_times.append(exact_ms)
                ann_times.append(ann_ms)
                reference = set(exact_ids)
                recalls.append(len(reference.intersection(ann_ids)) / len(reference))
            planner = "HNSW" if _planner_uses_hnsw(session, repository.id, query, args.k) else "other"
            print(
                f"{size:6d}  {args.k:2d}  {mean(exact_times):8.3f}  "
                f"{mean(ann_times):6.3f}  {mean(recalls):12.3f}  {planner}"
            )
        print(
            "\nTimings include validation/filtering queries and are local-machine evidence only. "
            "All benchmark rows are rolled back."
        )
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
        engine.dispose()


if __name__ == "__main__":
    main()
