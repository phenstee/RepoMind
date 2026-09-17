"""Opt-in real-pgvector Retrieval V2 strategy matrix."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from benchmarks.repo_eval_v2 import VERSION, _hashed_embedding, _source, _suite
from repomind.db import (
    create_database_engine,
    persist_embedded_chunks,
    persist_repository_snapshot,
    pgvector_semantic_search,
    postgres_hybrid_search,
)
from repomind.evaluation import evaluate_retrieval, format_retrieval_comparison
from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    CodeChunk,
    RepositorySnapshot,
    chunk_source_file,
)
from repomind.retrieval import (
    EmbeddedChunk,
    EmbeddingVector,
    RankedChunk,
    SemanticSearchMode,
    chunk_identity,
    semantic_search,
)

_DIMENSIONS = 1536
_MODEL = "repo-eval-v2-hash-1536"


def _vector(text: str) -> EmbeddingVector:
    small = _hashed_embedding(text)
    return EmbeddingVector(
        values=(*small, *([0.0] * (_DIMENSIONS - len(small)))),
        model=_MODEL,
    )


def _embedded(chunks: Sequence[CodeChunk]) -> list[EmbeddedChunk]:
    return [EmbeddedChunk(chunk=chunk, embedding=_vector(chunk.content)) for chunk in chunks]


class _Provider:
    def embed_text(self, text: str) -> EmbeddingVector:
        return _vector(text)


def main() -> None:
    """Run the matrix in one rollback-only database transaction."""

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.repo_eval_v2_postgres",
        description=__doc__,
    )
    parser.parse_args()
    database_url = os.environ.get("REPOMIND_TEST_DATABASE_URL")
    if not database_url:
        raise SystemExit(
            "set REPOMIND_TEST_DATABASE_URL to an isolated migrated PostgreSQL database"
        )

    source = _source()
    line_chunks = chunk_source_file(
        source,
        ChunkingConfig(max_lines_per_chunk=8, overlap_lines=2),
    )
    structural_chunks = chunk_source_file(
        source,
        ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURAL,
            max_lines_per_chunk=20,
            overlap_lines=0,
            max_chars_per_chunk=2_000,
        ),
    )
    line_corpus = _embedded(line_chunks)
    provider = _Provider()

    def line_exact(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return semantic_search(query, line_corpus, provider, top_k=top_k)

    engine = create_database_engine(database_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    try:
        repository_snapshot = persist_repository_snapshot(
            session,
            RepositorySnapshot(
                root=source.relative_path.parent,
                name=f"repo-eval-v2-{uuid4().hex}",
                files=[source],
                skipped=[],
                file_count=1,
                total_size_bytes=source.size_bytes,
                languages={"python": 1},
            ),
        )
        persist_embedded_chunks(session, repository_snapshot.id, _embedded(structural_chunks))

        def database(mode: SemanticSearchMode):
            def retrieve(query: str, *, top_k: int) -> Sequence[RankedChunk]:
                return pgvector_semantic_search(
                    session,
                    repository_snapshot.id,
                    _vector(query),
                    top_k=top_k,
                    mode=mode,
                )

            return retrieve

        def ann_hybrid(query: str, *, top_k: int) -> Sequence[RankedChunk]:
            return postgres_hybrid_search(
                session,
                repository_snapshot.id,
                query,
                _vector(query),
                top_k=top_k,
                semantic_mode=SemanticSearchMode.ANN,
            )

        exact = database(SemanticSearchMode.EXACT)
        ann = database(SemanticSearchMode.ANN)
        reports = [
            evaluate_retrieval(_suite(line_chunks), "line_v1+exact", line_exact, k=3),
            evaluate_retrieval(
                _suite(structural_chunks), "python_ast_v1+exact", exact, k=3
            ),
        ]
        # Force only this rollback-only evaluation to exercise the eligible ANN
        # path despite its intentionally tiny fixture.
        session.execute(text("SET LOCAL enable_seqscan = off"))
        reports.extend(
            [
                evaluate_retrieval(
                    _suite(structural_chunks), "python_ast_v1+ann", ann, k=3
                ),
                evaluate_retrieval(
                    _suite(structural_chunks),
                    "python_ast_v1+ann+bm25_rrf",
                    ann_hybrid,
                    k=3,
                ),
            ]
        )
        recalls = []
        for case in _suite(structural_chunks).cases:
            exact_ids = {
                chunk_identity(result.chunk) for result in exact(case.query, top_k=3)
            }
            ann_ids = {chunk_identity(result.chunk) for result in ann(case.query, top_k=3)}
            recalls.append(len(exact_ids & ann_ids) / len(exact_ids))

        print(f"{VERSION.upper()} POSTGRESQL RETRIEVAL MATRIX")
        print(format_retrieval_comparison(reports))
        print(f"\nANN neighbor Recall@3 against exact: {sum(recalls) / len(recalls):.3f}")
        print(
            "HNSW eligibility is forced only inside this rollback-only evaluation; "
            "production leaves planner settings unchanged."
        )
        print("All benchmark rows are isolated by repository and rolled back.")
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
        engine.dispose()


if __name__ == "__main__":
    main()
