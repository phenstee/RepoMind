"""Optional bounded live smoke check for in-memory semantic search.

Run with::

    uv run python scripts/manual_semantic_search.py \
        "Where is retry logic implemented?" --max-chunks 10

The script exits before ingestion or network access when ``OPENAI_API_KEY`` is
not configured. It never embeds more source chunks than ``--max-chunks``.
"""

import argparse
from pathlib import Path

from repomind.config import get_settings
from repomind.ingestion import (
    RepositoryIngestionError,
    chunk_repository,
    ingest_repository,
)
from repomind.retrieval import (
    EmbeddingError,
    OpenAIEmbeddingClient,
    RetrievalError,
    semantic_search,
)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deliberately small live semantic search.",
    )
    parser.add_argument("query", help="Natural-language repository query")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root to inspect (default: current directory)",
    )
    parser.add_argument(
        "--max-chunks",
        type=_positive_integer,
        default=10,
        help="Maximum source chunks to embed (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.query.strip():
        print("Query must not be empty or whitespace-only. No API request was made.")
        return

    settings = get_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set. No semantic-search API request was made.")
        return

    try:
        snapshot = ingest_repository(args.root)
        chunks = chunk_repository(snapshot)[: args.max_chunks]
        if not chunks:
            print("No source chunks were found. No semantic-search API request was made.")
            return

        client = OpenAIEmbeddingClient(settings)
        embedded_chunks = client.embed_chunks(chunks)
        results = semantic_search(
            args.query,
            embedded_chunks,
            client,
            top_k=min(5, len(embedded_chunks)),
        )
    except (EmbeddingError, RepositoryIngestionError, RetrievalError) as exc:
        print(f"Semantic-search check failed: {exc}")
        return

    print(f"Embedded source chunks: {len(embedded_chunks)}")
    print("The query was embedded once.")
    for result in results:
        chunk = result.chunk
        print(
            f"{result.rank}. {result.score:.4f} "
            f"{chunk.relative_path.as_posix()}:{chunk.start_line}-{chunk.end_line}"
        )


if __name__ == "__main__":
    main()
