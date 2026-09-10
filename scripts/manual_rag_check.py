"""Optional bounded live smoke check for the basic repository RAG pipeline.

Run with::

    uv run python scripts/manual_rag_check.py \
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
from repomind.llm import LLMError, OpenAILLMClient
from repomind.rag import RAGConfig, RAGError, answer_repository_question
from repomind.retrieval import EmbeddingError, OpenAIEmbeddingClient


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
        description="Run a deliberately small live repository RAG check.",
    )
    parser.add_argument("question", help="Natural-language repository question")
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
    if not args.question.strip():
        print("Question must not be empty or whitespace-only. No API request was made.")
        return

    settings = get_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set. No RAG API request was made.")
        return

    try:
        snapshot = ingest_repository(args.root)
        chunks = chunk_repository(snapshot)[: args.max_chunks]
        if not chunks:
            print("No source chunks were found. No RAG API request was made.")
            return

        embedding_client = OpenAIEmbeddingClient(settings)
        embedded_chunks = embedding_client.embed_chunks(chunks)
        answer = answer_repository_question(
            args.question,
            embedded_chunks,
            embedding_client,
            OpenAILLMClient(settings),
            config=RAGConfig(top_k=min(5, len(embedded_chunks))),
        )
    except (EmbeddingError, LLMError, RAGError, RepositoryIngestionError) as exc:
        print(f"RAG check failed: {exc}")
        return

    print(f"Embedded source chunks: {len(embedded_chunks)}")
    print(f"Insufficient evidence: {answer.insufficient_evidence}")
    print(answer.answer)
    if answer.citations:
        print("Citations:")
        for citation in answer.citations:
            print(
                f"- {citation.relative_path.as_posix()}:"
                f"{citation.start_line}-{citation.end_line}"
            )


if __name__ == "__main__":
    main()
