"""Small manual smoke check for repository ingestion."""

import argparse
from collections import Counter

from repomind.ingestion import (
    InvalidRepositoryRootError,
    chunk_repository,
    ingest_repository,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the source files RepoMind would ingest from a repository.",
    )
    parser.add_argument("root", nargs="?", default=".", help="Repository root path")
    parser.add_argument(
        "--chunks",
        action="store_true",
        help="Print deterministic chunking statistics",
    )
    args = parser.parse_args()

    try:
        snapshot = ingest_repository(args.root)
    except InvalidRepositoryRootError as exc:
        print(exc)
        return

    print(f"Repository: {snapshot.name}")
    print(f"Source files: {snapshot.file_count}")
    print(f"Total source bytes: {snapshot.total_size_bytes:,}")
    print()
    for language, count in snapshot.languages.items():
        print(f"{language:<12} {count}")

    if snapshot.skipped:
        print(f"\nSkipped files: {len(snapshot.skipped)}")

    if args.chunks:
        chunks = chunk_repository(snapshot)
        print(f"\nChunks: {len(chunks)}")
        if chunks:
            line_counts = [
                chunk.end_line - chunk.start_line + 1
                for chunk in chunks
            ]
            print(f"Average chunk lines: {sum(line_counts) / len(line_counts):.1f}")
            print(f"Largest chunk: {max(line_counts)} lines")
            print()
            by_language = Counter(chunk.language or "unknown" for chunk in chunks)
            for language, count in sorted(by_language.items()):
                print(f"{language:<12} {count} chunks")
            first = chunks[0]
            print(
                f"\nFirst chunk: {first.relative_path.as_posix()} "
                f"lines {first.start_line}-{first.end_line}"
            )


if __name__ == "__main__":
    main()
