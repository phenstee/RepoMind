"""Small manual smoke check for repository ingestion."""

import argparse

from repomind.ingestion import InvalidRepositoryRootError, ingest_repository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the source files RepoMind would ingest from a repository.",
    )
    parser.add_argument("root", nargs="?", default=".", help="Repository root path")
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


if __name__ == "__main__":
    main()
