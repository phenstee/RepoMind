"""Run deterministic Retrieval V2 structural-chunking comparisons."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from repomind.evaluation import (
    RetrievalBenchmarkCase,
    RetrievalBenchmarkSuite,
    evaluate_retrieval,
    format_retrieval_comparison,
)
from repomind.ingestion import (
    ChunkingConfig,
    ChunkingStrategy,
    ChunkKind,
    CodeChunk,
    SourceFile,
    chunk_source_file,
)
from repomind.retrieval import (
    BM25Index,
    EmbeddedChunk,
    EmbeddingTextStrategy,
    EmbeddingVector,
    RankedChunk,
    chunk_identity,
    embedding_text_for_chunk,
    hybrid_search,
    semantic_search,
    tokenize_code,
)

VERSION = "repo-eval-v2"
_SOURCE = '''import secrets

MAX_LOGIN_ATTEMPTS = 5


def normalize_credentials(username: str) -> str:
    """Normalize user input before authentication."""
    return username.strip().casefold()


class UserService:
    def login(self, username: str, password: str) -> str:
        normalized = normalize_credentials(username)
        if not normalized or not password:
            raise ValueError("credentials are required")
        return secrets.token_urlsafe(16)

    def logout(self, token: str) -> None:
        if not token:
            raise ValueError("token is required")
        self._revoke(token)

    def _revoke(self, token: str) -> None:
        def audit_message() -> str:
            return f"revoked:{token}"
        print(audit_message())
'''

_CASES = (
    ("exact-login-symbol", "UserService.login", "def login"),
    (
        "behavioral-normalization",
        "Where are credentials normalized before authentication?",
        "def normalize_credentials",
    ),
    ("module-constant", "maximum login attempts", "MAX_LOGIN_ATTEMPTS"),
    ("nested-audit", "Which nested function builds the revocation audit message?", "audit_message"),
)


def _source() -> SourceFile:
    return SourceFile(
        relative_path=Path("src/auth/service.py"),
        language="python",
        content=_SOURCE,
        size_bytes=len(_SOURCE.encode()),
        line_count=len(_SOURCE.splitlines()),
    )


def _hashed_embedding(text: str) -> tuple[float, ...]:
    values = [0.0] * 64
    for token in tokenize_code(text):
        digest = hashlib.sha256(token.encode()).digest()
        slot = int.from_bytes(digest[:2], "big") % len(values)
        values[slot] += 1.0
    return tuple(values)


class _EmbeddingProvider:
    model = "deterministic-hash-v1"

    def embed_text(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(values=_hashed_embedding(text), model=self.model)


def _embedded(
    chunks: Sequence[CodeChunk],
    text_strategy: EmbeddingTextStrategy,
) -> list[EmbeddedChunk]:
    provider = _EmbeddingProvider()
    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=provider.embed_text(embedding_text_for_chunk(chunk, text_strategy)),
        )
        for chunk in chunks
    ]


def _suite(chunks: Sequence[CodeChunk]) -> RetrievalBenchmarkSuite:
    cases = []
    for case_id, query, marker in _CASES:
        relevant = tuple(chunk_identity(chunk) for chunk in chunks if marker in chunk.content)
        if not relevant:
            raise RuntimeError(f"fixture marker {marker!r} is not represented")
        cases.append(
            RetrievalBenchmarkCase(
                id=case_id,
                query=query,
                relevant_chunks=relevant,
            )
        )
    return RetrievalBenchmarkSuite(version=VERSION, cases=tuple(cases))


def _retrievers(
    chunks: Sequence[CodeChunk],
    text_strategy: EmbeddingTextStrategy,
) -> tuple[object, object]:
    provider = _EmbeddingProvider()
    corpus = _embedded(chunks, text_strategy)
    bm25 = BM25Index.from_chunks(chunks)

    def exact(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return semantic_search(query, corpus, provider, top_k=top_k)

    def hybrid(query: str, *, top_k: int) -> Sequence[RankedChunk]:
        return hybrid_search(query, corpus, bm25, provider, top_k=top_k)

    return exact, hybrid


def main() -> None:
    """Print offline quality evidence without model or database access."""

    line_chunks = chunk_source_file(
        _source(),
        ChunkingConfig(max_lines_per_chunk=8, overlap_lines=2),
    )
    structural_chunks = chunk_source_file(
        _source(),
        ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURAL,
            max_lines_per_chunk=20,
            overlap_lines=0,
            max_chars_per_chunk=2_000,
        ),
    )
    line_exact, _ = _retrievers(line_chunks, EmbeddingTextStrategy.RAW_SOURCE)
    structural_exact, structural_hybrid = _retrievers(
        structural_chunks,
        EmbeddingTextStrategy.RAW_SOURCE,
    )
    structural_context, _ = _retrievers(
        structural_chunks,
        EmbeddingTextStrategy.STRUCTURAL_CONTEXT,
    )
    reports = [
        evaluate_retrieval(_suite(line_chunks), "line_v1+exact", line_exact, k=3),
        evaluate_retrieval(
            _suite(structural_chunks),
            "python_ast_v1+exact",
            structural_exact,
            k=3,
        ),
        evaluate_retrieval(
            _suite(structural_chunks),
            "python_ast_v1+exact+bm25_rrf",
            structural_hybrid,
            k=3,
        ),
        evaluate_retrieval(
            _suite(structural_chunks),
            "python_ast_v1+context+exact",
            structural_context,
            k=3,
        ),
    ]
    whole_symbols = {
        chunk.qualified_symbol_name
        for chunk in structural_chunks
        if chunk.chunk_kind in {ChunkKind.FUNCTION, ChunkKind.METHOD}
    }

    print("REPO-EVAL-V2 OFFLINE STRUCTURAL RETRIEVAL")
    print(format_retrieval_comparison(reports))
    print(
        "\nWhole-symbol containment: "
        f"{len(whole_symbols & {'normalize_credentials', 'UserService.login', 'UserService.logout', 'UserService._revoke'})}/4"
    )
    print(
        "These deterministic hashed embeddings isolate chunking/evaluation plumbing; "
        "they are not evidence about OpenAI embedding quality."
    )


if __name__ == "__main__":
    main()
